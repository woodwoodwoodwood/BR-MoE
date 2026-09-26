"""BR-MoE int3 的 vLLM MoE 量化方法。

实现要点
--------
* 只接管 **RoutedExperts**（专家层）；attention / dense / shared_experts 在转换阶段
  已反量化成 fp16，交给 vLLM 的 `UnquantizedLinearMethod` 处理（见 config.get_quant_method）。
* 走 **非 monolithic** 路径：`is_monolithic=False`（`moe_kernel` 保持 None），
  vLLM 的 runner 会用自带 router 算出 `topk_weights/topk_ids` 再调 `apply` ——
  正好匹配 `fused_moe_int3` 的签名（它把路由当**入参**，不做路由）。
* 权重张量直接用转换器产出的布局，**不需要 `process_weights_after_loading` 重排**。
* 共享专家由 runner 以 `NO_OVERLAP` 方式跑好并负责加回输出，这里忽略即可
  （`moe_runner.py::_maybe_apply_shared_experts` + `_apply_quant_method` 的返回值处理）。
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .kernel import brmoe_int3_moe

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)


@CustomOp.register("brmoe_int3_moe")
class BRMoEInt3MoEMethod(FusedMoEMethodBase, CustomOp):
    """按 BR-MoE 的 grouped int3 kernel 执行 MoE。

    Args:
        moe: vLLM 的 MoE 配置（专家数、hidden/intermediate、并行策略等）。
        quant_config: `BRMoEInt3Config`，携带 bits / group_size。
    """

    def __init__(self, quant_config, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = int(quant_config.group_size)
        self.bits = int(quant_config.bits)

        tp_size = moe.moe_parallel_config.tp_size
        if tp_size > 1:
            # 转换器没做张量并行切分；TP>1 需要在 create_weights 里按 tp_rank 切
            # w13（列并行）/ w2（行并行），并重打包 int3（切分边界要落在 32 的倍数上）。
            raise NotImplementedError(
                f"brmoe_int3 目前只支持 tp_size=1，当前 tp_size={tp_size}。"
            )
        if not moe.is_act_and_mul:
            raise NotImplementedError("brmoe_int3 假定 MoE 使用 act_and_mul (gate||up)。")

    # ------------------------------------------------------------------
    # 权重
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        E = int(num_experts)
        K = int(hidden_size)
        I = int(intermediate_size_per_partition)
        two_i = 2 * I
        gs = self.group_size

        for name, dim, mod in (("K", K, 32), ("I", I, 32),
                               ("K", K, gs), ("I", I, gs)):
            if dim % mod:
                raise ValueError(f"{name}={dim} 不是 {mod} 的整数倍，int3 打包/分组不成立")

        # 转换器输出的张量布局: 未转置 (w_transposed=False)
        specs = {
            "w13_q": ((E, two_i, K // 32 * 3), torch.int32),
            "w13_s": ((E, K // gs, two_i), torch.float16),
            "w13_z": ((E, K // gs, two_i), torch.float16),
            "w2_q": ((E, K, I // 32 * 3), torch.int32),
            "w2_s": ((E, I // gs, K), torch.float16),
            "w2_z": ((E, I // gs, K), torch.float16),
        }

        # checkpoint 里专家权重**已经是 fused 的** (`experts.w13_q`)，不是
        # vLLM 惯用的 `experts.{i}.gate_proj.weight` 逐专家形式。因此显式覆盖成
        # default_weight_loader（逐张量同名拷贝），避免 RoutedExperts.weight_loader
        # 的 shard_id/w1-w2-w3 融合逻辑把数据搬错位置。
        attrs = dict(extra_weight_attrs)
        attrs["weight_loader"] = default_weight_loader

        for name, (shape, dtype) in specs.items():
            p = Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)

        logger.info_once(
            "brmoe_int3: E=%d, K=%d, I=%d, group_size=%d  "
            "w13_q%s w2_q%s",
            E, K, I, gs, tuple(specs["w13_q"][0]), tuple(specs["w2_q"][0]),
        )

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)
        for name in ("w13_q", "w13_s", "w13_z", "w2_q", "w2_s", "w2_z"):
            p = getattr(layer, name)
            p.data = p.data.contiguous()
        # CUDA (Marlin 布局) 副本: 供 M 中间区间的 tile 级融合 kernel。
        # 必须先于 K-major 转置 (repack 的输入契约是 N-major 原件)。
        # 注意: 数值尚未校验 (verify_moe_cuda.py 未过), 仅用于性能测量。
        from .kernel import build_cuda_packed
        layer.brmoe_cuda_packed = build_cuda_packed(layer, self.group_size)
        logger.info_once(
            "brmoe_int3: CUDA (Marlin 布局) 副本 %s",
            "已构建" if layer.brmoe_cuda_packed is not None
            else "未构建 (扩展不可用), MoE 全程走 Triton")

        # 转置成 K-major ([E, Kpack, N]): GEMV 的 w 载入沿 n 连续 -> 合并访存。
        # A100 实测 (bench/micro_moe.py --gemv-sweep, job 38912/38913):
        #   GEMV 路径 +5~19%; TC 路径 +5~11% (M=2/4/8)。scale/zero 两种布局同形,
        #   不用动。kernel 侧 (GEMV 与 grouped GEMM) 都有 W_T 分支。
        layer.w13_q.data = layer.w13_q.data.permute(0, 2, 1).contiguous()
        layer.w2_q.data = layer.w2_q.data.permute(0, 2, 1).contiguous()
        layer.w_transposed = True

    # ------------------------------------------------------------------
    # 量化配置 (非 MK 路径下仅作信息传递)
    # ------------------------------------------------------------------

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        # 权重本身是打包的 int32，scale/zero 的布局也是自定义的 per-(E, K//gs, N)，
        # 与 FusedMoEQuantConfig 的标准语义（w1_scale/w2_scale 用于模块化 kernel）
        # 不一致。我们的 kernel 直接从 layer 参数取这些东西，所以这里只报存储类型。
        return FusedMoEQuantConfig.make(weight_dtype=torch.int32)

    @property
    def supports_eplb(self) -> bool:
        # EPLB 会重排专家权重的物理位置，需要额外的映射处理
        return False

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 注意: 非重叠路径下 runner 已经把共享专家算完并在外面加回
        # (moe_runner._maybe_apply_shared_experts(NO_OVERLAP))，这里不要再算一遍。
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    def forward_native(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self._run(layer, x, topk_weights, topk_ids)

    def forward_cuda(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self._run(layer, x, topk_weights, topk_ids)

    def _run(self, layer, x, topk_weights, topk_ids) -> torch.Tensor:
        # 输入可能是 3D (T, 1, K) 或多维，kernel 只吃 2D
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]) if x.dim() != 2 else x
        out = brmoe_int3_moe(
            x2, topk_weights, topk_ids, layer, self.group_size, out_dtype=x.dtype
        )
        if x.dim() != 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out
