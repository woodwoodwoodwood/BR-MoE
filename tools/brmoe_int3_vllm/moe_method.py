"""BR-MoE int3 的 vLLM MoE 量化方法。

实现要点
--------
* 只接管 **RoutedExperts**（专家层）；全 INT3 模型的 attention / dense /
  shared_experts 由 `BRMoEInt3LinearMethod` 处理（见 config.get_quant_method）。
* 走 **非 monolithic** 路径：`is_monolithic=False`（`moe_kernel` 保持 None），
  vLLM 的 runner 会用自带 router 算出 `topk_weights/topk_ids` 再调 `apply` ——
  正好匹配 `fused_moe_int3` 的签名（它把路由当**入参**，不做路由）。
* 加载后准备 Triton K-major 与 CUDA Marlin 重排布局，decode 不重复打包。
* 共享专家的执行、stream 同步与输出相加由 vLLM runner 负责；这里仅返回 routed
  结果。runner 根据配置选择辅助 stream 重叠或 `NO_OVERLAP`。
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
        self.bits = quant_config.bits

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
        lspec = self.quant_config.get_moe_layer_spec(getattr(layer, "layer_name", ""))
        gs = int(lspec.get("group_size", self.group_size))
        total_expert_bits = int(lspec.get("total_expert_bits", 0))
        default_b = self.bits if isinstance(self.bits, int) else 3
        w13_bits = int(lspec.get("w13_bits") or default_b)
        w2_bits = int(lspec.get("w2_bits") or default_b)
        w13_rank = int(lspec.get("w13_rank") or 0)
        w2_rank = int(lspec.get("w2_rank") or 0)
        num_4bit = int(lspec.get("num_4bit_experts", 0))
        total_comp_rank = int(lspec.get("total_comp_rank", 0))
        max_comp_rank = int(lspec.get("max_comp_rank", 0))

        layer.brmoe_group_size = gs
        layer.brmoe_total_expert_bits = total_expert_bits
        layer.brmoe_w13_bits = w13_bits
        layer.brmoe_w2_bits = w2_bits
        layer.brmoe_w13_rank = w13_rank
        layer.brmoe_w2_rank = w2_rank
        layer.brmoe_num_4bit_experts = num_4bit
        layer.brmoe_total_comp_rank = total_comp_rank
        layer.brmoe_max_comp_rank = max_comp_rank

        for name, dim, mod in (("K", K, 32), ("I", I, 32),
                               ("K", K, gs), ("I", I, gs)):
            if dim % mod:
                raise ValueError(f"{name}={dim} 不是 {mod} 的整数倍，int3 打包/分组不成立")

        # 转换器输出的张量布局: 未转置 (w_transposed=False)
        # * Per-3D-Tensor 模式 (total_expert_bits == 0, w13_bits/w2_bits in {2, 3, 4}):
        #   w13_q 为 [E, 2I, (K // 32) * w13_bits], w2_q 为 [E, K, (I // 32) * w2_bits], 单一静态大 Kernel 发射!
        # * Per-Expert 异构兜底模式 (total_expert_bits > 0):
        #   w13_q 为 [total_expert_bits, 2I, K//32], w2_q 为 [total_expert_bits, K, I//32] + bit_offsets [E+1]
        if total_expert_bits > 0:
            specs = {
                "w13_q": ((total_expert_bits, two_i, K // 32), torch.int32),
                "w13_s": ((E, K // gs, two_i), torch.float16),
                "w13_z": ((E, K // gs, two_i), torch.float16),
                "w2_q": ((total_expert_bits, K, I // 32), torch.int32),
                "w2_s": ((E, I // gs, K), torch.float16),
                "w2_z": ((E, I // gs, K), torch.float16),
                "bit_offsets": ((E + 1,), torch.int32),
            }
        else:
            specs = {
                "w13_q": ((E, two_i, (K // 32) * w13_bits), torch.int32),
                "w13_s": ((E, K // gs, two_i), torch.float16),
                "w13_z": ((E, K // gs, two_i), torch.float16),
                "w2_q": ((E, K, (I // 32) * w2_bits), torch.int32),
                "w2_s": ((E, I // gs, K), torch.float16),
                "w2_z": ((E, I // gs, K), torch.float16),
            }
        if w13_rank > 0:
            specs["w1_v"] = ((E, w13_rank, K), torch.float16)
            specs["w1_u"] = ((E, I, w13_rank), torch.float16)
            specs["w3_v"] = ((E, w13_rank, K), torch.float16)
            specs["w3_u"] = ((E, I, w13_rank), torch.float16)
        if w2_rank > 0:
            specs["w2_v"] = ((E, w2_rank, I), torch.float16)
            specs["w2_u"] = ((E, K, w2_rank), torch.float16)
        if total_comp_rank > 0 and w13_rank == 0 and w2_rank == 0:
            specs["comp_offsets"] = ((E + 1,), torch.int32)
            specs["w13_v"] = ((2, total_comp_rank, K), torch.float16)
            specs["w13_u"] = ((2, total_comp_rank, I), torch.float16)
            specs["w2_v"] = ((total_comp_rank, I), torch.float16)
            specs["w2_u"] = ((total_comp_rank, K), torch.float16)

        attrs = dict(extra_weight_attrs)
        attrs["weight_loader"] = default_weight_loader

        for name, (shape, dtype) in specs.items():
            p = Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)

        logger.info_once(
            "brmoe_int3: E=%d, K=%d, I=%d, group_size=%d, total_bits=%d, total_rank=%d  "
            "w13_q%s w2_q%s",
            E, K, I, gs, total_expert_bits or (3 * E), total_comp_rank,
            tuple(specs["w13_q"][0]), tuple(specs["w2_q"][0]),
        )

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)
        base_names = ["w13_q", "w13_s", "w13_z", "w2_q", "w2_s", "w2_z"]
        opt_names = [
            "bit_offsets", "msb_map", "w13_msb", "w2_msb",
            "comp_offsets", "w13_v", "w13_u",
            "w1_v", "w1_u", "w3_v", "w3_u",
            "w2_v", "w2_u",
        ]
        for name in base_names + opt_names:
            p = getattr(layer, name, None)
            if p is not None:
                p.data = p.data.contiguous()

        has_varbit = getattr(layer, "bit_offsets", None) is not None
        has_msb = getattr(layer, "w13_msb", None) is not None
        has_comp = (
            getattr(layer, "w13_v", None) is not None
            or getattr(layer, "w1_v", None) is not None
            or getattr(layer, "w2_v", None) is not None
        )
        w13_bits = getattr(layer, "brmoe_w13_bits", 3)
        w2_bits = getattr(layer, "brmoe_w2_bits", 3)
        gs = getattr(layer, "brmoe_group_size", self.group_size)

        if not has_varbit and not has_msb and not has_comp and w13_bits == 3 and w2_bits == 3:
            from .kernel import build_cuda_packed
            layer.brmoe_cuda_packed = build_cuda_packed(layer, gs)
        else:
            layer.brmoe_cuda_packed = None

        logger.info_once(
            "brmoe_int3: CUDA (Marlin 布局) 副本 %s",
            "已构建" if layer.brmoe_cuda_packed is not None
            else "未构建 (含混合精度/低秩补偿或扩展不可用), MoE 走 Triton")

        # 转置成 K-major ([E, Kpack, N]): GEMV 与 grouped GEMM 的 w 载入沿 n 连续 -> 合并访存
        layer.w13_q.data = layer.w13_q.data.permute(0, 2, 1).contiguous()
        layer.w2_q.data = layer.w2_q.data.permute(0, 2, 1).contiguous()
        if has_msb:
            layer.w13_msb.data = layer.w13_msb.data.permute(0, 2, 1).contiguous()
            layer.w2_msb.data = layer.w2_msb.data.permute(0, 2, 1).contiguous()
        if has_comp and getattr(layer, "comp_offsets", None) is not None and getattr(layer, "brmoe_max_comp_rank", 0) <= 0:
            offs = layer.comp_offsets.data
            layer.brmoe_max_comp_rank = int((offs[1:] - offs[:-1]).max().item())
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
        gs = getattr(layer, "brmoe_group_size", self.group_size)
        out = brmoe_int3_moe(
            x2, topk_weights, topk_ids, layer, gs, out_dtype=x.dtype
        )
        if x.dim() != 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out
