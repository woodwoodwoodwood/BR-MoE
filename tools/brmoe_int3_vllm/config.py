"""BR-MoE int3 的 vLLM 量化配置。

只量化 **专家层**；attention / dense / shared_experts 在转换阶段已反量化成 fp16，
所以本配置对非 RoutedExperts 层返回 None，让 vLLM 回退到 `UnquantizedLinearMethod`。

    LinearBase.__init__:
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            self.quant_method = quant_method
        ...
        else:  # 取名失败 / 返回 None
            self.quant_method = UnquantizedLinearMethod()
"""

from typing import Any

import torch

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


@register_quantization_config("brmoe_int3")
class BRMoEInt3Config(QuantizationConfig):
    """BR-MoE 的 3-bit (真 int3, 3.0 bpw) 分组量化配置。

    对应 checkpoint 里 `config.json` 的:
        "quantization_config": {
            "quant_method": "brmoe_int3", "bits": 3, "group_size": 64,
            "dense_mode": "fp16", ...
        }
    """

    def __init__(
        self,
        bits: int = 3,
        group_size: int = 64,
        dense_mode: str = "fp16",
        full_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if bits != 3:
            raise ValueError(
                f"brmoe_int3 只支持 bits=3，收到 bits={bits}。"
                " (4-bit 请改用 vLLM 内置的 moe_wna16 / gptq_marlin)"
            )
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.dense_mode = dense_mode
        self.full_config = full_config or {}

    # ------------------------------------------------------------------
    # QuantizationConfig 抽象接口
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "brmoe_int3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # kernel 的反量化路径固定产出 fp16
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton int3 grouped kernel 以 sm_75 为下限实测通过
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        # 量化参数直接写在 config.json 的 quantization_config 里，无需额外文件
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BRMoEInt3Config":
        return cls(
            bits=int(config.get("bits", 3)),
            group_size=cls.get_from_keys(config, ["group_size"]),
            dense_mode=config.get("dense_mode", "fp16"),
            full_config=config,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # 延迟导入，避免插件加载期就拉起整个 MoE 栈
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            from .moe_method import BRMoEInt3MoEMethod

            return BRMoEInt3MoEMethod(self, layer.moe_config)

        # 其余层的权重已在转换时反量化成 fp16，按"未量化"处理。
        #
        # 各调用点对 None 的态度**不一致**，必须分流:
        #
        #   LinearBase (linear.py:281)
        #       if quant_config is None:  -> UnquantizedLinearMethod()
        #       elif quant_method := ...: -> 用返回值
        #       else:                     -> raise ValueError(
        #             "All linear layers should support quant method.")
        #       => 返回 None 会**抛异常**，必须显式给 UnquantizedLinearMethod
        #
        #   VocabParallelEmbedding (vocab_parallel_embedding.py:293)
        #       if quant_method is None: -> UnquantizedEmbeddingMethod()
        #       => 返回 None 是**正确**的；若返回 Linear 方法，
        #          会因 "must implement the 'embedding' method" 抛异常
        #
        #   ParallelLMHead / GateLinear 等: 同 Embedding，None 会被兜住
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            # dense_mode="int3": 非专家线性层 (attention / dense MLP /
            # shared_experts) 在转换阶段被保留成 int3 打包权重 (qweight/scales/
            # zeros)，交给 BRMoEInt3LinearMethod 走 int3_moe_gemm (E=1, W_T=True)。
            # dense_mode="fp16": 这些层已被反量化成 fp16，按未量化处理。
            if getattr(self, "dense_mode", "fp16") == "int3":
                from .linear_method import BRMoEInt3LinearMethod
                return BRMoEInt3LinearMethod(self)
            return UnquantizedLinearMethod()
        return None

    def __repr__(self) -> str:
        return (f"BRMoEInt3Config(bits={self.bits}, group_size={self.group_size}, "
                f"dense_mode={self.dense_mode!r})")
