"""BR-MoE int3 的 vLLM 插件。

安装 (在工作目录跑，会注册 `vllm.general_plugins` entry point)
--------------------------------------------------------------
    pip install -e tools/brmoe_int3_vllm

使用
----
    python -m vllm.entrypoints.openai.api_server \
        --model /mnt/709/data3/home/jianglei/models/brmoe-3bit-vllm \
        --trust-remote-code \
        --quantization brmoe_int3 \
        --enforce-eager            # 第一步先关 CUDA Graph，跑通后再打开

`--trust-remote-code` 是必须的: 该 checkpoint 的 config.json 里
`model_type="deepseek"`，而 vLLM 在 `deepseek_v2.py` 里靠这个值选择
**标准 MHA** 分支（DeepSeek-MoE-16B 不是 MLA）。

`--quantization brmoe_int3` 一般可以省略（vLLM 会从 config.json 的
`quantization_config.quant_method` 自动识别），显式给上更稳。

组件
----
    config.py       BRMoEInt3Config      -> get_quant_method 分发
    moe_method.py   BRMoEInt3MoEMethod   -> create_weights / apply
    kernel.py       对 BR-MoE fused_moe_int3 的包装
"""

__all__ = ["register"]


def _patch_routed_experts_loader() -> None:
    """确保 vLLM 的 RoutedExperts.load_weights 支持直接加载预打包参数 (routed_experts.w13_q 等)。

    在 vLLM 新架构中，AutoWeightsLoader 会把 `mlp.experts.*` 转发给
    `MoERunner.load_weights -> RoutedExperts.load_weights`。而原生
    `RoutedExperts.load_weights` 仅匹配逐专家 `experts.{e}.gate_proj` 或
    3D `experts.gate_up_proj`，遇到预打包的 `routed_experts.w13_q`、`w13_v`、
    `w13_msb` 等直接参数名时会因未匹配而丢弃。这里为其补上直接参数分支。
    """
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    except Exception:
        return

    if getattr(RoutedExperts.load_weights, "_brmoe_direct_patched", False):
        return

    orig_load_weights = RoutedExperts.load_weights

    def _brmoe_routed_experts_load_weights(self, weights):
        remaining = []
        for expert_name, loaded_weight in weights:
            direct_name = expert_name.removeprefix("routed_experts.")
            param = getattr(self, direct_name, None)
            if isinstance(param, torch.nn.Parameter) and direct_name not in (
                "w13_weight",
                "w2_weight",
            ):
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                yield direct_name
            else:
                remaining.append((expert_name, loaded_weight))
        if remaining:
            yield from orig_load_weights(self, remaining)

    _brmoe_routed_experts_load_weights._brmoe_direct_patched = True
    RoutedExperts.load_weights = _brmoe_routed_experts_load_weights


def register() -> None:
    """vLLM `vllm.general_plugins` entry point 的入口。

    只做一件事: 导入 config 模块，触发 `@register_quantization_config("brmoe_int3")`，
    并打上 RoutedExperts 预打包参数加载补丁。
    该方法可能在同一进程里被多次调用，导入本身是幂等的。
    """
    from . import config as _config  # noqa: F401

    _patch_routed_experts_loader()
    assert _config.BRMoEInt3Config.get_name() == "brmoe_int3"
