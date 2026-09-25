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


def register() -> None:
    """vLLM `vllm.general_plugins` entry point 的入口。

    只做一件事: 导入 config 模块，触发 `@register_quantization_config("brmoe_int3")`。
    该方法可能在同一进程里被多次调用，导入本身是幂等的。
    """
    from . import config as _config  # noqa: F401

    assert _config.BRMoEInt3Config.get_name() == "brmoe_int3"
