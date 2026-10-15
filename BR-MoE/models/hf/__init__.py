from .mixtral import MixtralBRMoE, MixtralPatch
from .deepseek import DeepSeekMoEBRMoE, DeepSeekV2MoEBRMoE, DeepSeekV3MoEBRMoE, DeepSeekMoEPatch
from .qwen import Qwen15MoEBRMoE, Qwen2MoEBRMoE, Qwen15MoEPatch, Qwen2MoEPatch
from .qwen3 import Qwen3MoEBRMoE, Qwen3MoEPatch
from .qwen3_next import (
    Qwen3NextMoEBRMoE,
    Qwen3NextMoEPatch,
    Qwen3_5MoEBRMoE,
    Qwen3_5MoEPatch,
    Qwen3_6MoEBRMoE,
    Qwen3_6MoEPatch,
)
from .universal_moe import (
    UniversalMoEBRMoE,
    UniversalMoEPatch,
    GLM4MoEBRMoE,
    GLM4MoEPatch,
    OLMoEBRMoE,
    OLMoEPatch,
    PhiMoEBRMoE,
    PhiMoEPatch,
    ensure_unfused_moe_experts,
)
