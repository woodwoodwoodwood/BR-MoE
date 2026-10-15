from .base import BaseBRMoEHFModel
from .universal_moe import UniversalMoEPatch


class Qwen15MoEPatch(UniversalMoEPatch):
    """Patcher for Qwen1.5-MoE (e.g. Qwen1.5-MoE-A2.7B) and Qwen2-MoE (e.g. Qwen2-57B-A14B)."""

    DEFAULT_NUM_LAYERS = 24
    DEFAULT_NUM_EXPERTS = 60
    HAS_SHARED_EXPERT = True
    SHARED_EXPERT_ATTR = "shared_expert"
    MOE_BLOCK_ATTR = "mlp"
    EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class Qwen15MoEBRMoE(Qwen15MoEPatch, BaseBRMoEHFModel):
    pass


Qwen2MoEPatch = Qwen15MoEPatch
Qwen2MoEBRMoE = Qwen15MoEBRMoE
