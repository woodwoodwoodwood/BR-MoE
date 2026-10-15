from .base import BaseBRMoEHFModel
from .universal_moe import UniversalMoEPatch


class Qwen3MoEPatch(UniversalMoEPatch):
    """Patcher for Qwen3-MoE (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B) and Qwen3.5/3.6-MoE."""

    DEFAULT_NUM_LAYERS = 48
    DEFAULT_NUM_EXPERTS = 128
    HAS_SHARED_EXPERT = False
    SHARED_EXPERT_ATTR = "shared_expert"
    MOE_BLOCK_ATTR = "mlp"
    EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class Qwen3MoEBRMoE(Qwen3MoEPatch, BaseBRMoEHFModel):
    pass
