from .base import BaseBRMoEHFModel
from .universal_moe import UniversalMoEPatch


class MixtralPatch(UniversalMoEPatch):
    """Patcher for Mixtral-8x7B and Mixtral-8x22B."""

    DEFAULT_NUM_LAYERS = 32
    DEFAULT_NUM_EXPERTS = 8
    HAS_SHARED_EXPERT = False
    MOE_BLOCK_ATTR = "block_sparse_moe"
    EXPERT_PROJ_NAMES = ("w1", "w2", "w3")


class MixtralBRMoE(MixtralPatch, BaseBRMoEHFModel):
    pass
