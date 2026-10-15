from .base import BaseBRMoEHFModel
from .universal_moe import UniversalMoEPatch


class DeepSeekMoEPatch(UniversalMoEPatch):
    """Patcher for DeepSeek-MoE-16B, DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3, and DeepSeek-R1."""

    DEFAULT_NUM_LAYERS = 28
    DEFAULT_NUM_EXPERTS = 64
    HAS_SHARED_EXPERT = True
    SHARED_EXPERT_ATTR = "shared_experts"
    MOE_BLOCK_ATTR = "mlp"
    EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

    @classmethod
    def get_leaf_linear_tags(cls, model=None):
        if model is not None:
            return super().get_leaf_linear_tags(model=model)

        leaf_tags = []
        num_layers = cls.DEFAULT_NUM_LAYERS
        for i in range(num_layers):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
                leaf_tags.append(f"model.layers.{i}.self_attn.{proj}")
            if i == 0:
                leaf_tags.append(f"model.layers.{i}.mlp.gate_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.up_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.down_proj")
            else:
                for k in range(cls.DEFAULT_NUM_EXPERTS):
                    leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.gate_proj")
                    leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.up_proj")
                    leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.down_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.shared_experts.gate_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.shared_experts.up_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.shared_experts.down_proj")
        return leaf_tags


class DeepSeekMoEBRMoE(DeepSeekMoEPatch, BaseBRMoEHFModel):
    pass


DeepSeekV2MoEBRMoE = DeepSeekMoEBRMoE
DeepSeekV3MoEBRMoE = DeepSeekMoEBRMoE
