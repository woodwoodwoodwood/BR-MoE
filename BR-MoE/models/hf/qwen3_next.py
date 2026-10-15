"""Qwen3.6 / Qwen3.5-MoE / Qwen3-Next Adapter for BR-MoE.

Covers:
- `Qwen/Qwen3.6-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM`)
- `Qwen/Qwen3.5-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM`)
- `Qwen/Qwen3-Next-80B-A3B` (`Qwen3NextForCausalLM`)

Architecture highlights handled automatically:
1. Hybrid attention stack (`config.layer_types`):
   - `"full_attention"` layers with `self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}`
   - `"linear_attention"` layers (Gated DeltaNet) with
     `linear_attn.{in_proj_qkvz, in_proj_ba, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, conv1d, norm, dt_bias, A_log}`
2. Multimodal ConditionalGeneration wrapper (`model.language_model.layers.{i}`) as well as
   text-only CausalLM (`model.layers.{i}`), with automatic cross-prefix matching.
3. 3D fused expert weights (`experts.gate_up_proj`, `experts.down_proj`) unpacked on-the-fly
   into per-expert `gate_proj`, `up_proj`, `down_proj` via `ensure_unfused_moe_experts`.
4. Shared expert (`mlp.shared_expert.{gate_proj, up_proj, down_proj}`) + `mlp.shared_expert_gate`.
"""

from .base import BaseBRMoEHFModel
from .universal_moe import UniversalMoEPatch


class Qwen3NextMoEPatch(UniversalMoEPatch):
    DEFAULT_NUM_LAYERS = 40
    DEFAULT_NUM_EXPERTS = 256
    HAS_SHARED_EXPERT = True
    SHARED_EXPERT_ATTR = "shared_expert"
    MOE_BLOCK_ATTR = "mlp"
    EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class Qwen3NextMoEBRMoE(Qwen3NextMoEPatch, BaseBRMoEHFModel):
    pass


# Aliases for Qwen3.5-MoE and Qwen3.6-MoE (e.g., Qwen/Qwen3.6-35B-A3B)
Qwen3_5MoEPatch = Qwen3NextMoEPatch
Qwen3_5MoEBRMoE = Qwen3NextMoEBRMoE
Qwen3_6MoEPatch = Qwen3NextMoEPatch
Qwen3_6MoEBRMoE = Qwen3NextMoEBRMoE
