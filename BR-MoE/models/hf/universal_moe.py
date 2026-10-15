"""Universal MoE Model Adapter for BR-MoE.

Supports dynamic introspection and patching for modern HuggingFace MoE models:
- Qwen3.6-35B-A3B / Qwen3.5-35B-A3B (Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM)
- Qwen3-Next (Qwen3NextForCausalLM)
- Qwen3-MoE (Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3MoeForCausalLM)
- Qwen1.5-MoE / Qwen2-MoE (Qwen2MoeForCausalLM)
- DeepSeek-MoE / DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1 (DeepseekForCausalLM, DeepseekV2ForCausalLM, DeepseekV3ForCausalLM)
- GLM-4.5 / 4.6 / 4.7 MoE (Glm4MoeForCausalLM)
- OLMoE (OlmoeForCausalLM)
- Phi-MoE (PhimoeForCausalLM)
- Mixtral (MixtralForCausalLM)
- Llama-4 MoE, Ernie-4.5-MoE, Bailing-MoE, MiniMax-MoE, etc.

Key features:
1. Automatic detection of text decoder layers across CausalLM (`model.layers`) and
   Multimodal/VLM wrappers (`model.language_model.layers`, `language_model.model.layers`).
2. Automatic unpacking of 3D fused expert parameters (`experts.gate_up_proj`, `experts.down_proj`)
   into per-expert `nn.Linear` modules (`gate_proj`, `up_proj`, `down_proj`) while preserving
   both `ModuleList` indexing (`experts[k](x)`) and fused call (`experts(x, topk_ids, topk_weights)`).
3. Full support for hybrid attention architectures (standard `self_attn`, DeepSeek MLA,
   and Qwen3.5/3.6/Next Gated DeltaNet `linear_attn`).
4. Cross-prefix config lookup so `model.layers.{i}...` keys in `compress_config` seamlessly
   match `model.language_model.layers.{i}...` in VLM checkpoints.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..base import BasePatch, _QUANT_LAYERS
from .base import BaseBRMoEHFModel


# ---------------------------------------------------------------------------
# 1. 3D Fused Experts Unpacking (for Qwen3.5-MoE / Qwen3.6-35B-A3B / Llama4 / HF v5)
# ---------------------------------------------------------------------------

class UnfusedExpertMLP(nn.Module):
    """Per-expert SwiGLU MLP unpacked from 3D fused expert parameters."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        act_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Unfused3DExpertsModule(nn.ModuleList):
    """Drop-in replacement for 3D-parameter HF `Experts` modules (e.g. `Qwen3_5MoeTextExperts`).

    Inherits from `nn.ModuleList` so:
    - `len(experts)` and `experts[k](x)` work identically to standard `ModuleList` experts.
    - `experts.{k}.{gate_proj, up_proj, down_proj}` appear in `model.named_modules()`,
      enabling per-expert mixed-bit / mixed-rank quantization (`BRMoELinear`) and RCO search.
    - Calling `experts(hidden_states, top_k_index, top_k_weights)` directly (as done in
      HF Transformers v4.57+/v5 `Qwen3_5MoeSparseMoeBlock` / `Qwen3VLMoeSparseMoeBlock`)
      also works transparently.
    """

    def __init__(
        self,
        experts: List[nn.Module],
        is_transposed: bool = False,
    ) -> None:
        super().__init__(experts)
        self.is_transposed = is_transposed
        self.num_experts = len(experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: Optional[torch.Tensor] = None,
        top_k_weights: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Support kwargs naming variants across transformers versions
        if top_k_index is None:
            top_k_index = kwargs.get(
                "top_k_index",
                kwargs.get("selected_experts", kwargs.get("router_indices", None)),
            )
        if top_k_weights is None:
            top_k_weights = kwargs.get(
                "top_k_weights",
                kwargs.get("routing_weights", kwargs.get("topk_weights", None)),
            )

        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        final_hidden = torch.zeros_like(flat_hidden)

        if top_k_index is None or top_k_weights is None:
            raise ValueError(
                "Unfused3DExpertsModule.forward requires (hidden_states, top_k_index, top_k_weights)"
            )

        # Case A: top_k_index is [num_tokens, top_k] and top_k_weights is [num_tokens, top_k]
        # or [num_tokens, num_experts]
        if top_k_index.dim() == 2 and top_k_index.shape[0] == flat_hidden.shape[0]:
            expert_mask = F.one_hot(
                top_k_index.long(), num_classes=self.num_experts
            ).permute(2, 1, 0)  # [E, top_k, T]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for idx_tensor in expert_hit:
                expert_idx = int(idx_tensor[0].item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = flat_hidden[token_idx]
                expert_out = self[expert_idx](current_state)
                if top_k_weights.shape == top_k_index.shape:
                    w = top_k_weights[token_idx, top_k_pos, None]
                else:
                    w = top_k_weights[token_idx, expert_idx, None]
                final_hidden.index_add_(
                    0, token_idx, (expert_out * w.to(expert_out.dtype)).to(final_hidden.dtype)
                )
        # Case B: top_k_index is already one-hot expert_mask [E, top_k, T]
        elif top_k_index.dim() == 3 and top_k_index.shape[0] == self.num_experts:
            expert_mask = top_k_index
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for idx_tensor in expert_hit:
                expert_idx = int(idx_tensor[0].item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = flat_hidden[token_idx]
                expert_out = self[expert_idx](current_state)
                w = top_k_weights[token_idx, top_k_pos, None]
                final_hidden.index_add_(
                    0, token_idx, (expert_out * w.to(expert_out.dtype)).to(final_hidden.dtype)
                )
        else:
            # Fallback: dense routing weights [T, E]
            for expert_idx in range(self.num_experts):
                token_idx = torch.nonzero(top_k_weights[:, expert_idx] > 0, as_tuple=False).squeeze(-1)
                if token_idx.numel() == 0:
                    continue
                current_state = flat_hidden[token_idx]
                expert_out = self[expert_idx](current_state)
                w = top_k_weights[token_idx, expert_idx, None]
                final_hidden.index_add_(
                    0, token_idx, (expert_out * w.to(expert_out.dtype)).to(final_hidden.dtype)
                )

        return final_hidden.view(orig_shape)


def _is_3d_fused_experts_module(module: nn.Module) -> bool:
    """Detect whether an `experts` module stores 3D fused expert tensors."""
    if isinstance(module, nn.ModuleList):
        return False
    has_gate_up = hasattr(module, "gate_up_proj") and isinstance(
        getattr(module, "gate_up_proj"), (nn.Parameter, torch.Tensor)
    )
    has_down = hasattr(module, "down_proj") and isinstance(
        getattr(module, "down_proj"), (nn.Parameter, torch.Tensor)
    )
    if has_gate_up and has_down:
        return getattr(module, "gate_up_proj").ndim == 3
    return False


@torch.no_grad()
def unpack_single_3d_experts_module(experts_module: nn.Module) -> Unfused3DExpertsModule:
    """Convert a 3D fused `experts` module into an `Unfused3DExpertsModule` of `nn.Linear`s."""
    gate_up: torch.Tensor = experts_module.gate_up_proj.data
    down: torch.Tensor = experts_module.down_proj.data
    num_experts = gate_up.shape[0]

    # Determine whether weights are transposed:
    # Standard (non-transposed): gate_up is [E, 2*I, H], down is [E, H, I]
    # Transposed: gate_up is [E, H, 2*I], down is [E, I, H]
    is_transposed = getattr(experts_module, "is_transposed", None)
    if is_transposed is None:
        # In gate_up and down, H appears once in each and I vs 2*I differs by factor of 2.
        if gate_up.shape[1] == 2 * down.shape[2] and gate_up.shape[2] == down.shape[1]:
            is_transposed = False
        elif gate_up.shape[2] == 2 * down.shape[1] and gate_up.shape[1] == down.shape[2]:
            is_transposed = True
        else:
            is_transposed = False

    if is_transposed:
        # Transpose to [E, out_features, in_features]
        gate_up = gate_up.transpose(-1, -2).contiguous()
        down = down.transpose(-1, -2).contiguous()

    two_inter, hidden_size = gate_up.shape[1], gate_up.shape[2]
    inter_size = two_inter // 2
    dtype = gate_up.dtype
    device = gate_up.device
    is_meta = device.type == "meta"

    gate_up_bias = getattr(experts_module, "gate_up_proj_bias", None)
    down_bias = getattr(experts_module, "down_proj_bias", None)
    act_fn = getattr(experts_module, "act_fn", None)

    expert_list: List[nn.Module] = []
    for e in range(num_experts):
        gate_lin = nn.Linear(
            hidden_size, inter_size, bias=(gate_up_bias is not None), device=device, dtype=dtype
        )
        up_lin = nn.Linear(
            hidden_size, inter_size, bias=(gate_up_bias is not None), device=device, dtype=dtype
        )
        down_lin = nn.Linear(
            inter_size, hidden_size, bias=(down_bias is not None), device=device, dtype=dtype
        )
        if not is_meta:
            g_w, u_w = gate_up[e].chunk(2, dim=0)
            gate_lin.weight.data.copy_(g_w)
            up_lin.weight.data.copy_(u_w)
            down_lin.weight.data.copy_(down[e])
            if gate_up_bias is not None:
                g_b, u_b = gate_up_bias.data[e].chunk(2, dim=0)
                gate_lin.bias.data.copy_(g_b)
                up_lin.bias.data.copy_(u_b)
            if down_bias is not None:
                down_lin.bias.data.copy_(down_bias.data[e])

        expert_list.append(UnfusedExpertMLP(gate_lin, up_lin, down_lin, act_fn=act_fn))

    return Unfused3DExpertsModule(expert_list, is_transposed=bool(is_transposed))


def ensure_unfused_moe_experts(model: nn.Module) -> int:
    """Scan `model` and unpack any 3D fused `experts` modules into `Unfused3DExpertsModule`.

    Returns the number of MoE layers unpacked.
    """
    _, layers, _ = get_text_backbone_and_prefix(model)
    unpacked_count = 0
    for layer in layers:
        moe_block = get_moe_block_from_layer(layer)
        if moe_block is None:
            continue
        experts = getattr(moe_block, "experts", None)
        if experts is not None and _is_3d_fused_experts_module(experts):
            moe_block.experts = unpack_single_3d_experts_module(experts)
            unpacked_count += 1
    return unpacked_count


# ---------------------------------------------------------------------------
# 2. Backbone / Layer / Config Introspection Helpers
# ---------------------------------------------------------------------------

def get_text_config(config: Any) -> Any:
    """Extract the text backbone config from a top-level or VLM HF config."""
    if config is None:
        return None
    if hasattr(config, "get_text_config") and callable(config.get_text_config):
        try:
            return config.get_text_config()
        except Exception:
            pass
    for attr in ("text_config", "language_config", "llm_config"):
        sub = getattr(config, attr, None)
        if sub is not None:
            return sub
    return config


def get_text_backbone_and_prefix(
    model: nn.Module,
) -> Tuple[nn.Module, nn.ModuleList, str]:
    """Locate the text decoder backbone, its `layers` ModuleList, and its module path prefix.

    Supports:
    - `model.model.layers` -> prefix `"model.layers"` (standard CausalLM)
    - `model.model.language_model.layers` -> prefix `"model.language_model.layers"` (Qwen3.5/3.6 VLM)
    - `model.language_model.model.layers` -> prefix `"language_model.model.layers"` (VLM wrapper)
    - `model.language_model.layers` -> prefix `"language_model.layers"`
    - `model.layers` -> prefix `"layers"`
    """
    candidates = [
        ("model.language_model", "model.language_model.layers"),
        ("language_model.model", "language_model.model.layers"),
        ("language_model", "language_model.layers"),
        ("model", "model.layers"),
        ("", "layers"),
    ]
    for backbone_path, layers_prefix in candidates:
        curr = model
        ok = True
        if backbone_path:
            for part in backbone_path.split("."):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    ok = False
                    break
        if ok and hasattr(curr, "layers") and isinstance(curr.layers, nn.ModuleList):
            return curr, curr.layers, layers_prefix

    # Fallback: search named_modules for the first ModuleList named `layers` outside vision tower
    for name, mod in model.named_modules():
        if (
            name.endswith("layers")
            and isinstance(mod, nn.ModuleList)
            and "visual" not in name
            and "vision" not in name
            and "audio" not in name
        ):
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_path) if parent_path else model
            return parent, mod, name

    raise RuntimeError(f"Could not locate decoder layers ModuleList in model {type(model).__name__}")


def get_moe_block_from_layer(layer: nn.Module) -> Optional[nn.Module]:
    """Return the MoE or MLP block of a decoder layer (`mlp`, `block_sparse_moe`, `feed_forward`, `moe`)."""
    for attr in ("mlp", "block_sparse_moe", "feed_forward", "moe"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def get_moe_attr_name(layer: nn.Module) -> Optional[str]:
    for attr in ("mlp", "block_sparse_moe", "feed_forward", "moe"):
        if hasattr(layer, attr):
            return attr
    return None


def _lookup_patch_param(
    patch_params: Optional[Dict[str, Any]],
    layer_prefix: str,
    layer_idx: int,
    rel_tag: str,
    generic_tag: str,
) -> Any:
    """Look up quantization config for a module across all canonical prefix conventions.

    For example, if `layer_prefix` is `"model.language_model.layers"` (as in
    `Qwen3_5MoeForConditionalGeneration` for `Qwen3.6-35B-A3B`), this checks:
      1. `"model.language_model.layers.{i}.{rel_tag}"`
      2. `"model.layers.{i}.{rel_tag}"`
      3. `"language_model.model.layers.{i}.{rel_tag}"`
      4. `"layers.{i}.{rel_tag}"`
      5. `"{generic_tag}"` (e.g. `"mlp.experts.gate_proj"` or `"self_attn.q_proj"`)
    """
    if not patch_params:
        return None

    candidate_keys = [
        f"{layer_prefix}.{layer_idx}.{rel_tag}",
        f"model.layers.{layer_idx}.{rel_tag}",
        f"model.language_model.layers.{layer_idx}.{rel_tag}",
        f"language_model.model.layers.{layer_idx}.{rel_tag}",
        f"layers.{layer_idx}.{rel_tag}",
    ]
    for k in candidate_keys:
        if k in patch_params and patch_params[k] is not None:
            return patch_params[k]
    # If explicitly set to None in patch_params (e.g. skip_attention / skip_shared_expert),
    # respect the explicit None when the generic tag is not present or also None.
    for k in candidate_keys:
        if k in patch_params:
            return patch_params[k]

    return patch_params.get(generic_tag, None)


# ---------------------------------------------------------------------------
# 3. UniversalMoEPatch & UniversalMoEBRMoE
# ---------------------------------------------------------------------------

class UniversalMoEPatch(BasePatch):
    """Dynamic model patcher supporting all modern MoE architectures."""

    DEFAULT_NUM_LAYERS: int = 40
    DEFAULT_NUM_EXPERTS: int = 64
    HAS_SHARED_EXPERT: bool = True
    SHARED_EXPERT_ATTR: str = "shared_expert"
    MOE_BLOCK_ATTR: str = "mlp"
    EXPERT_PROJ_NAMES: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    @classmethod
    def get_linear_tags(cls) -> List[str]:
        moe_attr = cls.MOE_BLOCK_ATTR
        p1, p2, p3 = cls.EXPERT_PROJ_NAMES
        tags = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # DeepSeek MLA projections
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
            # Qwen3.5 / Qwen3.6 / Qwen3-Next Gated DeltaNet linear_attn projections
            "linear_attn.in_proj_qkvz",
            "linear_attn.in_proj_ba",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
            # Dense MLP (for hybrid dense+MoE layers like first_k_dense_replace)
            f"{moe_attr}.{p1}",
            f"{moe_attr}.{p2}",
            f"{moe_attr}.{p3}",
            # Routed experts
            f"{moe_attr}.experts.{p1}",
            f"{moe_attr}.experts.{p2}",
            f"{moe_attr}.{p3}" if False else f"{moe_attr}.experts.{p3}",
            # Shared expert (both singular & plural conventions)
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ]
        return tags

    @classmethod
    def get_leaf_linear_tags(cls, model: Optional[nn.Module] = None) -> List[str]:
        """Return all compressible leaf linear module names.

        When `model` is provided (recommended), inspects `model` dynamically and also
        includes canonical `model.layers.{i}...` aliases so `compress_config` keys match
        regardless of whether `model` uses `model.layers` or `model.language_model.layers`.
        """
        if model is not None:
            ensure_unfused_moe_experts(model)
            _, layers, layer_prefix = get_text_backbone_and_prefix(model)
            leaf_tags: List[str] = []
            seen = set()

            def _add(tag: str) -> None:
                if tag not in seen:
                    seen.add(tag)
                    leaf_tags.append(tag)

            for i, layer in enumerate(layers):
                for sub_name, sub_mod in layer.named_modules():
                    if not sub_name:
                        continue
                    if type(sub_mod) in _QUANT_LAYERS:
                        # Exclude router gates (`mlp.gate`, `block_sparse_moe.gate`, `shared_expert_gate`)
                        last_part = sub_name.split(".")[-1]
                        if last_part in ("gate", "shared_expert_gate", "e_score_correction_bias"):
                            continue
                        full_actual = f"{layer_prefix}.{i}.{sub_name}"
                        full_canon = f"model.layers.{i}.{sub_name}"
                        _add(full_actual)
                        _add(full_canon)
            return leaf_tags

        # Fallback static generation when model is not provided
        leaf_tags = []
        moe_attr = cls.MOE_BLOCK_ATTR
        p1, p2, p3 = cls.EXPERT_PROJ_NAMES
        for i in range(cls.DEFAULT_NUM_LAYERS):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                leaf_tags.append(f"model.layers.{i}.self_attn.{proj}")
            for proj in ("in_proj_qkvz", "in_proj_ba", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                leaf_tags.append(f"model.layers.{i}.linear_attn.{proj}")
            for k in range(cls.DEFAULT_NUM_EXPERTS):
                leaf_tags.append(f"model.layers.{i}.{moe_attr}.experts.{k}.{p1}")
                leaf_tags.append(f"model.layers.{i}.{moe_attr}.experts.{k}.{p2}")
                leaf_tags.append(f"model.layers.{i}.{moe_attr}.experts.{k}.{p3}")
            if cls.HAS_SHARED_EXPERT:
                for sh_attr in ("shared_expert", "shared_experts"):
                    leaf_tags.append(f"model.layers.{i}.mlp.{sh_attr}.gate_proj")
                    leaf_tags.append(f"model.layers.{i}.mlp.{sh_attr}.up_proj")
                    leaf_tags.append(f"model.layers.{i}.mlp.{sh_attr}.down_proj")
        return leaf_tags

    @classmethod
    def get_ignore_layers(cls, model: nn.Module) -> List[str]:
        layers = {""}
        for name, module in model.named_modules():
            if len(module._modules) > 0:
                layers.add(name)
            # Ignore multimodal vision/audio towers during text MoE quantization
            if any( name.startswith(prefix) or f".{prefix}." in name for prefix in ("visual", "vision_tower", "vision_model", "audio_tower", "multi_modal_projector")):
                layers.add(name)
        return list(layers)

    @classmethod
    def patch_nonlinearlayers(cls, model: nn.Module, patch_fct: Callable, verbose: bool = True) -> None:
        ensure_unfused_moe_experts(model)
        base_model, layers, _ = get_text_backbone_and_prefix(model)

        # Top-level embeddings, norm, rotary_emb, lm_head, and optional VLM visual tower
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head = patch_fct(model.lm_head)
        elif hasattr(model, "language_model") and hasattr(model.language_model, "lm_head") and model.language_model.lm_head is not None:
            model.language_model.lm_head = patch_fct(model.language_model.lm_head)

        for top_attr in ("embed_tokens", "norm", "rotary_emb"):
            if hasattr(base_model, top_attr) and getattr(base_model, top_attr) is not None:
                setattr(base_model, top_attr, patch_fct(getattr(base_model, top_attr)))

        for vlm_attr in ("visual", "vision_tower", "vision_model", "multi_modal_projector"):
            if hasattr(model, vlm_attr) and getattr(model, vlm_attr) is not None:
                setattr(model, vlm_attr, patch_fct(getattr(model, vlm_attr)))
            elif hasattr( base_model, vlm_attr) and getattr(base_model, vlm_attr) is not None:
                setattr(base_model, vlm_attr, patch_fct(getattr(base_model, vlm_attr)))

        for i in tqdm(range(len(layers)), disable=not verbose, desc="Patching non-linear layers"):
            layer = layers[i]

            # Layer norms & layer scales
            for norm_attr in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "attn_layer_scale",
                "ffn_layer_scale",
            ):
                if hasattr(layer, norm_attr) and getattr(layer, norm_attr) is not None:
                    val = getattr(layer, norm_attr)
                    if isinstance(val, nn.Module):
                        setattr(layer, norm_attr, patch_fct(val))

            # Full attention non-linear submodules (rotary_emb, q_norm, k_norm, MLA layernorms)
            if hasattr(layer, "self_attn") and layer.self_attn is not None:
                attn = layer.self_attn
                for attn_sub in ("rotary_emb", "q_norm", "k_norm", "q_a_layernorm", "kv_a_layernorm"):
                    if hasattr(attn, attn_sub) and getattr(attn, attn_sub) is not None:
                        setattr(attn, attn_sub, patch_fct(getattr(attn, attn_sub)))

            # Hybrid Linear Attention (Gated DeltaNet in Qwen3.6-35B-A3B / Qwen3.5-MoE / Qwen3-Next)
            if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                lin_attn = layer.linear_attn
                for la_sub in ("conv1d", "norm", "act", "act_fn"):
                    if hasattr(lin_attn, la_sub) and getattr(lin_attn, la_sub) is not None:
                        sub_obj = getattr(lin_attn, la_sub)
                        if isinstance(sub_obj, nn.Module):
                            setattr(lin_attn, la_sub, patch_fct(sub_obj))

            # MoE / MLP block non-linear submodules (router gate, shared_expert_gate, act_fn)
            moe_block = get_moe_block_from_layer(layer)
            if moe_block is not None:
                for gate_attr in ("gate", "shared_expert_gate", "router"):
                    if hasattr(moe_block, gate_attr) and getattr(moe_block, gate_attr) is not None:
                        setattr(moe_block, gate_attr, patch_fct(getattr(moe_block, gate_attr)))

                if hasattr(moe_block, "act_fn") and isinstance(getattr(moe_block, "act_fn"), nn.Module):
                    moe_block.act_fn = patch_fct(moe_block.act_fn)

                if hasattr(moe_block, "experts") and isinstance(moe_block.experts, nn.ModuleList):
                    for k in range(len(moe_block.experts)):
                        exp = moe_block.experts[k]
                        if hasattr(exp, "act_fn") and isinstance(getattr(exp, "act_fn"), nn.Module):
                            exp.act_fn = patch_fct(exp.act_fn)

                for sh_attr in ("shared_expert", "shared_experts", "shared_mlp"):
                    if hasattr(moe_block, sh_attr) and getattr(moe_block, sh_attr) is not None:
                        sh_mod = getattr(moe_block, sh_attr)
                        if hasattr(sh_mod, "act_fn") and isinstance(getattr(sh_mod, "act_fn"), nn.Module):
                            sh_mod.act_fn = patch_fct(sh_mod.act_fn)

    @classmethod
    def patch_linearlayers(
        cls,
        model: nn.Module,
        patch_fct: Callable,
        patch_params: Optional[Dict[str, Any]],
        verbose: bool = True,
    ) -> None:
        """Unified linear layer patcher supporting both uniform tags and per-expert MixBit configs."""
        ensure_unfused_moe_experts(model)
        _, layers, layer_prefix = get_text_backbone_and_prefix(model)

        for i in tqdm(range(len(layers)), disable=not verbose, desc="Patching linear layers"):
            layer = layers[i]

            # 1. Standard & MLA Self-Attention projections
            if hasattr(layer, "self_attn") and layer.self_attn is not None:
                attn = layer.self_attn
                for proj in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "qkv_proj",
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                ):
                    if hasattr(attn, proj) and getattr(attn, proj) is not None:
                        rel_tag = f"self_attn.{proj}"
                        cfg = _lookup_patch_param(patch_params, layer_prefix, i, rel_tag, rel_tag)
                        setattr(attn, proj, patch_fct(getattr(attn, proj), cfg))

            # 2. Hybrid Linear Attention (Gated DeltaNet in Qwen3.6-35B-A3B / Qwen3.5-MoE / Qwen3-Next)
            if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                lin_attn = layer.linear_attn
                for proj in (
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ):
                    if hasattr(lin_attn, proj) and getattr(lin_attn, proj) is not None:
                        rel_tag = f"linear_attn.{proj}"
                        cfg = _lookup_patch_param(patch_params, layer_prefix, i, rel_tag, rel_tag)
                        setattr(lin_attn, proj, patch_fct(getattr(lin_attn, proj), cfg))

            # 3. MoE / MLP block
            moe_attr = get_moe_attr_name(layer)
            if moe_attr is None:
                continue
            moe_block = getattr(layer, moe_attr)

            # 3a. Dense MLP layer (e.g. first_k_dense_replace or mlp_only_layers)
            for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2"):
                if hasattr(moe_block, proj) and isinstance(getattr(moe_block, proj), tuple(_QUANT_LAYERS)):
                    rel_tag = f"{moe_attr}.{proj}"
                    cfg = _lookup_patch_param(patch_params, layer_prefix, i, rel_tag, rel_tag)
                    setattr(moe_block, proj, patch_fct(getattr(moe_block, proj), cfg))

            # 3b. Routed Experts
            if hasattr(moe_block, "experts") and isinstance(moe_block.experts, nn.ModuleList):
                for k in range(len(moe_block.experts)):
                    expert = moe_block.experts[k]
                    for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2"):
                        if hasattr(expert, proj) and getattr(expert, proj) is not None:
                            rel_tag = f"{moe_attr}.experts.{k}.{proj}"
                            generic_tag = f"{moe_attr}.experts.{proj}"
                            cfg = _lookup_patch_param(patch_params, layer_prefix, i, rel_tag, generic_tag)
                            setattr(expert, proj, patch_fct(getattr(expert, proj), cfg))

            # 3c. Shared Expert(s)
            for sh_attr in ("shared_expert", "shared_experts", "shared_mlp"):
                if hasattr(moe_block, sh_attr) and getattr(moe_block, sh_attr) is not None:
                    sh_mod = getattr(moe_block, sh_attr)
                    for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2"):
                        if hasattr(sh_mod, proj) and getattr(sh_mod, proj) is not None:
                            rel_tag = f"{moe_attr}.{sh_attr}.{proj}"
                            # Support both singular and plural fallback in patch_params
                            alt_sh = "shared_experts" if sh_attr == "shared_expert" else "shared_expert"
                            cfg = _lookup_patch_param(
                                patch_params,
                                layer_prefix,
                                i,
                                rel_tag,
                                f"{moe_attr}.{sh_attr}.{proj}",
                            )
                            if cfg is None and patch_params:
                                cfg = _lookup_patch_param(
                                    patch_params,
                                    layer_prefix,
                                    i,
                                    f"{moe_attr}.{alt_sh}.{proj}",
                                    f"{moe_attr}.{alt_sh}.{proj}",
                                )
                            setattr(sh_mod, proj, patch_fct(getattr(sh_mod, proj), cfg))

    @classmethod
    def patch_linearlayers_mixbit(
        cls,
        model: nn.Module,
        patch_fct: Callable,
        patch_params: Optional[Dict[str, Any]],
        verbose: bool = True,
    ) -> None:
        return cls.patch_linearlayers(model, patch_fct, patch_params, verbose=verbose)


class UniversalMoEBRMoE(UniversalMoEPatch, BaseBRMoEHFModel):
    """Universal BR-MoE HuggingFace model class for any MoE architecture."""
    pass


class GLM4MoEPatch(UniversalMoEPatch):
    DEFAULT_NUM_LAYERS = 46
    DEFAULT_NUM_EXPERTS = 128
    HAS_SHARED_EXPERT = True
    SHARED_EXPERT_ATTR = "shared_experts"


class GLM4MoEBRMoE(GLM4MoEPatch, BaseBRMoEHFModel):
    pass


class OLMoEPatch(UniversalMoEPatch):
    DEFAULT_NUM_LAYERS = 16
    DEFAULT_NUM_EXPERTS = 64
    HAS_SHARED_EXPERT = False


class OLMoEBRMoE(OLMoEPatch, BaseBRMoEHFModel):
    pass


class PhiMoEPatch(UniversalMoEPatch):
    DEFAULT_NUM_LAYERS = 32
    DEFAULT_NUM_EXPERTS = 16
    HAS_SHARED_EXPERT = False
    MOE_BLOCK_ATTR = "block_sparse_moe"
    EXPERT_PROJ_NAMES = ("w1", "w2", "w3")


class PhiMoEBRMoE(PhiMoEPatch, BaseBRMoEHFModel):
    pass
