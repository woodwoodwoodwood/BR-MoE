"""Model Adapters & Automatic Memory Cost Calculator for BR-MoE / RCO Solver.

Provides built-in architectural specifications and dynamic `config.json` parsing
for modern MoE models (including `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-35B-A3B`,
`Qwen/Qwen3-Next-80B-A3B`, `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B`,
`DeepSeek-V2/V3/R1`, `GLM-4.5/4.6-Air`, `OLMoE-1B-7B`, `Mixtral-8x7B/8x22B`, etc.).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class MoEModelSpec:
    """Normalized architectural metadata for an MoE model."""

    model_id: str
    arch_family: str
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    first_k_dense_replace: int = 0
    shared_expert_intermediate_size: int = 0
    num_shared_experts: int = 0
    is_hybrid_linear_attn: bool = False
    is_multimodal_wrapper: bool = False
    moe_block_attr: str = "mlp"
    expert_proj_names: Tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj")
    shared_expert_attr: Optional[str] = "shared_expert"
    mlp_only_layers: List[int] = field(default_factory=list)

    @property
    def moe_layer_indices(self) -> List[int]:
        dense_set = set(range(self.first_k_dense_replace)) | set(self.mlp_only_layers)
        return [i for i in range(self.num_hidden_layers) if i not in dense_set]

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    @property
    def total_routed_experts(self) -> int:
        return self.num_moe_layers * self.num_experts

    @property
    def expert_params_count(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def has_linear_attn(self) -> bool:
        return bool(self.is_hybrid_linear_attn)

    @property
    def has_shared_expert(self) -> bool:
        return bool(self.num_shared_experts > 0 or self.shared_expert_intermediate_size > 0 or self.shared_expert_attr is not None)

    @property
    def expert_fp16_mb(self) -> float:
        return (self.expert_params_count * 2.0) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Built-in Presets for Popular & Recent MoE Models
# ---------------------------------------------------------------------------

BUILTIN_MOE_PRESETS: Dict[str, MoEModelSpec] = {
    # Qwen3.6-35B-A3B (VLM / Hybrid Gated DeltaNet + Full Attention, 256 routed experts + 1 shared expert)
    "qwen3.6-35b-a3b": MoEModelSpec(
        model_id="Qwen/Qwen3.6-35B-A3B",
        arch_family="qwen3_next",
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_shared_experts=1,
        is_hybrid_linear_attn=True,
        is_multimodal_wrapper=True,
        moe_block_attr="mlp",
        shared_expert_attr="shared_expert",
    ),
    # Qwen3.5-35B-A3B (Same family as Qwen3.6-35B-A3B)
    "qwen3.5-35b-a3b": MoEModelSpec(
        model_id="Qwen/Qwen3.5-35B-A3B",
        arch_family="qwen3_next",
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_shared_experts=1,
        is_hybrid_linear_attn=True,
        is_multimodal_wrapper=True,
        moe_block_attr="mlp",
        shared_expert_attr="shared_expert",
    ),
    # Qwen3-Next-80B-A3B (512 routed experts, top-10, hybrid linear_attn + self_attn)
    "qwen3-next-80b-a3b": MoEModelSpec(
        model_id="Qwen/Qwen3-Next-80B-A3B",
        arch_family="qwen3_next",
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        hidden_size=2048,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_shared_experts=1,
        is_hybrid_linear_attn=True,
        is_multimodal_wrapper=False,
        moe_block_attr="mlp",
        shared_expert_attr="shared_expert",
    ),
    # Qwen3-30B-A3B (48 layers, 128 routed experts, top-8, no shared expert)
    "qwen3-30b-a3b": MoEModelSpec(
        model_id="Qwen/Qwen3-30B-A3B",
        arch_family="qwen3_moe",
        num_hidden_layers=48,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=768,
        shared_expert_intermediate_size=0,
        num_shared_experts=0,
        moe_block_attr="mlp",
        shared_expert_attr=None,
    ),
    # Qwen3-235B-A22B (94 layers, 128 routed experts, top-8, no shared expert)
    "qwen3-235b-a22b": MoEModelSpec(
        model_id="Qwen/Qwen3-235B-A22B",
        arch_family="qwen3_moe",
        num_hidden_layers=94,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=1536,
        shared_expert_intermediate_size=0,
        num_shared_experts=0,
        moe_block_attr="mlp",
        shared_expert_attr=None,
    ),
    # Qwen1.5-MoE-A2.7B (24 layers, 60 routed experts, top-4, shared expert)
    "qwen1.5-moe": MoEModelSpec(
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
        arch_family="qwen2_moe",
        num_hidden_layers=24,
        num_experts=60,
        num_experts_per_tok=4,
        hidden_size=2048,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        num_shared_experts=1,
        moe_block_attr="mlp",
        shared_expert_attr="shared_expert",
    ),
    # Qwen2-57B-A14B (28 layers, 64 routed experts, top-8, shared expert)
    "qwen2-57b-a14b": MoEModelSpec(
        model_id="Qwen/Qwen2-57B-A14B",
        arch_family="qwen2_moe",
        num_hidden_layers=28,
        num_experts=64,
        num_experts_per_tok=8,
        hidden_size=3584,
        moe_intermediate_size=2560,
        shared_expert_intermediate_size=20480,
        num_shared_experts=1,
        moe_block_attr="mlp",
        shared_expert_attr="shared_expert",
    ),
    # DeepSeek-MoE-16B (28 layers, layer 0 dense, 64 routed experts, 2 shared experts)
    "deepseek-moe-16b": MoEModelSpec(
        model_id="deepseek-ai/deepseek-moe-16b-base",
        arch_family="deepseek",
        num_hidden_layers=28,
        num_experts=64,
        num_experts_per_tok=6,
        hidden_size=2048,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        shared_expert_intermediate_size=2816,
        num_shared_experts=2,
        moe_block_attr="mlp",
        shared_expert_attr="shared_experts",
    ),
    # DeepSeek-V2-Lite (27 layers, layer 0 dense, 64 routed experts, MLA)
    "deepseek-v2-lite": MoEModelSpec(
        model_id="deepseek-ai/DeepSeek-V2-Lite",
        arch_family="deepseek_v2",
        num_hidden_layers=27,
        num_experts=64,
        num_experts_per_tok=6,
        hidden_size=2048,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        shared_expert_intermediate_size=2816,
        num_shared_experts=2,
        moe_block_attr="mlp",
        shared_expert_attr="shared_experts",
    ),
    # DeepSeek-V3 / R1 (61 layers, layers 0..2 dense, 256 routed experts, MLA)
    "deepseek-v3": MoEModelSpec(
        model_id="deepseek-ai/DeepSeek-V3",
        arch_family="deepseek_v3",
        num_hidden_layers=61,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=7168,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        shared_expert_intermediate_size=2048,
        num_shared_experts=1,
        moe_block_attr="mlp",
        shared_expert_attr="shared_experts",
    ),
    # GLM-4.5-Air / GLM-4.6-Air (46 layers, layer 0 dense, 128 routed experts)
    "glm-4.5-air": MoEModelSpec(
        model_id="zai-org/GLM-4.5-Air",
        arch_family="glm4_moe",
        num_hidden_layers=46,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        shared_expert_intermediate_size=1408,
        num_shared_experts=1,
        moe_block_attr="mlp",
        shared_expert_attr="shared_experts",
    ),
    # OLMoE-1B-7B (16 layers, 64 routed experts, top-8, no shared expert)
    "olmoe-1b-7b": MoEModelSpec(
        model_id="allenai/OLMoE-1B-7B-0924",
        arch_family="olmoe",
        num_hidden_layers=16,
        num_experts=64,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=1024,
        moe_block_attr="mlp",
        shared_expert_attr=None,
    ),
    # Mixtral-8x7B (32 layers, 8 routed experts, top-2)
    "mixtral-8x7b": MoEModelSpec(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        arch_family="mixtral",
        num_hidden_layers=32,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=4096,
        moe_intermediate_size=14336,
        moe_block_attr="block_sparse_moe",
        expert_proj_names=("w1", "w2", "w3"),
        shared_expert_attr=None,
    ),
    # Mixtral-8x22B (56 layers, 8 routed experts, top-2)
    "mixtral-8x22b": MoEModelSpec(
        model_id="mistralai/Mixtral-8x22B-v0.1",
        arch_family="mixtral",
        num_hidden_layers=56,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=6144,
        moe_intermediate_size=16384,
        moe_block_attr="block_sparse_moe",
        expert_proj_names=("w1", "w2", "w3"),
        shared_expert_attr=None,
    ),
}

# Aliases mapping common CLI / Hub names to preset keys
_PRESET_ALIASES: Dict[str, str] = {
    "qwen3.6": "qwen3.6-35b-a3b",
    "qwen3_6": "qwen3.6-35b-a3b",
    "qwen3.6-moe": "qwen3.6-35b-a3b",
    "qwen/qwen3.6-35b-a3b": "qwen3.6-35b-a3b",
    "qwen3.5": "qwen3.5-35b-a3b",
    "qwen3_5": "qwen3.5-35b-a3b",
    "qwen3.5-moe": "qwen3.5-35b-a3b",
    "qwen3_5_moe": "qwen3.5-35b-a3b",
    "qwen/qwen3.5-35b-a3b": "qwen3.5-35b-a3b",
    "qwen3-next": "qwen3-next-80b-a3b",
    "qwen3_next": "qwen3-next-80b-a3b",
    "qwen/qwen3-next-80b-a3b": "qwen3-next-80b-a3b",
    "qwen3": "qwen3-30b-a3b",
    "qwen3-moe": "qwen3-30b-a3b",
    "qwen3_moe": "qwen3-30b-a3b",
    "qwen/qwen3-30b-a3b": "qwen3-30b-a3b",
    "qwen/qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen": "qwen1.5-moe",
    "qwen1.5": "qwen1.5-moe",
    "qwen15": "qwen1.5-moe",
    "qwen/qwen1.5-moe-a2.7b": "qwen1.5-moe",
    "qwen2-moe": "qwen2-57b-a14b",
    "deepseek": "deepseek-moe-16b",
    "deepseek-moe": "deepseek-moe-16b",
    "deepseek-ai/deepseek-moe-16b-base": "deepseek-moe-16b",
    "deepseek-v2": "deepseek-v2-lite",
    "deepseek-r1": "deepseek-v3",
    "glm4-moe": "glm-4.5-air",
    "glm4_moe": "glm-4.5-air",
    "olmoe": "olmoe-1b-7b",
    "mixtral": "mixtral-8x7b",
    "mistralai/mixtral-8x7b-v0.1": "mixtral-8x7b",
}


def _first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_moe_spec_from_config_dict(
    raw_cfg: Dict[str, Any], model_id: str = "custom_moe"
) -> MoEModelSpec:
    """Parse an `MoEModelSpec` directly from a HuggingFace `config.json` dict."""
    archs = raw_cfg.get("architectures") or []
    arch_name = archs[0] if archs else ""
    is_mm = (
        " text_config" in raw_cfg
        or "text_config" in raw_cfg
        or "ConditionalGeneration" in arch_name
    )
    text_cfg = raw_cfg.get("text_config") or raw_cfg.get("language_config") or raw_cfg

    model_type = str(text_cfg.get("model_type") or raw_cfg.get("model_type") or "").lower()
    num_layers = int(_first_present(text_cfg, ("num_hidden_layers", "num_layers", "n_layer"), 32))
    num_experts = int(
        _first_present(
            text_cfg,
            ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"),
            64,
        )
    )
    top_k = int(
        _first_present(
            text_cfg,
            ("num_experts_per_tok", "num_experts_per_token", "top_k", "moe_topk", "moe_top_k"),
            8,
        )
    )
    hidden_size = int(_first_present(text_cfg, ("hidden_size", "d_model"), 2048))
    moe_inter = int(
        _first_present(
            text_cfg,
            ("moe_intermediate_size", "ffn_dim", "intermediate_size"),
            1408,
        )
    )
    first_k_dense = int(_first_present(text_cfg, ("first_k_dense_replace",), 0))
    mlp_only_layers = list(text_cfg.get("mlp_only_layers") or [])

    num_shared = int(
        _first_present(text_cfg, ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts"), 0)
    )
    shared_inter = int(
        _first_present(text_cfg, ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"), 0)
    )
    if shared_inter > 0 and num_shared == 0:
        num_shared = 1
    elif num_shared > 0 and shared_inter == 0:
        shared_inter = moe_inter * num_shared

    layer_types = text_cfg.get("layer_types") or []
    is_hybrid = (
        any("linear" in str(lt) for lt in layer_types)
        or "qwen3_5" in model_type
        or "qwen3_next" in model_type
        or "linear_num_key_heads" in text_cfg
    )

    if "mixtral" in model_type or "phimoe" in model_type:
        moe_block_attr = "block_sparse_moe"
        expert_proj_names = ("w1", "w2", "w3")
    else:
        moe_block_attr = "mlp"
        expert_proj_names = ("gate_proj", "up_proj", "down_proj")

    if "deepseek" in model_type or "glm" in model_type:
        shared_attr = "shared_experts" if num_shared > 0 else None
    else:
        shared_attr = "shared_expert" if num_shared > 0 else None

    return MoEModelSpec(
        model_id=model_id,
        arch_family=model_type or arch_name,
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_inter,
        first_k_dense_replace=first_k_dense,
        shared_expert_intermediate_size=shared_inter,
        num_shared_experts=num_shared,
        is_hybrid_linear_attn=is_hybrid,
        is_multimodal_wrapper=is_mm,
        moe_block_attr=moe_block_attr,
        expert_proj_names=expert_proj_names,
        shared_expert_attr=shared_attr,
        mlp_only_layers=mlp_only_layers,
    )


def resolve_moe_model_spec(
    model_or_name: Union[str, Dict[str, Any], MoEModelSpec],
) -> MoEModelSpec:
    """Resolve an `MoEModelSpec` from a preset name, HF repo id, local path, or config dict."""
    if isinstance(model_or_name, MoEModelSpec):
        return model_or_name
    if isinstance(model_or_name, dict):
        return parse_moe_spec_from_config_dict(model_or_name)

    key = str(model_or_name).strip()
    key_lower = key.lower()

    if key_lower in BUILTIN_MOE_PRESETS:
        return BUILTIN_MOE_PRESETS[key_lower]
    if key_lower in _PRESET_ALIASES:
        return BUILTIN_MOE_PRESETS[_PRESET_ALIASES[key_lower]]

    # Check if key is a local file or directory with config.json
    if os.path.isfile(key):
        with open(key, "r", encoding="utf-8") as f:
            return parse_moe_spec_from_config_dict(json.load(f), model_id=key)
    if os.path.isdir(key):
        cfg_file = os.path.join(key, "config.json")
        if os.path.isfile(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                return parse_moe_spec_from_config_dict(json.load(f), model_id=key)

    # Substring match against built-in presets
    for alias, target in _PRESET_ALIASES.items():
        if alias in key_lower:
            return BUILTIN_MOE_PRESETS[target]
    for preset_name, spec in BUILTIN_MOE_PRESETS.items():
        if preset_name in key_lower or spec.model_id.lower() in key_lower:
            return spec

    raise ValueError(
        f"Unrecognized MoE model {model_or_name!r}. "
        f"Available presets: {sorted(BUILTIN_MOE_PRESETS.keys())} or pass a path to config.json."
    )


def compute_expert_memory_table(
    spec: Union[str, MoEModelSpec],
    bits_list: Sequence[int] = (2, 3, 4),
    ranks_list: Sequence[int] = (0, 16, 32, 64, 128, 256, 512, 1024),
    weight_group_size: int = 64,
    comp_bits: int = 3,
    comp_group_size: int = 64,
    include_pruned: bool = False,
    group_size: Optional[int] = None,
    comp_gs: Optional[int] = None,
) -> Dict[str, float]:
    """Compute analytical per-expert memory footprint (in MB) for all `(bits, rank)` options.

    Matches the exact BR-MoE storage layout:
    - Base quantized weight (3 matrices of shape `H x I`, group size `g_w`, FP16 scale & zero):
      `MB_base(b) = 3 * H * I * (b + 32 / g_w) / (8 * 2^20)`
    - Low-rank compensator `U, V` (3 pairs of shape `H x r` and `r x I`, quantized to `comp_bits`
      with group size `comp_group_size`):
      `MB_comp(r) = 3 * (H + I) * r * (comp_bits + 32 / comp_group_size) / (8 * 2^20)`
    """
    if group_size is not None:
        weight_group_size = int(group_size)
    if comp_gs is not None:
        comp_group_size = int(comp_gs)
    spec_obj = resolve_moe_model_spec(spec)
    H = float(spec_obj.hidden_size)
    I = float(spec_obj.moe_intermediate_size)

    num_weight_elems = 3.0 * H * I
    bits_overhead = 32.0 / float(weight_group_size) if weight_group_size > 0 else 0.0
    comp_eff_bits = float(comp_bits) + (32.0 / float(comp_group_size) if comp_group_size > 0 else 0.0)

    table: Dict[str, float] = {}
    if include_pruned:
        table["0bit_rank0"] = 0.0

    for b in bits_list:
        base_mb = (num_weight_elems * (float(b) + bits_overhead)) / (8.0 * 1024.0 * 1024.0)
        for r in ranks_list:
            comp_elems = 3.0 * (H + I) * float(r)
            comp_mb = (comp_elems * comp_eff_bits) / (8.0 * 1024.0 * 1024.0)
            table[f"{b}bit_rank{r}"] = round(base_mb + comp_mb, 6)

    return table


def compute_budget_from_avg_bits(
    spec: Union[str, MoEModelSpec],
    avg_bits: float = 3.2,
    weight_group_size: int = 64,
) -> float:
    """Convert target average bits-per-weight (including scale/zero/compensator) into total expert MB budget."""
    spec_obj = resolve_moe_model_spec(spec)
    bits_overhead = 32.0 / float(weight_group_size) if weight_group_size > 0 else 0.0
    per_expert_mb = (
        float(spec_obj.expert_params_count) * (float(avg_bits) + bits_overhead)
    ) / (8.0 * 1024.0 * 1024.0)
    return round(per_expert_mb * float(spec_obj.total_routed_experts), 2)


def synthesize_problem_tensors_for_model(
    spec: Union[str, MoEModelSpec],
    memory_usage: Optional[Dict[str, float]] = None,
    expert_freq: Optional[Dict[str, float]] = None,
    min_rank: int = 0,
    max_rank: Optional[int] = 256,
    allowed_bits: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], Any, Any, Any]:
    """Build `(experts, configs, loss_mat, cost_vec, freq_vec)` for any `MoEModelSpec`.

    Used when running RCO search on a new MoE model (e.g. `Qwen3.6-35B-A3B`) prior to
    full per-expert L2 impact profiling, combining measured/Zipfian routing frequencies
    with analytical SVD quantization distortion curves.
    """
    import numpy as np

    spec_obj = resolve_moe_model_spec(spec)
    if memory_usage is None:
        memory_usage = compute_expert_memory_table(spec_obj)

    configs: List[str] = []
    parsed_cfgs: List[Tuple[int, int]] = []
    for cfg_name in memory_usage.keys():
        parts = cfg_name.split("bit_rank")
        if len(parts) != 2:
            continue
        b, r = int(parts[0]), int(parts[1])
        if allowed_bits is not None and b not in allowed_bits:
            continue
        if r < min_rank or (max_rank is not None and r > max_rank):
            continue
        configs.append(cfg_name)
        parsed_cfgs.append((b, r))

    moe_layers = spec_obj.moe_layer_indices
    E = spec_obj.num_experts
    experts: List[str] = [f"L{l}_E{e}" for l in moe_layers for e in range(E)]

    rng = np.random.default_rng(seed)
    n_experts_total = len(experts)
    n_configs = len(configs)

    # Routing frequencies: use provided `expert_freq` if available, else realistic Zipfian routing distribution
    freq_vec = np.zeros(n_experts_total, dtype=np.float64)
    for idx, exp_key in enumerate(experts):
        if expert_freq and exp_key in expert_freq:
            freq_vec[idx] = float(expert_freq[exp_key])
        else:
            l_idx = int(exp_key.split("_")[0][1:])
            e_idx = int(exp_key.split("_")[1][1:])
            # Deterministic per-(layer, expert) Zipf-like routing frequency + log-normal spread
            perm_rank = ((e_idx * 73 + l_idx * 19) % E) + 1
            zipf_w = 1.0 / ((perm_rank + 2.0) ** 0.85)
            jitter = float(rng.lognormal(mean=0.0, sigma=0.18))
            freq_vec[idx] = zipf_w * jitter

    # Normalize frequencies per layer so each layer sums to top_k
    for l_pos in range(len(moe_layers)):
        s = l_pos * E
        e = s + E
        layer_sum = freq_vec[s:e].sum()
        if layer_sum > 0:
            freq_vec[s:e] = freq_vec[s:e] / layer_sum * float(spec_obj.num_experts_per_tok)

    # Layer-depth sensitivity: early and final layers have higher quantization sensitivity
    I_dim = float(spec_obj.moe_intermediate_size)
    loss_mat = np.zeros((n_experts_total, n_configs), dtype=np.float64)

    for idx, exp_key in enumerate(experts):
        l_idx = int(exp_key.split("_")[0][1:])
        rel_depth = float(l_idx) / max(1.0, float(spec_obj.num_hidden_layers - 1))
        depth_factor = 1.0 + 0.45 * ((2.0 * rel_depth - 1.0) ** 2)
        intrinsic_sens = float(rng.uniform(0.85, 1.15)) * depth_factor

        for c_idx, (b, r) in enumerate(parsed_cfgs):
            if b == 0:
                dist = 1.0
            else:
                # Rate-distortion + Marchenko-Pastur low-rank residual decay
                bit_dist = 2.0 ** (-1.85 * float(b - 2))
                rank_frac = min(1.0, float(r) / max(1.0, I_dim))
                comp_decay = (1.0 - 0.72 * (rank_frac ** 0.55))
                dist = bit_dist * max(0.05, comp_decay)
            loss_mat[idx, c_idx] = freq_vec[idx] * intrinsic_sens * dist

    cost_vec = np.array([float(memory_usage[c]) for c in configs], dtype=np.float64)
    return experts, configs, loss_mat, cost_vec, freq_vec

