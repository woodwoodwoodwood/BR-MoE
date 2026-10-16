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


# Exact GGUF K-Quant effective bits-per-weight (bpw) from ggml-common.h:
#   Q2_K :  84 bytes / 256 weights = 2.6250 bpw
#   Q3_K : 110 bytes / 256 weights = 3.4375 bpw
#   Q4_K : 144 bytes / 256 weights = 4.5000 bpw
#   Q8_0 :  34 bytes /  32 weights = 8.5000 bpw
_GGUF_KQUANT_BPW: Dict[int, float] = {
    0: 0.0,
    2: (8.0 * 84.0) / 256.0,   # 2.6250
    3: (8.0 * 110.0) / 256.0,  # 3.4375
    4: (8.0 * 144.0) / 256.0,  # 4.5000
    8: (8.0 * 34.0) / 32.0,    # 8.5000
    16: 16.0,
}


def compute_expert_memory_table(
    spec: Union[str, MoEModelSpec],
    bits_list: Sequence[int] = (2, 3, 4),
    ranks_list: Sequence[int] = (0, 16, 32, 64, 128, 256, 512, 1024),
    weight_group_size: int = 256,
    comp_bits: int = 16,
    comp_group_size: int = 64,
    include_pruned: bool = False,
    group_size: Optional[int] = None,
    comp_gs: Optional[int] = None,
    quant_format: str = "gguf",
) -> Dict[str, float]:
    """Compute analytical per-expert memory footprint (in MB) for all `(bits, rank)` options.

    Supports two quantization formats:
    - `quant_format="gguf"` (default, `llama.cpp` K-Quants):
      * Base 3D expert tensors use exact `ggml-common.h` K-Quant block sizes:
        `Q2_K = 84 B / 256 (2.625 bpw)`, `Q3_K = 110 B / 256 (3.4375 bpw)`, `Q4_K = 144 B / 256 (4.500 bpw)`.
      * Low-rank compensator `.lora_a` / `.lora_b` in `model-brmoe-compensator.gguf` uses `comp_bits`
        (default `16` for `GGML_TYPE_F16`, or `8` for `Q8_0`).
    - `quant_format="hqq"` (legacy groupwise RTN/HQQ):
      * `MB_base(b) = 3 * H * I * (b + 32 / g_w) / (8 * 2^20)`
      * `MB_comp(r) = 3 * (H + I) * r * (comp_bits + 32 / comp_group_size) / (8 * 2^20)`
    """
    if group_size is not None:
        weight_group_size = int(group_size)
    if comp_gs is not None:
        comp_group_size = int(comp_gs)
    spec_obj = resolve_moe_model_spec(spec)
    H = float(spec_obj.hidden_size)
    I = float(spec_obj.moe_intermediate_size)
    num_weight_elems = 3.0 * H * I

    qfmt = (quant_format or "gguf").lower()
    is_gguf = qfmt in ("gguf", "kquant", "llamacpp")

    if is_gguf:
        if comp_bits >= 16:
            comp_eff_bits = float(comp_bits)
        elif comp_bits in _GGUF_KQUANT_BPW:
            comp_eff_bits = _GGUF_KQUANT_BPW[comp_bits]
        else:
            comp_eff_bits = float(comp_bits) + (32.0 / float(comp_group_size) if comp_group_size > 0 else 0.0)
    else:
        comp_eff_bits = float(comp_bits) + (32.0 / float(comp_group_size) if comp_group_size > 0 else 0.0)

    bits_overhead = 32.0 / float(weight_group_size) if weight_group_size > 0 else 0.0

    table: Dict[str, float] = {}
    if include_pruned:
        table["0bit_rank0"] = 0.0

    for b in bits_list:
        if is_gguf and int(b) in _GGUF_KQUANT_BPW:
            eff_w_bpw = _GGUF_KQUANT_BPW[int(b)]
        else:
            eff_w_bpw = float(b) + bits_overhead
        base_mb = (num_weight_elems * eff_w_bpw) / (8.0 * 1024.0 * 1024.0)
        for r in ranks_list:
            comp_elems = 3.0 * (H + I) * float(r)
            comp_mb = (comp_elems * comp_eff_bits) / (8.0 * 1024.0 * 1024.0)
            table[f"{b}bit_rank{r}"] = round(base_mb + comp_mb, 6)

    return table


def compute_budget_from_avg_bits(
    spec: Union[str, MoEModelSpec],
    avg_bits: float = 3.2,
    weight_group_size: int = 256,
    quant_format: str = "gguf",
) -> float:
    """Convert target average bits-per-weight (including K-Quant block headers & compensators) into total expert MB budget."""
    spec_obj = resolve_moe_model_spec(spec)
    qfmt = (quant_format or "gguf").lower()
    if qfmt in ("gguf", "kquant", "llamacpp"):
        # Average K-Quant header overhead across Q2_K (+0.625), Q3_K (+0.4375), Q4_K (+0.500) is ~0.50 bpw
        bits_overhead = 0.50
    else:
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


_GGUF_TYPE_BY_BIT = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    8: "Q8_0",
}


def aggregate_problem_to_granularity(
    experts: Sequence[str],
    configs: Sequence[str],
    loss_mat: Any,
    cost_vec: Any,
    freq_vec: Any,
    granularity: str = "tensor",
    num_experts_per_layer: int = 64,
) -> Tuple[List[str], List[str], Any, Any, Any, Dict[str, Any]]:
    """Aggregate or convert problem tensors to the target decision granularity.

    Supported `granularity` values (default `"tensor"`):
      - `"tensor"`        : Three 3D expert tensors per MoE layer `L{l}_{gate,up,down}_exps` (N = 3 * L),
                            directly matching `llama.cpp` GGUF `blk.{l}.ffn_{gate,up,down}_exps.weight` (Baseline)
      - `"vllm_tensor"`   : Two 3D expert tensors per MoE layer `L{l}_w13_exps`, `L{l}_w2_exps` (N = 2 * L),
                            directly matching `vLLM` `RoutedExperts` (`w13_q` and `w2_q`)
      - `"layer"`         : One 3D expert group per MoE layer `L{l}_experts` (N = L)
      - `"expert"`        : Per-expert decision units `L{l}_E{e}` (N = L * E)
      - `"expert_tensor"` : Unconstrained per-expert per-projection `L{l}_E{e}_{gate,up,down}` (N = 3 * L * E)
    """
    import re
    from collections import defaultdict
    import numpy as np

    loss_np = np.asarray(loss_mat, dtype=np.float64)
    cost_np = np.asarray(cost_vec, dtype=np.float64)
    freq_np = np.asarray(freq_vec, dtype=np.float64)
    base_cost_1d = cost_np if cost_np.ndim == 1 else cost_np.mean(axis=0)

    # Detect whether input `experts` is already Per-3D-Tensor (`L{l}_{gate,up,down}_exps`)
    first_key = str(experts[0]) if len(experts) > 0 else ""
    is_input_3d_tensor = bool(
        re.match(r"L(\d+)_(gate_exps|up_exps|down_exps)$", first_key)
    )

    if is_input_3d_tensor:
        by_layer_proj: Dict[int, Dict[str, int]] = defaultdict(dict)
        for idx, key_s in enumerate(experts):
            mt = re.match(r"L(\d+)_(gate_exps|up_exps|down_exps)$", str(key_s))
            if mt:
                by_layer_proj[int(mt.group(1))][mt.group(2)] = idx
        sorted_layers = sorted(by_layer_proj.keys())
        E_l = float(max(1, int(num_experts_per_layer)))
        layer_to_exp_names = {
            lid: [f"L{lid}_E{e}" for e in range(int(E_l))] for lid in sorted_layers
        }

        units: List[str] = []
        loss_rows: List[np.ndarray] = []
        cost_rows: List[np.ndarray] = []
        freq_rows: List[float] = []

        if granularity == "tensor":
            for lid in sorted_layers:
                for ptag in ("gate_exps", "up_exps", "down_exps"):
                    if ptag in by_layer_proj[lid]:
                        row_i = by_layer_proj[lid][ptag]
                        units.append(f"L{lid}_{ptag}")
                        loss_rows.append(loss_np[row_i])
                        cost_rows.append((E_l / 3.0) * base_cost_1d)
                        freq_rows.append(float(freq_np[row_i]))
        elif granularity == "vllm_tensor":
            for lid in sorted_layers:
                pmap = by_layer_proj[lid]
                g_loss = loss_np[pmap["gate_exps"]] if "gate_exps" in pmap else 0.0
                u_loss = loss_np[pmap["up_exps"]] if "up_exps" in pmap else 0.0
                d_loss = loss_np[pmap["down_exps"]] if "down_exps" in pmap else 0.0
                units.append(f"L{lid}_w13_exps")
                loss_rows.append(g_loss + u_loss)
                cost_rows.append((2.0 * E_l / 3.0) * base_cost_1d)
                freq_rows.append(1.0)
                units.append(f"L{lid}_w2_exps")
                loss_rows.append(d_loss)
                cost_rows.append((E_l / 3.0) * base_cost_1d)
                freq_rows.append(1.0)
        elif granularity == "layer":
            for lid in sorted_layers:
                pmap = by_layer_proj[lid]
                tot_loss = sum(loss_np[i] for i in pmap.values())
                units.append(f"L{lid}_experts")
                loss_rows.append(tot_loss)
                cost_rows.append(E_l * base_cost_1d)
                freq_rows.append(1.0)
        else:
            raise ValueError(
                f"Input impact file is at 3D-tensor granularity ({first_key}); "
                f"cannot disaggregate to per-expert granularity={granularity!r} without per-expert impact file."
            )

        meta = {
            "granularity": granularity,
            "sorted_layers": sorted_layers,
            "layer_to_experts": layer_to_exp_names,
        }
        return (
            units,
            list(configs),
            np.stack(loss_rows, axis=0),
            np.stack(cost_rows, axis=0),
            np.array(freq_rows, dtype=np.float64),
            meta,
        )

    if granularity == "expert":
        cost_2d = np.broadcast_to(
            cost_np[None, :] if cost_np.ndim == 1 else cost_np,
            loss_np.shape,
        ).copy()
        meta = {"granularity": "expert", "layer_to_experts": {}}
        return list(experts), list(configs), loss_np, cost_2d, freq_np, meta

    by_layer: Dict[int, List[int]] = defaultdict(list)
    layer_to_exp_names: Dict[int, List[str]] = defaultdict(list)
    for idx, exp_key in enumerate(experts):
        m = re.match(r"L(\d+)_E(\d+)", str(exp_key))
        lid = int(m.group(1)) if m else 0
        by_layer[lid].append(idx)
        layer_to_exp_names[lid].append(str(exp_key))

    sorted_layers = sorted(by_layer.keys())

    units = []
    loss_rows = []
    cost_rows = []
    freq_rows = []

    def _cfg_br(cfg_str: str) -> Tuple[int, int]:
        bm = re.search(r"(\d+)bit", str(cfg_str))
        rm = re.search(r"rank(\d+)", str(cfg_str))
        return (int(bm.group(1)) if bm else 3), (int(rm.group(1)) if rm else 0)

    cfg_br_list = [_cfg_br(c) for c in configs]
    # Projection-specific distortion profile reflecting physical MoE activation statistics:
    # - gate/up (w13) receive smooth RMSNorm(x) inputs -> low-bit + low-rank SVD compensation is highly effective
    # - down (w2) receives heavy-tailed post-SwiGLU activations -> higher sensitivity to low bit-width, strong benefit from higher bit/rank
    raw_gate = 0.28 * np.array(
        [
            (0.82 if b <= 2 else (0.92 if b == 3 else 1.05))
            * (0.78 if r > 0 else 1.08)
            for (b, r) in cfg_br_list
        ],
        dtype=np.float64,
    )
    raw_up = 0.30 * np.array(
        [
            (0.86 if b <= 2 else (0.94 if b == 3 else 1.02))
            * (0.80 if r > 0 else 1.06)
            for (b, r) in cfg_br_list
        ],
        dtype=np.float64,
    )
    raw_down = 0.42 * np.array(
        [
            (1.65 if b <= 2 else (1.15 if b == 3 else 0.62))
            * (0.74 if r >= 32 else (0.85 if r > 0 else 1.15))
            for (b, r) in cfg_br_list
        ],
        dtype=np.float64,
    )
    # Normalize per configuration k so (w_gate[k] + w_up[k] + w_down[k] == 1.0),
    # guaranteeing identical objective scale across `layer`, `vllm_tensor`, `tensor`, `expert`, `expert_tensor`.
    raw_sum = np.maximum(raw_gate + raw_up + raw_down, 1e-12)
    w_gate = raw_gate / raw_sum
    w_up = raw_up / raw_sum
    w_down = raw_down / raw_sum
    w_w13 = w_gate + w_up

    if granularity == "layer":
        for lid in sorted_layers:
            idxs = by_layer[lid]
            E_l = float(len(idxs))
            units.append(f"L{lid}_experts")
            loss_rows.append(loss_np[idxs].sum(axis=0))
            cost_rows.append(E_l * base_cost_1d)
            freq_rows.append(float(freq_np[idxs].sum()))
    elif granularity == "tensor":
        # 3 3D expert tensors per layer: gate_exps (1/3 cost), up_exps (1/3 cost), down_exps (1/3 cost)
        proj_specs = [
            ("gate_exps", 1.0 / 3.0, 0.28, w_gate),
            ("up_exps", 1.0 / 3.0, 0.30, w_up),
            ("down_exps", 1.0 / 3.0, 0.42, w_down),
        ]
        for lid in sorted_layers:
            idxs = by_layer[lid]
            E_l = float(len(idxs))
            layer_loss_vec = loss_np[idxs].sum(axis=0)
            layer_freq = float(freq_np[idxs].sum())
            for proj_tag, cost_share, freq_share, prof in proj_specs:
                units.append(f"L{lid}_{proj_tag}")
                loss_rows.append(layer_loss_vec * prof)
                cost_rows.append((E_l * cost_share) * base_cost_1d)
                freq_rows.append(freq_share * layer_freq)
    elif granularity == "vllm_tensor":
        # 2 3D expert tensors per layer: w13_exps (2/3 cost, gate||up) and w2_exps (1/3 cost, down)
        proj_specs = [
            ("w13_exps", 2.0 / 3.0, 0.58, w_w13),
            ("w2_exps", 1.0 / 3.0, 0.42, w_down),
        ]
        for lid in sorted_layers:
            idxs = by_layer[lid]
            E_l = float(len(idxs))
            layer_loss_vec = loss_np[idxs].sum(axis=0)
            layer_freq = float(freq_np[idxs].sum())
            for proj_tag, cost_share, freq_share, prof in proj_specs:
                units.append(f"L{lid}_{proj_tag}")
                loss_rows.append(layer_loss_vec * prof)
                cost_rows.append((E_l * cost_share) * base_cost_1d)
                freq_rows.append(freq_share * layer_freq)
    elif granularity == "expert_tensor":
        # Full unconstrained per-expert per-projection (3 * L * E units)
        proj_specs = [
            ("gate_proj", 1.0 / 3.0, 0.28, w_gate),
            ("up_proj", 1.0 / 3.0, 0.30, w_up),
            ("down_proj", 1.0 / 3.0, 0.42, w_down),
        ]
        for idx, exp_key in enumerate(experts):
            for proj_tag, cost_share, freq_share, prof in proj_specs:
                units.append(f"{exp_key}.{proj_tag}")
                loss_rows.append(loss_np[idx] * prof)
                cost_rows.append(cost_share * base_cost_1d)
                freq_rows.append(freq_share * float(freq_np[idx]))
    else:
        raise ValueError(f"Unsupported granularity={granularity!r}")

    meta = {
        "granularity": granularity,
        "sorted_layers": sorted_layers,
        "layer_to_experts": {lid: layer_to_exp_names[lid] for lid in sorted_layers},
    }
    return (
        units,
        list(configs),
        np.stack(loss_rows, axis=0),
        np.stack(cost_rows, axis=0),
        np.array(freq_rows, dtype=np.float64),
        meta,
    )


def build_tensor_allocation_and_expanded_config(
    unit_config_map: Dict[str, str],
    meta: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Convert Per-Layer / Per-3D-Tensor RCO decision map into:
      1. `expanded_expert_config`: Per-expert map `{"L0_E0": "3bit_rank16", ...}` compatible
         with existing `BR-MoE` compression scripts (`BR_compress_auto.py`).
      2. `tensor_allocation`: Structured allocation dict for `llama.cpp` GGUF and `vLLM`
         (`gguf_tensors`, `vllm_layer_configs`, `projection_configs`).
    """
    import re

    granularity = meta.get("granularity", "tensor")
    if granularity in ("expert", "expert_tensor"):
        return dict(unit_config_map), {
            "granularity": granularity,
            "expert_configs": dict(unit_config_map),
        }

    def _parse_br(cfg_str: str) -> Tuple[int, int]:
        bm = re.search(r"(\d+)bit", cfg_str)
        rm = re.search(r"rank(\d+)", cfg_str)
        return (int(bm.group(1)) if bm else 3), (int(rm.group(1)) if rm else 0)

    layer_to_experts: Dict[int, List[str]] = meta.get("layer_to_experts", {})
    sorted_layers: List[int] = meta.get("sorted_layers", sorted(layer_to_experts.keys()))

    expanded_expert_config: Dict[str, str] = {}
    gguf_tensors: Dict[str, Dict[str, Any]] = {}
    vllm_layer_configs: Dict[str, Dict[str, int]] = {}
    projection_configs: Dict[str, Dict[str, str]] = {}

    for lid in sorted_layers:
        exp_names = layer_to_experts.get(lid, [])
        if granularity == "layer":
            cfg_l = unit_config_map[f"L{lid}_experts"]
            g_cfg = u_cfg = d_cfg = cfg_l
        elif granularity == "vllm_tensor":
            w13_cfg = unit_config_map[f"L{lid}_w13_exps"]
            w2_cfg = unit_config_map[f"L{lid}_w2_exps"]
            g_cfg = u_cfg = w13_cfg
            d_cfg = w2_cfg
        else:
            g_cfg = unit_config_map[f"L{lid}_gate_exps"]
            u_cfg = unit_config_map[f"L{lid}_up_exps"]
            d_cfg = unit_config_map[f"L{lid}_down_exps"]

        bg, rg = _parse_br(g_cfg)
        bu, ru = _parse_br(u_cfg)
        bd, rd = _parse_br(d_cfg)

        w13_bits = max(bg, bu)
        w13_rank = max(rg, ru)
        w2_bits = bd
        w2_rank = rd

        gguf_tensors[f"blk.{lid}.ffn_gate_exps.weight"] = {
            "bits": bg,
            "rank": rg,
            "gguf_type": _GGUF_TYPE_BY_BIT.get(bg, "IQ3_S"),
            "ggml_type": _GGUF_TYPE_BY_BIT.get(bg, "IQ3_S"),
            "config": g_cfg,
        }
        gguf_tensors[f"blk.{lid}.ffn_up_exps.weight"] = {
            "bits": bu,
            "rank": ru,
            "gguf_type": _GGUF_TYPE_BY_BIT.get(bu, "IQ3_S"),
            "ggml_type": _GGUF_TYPE_BY_BIT.get(bu, "IQ3_S"),
            "config": u_cfg,
        }
        gguf_tensors[f"blk.{lid}.ffn_down_exps.weight"] = {
            "bits": bd,
            "rank": rd,
            "gguf_type": _GGUF_TYPE_BY_BIT.get(bd, "IQ3_S"),
            "ggml_type": _GGUF_TYPE_BY_BIT.get(bd, "IQ3_S"),
            "config": d_cfg,
        }
        vllm_layer_configs[f"model.layers.{lid}.mlp.experts"] = {
            "w13_bits": w13_bits,
            "w13_rank": w13_rank,
            "w2_bits": w2_bits,
            "w2_rank": w2_rank,
        }
        projection_configs[f"L{lid}"] = {
            "gate_proj": g_cfg,
            "up_proj": u_cfg,
            "down_proj": d_cfg,
        }

        rep_b = max(w13_bits, w2_bits)
        rep_r = max(w13_rank, w2_rank)
        rep_cfg = g_cfg if (g_cfg == u_cfg == d_cfg) else f"{rep_b}bit_rank{rep_r}"
        for exp_key in exp_names:
            expanded_expert_config[exp_key] = rep_cfg

    tensor_allocation = {
        "granularity": granularity,
        "tensor_configs": dict(unit_config_map),
        "projection_configs": projection_configs,
        "gguf_tensors": gguf_tensors,
        "vllm_layer_configs": vllm_layer_configs,
        "vllm_layers": vllm_layer_configs,
    }
    return expanded_expert_config, tensor_allocation

