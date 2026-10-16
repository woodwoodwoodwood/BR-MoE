import json
import re
import importlib.resources as pkg_resources
from typing import Any, Dict, Optional

try:
    from ..core.quantize import BRMoELinear
    from .utils import compensator_dequantize
except Exception:
    BRMoELinear = None
    compensator_dequantize = None
try:
    from ..core import model_statistics
except Exception:
    model_statistics = None

MIXTRAL_LAYERS = {
    "dense": ["self_attn"],
    "sparse": ["experts"],
    "layer_count": 32,
}
DEEPSEEK_LAYERS = {
    "dense": ["self_attn", "shared", "layers.0.mlp"],
    "sparse": ["mlp.experts"],
    "layer_count": 27,
}
QWEN15MOE_LAYERS = {
    "dense": ["self_attn", "mlp.shared_expert", "mlp.shared_experts"],
    "sparse": ["mlp.experts"],
    "layer_count": 24,
}
QWEN3MOE_LAYERS = {
    "dense": ["self_attn", "mlp.shared_expert", "mlp.shared_experts"],
    "sparse": ["mlp.experts"],
    "layer_count": 48,
}
# Dense FFN projections are the concatenated expert matrix (no `experts.{i}` module).
QWEN3_DENSE_LAYERS = {
    "dense": ["self_attn", "linear_attn"],
    "sparse": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "layer_count": 36,
}
QWEN3_NEXT_LAYERS = {
    # Covers Qwen3.6-35B-A3B, Qwen3.5-35B-A3B, Qwen3-Next-80B-A3B (hybrid self_attn + linear_attn)
    "dense": [
        "self_attn",
        "linear_attn",
        "mlp.shared_expert",
        "mlp.shared_experts",
    ],
    "sparse": ["mlp.experts"],
    "layer_count": 40,
}
GLM4MOE_LAYERS = {
    "dense": ["self_attn", "mlp.shared_experts", "mlp.shared_expert", "layers.0.mlp"],
    "sparse": ["mlp.experts"],
    "layer_count": 46,
}
OLMOE_LAYERS = {
    "dense": ["self_attn"],
    "sparse": ["mlp.experts"],
    "layer_count": 16,
}
PHIMOE_LAYERS = {
    "dense": ["self_attn"],
    "sparse": ["block_sparse_moe.experts", "experts"],
    "layer_count": 32,
}
UNIVERSAL_MOE_LAYERS = {
    "dense": [
        "self_attn",
        "linear_attn",
        "mlp.shared_expert",
        "mlp.shared_experts",
        "shared_mlp",
    ],
    "sparse": ["mlp.experts", "block_sparse_moe.experts", "experts"],
    "layer_count": 40,
}


def _is_dense_qwen(combined: str) -> bool:
    """True for dense Qwen3 / Qwen3.5 / Qwen3.8 text models, not their MoE siblings."""
    if any(tag in combined for tag in ("qwen3-8b", "qwen3.8", "qwen3_8")):
        return True
    if any(tag in combined for tag in ("moe", "a3b", "a22b", "a17b", "qwen3-next", "qwen3_next", "-next")):
        return False
    return "qwen3_5_text" in combined or "qwen3forcausallm" in combined


def get_model_layer_info(model_id: Optional[str], model: Any = None) -> Dict[str, Any]:
    """Resolve layer categorization for any known or custom MoE model."""
    mid = (model_id or "").lower()
    arch_str = ""
    model_type = ""
    num_layers = None

    if model is not None and hasattr(model, "config"):
        cfg = model.config
        archs = getattr(cfg, "architectures", None) or []
        if archs:
            arch_str = archs[0].lower()
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
        model_type = getattr(text_cfg, "model_type", getattr(cfg, "model_type", "")).lower()
        num_layers = getattr(text_cfg, "num_hidden_layers", None)

    combined = f"{mid} {arch_str} {model_type}"

    if "mixtral" in combined:
        info = dict(MIXTRAL_LAYERS)
    elif "deepseek" in combined:
        info = dict(DEEPSEEK_LAYERS)
    elif _is_dense_qwen(combined):
        info = dict(QWEN3_DENSE_LAYERS)
        if "qwen3.8" in combined or "qwen3_8" in combined:
            info["layer_count"] = 64
    elif any(k in combined for k in ("qwen3.6", "qwen3_6", "qwen3.5", "qwen3_5", "qwen3_next", "qwen3-next", "35b-a3b", "80b-a3b")):
        info = dict(QWEN3_NEXT_LAYERS)
    elif any(k in combined for k in ("qwen3", "30b-a3b", "235b-a22b")):
        info = dict(QWEN3MOE_LAYERS)
    elif "qwen" in combined:
        info = dict(QWEN15MOE_LAYERS)
    elif "glm" in combined:
        info = dict(GLM4MOE_LAYERS)
    elif "olmoe" in combined:
        info = dict(OLMOE_LAYERS)
    elif "phi" in combined:
        info = dict(PHIMOE_LAYERS)
    else:
        info = dict(UNIVERSAL_MOE_LAYERS)

    if num_layers is not None:
        info["layer_count"] = int(num_layers)
    return info


def resolve_module_rank(
    module_name: str,
    ranks: Dict[str, int],
    default_sparse_rank: Optional[int] = None,
    default_dense_rank: Optional[int] = None,
) -> Optional[int]:
    """Boundary-aware rank lookup that avoids `experts.1` matching `experts.10`."""
    if not ranks:
        if "experts" in module_name and "shared" not in module_name:
            return default_sparse_rank
        return default_dense_rank

    # 1. Exact match & canonical prefix variants (e.g. VLM `model.language_model.layers` <-> `model.layers`)
    candidates = [
        module_name,
        module_name.replace("model.language_model.layers.", "model.layers."),
        module_name.replace("language_model.model.layers.", "model.layers."),
        module_name.replace("language_model.layers.", "model.layers."),
    ]
    if not module_name.startswith("model."):
        candidates.append(f"model.{module_name}")

    for cand in candidates:
        if cand in ranks:
            return ranks[cand]

    # 2. Check parent prefix matches in O(L) (e.g. `model.layers.0.mlp.experts.0.` or `layers.0.mlp`)
    for cand in candidates:
        parts = cand.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in ranks:
                return ranks[prefix]
            if (prefix + ".") in ranks:
                return ranks[prefix + "."]

    # 3. Boundary-safe substring match on short pattern keys (e.g. "self_attn", "linear_attn", "mlp.experts")
    pattern_items = [
        (k, v) for k, v in ranks.items()
        if k and not k.startswith("model.layers.") and not k.startswith("model.language_model.layers.")
    ]
    pattern_items.sort(key=lambda kv: len(kv[0]), reverse=True)
    for cand in candidates:
        is_routed_expert = (".experts." in cand) and ("shared" not in cand)
        for key, value in pattern_items:
            if is_routed_expert and ("experts" not in key):
                continue
            idx = cand.find(key)
            if idx != -1:
                end_idx = idx + len(key)
                # Ensure we don't match a partial number (e.g. `experts.1` inside `experts.10`)
                if end_idx < len(cand) and key[-1].isdigit() and cand[end_idx].isdigit():
                    continue
                return value

    # 3. Fallback to default sparse/dense rank if provided
    if "experts" in module_name and "shared" not in module_name:
        return default_sparse_rank
    return default_dense_rank


def rank_generate(
    model_id: str,
    compress_config: Dict[str, Any],
    sparse_rank: int,
    dense_rank: int,
    strategy: Optional[str],
    model: Any = None,
) -> Dict[str, int]:
    model_layer_info = get_model_layer_info(model_id, model=model)
    mid_lower = (model_id or "").lower()

    if strategy is None or strategy == "uniform":
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
            **{name: sparse_rank for name in model_layer_info["sparse"]},
        }
    elif strategy == "zero":
        ranks = {
            **{name: 0 for name in model_layer_info["dense"]},
            **{name: 0 for name in model_layer_info["sparse"]},
        }
    elif strategy == "custom":
        print("Using 'custom' rank strategy: extracting ranks from per-module configs.")
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
            **{name: sparse_rank for name in model_layer_info["sparse"]},
        }

        for name, config in compress_config.items():
            if isinstance(config, dict) and "compensator_params" in config:
                comp_params = config["compensator_params"]
                if "experts" in name and "shared" not in name:
                    rank_to_use = comp_params.get("sparse_rank", sparse_rank)
                else:
                    rank_to_use = comp_params.get("dense_rank", dense_rank)
                ranks[name] = rank_to_use
    elif strategy == "frequency" and "deepseek" in mid_lower:
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        with pkg_resources.files(model_statistics).joinpath("DeepSeek_expt_freq.json").open("r") as f:
            data = json.load(f)
        for layer_index in range(len(data)):
            freq = data[layer_index]
            freq_sum = sum(freq)
            for expert_index in range(len(freq)):
                rank = int(round(freq[expert_index] / freq_sum * (sparse_rank * len(freq))))
                ranks[f"layers.{layer_index + 1}.mlp.experts.{expert_index}."] = rank
    elif strategy == "frequency" and "qwen" in mid_lower:
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        try:
            with pkg_resources.files(model_statistics).joinpath("Qwen15_expt_freq.json").open("r") as f:
                data = json.load(f)
            print("Loaded Qwen MoE expert frequency data")
            for key, freq in data.items():
                if "-" in key:
                    layer_idx, expert_idx = key.split("-")
                    layer_idx, expert_idx = int(layer_idx), int(expert_idx)
                    rank = int(round(freq * sparse_rank * 8))
                    ranks[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."] = rank
        except FileNotFoundError:
            print("Warning: Qwen15_expt_freq.json not found, using uniform ranks for all experts")
            ranks.update({name: sparse_rank for name in model_layer_info["sparse"]})
    elif strategy == "Kurtosis" and "mixtral" in mid_lower:
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        if sparse_rank == 16:
            k = 2
        elif sparse_rank == 32:
            k = 3
        else:
            raise NotImplementedError("Currently Mixtral Kurtosis strategy only supports avg rank of 16 and 32")
        with pkg_resources.files(model_statistics).joinpath("Mixtral_kurtosis_values.json").open("r") as f:
            data = json.load(f)
        for name, kurtosis in data.items():
            kurtosis = round(float(kurtosis))
            if "self_attn" in name:
                continue
            if kurtosis < 1:
                rank = 0
            elif kurtosis > 9:
                rank = 1024
            else:
                rank = 2 ** (kurtosis + k)
            text = name.replace(".weight", "").strip()
            ranks[text] = rank
    else:
        # Graceful fallback for any other model/strategy combination
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
            **{name: sparse_rank for name in model_layer_info["sparse"]},
        }
    return ranks


def load_compensators(model, compensators, ranks):
    for name, module in model.named_modules():
        if type(module) is BRMoELinear:
            UV_quantized = compensators.pop(name, None)
            orig_shape = module.meta["shape"]
            default_sr = module.compress_config.get("compensator_params", {}).get("sparse_rank", 0)
            default_dr = module.compress_config.get("compensator_params", {}).get("dense_rank", 0)
            rank = resolve_module_rank(name, ranks, default_sparse_rank=default_sr, default_dense_rank=default_dr)
            compensator_dtype = module.compress_config["compensator_params"]["compensator_dtype"]
            compensator_quantize_gs = module.compress_config["compensator_params"]["compensator_quant_gs"]
            if rank is not None and rank > 0 and UV_quantized is not None:
                module.U, module.V = compensator_dequantize(
                    UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype
                )
