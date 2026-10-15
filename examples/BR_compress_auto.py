"""Universal BR-MoE compression entry point for modern MoE models.

Supports:
  - Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3-Next-80B-A3B
    (including hybrid Gated DeltaNet `linear_attn` + `self_attn`, VLM `language_model` wrapper,
     and 3D fused `experts.gate_up_proj` / `down_proj` unpacking)
  - Qwen/Qwen3-30B-A3B, Qwen/Qwen3-235B-A22B
  - Qwen/Qwen1.5-MoE-A2.7B
  - deepseek-ai/DeepSeek-MoE-16b-base, deepseek-ai/DeepSeek-V2-Lite, deepseek-ai/DeepSeek-V3
  - zai-org/GLM-4.5-Air, allenai/OLMoE-1B-7B-0924, microsoft/Phi-3.5-MoE-instruct
  - mistralai/Mixtral-8x7B-v0.1, mistralai/Mixtral-8x22B-v0.1

Example usage:
  # 1. Generate optimal expert bit/rank allocation for Qwen3.6-35B-A3B via RCO solver:
  python -m RCO_solver.rco_search --model_preset Qwen/Qwen3.6-35B-A3B --avg_bits 3.15 --output best_config_qwen3_6.json

  # 2. Compress Qwen3.6-35B-A3B with the generated config:
  python examples/BR_compress_auto.py \
      --model_path /path/to/Qwen3.6-35B-A3B \
      --save_dir /path/to/Qwen3.6-35B-A3B-BRMoE \
      --config best_config_qwen3_6.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_INNER_PKG = _REPO_ROOT / "BR-MoE"
if _INNER_PKG.exists() and str(_INNER_PKG) not in sys.path:
    sys.path.insert(0, str(_INNER_PKG))

try:
    from BR_MoE.core.quantize import br_moe_base_compress_config
    from BR_MoE.engine.hf import AutoBRMoEHFModel
    from BR_MoE.models.hf.universal_moe import (
        UniversalMoEBRMoE,
        ensure_unfused_moe_experts,
        get_text_backbone_and_prefix,
    )
except ImportError:
    from core.quantize import br_moe_base_compress_config
    from engine.hf import AutoBRMoEHFModel
    from models.hf.universal_moe import (
        UniversalMoEBRMoE,
        ensure_unfused_moe_experts,
        get_text_backbone_and_prefix,
    )


logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("BR_compress_auto")


def load_hf_moe_model(model_path: str, dtype: torch.dtype = torch.float16):
    """Load causal LM or multimodal VLM MoE (e.g. Qwen3.6-35B-A3B) and unpack 3D experts."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = getattr(config, "architectures", None) or []
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception:
        model = None
        for arch_name in archs:
            cls_obj = getattr(transformers, arch_name, None)
            if cls_obj is not None:
                model = cls_obj.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                break
        if model is None and hasattr(transformers, "AutoModelForImageTextToText"):
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        if model is None:
            raise RuntimeError(f"Unable to load model from {model_path} (architectures={archs})")

    num_unpacked = ensure_unfused_moe_experts(model)
    if num_unpacked > 0:
        logger.info(f"Unpacked {num_unpacked} 3D fused expert layers into per-expert Linear modules.")
    return model


def build_auto_compress_config(
    model: torch.nn.Module,
    best_configs: Dict[str, Any],
    attn_bits: int = 3,
    attn_rank: int = 512,
    shared_expert_bits: int = 4,
    shared_expert_rank: int = 512,
    default_expert_bits: int = 3,
    default_expert_rank: int = 16,
    group_size: int = 128,
    compensator_dtype: str = "int3",
    compensator_quant_gs: int = 64,
    svd_iter: int = 20,
    skip_attention: bool = False,
    skip_shared_expert: bool = False,
) -> Dict[str, Any]:
    """Build a model-aware `compress_config` dictionary from `best_configs` and model structure."""
    config_cache: Dict[str, Dict[str, Any]] = {}

    def get_or_create_config(bits: int, rank: int) -> Dict[str, Any]:
        key = f"{bits}bit_rank{rank}"
        if key not in config_cache:
            cfg = br_moe_base_compress_config(
                nbits=bits,
                group_size=group_size,
                quant_zero=False,
                quant_scale=False,
                offload_meta=False,
                view_as_float=False,
                axis=1,
                iter=svd_iter,
                sparse_rank=rank,
                dense_rank=rank,
                rank_strategy="custom",
                compensator_dtype=compensator_dtype,
                compensator_quant_gs=compensator_quant_gs,
            )
            cfg["compensator_params"]["ranks"] = {}
            config_cache[key] = cfg
        return config_cache[key]

    compress_config: Dict[str, Any] = {
        "skip_attention": skip_attention,
        "skip_shared_expert": skip_shared_expert,
        "compensator_params": {
            "iter": svd_iter,
            "sparse_rank": default_expert_rank,
            "dense_rank": attn_rank,
            "rank_strategy": "custom",
            "compensator_dtype": compensator_dtype,
            "compensator_quant_gs": compensator_quant_gs,
            "ranks": {},
        },
    }

    # 1. Configure attention projections (standard GQA, DeepSeek MLA, and Qwen3.6/3.5/Next Gated DeltaNet linear_attn)
    if not skip_attention:
        attn_cfg = get_or_create_config(attn_bits, attn_rank)
        for attn_key in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
            "linear_attn.in_proj_qkvz",
            "linear_attn.in_proj_ba",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
        ):
            compress_config[attn_key] = attn_cfg

    _, layers_list, layer_prefix = get_text_backbone_and_prefix(model)

    # 2. Configure shared experts and dense MLP layers dynamically from model structure
    shared_cfg = get_or_create_config(shared_expert_bits, shared_expert_rank)
    for layer_idx, layer in enumerate(layers_list):
        lp = f"{layer_prefix}.{layer_idx}"
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if not skip_shared_expert:
            for sh_attr in ("shared_expert", "shared_experts"):
                sh_mod = getattr(mlp, sh_attr, None)
                if sh_mod is not None:
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        if hasattr(sh_mod, proj):
                            compress_config[f"{lp}.mlp.{sh_attr}.{proj}"] = shared_cfg
                            compress_config[f"model.layers.{layer_idx}.mlp.{sh_attr}.{proj}"] = shared_cfg
        # Dense MLP layer (e.g. DeepSeek / GLM4 first_k_dense_replace)
        if not hasattr(mlp, "experts"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if hasattr(mlp, proj):
                    compress_config[f"{lp}.mlp.{proj}"] = shared_cfg
                    compress_config[f"model.layers.{layer_idx}.mlp.{proj}"] = shared_cfg

    # 3. Parse per-expert configurations from `best_configs`
    config_stats: Dict[str, int] = {}
    parsed_experts = set()

    for expert_key, raw_val in best_configs.items():
        m_le = re.match(r"L(\d+)_E(\d+)", str(expert_key))
        if not m_le:
            continue
        layer_idx = int(m_le.group(1))
        expert_idx = int(m_le.group(2))

        cfg_str = raw_val["config"] if isinstance(raw_val, dict) and "config" in raw_val else str(raw_val)
        if cfg_str.startswith("prune") or cfg_str.startswith("0bit"):
            # Pruned expert: fallback to minimal 2-bit rank-0 if physical pruning is not active
            bits, rank = 2, 0
        else:
            m_br = re.match(r"(\d+)bit_rank(\d+)", cfg_str)
            if not m_br:
                continue
            bits = int(m_br.group(1))
            rank = int(m_br.group(2))

        cfg_obj = get_or_create_config(bits, rank)
        config_stats[cfg_str] = config_stats.get(cfg_str, 0) + 1
        parsed_experts.add((layer_idx, expert_idx))

        for proj in ("gate_proj", "up_proj", "down_proj"):
            compress_config[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"] = cfg_obj
            if layer_prefix != "model.layers":
                compress_config[f"{layer_prefix}.{layer_idx}.mlp.experts.{expert_idx}.{proj}"] = cfg_obj
        for proj in ("w1", "w2", "w3"):
            compress_config[f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{proj}"] = cfg_obj

    # 4. Fill any remaining routed experts with default_expert_bits / default_expert_rank
    default_expert_cfg = get_or_create_config(default_expert_bits, default_expert_rank)
    for layer_idx, layer in enumerate(layers_list):
        lp = f"{layer_prefix}.{layer_idx}"
        moe_block = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if moe_block is None or not hasattr(moe_block, "experts"):
            continue
        moe_attr = "mlp" if hasattr(layer, "mlp") else "block_sparse_moe"
        experts = moe_block.experts
        if isinstance(experts, torch.nn.ModuleList):
            for expert_idx, exp_mod in enumerate(experts):
                if (layer_idx, expert_idx) in parsed_experts:
                    continue
                for proj in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"):
                    if hasattr(exp_mod, proj):
                        compress_config[f"model.layers.{layer_idx}.{moe_attr}.experts.{expert_idx}.{proj}"] = default_expert_cfg
                        if layer_prefix != "model.layers":
                            compress_config[f"{lp}.{moe_attr}.experts.{expert_idx}.{proj}"] = default_expert_cfg

    if config_stats:
        total = sum(config_stats.values())
        logger.info(f"Loaded expert configurations for {total} experts:")
        for cname, cnt in sorted(config_stats.items(), key=lambda kv: kv[1], reverse=True):
            logger.info(f"  {cname}: {cnt} ({cnt / total * 100:.1f}%)")
    else:
        logger.info(
            f"Using uniform routed expert configuration: {default_expert_bits}bit_rank{default_expert_rank}"
        )

    return compress_config


def main():
    parser = argparse.ArgumentParser(description="Universal BR-MoE Model Compression")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF repo ID of the MoE model")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for compressed model")
    parser.add_argument("--config", type=str, default="", help="Path to best_config.json (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for compression (cuda or cpu)")
    parser.add_argument("--attn_bits", type=int, default=3, help="Bit-width for attention layers")
    parser.add_argument("--attn_rank", type=int, default=512, help="Compensator rank for attention layers")
    parser.add_argument("--shared_bits", type=int, default=4, help="Bit-width for shared experts")
    parser.add_argument("--shared_rank", type=int, default=512, help="Compensator rank for shared experts")
    parser.add_argument("--default_bits", type=int, default=3, help="Default bit-width for routed experts")
    parser.add_argument("--default_rank", type=int, default=16, help="Default rank for routed experts")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--svd_iter", type=int, default=20, help="Randomized SVD iterations")
    parser.add_argument("--skip_attention", action="store_true", help="Keep attention layers in FP16/BF16")
    parser.add_argument("--skip_shared_expert", action="store_true", help="Keep shared experts in FP16/BF16")
    args = parser.parse_args()

    best_configs: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading optimal expert configurations from: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            best_configs = json.load(f)

    logger.info(f"Loading model from: {args.model_path}")
    model = load_hf_moe_model(args.model_path, dtype=torch.float16)

    compress_config = build_auto_compress_config(
        model=model,
        best_configs=best_configs,
        attn_bits=args.attn_bits,
        attn_rank=args.attn_rank,
        shared_expert_bits=args.shared_bits,
        shared_expert_rank=args.shared_rank,
        default_expert_bits=args.default_bits,
        default_expert_rank=args.default_rank,
        group_size=args.group_size,
        svd_iter=args.svd_iter,
        skip_attention=args.skip_attention,
        skip_shared_expert=args.skip_shared_expert,
    )

    model.mix_moe_bit = True
    logger.info("Starting BR-MoE model compression...")
    UniversalMoEBRMoE.compress_model(
        model=model,
        compress_config=compress_config,
        device=args.device,
    )

    logger.info(f"Saving compressed model to: {args.save_dir}")
    UniversalMoEBRMoE.save_compressed(model, args.save_dir)
    logger.info("Compression completed successfully!")


if __name__ == "__main__":
    main()
