#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import json
import os
import logging
import argparse
import time
import pandas as pd
import glob
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from BR_MoE.models.hf.qwen import Qwen15MoEBRMoE as AutoBRMoEHFModel
from BR_MoE.core.quantize import *

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
MODEL_PATH = "./models/qwen1.5-moe"
EXPERT_FREQ_PATH = "./expert_impact_results/qwen15_expert_freq.json"
RESULTS_DIR = "./expert_impact_results"
DATASET_PATH = "./datasets/wikitext-2-raw-v1"


def _get_model_input_device(model):
    try:
        return next(model.model.embed_tokens.parameters()).device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_expert_device(model, layer_idx: int, expert_idx: int) -> str:
    leaf_paths = [
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj",
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj",
        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj"
    ]
    
    device_map = getattr(model, "hf_device_map", None)
    dev = None
    
    if isinstance(device_map, dict):
        for leaf_path in leaf_paths:
            dev = device_map.get(leaf_path, None)
            if dev is not None:
                break
                
        if dev is None:
            target_prefix = f"model.layers.{layer_idx}"
            best_key = ""
            best_val = None
            for k, v in device_map.items():
                if k.startswith(target_prefix) or target_prefix.startswith(k):
                    if len(k) > len(best_key):
                        best_key, best_val = k, v
            dev = best_val

    if not dev or str(dev).startswith("meta"):
        dev = str(_get_model_input_device(model))
    return str(dev)


def _materialize_expert_if_needed(expert, device: torch.device) -> None:
    for lname in ["gate_proj", "up_proj", "down_proj"]:
        lin = getattr(expert, lname, None)
        if lin is None:
            continue
        try:
            w = lin.weight
        except Exception:
            w = None
        is_meta = getattr(w, "is_meta", False) if w is not None else False
        if is_meta:
            in_features = getattr(lin, "in_features", None)
            if in_features is None:
                logger.warning(f"Cannot get in_features for {lname}, skipping materialize attempt")
                continue
            dtype = w.dtype if (w is not None and w.dtype.is_floating_point) else torch.float16
            dummy = torch.zeros(1, in_features, device=device, dtype=dtype)
            try:
                with torch.inference_mode():
                    _ = lin(dummy)
            except Exception as e:
                logger.warning(f"Failed to materialize {lname}: {e}")


def create_quant_config(nbits, rank, group_size=128):
    config = BR_moe_base_compress_config(
        nbits=nbits, group_size=group_size, quant_zero=False, quant_scale=False,
        offload_meta=False, view_as_float=False, axis=1, iter=10, 
        sparse_rank=rank, dense_rank=0, rank_strategy=None, 
        compensator_dtype="int3", compensator_quant_gs=64
    )
    config["compensator_params"]["ranks"] = {}
    return config


def prepare_calibration_data(tokenizer, num_samples=32, seq_len=512, dataset_path=DATASET_PATH, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading calibration data from {dataset_path}")
    logger.info(f"Target: {num_samples} samples, each length {seq_len}")
    
    all_parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
    if not all_parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")
    
    parquet_files = [f for f in all_parquet_files if os.path.basename(f).startswith("test")]
    if not parquet_files:
        logger.warning(f"No test parquet files found in {dataset_path}, using all files")
        parquet_files = all_parquet_files
    
    logger.info(f"Found {len(parquet_files)} qualifying parquet files")
    
    texts = []
    for parquet_file in parquet_files:
        logger.info(f"Loading: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        
        if 'text' in df.columns:
            column_name = 'text'
        else:
            column_name = df.columns[0]
            logger.warning(f"'text' column not found, using column: {column_name}")
        
        batch_texts = df[column_name].dropna().tolist()
        texts.extend(batch_texts)
    
    if len(texts) < num_samples:
        logger.warning(f"Dataset has only {len(texts)} records, less than requested {num_samples}")
        while len(texts) < num_samples:
            texts.extend(texts[:num_samples-len(texts)])
    
    selected_texts = texts[:num_samples]
    
    logger.info("Sample examples:")
    num_examples = min(3, len(selected_texts))
    for i in range(num_examples):
        logger.info(f"\nSample {i+1}:\n{selected_texts[i][:200]}...")
    
    logger.info("Tokenizing text...")
    inputs = tokenizer(
        selected_texts, 
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="pt"
    )
    
    logger.info("Tokenized examples:")
    for i in range(min(2, len(selected_texts))):
        tokens = inputs.input_ids[i][:20].tolist()
        decoded = tokenizer.decode(tokens)
        logger.info(f"\nTokenized sample {i+1} (first 20 tokens):\n{decoded}...")
    
    logger.info(f"Calibration data ready, shape: {inputs.input_ids.shape}")
    return inputs.input_ids.to(device)


def get_and_save_reference_outputs_with_hooks(model, input_ids, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Computing FP16 reference outputs using hooks, caching to: {cache_dir}")
    
    num_layers = model.config.num_hidden_layers
    if all(os.path.exists(os.path.join(cache_dir, f'layer_{i}_output.pt')) for i in range(num_layers)):
        logger.info("All reference output files exist in cache, skipping computation.")
        return

    hooks = []
    captured_outputs = [None] * num_layers

    def create_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured_outputs[layer_idx] = output[0].clone().cpu()
            else:
                captured_outputs[layer_idx] = output.clone().cpu()
        return hook

    for i, layer in enumerate(model.model.layers):
        hook_handle = layer.register_forward_hook(create_hook(i))
        hooks.append(hook_handle)

    with torch.inference_mode():
        input_dev = _get_model_input_device(model)
        model(input_ids.to(input_dev))

    for hook in hooks:
        hook.remove()
        
    for i, output_tensor in enumerate(captured_outputs):
        if output_tensor is not None:
            torch.save(output_tensor, os.path.join(cache_dir, f'layer_{i}_output.pt'))
        else:
            logger.warning(f"Failed to capture output for layer {i}.")
            
    logger.info(f"Computed and cached reference outputs for {num_layers} layers.")


def get_quantized_output_with_hook(model, input_ids, layer_idx):
    quantized_output = None
    
    def hook(module, input, output):
        nonlocal quantized_output
        if isinstance(output, tuple):
            quantized_output = output[0].clone()
        else:
            quantized_output = output.clone()

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(hook)
    
    with torch.inference_mode():
        input_dev = _get_model_input_device(model)
        model(input_ids.to(input_dev))
        
    hook_handle.remove()
    return quantized_output


def evaluate_expert_impact(model, reference_output_path, input_ids, layer_idx, expert_idx, nbits, rank):
    logger.info(f"Evaluating: Layer-{layer_idx}, Expert-{expert_idx} | Config: {nbits}bit-rank{rank}")

    expert = model.model.layers[layer_idx].mlp.experts[expert_idx]
    resolved_dev_str = _resolve_expert_device(model, layer_idx, expert_idx)
    target_device = torch.device(resolved_dev_str)
    
    try:
        is_meta_any = any(getattr(getattr(expert, w).weight, "is_meta", False) 
                         for w in ["gate_proj", "up_proj", "down_proj"]) 
    except Exception:
        is_meta_any = False
    
    if is_meta_any:
        logger.warning(f"Detected meta weights: L{layer_idx} E{expert_idx}, attempting materialize on {resolved_dev_str}...")
        _materialize_expert_if_needed(expert, target_device)

    orig_gate_proj = expert.gate_proj
    orig_up_proj = expert.up_proj
    orig_down_proj = expert.down_proj

    leaf_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
    try:
        orig_gate_proj.name = f"{leaf_prefix}.gate_proj"
        orig_up_proj.name = f"{leaf_prefix}.up_proj"
        orig_down_proj.name = f"{leaf_prefix}.down_proj"
    except Exception:
        setattr(orig_gate_proj, "name", f"{leaf_prefix}.gate_proj")
        setattr(orig_up_proj, "name", f"{leaf_prefix}.up_proj")
        setattr(orig_down_proj, "name", f"{leaf_prefix}.down_proj")

    quant_cfg = create_quant_config(nbits, rank)
    quant_cfg["compensator_params"]["ranks"] = {
        f"{leaf_prefix}.gate_proj": rank,
        f"{leaf_prefix}.up_proj": rank,
        f"{leaf_prefix}.down_proj": rank,
    }

    l2_diff = -1.0
    rel_diff = -1.0
    q_gate_proj = q_up_proj = q_down_proj = None
    try:
        q_gate_proj = BRMoELinear(
            linear_layer=orig_gate_proj,
            compress_config=quant_cfg,
            compute_dtype=torch.float16,
            device=resolved_dev_str,
            del_orig=False,
        )
        q_up_proj = BRMoELinear(
            linear_layer=orig_up_proj,
            compress_config=quant_cfg,
            compute_dtype=torch.float16,
            device=resolved_dev_str,
            del_orig=False,
        )
        q_down_proj = BRMoELinear(
            linear_layer=orig_down_proj,
            compress_config=quant_cfg,
            compute_dtype=torch.float16,
            device=resolved_dev_str,
            del_orig=False,
        )

        expert.gate_proj = q_gate_proj
        expert.up_proj = q_up_proj
        expert.down_proj = q_down_proj

        quantized_output = get_quantized_output_with_hook(model, input_ids, layer_idx)
        reference_output = torch.load(reference_output_path, map_location=quantized_output.device)
        
        l2_diff = torch.norm(reference_output - quantized_output).item()
        rel_diff = (torch.norm(reference_output - quantized_output) / torch.norm(reference_output)).item()
        
        del reference_output

    finally:
        expert.gate_proj = orig_gate_proj
        expert.up_proj = orig_up_proj
        expert.down_proj = orig_down_proj
        
        del q_gate_proj, q_up_proj, q_down_proj
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "l2_norm": l2_diff,
        "relative_error": rel_diff
    }


def main():
    parser = argparse.ArgumentParser(description="Efficient GPU-based evaluation of Qwen1.5-MoE expert quantization impact using hooks")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Calibration dataset path")
    parser.add_argument("--expert_freq_path", type=str, default=EXPERT_FREQ_PATH, help="Expert frequency file path")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR, help="Results output directory")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of calibration data samples")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--start_layer", type=int, default=0, help="Start evaluation from which layer")
    parser.add_argument("--start_expert", type=int, default=0, help="Start evaluation from which expert in the layer")
    parser.add_argument("--bits", type=str, default="2,3,4", help="Quantization bits to test, comma separated")
    parser.add_argument("--ranks", type=str, default="0,16,32,64,256,1024", help="Sparse ranks to test, comma separated")
    parser.add_argument("--specific_layer", type=int, default=None, help="Evaluate only specific layer")
    parser.add_argument("--specific_expert", type=int, default=None, help="Evaluate only specific expert")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"qwen15_expert_impact_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    
    bit_configs = [int(b) for b in args.bits.split(',')]
    rank_configs = [int(r) for r in args.ranks.split(',')]
    
    logger.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
    
    input_device = _get_model_input_device(model)
    input_ids = prepare_calibration_data(tokenizer, args.num_samples, args.seq_len, dataset_path=args.dataset_path, device=input_device)
    
    ref_output_cache_dir = os.path.join(args.output_dir, "reference_outputs_cache_hook")
    get_and_save_reference_outputs_with_hooks(model, input_ids, ref_output_cache_dir)
    
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Successfully loaded existing results, {len(results)} experts evaluated.")
        except json.JSONDecodeError:
            logger.warning(f"Results file {results_file} is corrupted, creating new file.")

    expert_freq = {}
    if os.path.exists(args.expert_freq_path):
        try:
            with open(args.expert_freq_path, 'r') as f:
                expert_freq = json.load(f)
            logger.info(f"Loaded expert frequency file: {len(expert_freq)} records")
            
            sorted_experts = sorted(expert_freq.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 most frequent experts:")
            for i, (expert_key, freq) in enumerate(sorted_experts[:10]):
                if i >= 10:
                    break
                layer_idx = expert_key.split('_')[0][1:]
                expert_idx = expert_key.split('_')[1][1:]
                logger.info(f"#{i+1}: Layer {layer_idx} Expert {expert_idx} - Frequency {freq:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load expert frequency file: {e}")

    start_time = time.time()
    
    tasks = []
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts
    
    for layer_idx in range(num_layers):
        if args.specific_layer is not None and layer_idx != args.specific_layer:
            continue
        if layer_idx < args.start_layer:
            continue
            
        for expert_idx in range(num_experts):
            if args.specific_expert is not None and expert_idx != args.specific_expert:
                continue
            if layer_idx == args.start_layer and expert_idx < args.start_expert:
                continue

            expert_key = f"L{layer_idx}_E{expert_idx}"
            freq = expert_freq.get(expert_key, 0)
            
            for nbits in bit_configs:
                for rank in rank_configs:
                    config_key = f"{nbits}bit_rank{rank}"
                    
                    if expert_key in results and config_key in results[expert_key]:
                        continue
                    
                    tasks.append({
                        "layer": layer_idx,
                        "expert": expert_idx,
                        "nbits": nbits,
                        "rank": rank,
                        "freq": freq
                    })
    
    tasks.sort(key=lambda x: x["freq"], reverse=True)
    logger.info(f"Created {len(tasks)} evaluation tasks")
    
    for task_idx, task in enumerate(tqdm(tasks, desc="Evaluating experts")):
        layer_idx = task["layer"]
        expert_idx = task["expert"]
        nbits = task["nbits"]
        rank = task["rank"]
        
        expert_key = f"L{layer_idx}_E{expert_idx}"
        config_key = f"{nbits}bit_rank{rank}"
        
        if expert_key not in results:
            results[expert_key] = {}
        
        if config_key in results[expert_key]:
            logger.info(f"Skipping already evaluated config: {expert_key} -> {config_key}")
            continue
        
        logger.info(f"Task {task_idx+1}/{len(tasks)}: {expert_key} -> {config_key}")
        reference_output_path = os.path.join(ref_output_cache_dir, f'layer_{layer_idx}_output.pt')
        
        impact = evaluate_expert_impact(
            model, reference_output_path, input_ids,
            layer_idx, expert_idx, nbits, rank
        )
        
        results[expert_key][config_key] = impact
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Result: {expert_key} -> {config_key} -> L2={impact['l2_norm']:.6f}, Rel={impact['relative_error']:.6f}")
        
        if (task_idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    end_time = time.time()
    logger.info(f"Evaluation completed. Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
