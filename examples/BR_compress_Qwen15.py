import torch
from transformers import AutoModelForCausalLM
from BR_MoE.models.hf.qwen import Qwen15MoEBRMoE as AutoBRMoEHFModel
from BR_MoE.core.quantize import *
import json
import os
import logging
import re
import argparse

# Setup logging format
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration paths
model_path = ""
quant_model_dir = ""
best_configs_path = "./best_config.json"
device = "cuda"

# Load best configurations
logger.info(f"Loading best configurations: {best_configs_path}")
with open(best_configs_path, 'r') as f:
    best_configs = json.load(f)

# Create configuration cache
config_cache = {}

def get_or_create_config(bits, rank):
    """Get or create quantization configuration"""
    config_key = f"{bits}bit_rank{rank}"
    if config_key not in config_cache:
        config = br_moe_base_compress_config(
            nbits=bits, group_size=128, quant_zero=False, quant_scale=False, 
            offload_meta=False, view_as_float=False, axis=1, iter=20, 
            sparse_rank=rank, dense_rank=512, rank_strategy="custom", 
            compensator_dtype="int3", 
            compensator_quant_gs=64
        )
        
        # Set compensation matrix rank
        config["compensator_params"]["ranks"] = {}
        if rank > 0:
            # Set specific rank parameters if compensation matrix is needed
            pass
            
        config_cache[config_key] = config
    
    return config_cache[config_key]

# Create attention layer configuration (using 3-bit)
config_3bit_atten = br_moe_base_compress_config(
        nbits=3, group_size=128, quant_zero=False, quant_scale=False, 
        offload_meta=False, view_as_float=False, axis=1, iter=20, 
        sparse_rank=16, dense_rank=512, rank_strategy="custom", 
        compensator_dtype="int3", 
        compensator_quant_gs=64
    )
config_3bit_atten["compensator_params"]["ranks"] = {}

# Main compression configuration
compress_config = {
    "compensator_params": {
        "iter": 20, 
        "sparse_rank": 16, 
        "dense_rank": 512, 
        "rank_strategy": "custom",
        "compensator_dtype": "int3", 
        "compensator_quant_gs": 64, 
        "ranks": {}
    }
}

# Set attention layer configuration
compress_config["self_attn.q_proj"] = config_3bit_atten
compress_config["self_attn.k_proj"] = config_3bit_atten
compress_config["self_attn.v_proj"] = config_3bit_atten
compress_config["self_attn.o_proj"] = config_3bit_atten

# Configuration statistics
config_stats = {}
shared_expert_config = get_or_create_config(4, 512)  # Default shared expert uses 4bit_rank512

# Load optimal configuration from JSON file
def load_optimal_config(config_file):
    try:
        logger.info(f"Loading optimal configurations from file: {config_file}")
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return {}

# Parse configuration file path from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="best_config.json", 
                    help="Optimal quantization configuration file path")
config_args, _ = parser.parse_known_args()

# Load best configurations
best_configs = load_optimal_config(config_args.config)
if not best_configs:
    logger.warning(f"Failed to load configuration file or configuration is empty, will use default configuration")
    # Set a default configuration here if needed

logger.info(f"Loaded quantization configurations for {len(best_configs)} experts")

# Parse and apply best configuration for each expert
for expert_key, config_str in best_configs.items():
    # Parse expert key (e.g., L1_E0)
    match = re.match(r'L(\d+)_E(\d+)', expert_key)
    if not match:
        logger.warning(f"Unable to parse expert ID: {expert_key}, will skip")
        continue
        
    layer_idx = int(match.group(1))   # Zero-based index
    expert_idx = int(match.group(2))
    
    # Parse configuration (e.g., 3bit_rank0)
    # Check configuration format - adjust according to JSON format
    if isinstance(config_str, dict) and "config" in config_str:
        # If configuration is in dictionary format {"config": "3bit_rank0"}
        config_str = config_str["config"]
    
    match = re.match(r'(\d+)bit_rank(\d+)', config_str)
    if not match:
        logger.warning(f"Unable to parse configuration: {config_str}, will use default configuration for expert {expert_key}")
        continue
        
    bits = int(match.group(1))
    rank = int(match.group(2))
    
    # Get or create corresponding configuration
    config_to_use = get_or_create_config(bits, rank)
    
    # Statistics configuration usage
    if config_str not in config_stats:
        config_stats[config_str] = 0
    config_stats[config_str] += 1
    
    # Set configuration for expert's three matrices
    for weight_type in ['gate_proj', 'up_proj', 'down_proj']:
        key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{weight_type}"
        compress_config[key] = config_to_use
        
# Print configuration statistics
logger.info("Quantization configuration statistics:")
for config_name, count in sorted(config_stats.items(), key=lambda x: x[1], reverse=True):
    logger.info(f"  {config_name}: {count} experts ({count/sum(config_stats.values())*100:.1f}%)")

# Set configuration for shared experts
logger.info("Setting 3bit_rank512 quantization for all shared experts...")
for i in range(0, 24):  # DeepSeek-MoE internal index is 0-23 for MoE layers
    layer_idx = i  # Directly use 0-23 index
    for weight_type in ['gate_proj', 'up_proj', 'down_proj']:
        key = f"model.layers.{layer_idx}.mlp.shared_expert.{weight_type}"
        compress_config[key] = shared_expert_config

# Print configuration usage statistics
logger.info("Configuration usage statistics:")
for config_name, count in sorted(config_stats.items(), key=lambda x: x[1], reverse=True):
    logger.info(f"  {config_name}: {count} experts")

# Load and compress model
logger.info(f"Loading model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                         torch_dtype=torch.float16, 
                                         trust_remote_code=True)

# Ensure model path name is correct
model.config._name_or_path = "Qwen/Qwen1.5-MoE"

logger.info("Starting model compression")
AutoBRMoEHFModel.compress_model(model, 
                           compress_config=compress_config, 
                           device=device)

logger.info(f"Saving compressed model to: {quant_model_dir}")
AutoBRMoEHFModel.save_compressed(model, quant_model_dir)

logger.info("Model compression completed!")
