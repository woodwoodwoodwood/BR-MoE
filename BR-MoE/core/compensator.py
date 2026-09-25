import json
from ..core.quantize import BRMoELinear
from .utils import compensator_dequantize
import importlib.resources as pkg_resources
from ..core import model_statistics

MIXTRAL_LAYERS = {
    "dense": ["self_attn"],
    "sparse": ["experts"],
    "layer_count": 32
}
DEEPSEEK_LAYERS = {
    "dense": ['self_attn', 'shared', 'layers.0.mlp'],
    "sparse": ['mlp.experts'],
    "layer_count": 27
}
QWEN15MOE_LAYERS = {
    # include both naming variants to ease substring matching
    "dense": ["self_attn", "mlp.shared_expert", "mlp.shared_experts"],
    "sparse": ["mlp.experts"],
    "layer_count": 24
}
QWEN3MOE_LAYERS = {
    "dense": ["self_attn", "mlp.shared_expert", "mlp.shared_experts", "layers.0.mlp"],
    "sparse": ["mlp.experts"],
    "layer_count": 40
}

def rank_generate(model_id, compress_config,sparse_rank,dense_rank,strategy):
    if model_id == "mistralai/Mixtral-8x7B-v0.1":
        print("generate ranks for Mixtral-8x7B")
        model_layer_info = MIXTRAL_LAYERS
    elif model_id == "deepseek-ai/deepseek-moe-16b-base":
        model_layer_info = DEEPSEEK_LAYERS
    elif ("qwen3" in model_id.lower()) and ("moe" in model_id.lower()):
        print("generate ranks for Qwen3 MoE")
        model_layer_info = QWEN3MOE_LAYERS
    elif ("qwen" in model_id.lower()) and ("moe" in model_id.lower()):
        print("generate ranks for Qwen (Qwen2/Qwen1.5) MoE")
        model_layer_info = QWEN15MOE_LAYERS
    else:
        raise NotImplementedError
    
    if strategy == None:
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
        ranks = {}

        for name, config in compress_config.items():
            if isinstance(config, dict) and "compensator_params" in config:
                comp_params = config["compensator_params"]
                
                if "experts" in name:
                    rank_to_use = comp_params.get("sparse_rank", 0)
                else:
                    rank_to_use = comp_params.get("dense_rank", 0)
                
                ranks[name] = rank_to_use
    elif strategy == "frequency" and model_id == "deepseek-ai/deepseek-moe-16b-base":
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        with pkg_resources.files(model_statistics).joinpath("DeepSeek_expt_freq.json").open("r") as f:
            data = json.load(f)
        for layer_index in range(27):
            freq = data[layer_index]
            freq_sum = sum(freq)
            for expert_index in range(len(freq)):
                rank = int(round(freq[expert_index] / freq_sum * (sparse_rank*len(freq))))   # Assign rank based on the weight
                ranks[f'layers.{layer_index + 1}.mlp.experts.{expert_index}.'] = rank            
    elif strategy == "frequency" and (("qwen" in model_id.lower()) and ("moe" in model_id.lower())):
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        try:
            with pkg_resources.files(model_statistics).joinpath("Qwen15_expt_freq.json").open("r") as f:
                data = json.load(f)
            print("Loaded Qwen1.5-MoE expert frequency data")
            for key, freq in data.items():
                if "-" in key:
                    layer_idx, expert_idx = key.split("-")
                    layer_idx, expert_idx = int(layer_idx), int(expert_idx)
                    rank = int(round(freq * sparse_rank * 8))
                    ranks[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}'] = rank
                    
        except FileNotFoundError:
            print("Warning: Qwen15_expt_freq.json not found, using uniform ranks for all experts")
            num_layers = model_layer_info["layer_count"]
            num_experts = 60
            for layer_idx in range(num_layers):
                for expert_idx in range(num_experts):
                    ranks[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}'] = sparse_rank

    elif strategy == "Kurtosis" and model_id == "mistralai/Mixtral-8x7B-v0.1":
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]}
        }
        if sparse_rank == 16:
            k = 2
        elif sparse_rank == 32:
            k = 3
        else:
            raise NotImplementedError("Currently Mixtral Kurtosis strategy only support the avg rank of 16 and 32")
        with pkg_resources.files(model_statistics).joinpath("Mixtral_kurtosis_values.json").open("r") as f:
            data = json.load(f)
        for name, kurtosis in data.items():
            kurtosis = round(float(kurtosis))
            if "self_attn" in name:
                continue
            else:
                if kurtosis < 1:
                    rank = 0
                elif kurtosis > 9:
                    rank = 1024
                else:
                    rank = 2 ** (kurtosis+k)
            text = name.replace('.weight', '').strip()
            
            ranks[text] = rank
    else:
        raise NotImplementedError
    return ranks


# def quantize_full_to_int8(tensor_in):
#     max_val, _ = torch.max(tensor_in, dim=1, keepdim=True)
#     min_val, _ = torch.min(tensor_in, dim=1, keepdim=True)
#     max_min = max_val - min_val
#     max_min[max_min==0] = 255  #deal with the case max = min
#     scale = 255 / max_min
#     zero = - torch.round(scale * min_val) - 128  
#     tensor_int8 = torch.round(tensor_in * scale + zero).to(torch.int8)
#     return  scale,zero,tensor_int8



def load_compensators(model,compensators,ranks):
    for name, module in model.named_modules():
        if type(module) is BRMoELinear:
            UV_quantized = compensators.pop(name, None)
            orig_shape=module.meta['shape']
            # module.compress_config["compensator_params"]["ranks"] = ranks
            rank = next((value for key, value in ranks.items() if key in name), None)
            compensator_dtype = module.compress_config["compensator_params"]["compensator_dtype"]
            compensator_quantize_gs = module.compress_config["compensator_params"]["compensator_quant_gs"]
            if rank is not None and rank > 0:
                module.U, module.V = compensator_dequantize(UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype)
