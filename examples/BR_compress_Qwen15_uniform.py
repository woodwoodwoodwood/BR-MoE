from BR_MoE.models.hf.qwen import Qwen15MoEBRMoE as AutoBRMoEHFModel
from BR_MoE.core.quantize import *
from transformers import AutoModelForCausalLM
import torch

device = "cuda"
quant_model_dir = "./"
compress_config = BaseCompressConfig(
				# quantization config
				 nbits = 3, 
				 group_size = 64, 
				 quant_scale = False, 
				 quant_zero = False, 
				 axis = 1,
				# compensator config
				 iter = 20,
				 sparse_rank = 32,
				 dense_rank = 512,
				 rank_strategy = "frequency",
				 compensator_dtype  = "int3"
				 ) 
model = AutoModelForCausalLM.from_pretrained("model_path",
					 torch_dtype=torch.float16,
					 trust_remote_code=True)

# Ensure model type is correctly identified
model.config._name_or_path = "Qwen/Qwen1.5-MoE"

AutoBRMoEHFModel.compress_model(model, 
			   compress_config=compress_config, 
			   device=device)    
AutoBRMoEHFModel.save_compressed(model, quant_model_dir)
