import torch
from transformers import AutoModelForCausalLM
from BR_MoE.models.hf.mixtral import MixtralBRMoE as AutoBRMoEHFModel
from BR_MoE.core.quantize import *


def main():

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
                                         iter = 10,
                                         sparse_rank = 16,
                                         dense_rank = 512,
                                         rank_strategy = None,
                                         compensator_dtype  = "int3"

                                         ) 
    model = AutoModelForCausalLM.from_pretrained("model_path",
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    AutoBRMoEHFModel.compress_model(model, 
                                   compress_config=compress_config, 
                                   device=device)    
    AutoBRMoEHFModel.save_compressed(model, quant_model_dir)



if __name__ == "__main__":
    main()

