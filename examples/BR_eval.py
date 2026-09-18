from BR_MoE.core.quantize import *
from BR_MoE.models.hf.mixtral import MixtralBRMoE as AutoBRMoEHFModel
# from BR_MoE.models.hf.deepseek import DeepSeekMoEBRMoE
from BR_MoE.engine.hf import AutoTokenizer
from evaluation.eval_wikitext2_ppl import eval_wikitext2_perplexity
from evaluation.eval_fewshots import eval_fewshots
from evaluation.eval_zeroshot import eval_zeroshot


def main():

    quant_model_dir = "quant_model_path"
    model_id = "mistralai/Mixtral-8x7B-v0.1" 
    model = AutoBRMoEHFModel.from_compressed(quant_model_dir)
    tokenizer  = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token
    eval_zeroshot(model,tokenizer,quant_model_dir)
    eval_wikitext2_perplexity(model,tokenizer,quant_model_dir)
    eval_fewshots(model,tokenizer,quant_model_dir)
    return

if __name__ == "__main__":
    main()




