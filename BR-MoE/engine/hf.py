import transformers
import torch
from .base import BRMoEWrapper
from BR_MoE.models.hf.mixtral import MixtralBRMoE
from BR_MoE.models.hf.deepseek import DeepSeekMoEBRMoE, DeepSeekV2MoEBRMoE, DeepSeekV3MoEBRMoE
from BR_MoE.models.hf.qwen import Qwen15MoEBRMoE, Qwen2MoEBRMoE
from BR_MoE.models.hf.qwen3 import Qwen3MoEBRMoE
from BR_MoE.models.hf.qwen3_next import (
    Qwen3NextMoEBRMoE,
    Qwen3_5MoEBRMoE,
    Qwen3_6MoEBRMoE,
)
from BR_MoE.models.hf.universal_moe import (
    UniversalMoEBRMoE,
    GLM4MoEBRMoE,
    OLMoEBRMoE,
    PhiMoEBRMoE,
)


_BRMOE_REGISTRY = {
    # Mixtral
    "MixtralForCausalLM": MixtralBRMoE,
    # DeepSeek-MoE / V2 / V3 / R1
    "DeepseekForCausalLM": DeepSeekMoEBRMoE,
    "DeepSeekForCausalLM": DeepSeekMoEBRMoE,
    "DeepseekMoEForCausalLM": DeepSeekMoEBRMoE,
    "DeepseekV2ForCausalLM": DeepSeekV2MoEBRMoE,
    "DeepseekV3ForCausalLM": DeepSeekV3MoEBRMoE,
    # Qwen1.5-MoE & Qwen2-MoE
    "Qwen2MoeForCausalLM": Qwen2MoEBRMoE,
    "Qwen2MoEForCausalLM": Qwen2MoEBRMoE,
    "Qwen1_5_MoEForCausalLM": Qwen15MoEBRMoE,
    # Qwen3-MoE (e.g., Qwen3-30B-A3B, Qwen3-235B-A22B)
    "Qwen3MoeForCausalLM": Qwen3MoEBRMoE,
    "Qwen3MoEForCausalLM": Qwen3MoEBRMoE,
    "Qwen3VLMoeForConditionalGeneration": Qwen3MoEBRMoE,
    # Qwen3.5-MoE / Qwen3.6-MoE (e.g., Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.5-35B-A3B) & Qwen3-Next
    "Qwen3NextForCausalLM": Qwen3NextMoEBRMoE,
    "Qwen3_5MoeForCausalLM": Qwen3_5MoEBRMoE,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_6MoEBRMoE,
    "Qwen3_6MoeForCausalLM": Qwen3_6MoEBRMoE,
    "Qwen3_6MoeForConditionalGeneration": Qwen3_6MoEBRMoE,
    # GLM-4.5 / 4.6 / 4.7 MoE, OLMoE, Phi-MoE, Llama-4, Ernie-4.5, Bailing, MiniMax
    "Glm4MoeForCausalLM": GLM4MoEBRMoE,
    "OlmoeForCausalLM": OLMoEBRMoE,
    "PhimoeForCausalLM": PhiMoEBRMoE,
    "Llama4ForConditionalGeneration": UniversalMoEBRMoE,
    "Llama4ForCausalLM": UniversalMoEBRMoE,
    "Ernie4_5_MoeForCausalLM": UniversalMoEBRMoE,
    "BailingMoeForCausalLM": UniversalMoEBRMoE,
    "BailingMoeV2ForCausalLM": UniversalMoEBRMoE,
    "MiniMaxM1ForCausalLM": UniversalMoEBRMoE,
    "MiniMaxM2ForCausalLM": UniversalMoEBRMoE,
}


# Alias
AutoTokenizer = transformers.AutoTokenizer

# Used to call super() on classmethods
_Parent = transformers.AutoModelForCausalLM


class BRMoEModelForCausalLM(_Parent, BRMoEWrapper):
    _HQQ_REGISTRY = _BRMOE_REGISTRY
    _DEFAULT_BRMOE_CLASS = UniversalMoEBRMoE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _make_quantizable(cls, model, quantized: bool) -> None:
        model.brmoe_compressed = quantized
        model.arch_key = model.config.architectures[0]
        model.quantize_model = (
            lambda quant_config,
            compute_dtype=torch.float16,
            device="cuda": cls.quantize_model_(
                model=model,
                quant_config=quant_config,
                compute_dtype=compute_dtype,
                device=device,
            )
        )
        model.save_quantized = lambda save_dir: cls.save_quantized_(
            model=model, save_dir=save_dir
        )
        model.cuda = lambda *args, **kwargs: model if (quantized) else model.cuda
        model.to = lambda *args, **kwargs: model if (quantized) else model.to
        model.float = lambda *args, **kwargs: model if (quantized) else model.float
        model.half = lambda *args, **kwargs: model if (quantized) else model.half
        model.base_class = cls._get_hqq_class(model)

    # Force loading the model on CPU and unquantized
    @classmethod
    def _validate_params(cls, params: dict) -> None:
        for p in ["load_in_4bit", "load_in_8bit"]:  # ignore these
            if p in params:
                params[p] = False
        params["device_map"] = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls._validate_params(kwargs)
        model = super(_Parent, cls).from_pretrained(*args, **kwargs)
        cls._make_quantizable(model, quantized=False)
        return model

    @classmethod
    def _get_arch_key_from_save_dir(cls, save_dir: str):
        config = transformers.AutoConfig.from_pretrained(save_dir, trust_remote_code=True)
        return config.architectures[0]
