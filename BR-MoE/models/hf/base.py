import os
import transformers
from accelerate import init_empty_weights
from ..base import BaseBRMoEModel, BasePatch


class BaseBRMoEHFModel(BaseBRMoEModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        print(cls.get_config_file(save_dir))
        _local_cfg = cls.get_config_file(save_dir)
        _cfg_src = (
            save_dir
            if os.path.exists(_local_cfg)
            else (
                "deepseek-ai/deepseek-moe-16b-base"
                if "deepseek" in save_dir
                else save_dir
            )
        )
        config = transformers.AutoConfig.from_pretrained(
            _cfg_src, trust_remote_code=True
        )

        auto_class = transformers.AutoModel
        archs = getattr(config, "architectures", None) or []
        if len(archs) >= 1:
            arch_name = archs[0]
            # First try resolving exact class on transformers (e.g. Qwen3_5MoeForConditionalGeneration)
            exact_cls = getattr(transformers, arch_name, None)
            if exact_cls is not None:
                auto_class = exact_cls
            elif "CausalLM" in arch_name:
                auto_class = transformers.AutoModelForCausalLM
            elif "ConditionalGeneration" in arch_name or "Vision2Seq" in arch_name or "ImageTextToText" in arch_name:
                auto_class = getattr(
                    transformers,
                    "AutoModelForImageTextToText",
                    getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModelForCausalLM),
                )

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs, trust_remote_code=True)

        # Unpack 3D fused experts if present so per-expert BRMoELinear state_dict can be loaded
        try:
            from .universal_moe import ensure_unfused_moe_experts
            ensure_unfused_moe_experts(model)
        except Exception:
            pass

        return model


# Auto class used for HF models if no architecture was manually setup
class AutoBRMoEHFModel(BaseBRMoEHFModel, BasePatch):
    pass
