
import os
import transformers
from accelerate import init_empty_weights
from ..base import BaseBRMoEModel, BasePatch

class BaseBRMoEHFModel(BaseBRMoEModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        print(cls.get_config_file(save_dir))
        # 必须传「目录」而不是 config.json 的文件路径: transformers 在解析 auto_map 时
        # 会把非目录参数当成 Hub repo id 校验, 从而抛 HFValidationError。
        # 本地存在 config.json 时直接用 save_dir(离线可用), 否则回退到官方 Hub 配置。
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

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1 and ("CausalLM" in archs[0]):
            auto_class = transformers.AutoModelForCausalLM

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs, trust_remote_code=True)

        return model


# Auto class used for HF models if no architecture was manually setup
class AutoBRMoEHFModel(BaseBRMoEHFModel, BasePatch):
    pass
