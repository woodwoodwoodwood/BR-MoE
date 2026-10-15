import os
import torch
from torch import nn,float16,uint8, int32, Tensor
from os.path import join as pjoin
from typing import Callable
from tqdm import tqdm
from abc import abstractmethod
from functools import partial
from typing import Union
from huggingface_hub import snapshot_download
from ..core.utils import cleanup
from ..core.quantize import BRMoELinear
from ..core.compensator import rank_generate,load_compensators
import json
import copy



_BRMOE_BACKEND_CLASSES =[]
try:
    from ..backends.marlin import MarlinLinear

    _BRMOE_BACKEND_CLASSES.append(MarlinLinear)
except Exception:
    pass


# Defined what is qualified as "linear layer"
_QUANT_LAYERS = [nn.Linear, BRMoELinear] +  _BRMOE_BACKEND_CLASSES
_IGNORE_LINEAR = ["lm_head"]


# Finds the parent of a node module named "name"
def find_parent(model, name: str) -> nn.Module:
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


# checks if a module is a leaf: doesn't have another module inside
def is_leaf_module(module) -> bool:
    return len(module._modules) == 0


# Get the linear_tag from a modul name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name: str) -> str:
    return ".".join(
        [
            n
            for n in name.split(".")
            if ((n not in ["model", "language_model", "layers"]) and (not n.isnumeric()))
        ]
    )


# returns all children nodes from model
def get_all_children_from_model(model, ignore: list = []) -> list:
    tags = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and (name.split(".")[-1] not in ignore):
            tags.append(name)
    return tags


# Get all linear tags available
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)


def forward_device_hooked(self, *args, **kwargs):
    args = list(args)

    # eddit this to make torch.compile compatible
    for i in range(len(args)):
        if isinstance(
            args[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(args[i], "to"):
            args[i] = args[i].to(self.device)

    for i in kwargs:
        if isinstance(
            kwargs[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(kwargs[i], "to"):
            kwargs[i] = kwargs[i].to(self.device)

    # return self.__class__.forward(self, *args, **kwargs)
    return self.forward_orig(*args, **kwargs)


# Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function.
class BasePatch:
    # Override these OR override the main patch_model() function
    ############################################
    # This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_nonlinearlayers(
        cls, model, patch_fct: Callable, verbose: bool = True
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) not in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name]),
            )

        cleanup()

    # This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name], patch_param),
            )

        cleanup()

    ############################################
    # These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod
    def set_auto_linear_tags(cls, model, ignore: list = _IGNORE_LINEAR) -> None:
        if hasattr(model, "linear_tags") is False:
            linear_tags = cls.get_linear_tags()
            model.linear_tags = (
                linear_tags
                if len(linear_tags) > 0
                else get_linear_tags_from_model(model, ignore=ignore)
            )
            model.base_class = cls

    # Returns the current linear tags
    @classmethod
    def get_linear_tags(cls) -> list:
        return []

    @classmethod
    def get_ignore_layers(cls, model) -> list:
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)

    # Autmatically name modules. This is very important to save/load the weights
    @classmethod
    def autoname_modules(cls, model) -> None:
        for name, module in model.named_modules():
            module.name = name

    # Freeze all layers
    @classmethod
    def freeze_model(cls, model) -> None:
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

    # Main patching function
    @classmethod
    def patch_model(
        cls,
        model,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        cleanup()


class BaseBRMoEModel:
    # Override these
    ############################################
    # This method creates and empty model based on the specfied architecture
    @abstractmethod
    def create_model(cls, save_dir, kwargs):
        pass

    # This method saves the model architecture only without inculding the weights (for example to a config.json)
    @abstractmethod
    def cache_model(cls, model, save_dir: str):
        pass

    ############################################

    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    # Save weights to disk
    @classmethod
    def save_weights(cls, weights: dict, save_dir: str) -> None:
        torch.save(weights, cls.get_weight_file(save_dir))
        print("> Compressed Weight saved")

    # Load weights from disk
    @classmethod
    def load_weights(cls, save_dir: str, map_location=None):
        return torch.load(cls.get_weight_file(save_dir), map_location=map_location)

    # Set-up model with the necessary data
    @classmethod
    def setup_model(cls, model):
        cls.autoname_modules(model)
        cls.set_auto_linear_tags(model)

    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def compress_model(
        cls,
        model,
        compress_config: dict,
        compute_dtype: torch.dtype = torch.float16,
        device: Union[str, list, dict] = "cuda",
    ):
        # # Check if the model was already quantized
        # if getattr(model, "brmoe_compressed", False):
        #     print("Model was already quantized")
        #     return

        # Unpack 3D fused experts if present (e.g. Qwen3.6-35B-A3B / Qwen3.5-MoE / Llama4)
        try:
            from .hf.universal_moe import ensure_unfused_moe_experts, get_text_backbone_and_prefix, UniversalMoEPatch
            ensure_unfused_moe_experts(model)
        except Exception:
            UniversalMoEPatch = None
            get_text_backbone_and_prefix = None

        # Set linear tags automatically
        cls.setup_model(model)
        model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
        compress_config["compensator_params"]["ranks"] = rank_generate(
            model_name_or_path,
            compress_config,
            compress_config["compensator_params"]["sparse_rank"],
            compress_config["compensator_params"]["dense_rank"],
            compress_config["compensator_params"]["rank_strategy"],
            model=model,
        )
        with open("ranks.json", "w", encoding="utf-8") as f:
            json.dump(compress_config["compensator_params"]["ranks"], f, ensure_ascii=False, indent=4)

        has_per_layer_keys = any(
            isinstance(k, str) and ("layers." in k) for k in compress_config.keys()
        )
        if (hasattr(model, "mix_moe_bit") and model.mix_moe_bit) or has_per_layer_keys:
            print(" (mix_moe_bit=True)")

            model_name = model_name_or_path.lower()
            arch_list = getattr(getattr(model, "config", None), "architectures", None) or []
            arch_str = arch_list[0].lower() if arch_list else ""
            model_type_str = getattr(getattr(model, "config", None), "model_type", "").lower()
            combined_id = f"{model_name} {arch_str} {model_type_str}"

            if UniversalMoEPatch is not None and issubclass(cls, UniversalMoEPatch):
                ModelPatch = cls
            elif "mixtral" in combined_id:
                print("MixtralPatch")
                from .hf.mixtral import MixtralPatch as ModelPatch
            elif "deepseek" in combined_id:
                print("DeepSeekMoEPatch")
                from .hf.deepseek import DeepSeekMoEPatch as ModelPatch
            elif any(k in combined_id for k in ("qwen3.6", "qwen3_6", "qwen3.5", "qwen3_5", "qwen3_next", "qwen3-next", "35b-a3b", "80b-a3b")):
                print("Qwen3NextMoEPatch")
                from .hf.qwen3_next import Qwen3NextMoEPatch as ModelPatch
            elif any(k in combined_id for k in ("qwen3", "30b-a3b", "235b-a22b")):
                print("Qwen3MoEPatch")
                from .hf.qwen3 import Qwen3MoEPatch as ModelPatch
            elif "qwen" in combined_id:
                print("Qwen15MoEPatch")
                from .hf.qwen import Qwen15MoEPatch as ModelPatch
            elif "glm" in combined_id:
                print("GLM4MoEPatch")
                from .hf.universal_moe import GLM4MoEPatch as ModelPatch
            elif "olmoe" in combined_id:
                print("OLMoEPatch")
                from .hf.universal_moe import OLMoEPatch as ModelPatch
            elif "phi" in combined_id:
                print("PhiMoEPatch")
                from .hf.universal_moe import PhiMoEPatch as ModelPatch
            else:
                print(f"Using UniversalMoEPatch for {model_name_or_path or arch_str}")
                from .hf.universal_moe import UniversalMoEPatch as ModelPatch

            try:
                leaf_tags = ModelPatch.get_leaf_linear_tags(model=model)
            except TypeError:
                leaf_tags = ModelPatch.get_leaf_linear_tags() if hasattr(ModelPatch, "get_leaf_linear_tags") else model.linear_tags
            patch_params = {key: None for key in leaf_tags}

            if hasattr(ModelPatch, "patch_linearlayers_mixbit"):
                ModelPatch.patch_linearlayers = ModelPatch.patch_linearlayers_mixbit

            attn_keys = [
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
            ]
            global_ranks = compress_config["compensator_params"]["ranks"]

            def _clone_cfg_with_ranks(cfg_dict):
                c = dict(cfg_dict)
                if "compensator_params" in c and isinstance(c["compensator_params"], dict):
                    cp = dict(c["compensator_params"])
                    cp["ranks"] = global_ranks
                    c["compensator_params"] = cp
                return c

            for key in attn_keys:
                if key in compress_config and isinstance(compress_config[key], dict):
                    config_copy = _clone_cfg_with_ranks(compress_config[key])
                    patch_params[key] = config_copy
                    for leaf_key in leaf_tags:
                        if key in leaf_key:
                            patch_params[leaf_key] = _clone_cfg_with_ranks(compress_config[key])

            for key in compress_config:
                if key in ("compensator_params", "skip_attention", "skip_shared_expert"):
                    continue
                val = compress_config[key]
                if isinstance(val, dict):
                    config_copy = _clone_cfg_with_ranks(val)
                    patch_params[key] = config_copy
                    # Also populate VLM language_model prefix alias if needed
                    if key.startswith("model.layers."):
                        alt_key = key.replace("model.layers.", "model.language_model.layers.", 1)
                        patch_params[alt_key] = _clone_cfg_with_ranks(val)
        else:
            if True in [(key in model.linear_tags) for key in compress_config.keys()]:
                global_ranks = compress_config["compensator_params"]["ranks"]
                patch_params = {key: None for key in model.linear_tags}
                for key, val in compress_config.items():
                    if isinstance(val, dict) and "compensator_params" in val:
                        val_copy = dict(val)
                        cp = dict(val["compensator_params"])
                        cp["ranks"] = global_ranks
                        val_copy["compensator_params"] = cp
                        patch_params[key] = val_copy
            else:
                patch_params = {k: compress_config for k in model.linear_tags}

        try:
            skip_attention = compress_config.get("skip_attention", False)
            skip_shared_expert = compress_config.get("skip_shared_expert", False)

            if skip_attention:
                for key in list(patch_params.keys()):
                    if ("self_attn." in key) or ("linear_attn." in key):
                        patch_params[key] = None

            if skip_shared_expert:
                for key in list(patch_params.keys()):
                    if ("shared_expert." in key) or ("shared_experts." in key) or ("shared_mlp." in key):
                        patch_params[key] = None
        except Exception as _e:
            print(f"[WARN] ：{_e}")

        # Save compact debug configs without duplicating global_ranks in every entry
        try:
            def _strip_nested_ranks(d):
                out = {}
                for k, v in d.items():
                    if k == "compensator_params" and isinstance(v, dict):
                        out[k] = {ck: cv for ck, cv in v.items() if ck != "ranks"}
                    elif isinstance(v, dict) and "compensator_params" in v:
                        vc = dict(v)
                        if isinstance(vc["compensator_params"], dict):
                            vc["compensator_params"] = {
                                ck: cv for ck, cv in vc["compensator_params"].items() if ck != "ranks"
                            }
                        out[k] = vc
                    else:
                        out[k] = v
                return out

            with open("full_compress_config.json", "w", encoding="utf-8") as f:
                json.dump(_strip_nested_ranks(compress_config), f, ensure_ascii=False, indent=2)
            with open("patch_params.json", "w", encoding="utf-8") as f:
                json.dump(_strip_nested_ranks(patch_params), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            if get_text_backbone_and_prefix is not None:
                _, layers_list, layer_prefix = get_text_backbone_and_prefix(model)
                num_blocks = len(layers_list)
                all_blocks = [f"{layer_prefix}.{i}" for i in range(num_blocks)]
            else:
                num_blocks = (
                    len(model.model.layers)
                    if hasattr(model, "model")
                    else len(model.layers)
                )
                all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = []
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # We replace the nn.Linear layers with BRMoELinear
        def _patch_linear(linear_layer, compress_config):   
            if type(linear_layer) is BRMoELinear:
                return linear_layer

            current_device = device_map[linear_layer.name]

            if compress_config is not None:
                out_module = BRMoELinear(
                    linear_layer,
                    compress_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)


        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        model.brmoe_compressed = True

        return model

    # Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                # disable state_dict encoding for safetensors
                module.encoded_state_dict = False
                state_dict = module.state_dict()
                
                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights

    @classmethod
    def save_compensators(cls,model,save_dir: str) -> None:
        weights = {}
        for name, module in model.named_modules():
            if type(module) is BRMoELinear:
                weights[name] = module.UV_quantized
        torch.save(weights, f"{save_dir}/compensators.pt")
        print("> Compensators saved")

    @classmethod
    def save_ranks(cls,model,save_dir: str) -> None:
        for name, module in model.named_modules():
            if type(module) is BRMoELinear:
                ranks = module.compress_config["compensator_params"]["ranks"]
                break
        with open(f"{save_dir}/ranks.json","w") as f:
            json.dump(ranks, f, indent=4)
            print("> Ranks saved")
        
    # Main function to save a quantized model
    @classmethod
    def save_compressed(cls, model, save_dir: str, verbose: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        # Save config
        cls.cache_model(model, save_dir)

        # Save compensators
        cls.save_compensators(model,save_dir)

        # Save ranks 
        cls.save_ranks(model,save_dir)

        # Serialization
        weights = cls.serialize_weights(model, verbose=verbose)

        # Save
        cls.save_weights(weights, save_dir)

    @classmethod
    def try_snapshot_download(
        cls, save_dir_or_hub: str, cache_dir: Union[str, None] = ""
    ):
        if cache_dir is None:
            save_dir = pjoin("", save_dir_or_hub)
        else:
            save_dir = pjoin(cache_dir, save_dir_or_hub)

        if not os.path.exists(save_dir):
            save_dir = snapshot_download(repo_id=save_dir_or_hub, cache_dir=cache_dir)
            save_dir = pjoin(save_dir)

        # Check
        if not os.path.exists(cls.get_weight_file(save_dir)):
            raise Exception("Weight file missing. Check your cache directory.")
        if not os.path.exists(cls.get_config_file(save_dir)):
            raise Exception("Config file missing. Check your cache directory.")

        return save_dir

    # This method is specfically designed in case we need to load some weights that are not part of any module
    @classmethod
    def post_module_load(cls, model, weights: dict):
        pass

    # Main function to load an HQQ quantized model from either HF hub or locally
    @classmethod
    def from_compressed(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str, None] = "",
        **kwargs,
    ):
        # Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        # Load model from config
        model = cls.create_model(save_dir, kwargs)

        # Track save directory
        model.save_dir = save_dir

        all_compensators = torch.load(f"{save_dir}/compensators.pt")

        # Name the layers
        cls.setup_model(model)

        # Load weights
        try:
            weights = cls.load_weights(save_dir)
            # if lorc_save_dir is not None:
            #     print(f"using partial lorc weights. tags are {lorc_tags}")
            #     lorc_weights = cls.load_weights(lorc_save_dir)
            #     for weight_name in weights:
            #         if any(lorc_tag in weight_name for lorc_tag in lorc_tags):
            #             weights[weight_name] = lorc_weights[weight_name]
            #             print(f"{weight_name} uses lorc")
            #     del lorc_weights

        except Exception:
            print("Failed to load the weights")
            raise FileNotFoundError

        
            
        # load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            state_dict = weights[module.name]
            if "W_q" in state_dict:
                module = BRMoELinear(
                    linear_layer=None,
                    compress_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                module.load_state_dict(state_dict)


            else:
                for key in state_dict:
                    setattr(
                        module,
                        key,
                        nn.Parameter(
                            state_dict[key].to(
                                device=device, dtype=compute_dtype, non_blocking=True
                            ),
                            requires_grad=False,
                        ),
                    )

            return module

        # Load modules
        cls.patch_model(
            model, _load_module, _load_module, {k: None for k in model.linear_tags}
        )

        with open(f"{save_dir}/ranks.json", "r") as f:
           ranks = json.load(f)
           
        load_compensators(model,all_compensators,ranks)
        

        # Load other weights that are not part of any module
        cls.post_module_load(model, weights)

        model.brmoe_compressed = True

        # Set base class
        model.base_class = cls


        return model

    quantize_model = compress_model
    save_quantized = save_compressed
    from_quantized = from_compressed
