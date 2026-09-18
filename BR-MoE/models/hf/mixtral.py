from ..base import BasePatch
from .base import BaseBRMoEHFModel
from tqdm import tqdm


# Patch  functions
class MixtralPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
    @classmethod
    def get_linear_tags(cls):
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "block_sparse_moe.experts.w1",
            "block_sparse_moe.experts.w2",
            "block_sparse_moe.experts.w3",
        ]

    @classmethod
    def get_leaf_linear_tags(cls):
        leaf_tags = []
        num_layers = 32

        for i in range(num_layers):
            leaf_tags.append(f"model.layers.{i}.self_attn.q_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.k_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.v_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.o_proj")
            
            n_experts = 8
            for k in range(n_experts):
                leaf_tags.append(f"model.layers.{i}.block_sparse_moe.experts.{k}.w1")
                leaf_tags.append(f"model.layers.{i}.block_sparse_moe.experts.{k}.w2")
                leaf_tags.append(f"model.layers.{i}.block_sparse_moe.experts.{k}.w3")
        
        # print(leaf_tags)
        return leaf_tags

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)  ###
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.norm = patch_fct(base_model.norm)

        layers = base_model.layers
        for i in tqdm(range(len(base_model.layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

            layers[i].block_sparse_moe.gate = patch_fct(
                layers[i].block_sparse_moe.gate
            )  # Keep MOE gate as fp16 because it's small

            n_experts = len(layers[i].block_sparse_moe.experts)
            for k in range(n_experts):
                layers[i].block_sparse_moe.experts[k].act_fn = patch_fct(
                    layers[i].block_sparse_moe.experts[k].act_fn
                )

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        print("Calling mixtral_patch_linearlayers()")
        base_model = model.model
        layers = base_model.layers
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].self_attn.q_proj = patch_fct(
                layers[i].self_attn.q_proj, patch_params["self_attn.q_proj"]
            )
            layers[i].self_attn.k_proj = patch_fct(
                layers[i].self_attn.k_proj, patch_params["self_attn.k_proj"]
            )
            layers[i].self_attn.v_proj = patch_fct(
                layers[i].self_attn.v_proj, patch_params["self_attn.v_proj"]
            )
            layers[i].self_attn.o_proj = patch_fct(
                layers[i].self_attn.o_proj, patch_params["self_attn.o_proj"]
            )

            n_experts = len(layers[i].block_sparse_moe.experts)
            for k in range(n_experts):
                layers[i].block_sparse_moe.experts[k].w1 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w1,
                    patch_params["block_sparse_moe.experts.w1"],
                )
                layers[i].block_sparse_moe.experts[k].w2 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w2,
                    patch_params["block_sparse_moe.experts.w2"],
                )
                layers[i].block_sparse_moe.experts[k].w3 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w3,
                    patch_params["block_sparse_moe.experts.w3"],
                )

    @classmethod
    def patch_linearlayers_mixbit(cls, model, patch_fct, patch_params, verbose=True):
        
        base_model = model.model
        layers = base_model.layers
        for i in tqdm(range(len(layers)), disable=not verbose):
            q_proj_key = f"model.layers.{i}.self_attn.q_proj"
            k_proj_key = f"model.layers.{i}.self_attn.k_proj"
            v_proj_key = f"model.layers.{i}.self_attn.v_proj"
            o_proj_key = f"model.layers.{i}.self_attn.o_proj"
            
            q_proj_config = patch_params.get(q_proj_key, patch_params.get("self_attn.q_proj"))
            k_proj_config = patch_params.get(k_proj_key, patch_params.get("self_attn.k_proj"))
            v_proj_config = patch_params.get(v_proj_key, patch_params.get("self_attn.v_proj"))
            o_proj_config = patch_params.get(o_proj_key, patch_params.get("self_attn.o_proj"))
            
            layers[i].self_attn.q_proj = patch_fct(
                layers[i].self_attn.q_proj, q_proj_config
            )
            layers[i].self_attn.k_proj = patch_fct(
                layers[i].self_attn.k_proj, k_proj_config
            )
            layers[i].self_attn.v_proj = patch_fct(
                layers[i].self_attn.v_proj, v_proj_config
            )
            layers[i].self_attn.o_proj = patch_fct(
                layers[i].self_attn.o_proj, o_proj_config
            )

            n_experts = len(layers[i].block_sparse_moe.experts)
            for k in range(n_experts):
                w1_key = f"model.layers.{i}.block_sparse_moe.experts.{k}.w1"
                w2_key = f"model.layers.{i}.block_sparse_moe.experts.{k}.w2"
                w3_key = f"model.layers.{i}.block_sparse_moe.experts.{k}.w3"
                
                w1_config = patch_params.get(w1_key, patch_params.get("block_sparse_moe.experts.w1"))
                w2_config = patch_params.get(w2_key, patch_params.get("block_sparse_moe.experts.w2"))
                w3_config = patch_params.get(w3_key, patch_params.get("block_sparse_moe.experts.w3"))

                layers[i].block_sparse_moe.experts[k].w1 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w1, w1_config
                )
                layers[i].block_sparse_moe.experts[k].w2 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w2, w2_config
                )
                layers[i].block_sparse_moe.experts[k].w3 = patch_fct(
                    layers[i].block_sparse_moe.experts[k].w3, w3_config
                )

class MixtralBRMoE(MixtralPatch, BaseBRMoEHFModel):
    pass
