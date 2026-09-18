from ..base import BasePatch
from .base import BaseBRMoEHFModel
from tqdm import tqdm


# Patch functions
class Qwen3MoEPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type
    @classmethod
    def get_linear_tags(cls):
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.experts.gate_proj",
            "mlp.experts.up_proj",
            "mlp.experts.down_proj",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj"
        ]

    @classmethod
    def get_leaf_linear_tags(cls):
        leaf_tags = []
        num_layers = 40

        for i in range(num_layers):
            leaf_tags.append(f"model.layers.{i}.self_attn.q_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.k_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.v_proj")
            leaf_tags.append(f"model.layers.{i}.self_attn.o_proj")
            
            n_experts = 60
            for k in range(n_experts):
                leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.gate_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.up_proj")
                leaf_tags.append(f"model.layers.{i}.mlp.experts.{k}.down_proj")
            
            leaf_tags.append(f"model.layers.{i}.mlp.shared_expert.gate_proj")
            leaf_tags.append(f"model.layers.{i}.mlp.shared_expert.up_proj")
            leaf_tags.append(f"model.layers.{i}.mlp.shared_expert.down_proj")
        
        return leaf_tags

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.norm = patch_fct(base_model.norm)

        layers = base_model.layers
        for i in tqdm(range(len(base_model.layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

            n_experts = len(layers[i].mlp.experts)
            for k in range(n_experts):
                layers[i].mlp.experts[k].act_fn = patch_fct(
                    layers[i].mlp.experts[k].act_fn
                )
            layers[i].mlp.shared_expert.act_fn = patch_fct(
                layers[i].mlp.shared_expert.act_fn
            )
            layers[i].mlp.gate = patch_fct(
                layers[i].mlp.gate
            )  # Keep MOE gate as fp16 because it's small
            layers[i].mlp.shared_expert_gate = patch_fct(
                layers[i].mlp.shared_expert_gate
            )

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
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

            if hasattr(layers[i].mlp, 'experts'):
                n_experts = len(layers[i].mlp.experts)
                for k in range(n_experts):
                    if hasattr(layers[i].mlp.experts[k], 'gate_proj'):
                        layers[i].mlp.experts[k].gate_proj = patch_fct(
                            layers[i].mlp.experts[k].gate_proj,
                            patch_params["mlp.experts.gate_proj"],
                        )
                        layers[i].mlp.experts[k].up_proj = patch_fct(
                            layers[i].mlp.experts[k].up_proj,
                            patch_params["mlp.experts.up_proj"],
                        )
                        layers[i].mlp.experts[k].down_proj = patch_fct(
                            layers[i].mlp.experts[k].down_proj,
                            patch_params["mlp.experts.down_proj"],
                        )
            
            if hasattr(layers[i].mlp, 'shared_expert'):
                if hasattr(layers[i].mlp.shared_expert, 'gate_proj'):
                    layers[i].mlp.shared_expert.gate_proj = patch_fct(
                        layers[i].mlp.shared_expert.gate_proj,
                        patch_params["mlp.shared_expert.gate_proj"],
                    )
                    layers[i].mlp.shared_expert.up_proj = patch_fct(
                        layers[i].mlp.shared_expert.up_proj,
                        patch_params["mlp.shared_expert.up_proj"],
                    )
                    layers[i].mlp.shared_expert.down_proj = patch_fct(
                        layers[i].mlp.shared_expert.down_proj,
                        patch_params["mlp.shared_expert.down_proj"],
                    )

    @classmethod
    def patch_linearlayers_mixbit(cls, model, patch_fct, patch_params, verbose=True):
        """
        Mixed Quantization for Qwen3-MoE
        """
        
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
            
            if q_proj_config is not None:
                layers[i].self_attn.q_proj = patch_fct(layers[i].self_attn.q_proj, q_proj_config)
            if k_proj_config is not None:
                layers[i].self_attn.k_proj = patch_fct(layers[i].self_attn.k_proj, k_proj_config)
            if v_proj_config is not None:
                layers[i].self_attn.v_proj = patch_fct(layers[i].self_attn.v_proj, v_proj_config)
            if o_proj_config is not None:
                layers[i].self_attn.o_proj = patch_fct(layers[i].self_attn.o_proj, o_proj_config)

            if hasattr(layers[i].mlp, 'experts'):
                n_experts = len(layers[i].mlp.experts)
                for k in range(n_experts):
                    if hasattr(layers[i].mlp.experts[k], 'gate_proj'):
                        gate_proj_key = f"model.layers.{i}.mlp.experts.{k}.gate_proj"
                        up_proj_key = f"model.layers.{i}.mlp.experts.{k}.up_proj"
                        down_proj_key = f"model.layers.{i}.mlp.experts.{k}.down_proj"
                        
                        gate_proj_config = patch_params.get(gate_proj_key, patch_params.get("mlp.experts.gate_proj"))
                        up_proj_config = patch_params.get(up_proj_key, patch_params.get("mlp.experts.up_proj"))
                        down_proj_config = patch_params.get(down_proj_key, patch_params.get("mlp.experts.down_proj"))
                        
                        if gate_proj_config is not None:
                            layers[i].mlp.experts[k].gate_proj = patch_fct(layers[i].mlp.experts[k].gate_proj, gate_proj_config)
                        if up_proj_config is not None:
                            layers[i].mlp.experts[k].up_proj = patch_fct(layers[i].mlp.experts[k].up_proj, up_proj_config)
                        if down_proj_config is not None:
                            layers[i].mlp.experts[k].down_proj = patch_fct(layers[i].mlp.experts[k].down_proj, down_proj_config)

            if hasattr(layers[i].mlp, 'shared_expert'):
                if hasattr(layers[i].mlp.shared_expert, 'gate_proj'):
                    shared_gate_proj_key = f"model.layers.{i}.mlp.shared_expert.gate_proj"
                    shared_up_proj_key = f"model.layers.{i}.mlp.shared_expert.up_proj"
                    shared_down_proj_key = f"model.layers.{i}.mlp.shared_expert.down_proj"
                    
                    shared_gate_proj_config = patch_params.get(shared_gate_proj_key, patch_params.get("mlp.shared_expert.gate_proj"))
                    shared_up_proj_config = patch_params.get(shared_up_proj_key, patch_params.get("mlp.shared_expert.up_proj"))
                    shared_down_proj_config = patch_params.get(shared_down_proj_key, patch_params.get("mlp.shared_expert.down_proj"))
                    
                    if shared_gate_proj_config is not None:
                        layers[i].mlp.shared_expert.gate_proj = patch_fct(
                            layers[i].mlp.shared_expert.gate_proj, shared_gate_proj_config)
                    if shared_up_proj_config is not None:
                        layers[i].mlp.shared_expert.up_proj = patch_fct(
                            layers[i].mlp.shared_expert.up_proj, shared_up_proj_config)
                    if shared_down_proj_config is not None:
                        layers[i].mlp.shared_expert.down_proj = patch_fct(
                            layers[i].mlp.shared_expert.down_proj, shared_down_proj_config)


class Qwen3MoEBRMoE(Qwen3MoEPatch, BaseBRMoEHFModel):
    pass
