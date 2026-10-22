"""CPU integration checks; build tools/build_ggml_quant.py before native tests."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brmoe_quant import FORMATS, get_format, parse_option, tensor_nbytes
from brmoe_quant.native import quantize, dequantize, library
from RCO_solver.tensor_units import discover_units, unit_costs, collect_importance


class Block(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(width, width, bias=False)
        self.mlp = nn.Module()
        self.mlp.shared_expert = nn.Module()
        self.mlp.shared_expert.up_proj = nn.Linear(width, width, bias=False)
        self.mlp.experts = nn.ModuleList([nn.Module(), nn.Module()])
        for expert in self.mlp.experts:
            expert.up_proj = nn.Linear(width, width, bias=False)
        self.mlp.gate = nn.Linear(width, 2, bias=False)

    def forward(self, x):
        x = self.self_attn.q_proj(x).tanh()
        return x + self.mlp.shared_expert.up_proj(x) + sum(e.up_proj(x) for e in self.mlp.experts) / 2


class ToyModel(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(width)])

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.model.layers[0](input_ids))


def test_inventory_costs_and_alignment():
    units = discover_units(ToyModel())
    assert [u.category for u in units] == ["attention", "shared", "routed"]
    assert len(units[-1].names) == 2
    opts = ["IQ4_NL_rank0", "IQ4_XS_rank0", "F16_rank64"]
    costs, valid = unit_costs(units, opts)
    assert valid[:, 0].all() and not valid[:, 1].any()
    assert costs[-1, 0] == 2 * costs[0, 0]
    assert round(costs[0, 2] * 1024**2) == (32*32 + 2*32*32)*2
    assert parse_option("IQ4_XS_rank16") == ("IQ4_XS", 16)
    assert tensor_nbytes((2, 256), "IQ4_XS") != tensor_nbytes((2, 256), "IQ4_NL")
    assert len(discover_units(ToyModel(), include_attention=False, include_shared=False)) == 1


@pytest.fixture
def native():
    try:
        return library()
    except RuntimeError as error:
        if "is missing" in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize("qtype", list(FORMATS))
def test_native_roundtrip_and_core_serialization(native, qtype):
    from examples.BR_compress_auto import br_moe_base_compress_config
    from BR_MoE.core.quantize import Quantizer, BRMoELinear
    x = np.random.default_rng(42).normal(size=(8, 256)).astype(np.float32)
    imp = np.linspace(0.1, 2, 256, dtype=np.float32)
    raw = quantize(x, qtype, imp)
    actual = dequantize(raw, x.shape, qtype)
    import gguf
    independent = gguf.dequantize(raw.reshape(x.shape[0], -1), gguf.GGMLQuantizationType[get_format(qtype).name])
    np.testing.assert_allclose(actual, independent, atol=1e-6, rtol=1e-6)
    assert raw.nbytes == tensor_nbytes(x.shape, qtype)
    assert np.isfinite(actual).all() and np.mean((x - actual)**2) < 0.2
    q, meta = Quantizer.quantize(torch.from_numpy(x), nbits=qtype, importance=imp, device="cpu")
    np.testing.assert_array_equal(q.numpy(), raw)
    np.testing.assert_array_equal(Quantizer.dequantize(q, meta).numpy(), actual)
    from BR_MoE.core.gguf_kquant import gguf_kquant_quant_dequant_np, gguf_kquant_quant_dequant
    np.testing.assert_array_equal(gguf_kquant_quant_dequant_np(x, qtype, imp), actual)
    np.testing.assert_array_equal(gguf_kquant_quant_dequant(torch.from_numpy(x), qtype, imp).numpy(), actual)
    cfg = br_moe_base_compress_config(nbits=qtype, importance=torch.from_numpy(imp), dense_rank=2, sparse_rank=2)
    lin = nn.Linear(256, 8, bias=False)
    lin.name = "model.layers.0.self_attn.q_proj"
    lin.weight.data.copy_(torch.from_numpy(x))
    packed = BRMoELinear(lin, cfg, compute_dtype=torch.float32, device="cpu")
    state = packed.state_dict()
    assert "importance" not in state
    restored = BRMoELinear(None, None, compute_dtype=torch.float32, device="cpu")
    restored.load_state_dict(copy.deepcopy(state))
    assert restored.meta["quant_type"] == qtype
    assert restored.compress_config["compensator_params"]["compensator_dtype"] == "fp16"
    from BR_MoE.core.compensator import load_compensators
    holder = nn.Module()
    holder.projection = restored
    load_compensators(holder, {"projection": packed.UV_quantized}, {"projection": 2})
    torch.testing.assert_close(restored.dequantize(), packed.dequantize())
    sample = torch.randn(2, 256)
    torch.testing.assert_close(restored.matmul(sample), packed.matmul(sample))


def test_missing_imatrix_fails_before_native_encode(native):
    with pytest.raises(ValueError, match="requires input-channel importance"):
        quantize(np.ones((2, 256)), "IQ2_XS")
    with pytest.raises(ValueError):
        quantize(np.ones((2, 33)), "IQ4_NL")


def test_calibration_masks_unseen_experts():
    from RCO_solver.full_model_search import prepare_candidates
    model = ToyModel(256)
    names = [name for u in discover_units(model) for name in u.names]
    importance = {name: np.ones(256, dtype=np.float32) for name in names[:-1]}
    units, costs, valid, _ = prepare_candidates(model, ["IQ2_XS_rank0", "F16_rank0"], [], importance=importance)
    assert valid[:2, 0].all()
    assert not valid[-1, 0]  # a single unseen expert invalidates the 3D IQ2 group
    assert valid[:, 1].all()


def test_explicit_projection_patcher():
    from examples.BR_compress_auto import build_auto_compress_config, UniversalMoEBRMoE
    model = ToyModel()
    block = model.model.layers[0]
    shared = block.mlp.shared_expert
    del block.mlp.shared_expert
    block.mlp.shared_experts = nn.ModuleList([shared, copy.deepcopy(shared)])
    block.self_attn.extra_proj = nn.Linear(32, 32)
    units = discover_units(model)
    configs = {unit.key: "IQ4_NL_rank0" for unit in units}
    patch_params = build_auto_compress_config(model, configs)
    for name, module in model.named_modules():
        module.name = name
    visited = set()
    def replace(module, cfg):
        if cfg is None:
            return module
        visited.add(module.name)
        return nn.Identity()
    UniversalMoEBRMoE.patch_linearlayers(model, replace, patch_params, verbose=False)
    assert visited == {name for unit in units for name in unit.names}


def test_e2e_attention_shared_export_to_compression(native):
    from RCO_solver.e2e_ste import run_e2e_kl_search
    from examples.BR_compress_auto import build_auto_compress_config
    torch.manual_seed(12)
    model = ToyModel()
    original = copy.deepcopy(model)
    inputs = [{"input_ids": torch.randn(1, 5, 32), "attention_mask": torch.ones(1, 5)}]
    units = discover_units(model)
    importance = collect_importance(model, inputs, units)
    assert len(importance) == 4
    options = ["IQ4_NL_rank0", "IQ4_NL_rank2", "IQ3_S_rank0", "F16_rank0"]
    costs, valid = unit_costs(units, options)
    budget = float(costs[:, 1].sum())
    chosen, info = run_e2e_kl_search(model, inputs, options, torch.tensor(costs), budget,
                                   steps=2, importance=importance, valid_options=valid)
    assert set(chosen) == {u.key for u in units}
    assert "IQ3_S_rank0" not in chosen.values()
    assert info["selected_cost_mib"] <= budget + 1e-8
    assert all(np.isfinite(h["kl_loss"]) for h in info["history"])
    cfg = build_auto_compress_config(original, chosen, importance=importance)
    for unit in units:
        qtype, rank = parse_option(chosen[unit.key])
        for name in unit.names:
            assert cfg[name]["weight_quant_params"]["quant_type"] == qtype
            assert cfg[name]["compensator_params"]["dense_rank"] == rank
    with pytest.raises(ValueError, match="unmatched"):
        build_auto_compress_config(original, {"L99.self_attn.q_proj": "F16_rank0"})


def test_proxy_search_masks_alignment(native):
    from RCO_solver.full_model_search import proxy_losses
    from RCO_solver.rco_search import rco_optimize_table
    model = ToyModel()
    units = discover_units(model)
    options = ["IQ4_NL_rank0", "IQ3_S_rank0", "F16_rank0"]
    costs, valid = unit_costs(units, options)
    losses = proxy_losses(model, units, options, valid, {})
    budget = costs[:, 0].sum() * 1.5
    chosen, info = rco_optimize_table([u.key for u in units], options, losses, costs,
                                     budget, steps=2, valid_options=valid, verbose=False)
    assert "IQ3_S_rank0" not in chosen.values()
    assert info["total_memory_mb"] <= budget + 1e-8


def test_gguf_iq_payload_and_2d_adapter(native, tmp_path):
    import gguf
    from tools.export_brmoe_to_gguf import MinimalGGUFWriter, export_brmoe_checkpoint_to_gguf, read_gguf_header_summary
    writer = MinimalGGUFWriter()
    x = np.random.default_rng(1).normal(size=(2, 8, 256)).astype(np.float32)
    for qtype in ("IQ3_S", "IQ4_XS", "IQ4_NL", "IQ2_XS"):
        writer.add_kquant_tensor(qtype, x.shape, qtype, quantize(x, qtype, np.ones(256)))
    target = tmp_path / "weights.gguf"
    writer.write(str(target))
    tensors = gguf.GGUFReader(str(target)).tensors
    assert len(tensors) == 4
    for tensor in tensors:
        assert tensor.tensor_type == get_format(tensor.name).type_id
        np.testing.assert_allclose(gguf.dequantize(tensor.data, tensor.tensor_type).reshape(x.shape),
                                   dequantize(tensor.data, x.shape, tensor.name), atol=1e-6)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    attention = "model.layers.0.self_attn.q_proj"
    shared = "model.layers.0.mlp.shared_expert.up_proj"
    torch.save({attention: (torch.randn(32, 2), torch.randn(2, 32)),
                shared: (torch.randn(32, 4), torch.randn(4, 32))}, checkpoint / "compensators.pt")
    torch.save({attention: {"nbits": 4, "quant_type": "IQ4_NL"},
                shared: {"nbits": 4, "quant_type": "IQ4_XS"}}, checkpoint / "qmodel.pt")
    output = tmp_path / "output"
    export_brmoe_checkpoint_to_gguf(str(checkpoint), str(output))
    import json
    recipe = json.loads((output / "brmoe_gguf_tensor_types.json").read_text())
    assert recipe["tensor_ggml_types"]["blk.0.attn_q.weight"] == "IQ4_NL"
    assert recipe["tensor_ggml_types"]["blk.0.ffn_up_shexp.weight"] == "IQ4_XS"
    tensors = gguf.GGUFReader(str(output / "model-brmoe-compensator.gguf")).tensors
    assert len(tensors) == 4


def test_real_hf_checkpoint_pipeline(native, tmp_path, monkeypatch):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    from examples.BR_compress_auto import build_auto_compress_config, ensure_unfused_moe_experts, UniversalMoEBRMoE
    from RCO_solver.tensor_units import allocation_report
    monkeypatch.chdir(tmp_path)
    config = Qwen2MoeConfig(hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
                            shared_expert_intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2, num_experts=2,
                            num_experts_per_tok=2, vocab_size=32)
    config._name_or_path = "tiny-qwen2moe"
    model = Qwen2MoeForCausalLM(config).eval()
    ensure_unfused_moe_experts(model)
    units = discover_units(model)
    assert {u.category for u in units} == {"attention", "shared", "routed"}
    costs, valid = unit_costs(units, ["IQ4_NL_rank2"])
    assert valid.all()
    chosen = {u.key: "IQ4_NL_rank2" for u in units}
    report = allocation_report(units, ["IQ4_NL_rank2"], costs, chosen)
    cfg = build_auto_compress_config(model, report)
    assert "self_attn.q_proj" not in cfg  # full manifest has no implicit defaults
    model.mix_moe_bit = True
    UniversalMoEBRMoE.compress_model(model, cfg, device="cpu", compute_dtype=torch.float32)
    tokens = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = model(tokens).logits
    checkpoint = tmp_path / "checkpoint"
    UniversalMoEBRMoE.save_compressed(model, str(checkpoint))
    weights = torch.load(checkpoint / "qmodel.pt", weights_only=False)
    comps = torch.load(checkpoint / "compensators.pt", weights_only=False)
    payload = 0
    for unit in units:
        for name in unit.names:
            packed = weights[name]["W_q"]
            payload += packed.numel() * packed.element_size()
            payload += sum(t.numel() * t.element_size() for t in comps[name])
    assert payload == report["total_payload_bytes"]
    restored = UniversalMoEBRMoE.from_compressed(str(checkpoint), compute_dtype=torch.float32, device="cpu")
    with torch.no_grad():
        actual = restored(tokens).logits
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_dense_ffn_inventory_preserves_concatenated_expert():
    model = ToyModel()
    mlp = model.model.layers[0].mlp
    del mlp.experts
    for projection in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, projection, nn.Linear(32, 32, bias=False))
    units = discover_units(model)
    routed = [unit for unit in units if unit.category == "routed"]
    assert {unit.key for unit in routed} == {"L0_gate_exps", "L0_up_exps", "L0_down_exps"}
    assert all(len(unit.names) == 1 for unit in routed)
    assert len([unit for unit in discover_units(model, "layer") if unit.category == "routed"]) == 1
