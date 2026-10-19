"""Discover real 2D/3D search units and charge their deployed payload sizes."""
from dataclasses import dataclass, asdict
import re
import numpy as np
from brmoe_quant import get_format, parse_option, tensor_nbytes


@dataclass
class TensorUnit:
    key: str
    category: str
    names: list
    shapes: list


def discover_units(model, granularity="tensor", include_attention=True, include_shared=True):
    import torch.nn as nn
    from RCO_solver.e2e_ste import _map_projection_to_unit_key
    units = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in name.split('.') for p in ("visual", "vision_tower", "vision_model", "audio_tower")):
            continue
        m = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", name)
        if not m:
            continue
        layer, tail = int(m[1]), m[2]
        routed = re.fullmatch(r"(?:mlp|block_sparse_moe|feed_forward|moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3|fc1|fc2)", tail)
        if routed:
            key = _map_projection_to_unit_key(layer, int(routed[1]), routed[2], granularity)
            category = "routed"
        elif include_attention and tail.split('.')[0] in ("self_attn", "linear_attn", "attention", "attn"):
            key, category = f"L{layer}.{tail}", "attention"
        elif include_shared and any(p in tail.split('.') for p in ("shared_expert", "shared_experts", "shared_mlp")):
            key, category = f"L{layer}.{tail}", "shared"
        else:
            continue
        unit = units.setdefault(key, TensorUnit(key, category, [], []))
        unit.names.append(name)
        unit.shapes.append(tuple(module.weight.shape))
    return list(units.values())


def unit_costs(units, option_names, compensator_dtype="fp16"):
    """Return MiB costs and valid-option mask; rank is clipped to matrix dimensions.

    Base and U/V row padding is never invented. Non-aligned candidates are masked.
    Fixed parameters, activations and KV cache are outside this search budget.
    """
    comp_type = {"fp16": "F16", "f16": "F16", "float16": "F16", "q8_0": "Q8_0"}.get(compensator_dtype.lower())
    if comp_type is None:
        raise ValueError("Shape-aware GGUF search supports fp16 or q8_0 compensators")
    costs = np.zeros((len(units), len(option_names)), dtype=np.float64)
    valid = np.ones_like(costs, dtype=bool)
    for i, unit in enumerate(units):
        for k, option in enumerate(option_names):
            qtype, rank = parse_option(option)
            if qtype == 0:
                # Pruning attention/shared projections is not part of this search.
                valid[i, k] = unit.category == "routed"
                continue
            try:
                size = 0
                for out_dim, in_dim in unit.shapes:
                    size += tensor_nbytes((out_dim, in_dim), qtype)
                    r = min(rank, out_dim, in_dim)
                    if r:
                        size += tensor_nbytes((out_dim, r), comp_type)
                        size += tensor_nbytes((r, in_dim), comp_type)
                costs[i, k] = size / (1024 ** 2)
            except ValueError:
                valid[i, k] = False
        if not valid[i].any():
            raise ValueError(f"No aligned quantization candidate for {unit.key} {unit.shapes}; include F16_rank0 as a fallback")
    return costs, valid


def collect_importance(model, inputs, units):
    """Mean squared input activation per channel, including routed expert inputs."""
    import torch
    stats, counts, handles = {}, {}, []
    def hook(name):
        def capture(_module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            if x.shape[0]:
                val = x.square().sum(0).cpu()
                stats[name] = stats.get(name, 0) + val
                counts[name] = counts.get(name, 0) + x.shape[0]
        return capture
    for unit in units:
        for name in unit.names:
            handles.append(model.get_submodule(name).register_forward_pre_hook(hook(name)))
    try:
        with torch.no_grad():
            for inp in inputs:
                model(**inp) if isinstance(inp, dict) else model(inp)
    finally:
        for h in handles:
            h.remove()
    # Unrouted experts have no empirical importance. Leave them absent so callers
    # mask formats requiring an imatrix instead of silently fabricating statistics.
    return {name: (value / counts[name]).numpy() for name, value in stats.items() if torch.isfinite(value).all() and (value > 0).any()}


def allocation_report(units, options, costs, chosen):
    result = {"schema_version": 1, "budget_scope": "selected_weight_payloads_and_compensators", "units": {}}
    for i, unit in enumerate(units):
        opt = chosen[unit.key]
        qtype, rank = parse_option(opt)
        result["units"][unit.key] = {
            **asdict(unit), "config": opt,
            "quant_type": "prune" if qtype == 0 else get_format(qtype).name,
            "rank": rank,
            "effective_ranks": [min(rank, *s) for s in unit.shapes],
            "payload_bytes": round(costs[i, options.index(opt)] * 1024 ** 2),
        }
    result["total_payload_bytes"] = sum(u["payload_bytes"] for u in result["units"].values())
    return result
