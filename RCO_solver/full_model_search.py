"""Shape-aware search over routed, shared and attention weight payloads."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from brmoe_quant import get_format, parse_option
from RCO_solver.tensor_units import discover_units, unit_costs, collect_importance, allocation_report


def prepare_candidates(model, options, inputs, granularity="tensor", include_attention=True,
                       include_shared=True, importance=None):
    units = discover_units(model, granularity, include_attention, include_shared)
    if not units:
        raise ValueError("No searchable linear weights found; unpack fused experts first")
    costs, valid = unit_costs(units, options)
    needs_importance = any(get_format(parse_option(o)[0]).requires_imatrix for o in options)
    if importance is None:
        importance = collect_importance(model, inputs, units) if needs_importance else {}
    importance = {name: np.asarray(value, dtype=np.float32) for name, value in importance.items()}
    for unit in units:
        for name, shape in zip(unit.names, unit.shapes):
            value = importance.get(name)
            if value is not None:
                if value.shape != (shape[-1],) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"Invalid importance vector for {name}")
                if not (value > 0).any():
                    del importance[name]
    for i, unit in enumerate(units):
        for j, option in enumerate(options):
            if get_format(parse_option(option)[0]).requires_imatrix:
                valid[i, j] &= all(
                    name in importance and np.asarray(importance[name]).shape == (shape[-1],)
                    and np.isfinite(importance[name]).all() and (importance[name] >= 0).all()
                    and (importance[name] > 0).any()
                    for name, shape in zip(unit.names, unit.shapes)
                )
    if not valid.any(axis=1).all():
        bad = [u.key for u, row in zip(units, valid) if not row.any()]
        raise ValueError(f"No valid candidates for {bad}; include F16_rank0 or supply calibration importance")
    return units, costs, valid, importance


@torch.no_grad()
def proxy_losses(model, units, options, valid, importance):
    """Activation-weighted squared residuals with the same native base/prefix SVD."""
    from RCO_solver.e2e_ste import RCOInterpolatedLinear
    losses = np.zeros(valid.shape, dtype=np.float64)
    shared = {"valid_options": valid, "importance": importance, "compensator_dtype": "fp16"}
    for i, unit in enumerate(units):
        for name in unit.names:
            original = model.get_submodule(name)
            layer = RCOInterpolatedLinear(original, i, options, shared, cpu_offload=True, module_name=name)
            channel_weight = torch.as_tensor(importance.get(name, np.ones(original.in_features)),
                                            device=original.weight.device, dtype=torch.float32)
            for k in np.flatnonzero(valid[i]):
                candidate = layer.get_candidate_weight(int(k), original.weight.device, original.weight.dtype)
                losses[i, k] += float(((original.weight.float() - candidate.float()).square() * channel_weight).sum())
    return losses


def _texts(path):
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    for file in files:
        if file.suffix == ".parquet":
            import pandas as pd
            frame = pd.read_parquet(file)
            yield from frame["text"].dropna().astype(str)
        elif file.suffix == ".jsonl":
            with file.open() as stream:
                for line in stream:
                    if line.strip():
                        yield str(json.loads(line)["text"])
        elif file.suffix == ".txt":
            yield from file.read_text().splitlines()


def run_cli(args, progress_callback=None):
    from transformers import AutoTokenizer
    from examples.BR_compress_auto import load_hf_moe_model, ensure_unfused_moe_experts
    from RCO_solver.e2e_ste import run_e2e_kl_search

    if args.compare_expert_freedom or args.only_3bit or args.max_compensated_fraction is not None:
        raise ValueError("Use --quant_types and --granularity for full-model e2e search; proxy-only flags are not supported")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    types = list(dict.fromkeys(get_format(t).name for t in args.quant_types.split(",")))
    ranks = sorted(set(int(r) for r in args.ranks.split(",") if args.min_rank <= int(r) <= args.max_rank))
    if not ranks or min(ranks) < 0:
        raise ValueError("--ranks must contain nonnegative ranks within --min_rank/--max_rank")
    options = [f"{t}_rank{r}" for t in types for r in ranks if t not in ("F16", "F32") or r == 0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_hf_moe_model(args.model_path, dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    model.config.use_cache = False
    ensure_unfused_moe_experts(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = []
    for text in _texts(args.dataset_path):
        if text.strip():
            # No padding: statistics and KL see only real calibration tokens.
            encoded = tokenizer(text, max_length=args.seq_len, truncation=True, return_tensors="pt")
            inputs.append({k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")})
        if len(inputs) >= args.num_calib_samples:
            break
    if not inputs or args.num_calib_samples < 1:
        raise ValueError("Calibration dataset is empty; provide text JSONL, TXT or Parquet")
    importance = None
    if args.imatrix_file:
        with np.load(args.imatrix_file, allow_pickle=False) as data:
            importance = dict(data)
    units, costs, valid, importance = prepare_candidates(
        model, options, inputs, args.granularity, not args.skip_attention,
        not args.skip_shared_expert, importance,
    )
    if progress_callback is not None:
        progress_callback("calibration", {"samples": len(inputs), "units": len(units),
            "module_count": sum(len(u.names) for u in units), "options": len(options)})
    # Preserve expensive calibration statistics even if a later search step fails.
    if importance:
        calibration_path = Path(args.output_file).with_suffix(".imatrix.npz")
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(calibration_path, **importance)
    numel = sum(int(np.prod(shape)) for unit in units for shape in unit.shapes)
    budget = args.budget
    if budget is None and args.avg_bits is not None:
        budget = numel * args.avg_bits / 8 / (1024 ** 2)
    if budget is None:
        raise ValueError("Specify --budget (selected payload MiB) or --avg_bits")
    lower = np.where(valid, costs, np.inf).min(1).sum()
    upper = np.where(valid, costs, -np.inf).max(1).sum()
    if not np.isfinite(budget) or budget < lower - 1e-8:
        raise ValueError(f"Budget {budget} MiB is below minimum feasible {lower} MiB")
    counts = {category: sum(u.category == category for u in units) for category in ("routed", "attention", "shared")}
    print(f"Search units: {counts}; selected payload budget {budget:.3f} MiB; minimum {lower:.3f} MiB", flush=True)
    if args.mode == "tensor_proxy":
        from RCO_solver.rco_search import rco_optimize_table
        losses = proxy_losses(model, units, options, valid, importance)
        best, info = rco_optimize_table([u.key for u in units], options, losses, costs,
                                        budget, steps=args.steps, lr=args.lr, valid_options=valid,
                                        seed=args.seed, gumbel_samples=args.gumbel_samples)
    else:
        best, info = run_e2e_kl_search(
            model, inputs, options, torch.tensor(costs, dtype=torch.float64, device=device),
            target_budget=min(budget, upper), steps=args.steps, lr=args.lr, granularity=args.granularity,
            include_attention=not args.skip_attention, include_shared=not args.skip_shared_expert,
            importance=importance, valid_options=valid,
            progress_callback=progress_callback,
        )
    report = allocation_report(units, options, costs, best)
    from brmoe_quant.native import REVISION
    report["codec"] = {"backend": "ggml_quantize_chunk", "revision": REVISION}
    report["calibration"] = {"samples": len(inputs), "max_seq_len": args.seq_len, "seed": args.seed}
    report.update(budget_mib=budget, selected_numel=numel, granularity=args.granularity, mode=args.mode, search=info)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    if importance:
        imatrix_path = out.with_suffix(".imatrix.npz")
        np.savez(imatrix_path, **importance)
        report["imatrix_file"] = imatrix_path.name
    out.write_text(json.dumps(best, indent=2) + "\n")
    manifest = Path(args.tensor_output_file) if args.tensor_output_file else out.with_name(out.stem + "_tensor_allocation.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if importance:
        report["imatrix_file"] = os.path.relpath(imatrix_path.resolve(), manifest.parent.resolve())
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {out}, {manifest}; selected payload {report['total_payload_bytes'] / 1024**2:.3f} MiB")
