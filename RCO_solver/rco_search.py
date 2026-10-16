#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RCO Solver for BR-MoE: Riemannian Constrained Optimization of Mixed-Precision
Quantization, Low-Rank Compensators, and Expert Pruning.

Implements the method from:
    Michael Helcig, Dan Alistarh (2026).
    "Model Compression with Exact Budget Constraints via Riemannian Manifolds"
    https://arxiv.org/abs/2605.00649

Supports three operating modes for BR-MoE:
  1. `proxy`:
     Uses BR-MoE's collected L2 impact & expert frequency tables (same inputs as
     `ILP_solver/solver.py`), optimizing over the Riemannian budget manifold with
     optional cross-expert non-decomposable interactions and multi-constraint budgets.
  2. `int3_codesign`:
     Multi-constraint RCO co-design specifically for `brmoe_int3_vllm` / `brmoe_grouped`
     inference backends: simultaneously enforces an exact memory budget AND an exact
     limit on the fraction of experts using low-rank compensators (`rank > 0`) or
     pruned experts, with zero penalty hyperparameters.
  3. `e2e_kl`:
     End-to-end calibration KL-divergence optimization directly on a HuggingFace MoE
     model using Gumbel-STE, budget-constrained DP, and streaming weight assembly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RCO_solver.manifold import (
    BudgetManifold,
    InequalityBudgetManifold,
    MultiBudgetManifold,
    NumpyAdam,
    _softmax_np,
    _to_numpy,
    dp_solve_mckp,
)


# ---------------------------------------------------------------------------
# Data Loading Utilities (compatible with ILP_solver/solver.py & model_statistics)
# ---------------------------------------------------------------------------


def load_expert_freq(freq_file_path: str) -> Dict[str, float]:
    """Load expert activation frequencies across all BR-MoE JSON formats."""
    with open(freq_file_path, "r", encoding="utf-8") as f:
        freq_data = json.load(f)

    expert_freq: Dict[str, float] = {}

    if isinstance(freq_data, dict):
        if "frequency" in freq_data:
            # Mixtral format: {"frequency": {"model.layers.0.block_sparse_moe": [...]}}
            freq_dict = freq_data["frequency"]
            for layer_key, experts_freq in freq_dict.items():
                layer_match = re.search(r"layers\.([0-9]+)", layer_key)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    for expert_idx, freq in enumerate(experts_freq):
                        expert_freq[f"L{layer_idx}_E{expert_idx}"] = float(freq)
        elif any(re.match(r"layer_\d+", str(k)) for k in freq_data.keys()):
            # Qwen format: {"layer_0": {"expert_51": 3398, ...}}
            for layer_key, experts_dict in freq_data.items():
                layer_match = re.search(r"layer_(\d+)", str(layer_key))
                if layer_match and isinstance(experts_dict, dict):
                    layer_idx = int(layer_match.group(1))
                    for exp_key, freq in experts_dict.items():
                        exp_match = re.search(r"expert_(\d+)", str(exp_key))
                        if exp_match:
                            expert_idx = int(exp_match.group(1))
                            expert_freq[f"L{layer_idx}_E{expert_idx}"] = float(freq)
        else:
            # Flat format: {"L0_E0": 0.05} or {"0-0": 0.05}
            expert_pattern = re.compile(r"L\d+_E\d+")
            for key, value in freq_data.items():
                if expert_pattern.match(str(key)):
                    expert_freq[str(key)] = float(value)
                elif "-" in str(key):
                    parts = str(key).split("-")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        expert_freq[f"L{parts[0]}_E{parts[1]}"] = float(value)
    elif isinstance(freq_data, list):
        # DeepSeek format: [[95066, 99845, ...], ...] (layers 1..27)
        for layer_idx, layer_freqs in enumerate(freq_data):
            for expert_idx, freq in enumerate(layer_freqs):
                expert_freq[f"L{layer_idx + 1}_E{expert_idx}"] = float(freq)

    # Normalize frequencies per layer so scale is consistent across models
    by_layer: Dict[int, List[str]] = defaultdict(list)
    for k in expert_freq:
        m = re.match(r"L(\d+)_E(\d+)", k)
        if m:
            by_layer[int(m.group(1))].append(k)
    for _, keys in by_layer.items():
        s = sum(expert_freq[k] for k in keys)
        if s > 0:
            for k in keys:
                expert_freq[k] = expert_freq[k] / s * len(keys)

    return expert_freq


def extract_bit_and_rank(config_name: str) -> Tuple[int, int]:
    if config_name.lower() == "prune":
        return 0, 0
    b_match = re.search(r"(\d+)bit", config_name)
    r_match = re.search(r"rank(\d+)", config_name)
    bit = int(b_match.group(1)) if b_match else 0
    rank = int(r_match.group(1)) if r_match else 0
    return bit, rank


def prepare_problem_tensors(
    l2_results: dict,
    memory_usage: dict,
    expert_freq: Dict[str, float],
    min_rank: int = 0,
    max_rank: Optional[int] = 256,
    allow_4bit_high_rank: bool = False,
    allowed_bits: Optional[List[int]] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build (experts, configs, loss_matrix [N, K], cost_vec [K], freq_vec [N])."""
    experts = list(l2_results.keys())
    first_expert = experts[0]
    available_configs = list(l2_results[first_expert].keys())

    configs = []
    for cfg in memory_usage.keys():
        if cfg not in available_configs:
            continue
        bit, rank = extract_bit_and_rank(cfg)
        if allowed_bits is not None and bit not in allowed_bits:
            continue
        if rank < min_rank:
            continue
        if max_rank is not None and rank > max_rank:
            continue
        if not allow_4bit_high_rank and bit == 4 and rank > 32:
            continue
        configs.append(cfg)

    if not configs:
        raise ValueError("No valid configurations remain after filtering!")

    N = len(experts)
    K = len(configs)
    loss_mat = np.zeros((N, K), dtype=np.float64)
    cost_vec = np.zeros(K, dtype=np.float64)
    freq_vec = np.zeros(N, dtype=np.float64)

    for k, cfg in enumerate(configs):
        cost_vec[k] = float(memory_usage[cfg])

    for i, exp_key in enumerate(experts):
        f_w = float(expert_freq.get(exp_key, 1.0))
        freq_vec[i] = f_w
        for k, cfg in enumerate(configs):
            raw_val = l2_results[exp_key][cfg]
            if isinstance(raw_val, dict):
                l2_val = float(
                    raw_val.get("l2_norm", raw_val.get("relative_error", 0.0))
                )
            else:
                l2_val = float(raw_val)
            loss_mat[i, k] = f_w * l2_val

    return experts, configs, loss_mat, cost_vec, freq_vec


# ---------------------------------------------------------------------------
# Algorithm 1: Riemannian Constrained Optimization (RCO) Search
# ---------------------------------------------------------------------------


def rco_optimize_table(
    experts: List[str],
    configs: List[str],
    loss_mat: np.ndarray,
    cost_vec: np.ndarray,
    target_budget: float,
    steps: int = 300,
    lr: float = 0.1,
    gumbel_samples: int = 0,
    tau_init: float = 1.0,
    tau_min: float = 0.01,
    use_slack: bool = False,
    warm_start: bool = True,
    interaction_strength: float = 0.0,
    max_compensated_fraction: Optional[float] = None,
    dp_resolution: int = 4000,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, str], dict]:
    """Run Riemannian Constrained Optimization (RCO) over the (N, K) configuration space."""
    rng = np.random.default_rng(seed)
    loss_np = _to_numpy(loss_mat).astype(np.float64)
    cost_raw = _to_numpy(cost_vec).astype(np.float64)
    N, K = loss_np.shape
    cost_np = np.broadcast_to(
        cost_raw[None, :] if cost_raw.ndim == 1 else cost_raw, (N, K)
    ).copy()

    min_budget = float(cost_np.min(axis=1).sum())
    max_budget = float(cost_np.max(axis=1).sum())
    if target_budget < min_budget:
        raise ValueError(
            f"Target budget {target_budget:.2f} MB is below minimum possible {min_budget:.2f} MB!"
        )
    eff_budget = min(float(target_budget), max_budget - 1e-3)

    # Group indices by layer for non-decomposable intra-layer interaction
    layer_groups: List[np.ndarray] = []
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for idx, exp_key in enumerate(experts):
        m = re.match(r"L(\d+)_", exp_key)
        layer_id = int(m.group(1)) if m else 0
        by_layer[layer_id].append(idx)
    for _, idxs in sorted(by_layer.items()):
        layer_groups.append(np.array(idxs, dtype=np.int64))

    # Initialize logits alpha_0
    if warm_start:
        norm_loss = loss_np / (loss_np.mean() + 1e-12)
        alpha = -np.log(np.clip(norm_loss, 1e-8, None))
        alpha = alpha - alpha.mean(axis=-1, keepdims=True)
    else:
        alpha = np.zeros((N, K), dtype=np.float64)

    min_avail_rank = min(extract_bit_and_rank(c)[1] for c in configs)
    comp_indicator = np.array(
        [1.0 if extract_bit_and_rank(c)[1] > min_avail_rank else 0.0 for c in configs],
        dtype=np.float64,
    )
    has_multi_budget = (
        max_compensated_fraction is not None
        and (comp_indicator.min() < comp_indicator.max())
    )

    if has_multi_budget:
        target_comp_count = float(max_compensated_fraction) * N
        manifold = MultiBudgetManifold(
            cost_list=[cost_np, comp_indicator],
            target_budgets=[eff_budget, target_comp_count],
            tol=1e-8,
        )
    elif use_slack:
        manifold = InequalityBudgetManifold(
            costs=cost_np, target_budget=eff_budget, tol=1e-8
        )
    else:
        manifold = BudgetManifold(
            costs=cost_np, target_budget=eff_budget, tol=1e-8
        )

    manifold.initialize(alpha)
    optimizer = NumpyAdam([alpha], lr=lr)

    def eval_obj_and_grad_wrt_probs(probs_in: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute objective L(p) and dL/dp in R^{N x K}."""
        per_expert = np.sum(probs_in * loss_np, axis=-1)  # [N]
        total = float(np.sum(per_expert))
        grad_p = loss_np.copy()
        if interaction_strength > 0.0:
            for l_idxs in layer_groups:
                layer_err = float(np.sum(per_expert[l_idxs]))
                scale_l = interaction_strength / max(1, len(l_idxs))
                total += scale_l * (layer_err ** 2)
                grad_p[l_idxs] += (2.0 * scale_l * layer_err) * loss_np[l_idxs]
        return total, grad_p

    history = []
    t0 = time.time()

    for step in range(steps):
        progress = step / max(steps - 1, 1)
        tau = max(tau_min, tau_init * ((tau_min / tau_init) ** progress))

        optimizer.zero_grad()

        if gumbel_samples <= 0:
            probs = _softmax_np(alpha, axis=-1)
            step_loss, grad_p = eval_obj_and_grad_wrt_probs(probs)
            # Chain rule through softmax: dL/d(alpha_{ik}) = p_{ik} * (grad_p_{ik} - <p_i, grad_p_i>)
            exp_gp = np.sum(probs * grad_p, axis=-1, keepdims=True)
            grad_alpha = probs * (grad_p - exp_gp)
            optimizer.set_grad(alpha, grad_alpha)
        else:
            accum_loss = 0.0
            accum_grad = np.zeros_like(alpha)
            for _ in range(gumbel_samples):
                u = np.clip(rng.random(alpha.shape), 1e-15, 1.0 - 1e-15)
                gumbel = -np.log(-np.log(u))
                noisy_logits = (alpha + gumbel) / tau
                soft_p = _softmax_np(noisy_logits, axis=-1)

                z_idx = dp_solve_mckp(
                    noisy_logits,
                    cost_np,
                    eff_budget,
                    resolution=min(1000, dp_resolution),
                )
                z_onehot = np.eye(K, dtype=np.float64)[z_idx]

                # Straight-through forward value is z_onehot, backward Jacobian is d(soft_p)/d(alpha)
                loss_s, grad_p = eval_obj_and_grad_wrt_probs(z_onehot)
                exp_gp = np.sum(soft_p * grad_p, axis=-1, keepdims=True)
                grad_s = (soft_p / tau) * (grad_p - exp_gp)
                accum_loss += loss_s / gumbel_samples
                accum_grad += grad_s / gumbel_samples
            step_loss = accum_loss
            optimizer.set_grad(alpha, accum_grad)

        info = manifold.step(alpha, optimizer)

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            p_cur = _softmax_np(alpha, axis=-1)
            committed = int(np.sum(np.max(p_cur, axis=-1) > 0.9))
            history.append(
                {
                    "step": step,
                    "loss": step_loss,
                    "violation": info["violation"],
                    "committed": committed,
                }
            )
            if verbose:
                print(
                    f"  [RCO Step {step:3d}/{steps}] obj={step_loss:.4f} | "
                    f"|C(alpha)-B|={info['violation']:.2e} | "
                    f"committed={committed}/{N}"
                )

    # Final discrete extraction (Algorithm 1, line 16)
    final_probs = _softmax_np(alpha, axis=-1)
    log_probs = np.log(np.clip(final_probs, 1e-12, None))

    if has_multi_budget:
        chosen_idx = _decode_multi_budget_dp(
            log_probs,
            loss_np,
            cost_np,
            comp_indicator,
            float(target_budget),
            float(max_compensated_fraction) * N,
            resolution=dp_resolution,
        )
    else:
        # Combine learned manifold log_probs with direct distortion scores
        scores = log_probs - (loss_np / (loss_np.mean() + 1e-12))
        chosen_idx = dp_solve_mckp(
            scores,
            cost_np,
            float(target_budget),
            resolution=dp_resolution,
        )

    chosen_idx = _local_feasible_polish(
        chosen_idx,
        loss_np,
        cost_np,
        float(target_budget),
        comp_indicator if has_multi_budget else None,
        float(max_compensated_fraction) * N if has_multi_budget else None,
        layer_groups=layer_groups if interaction_strength > 0 else None,
        interaction_strength=interaction_strength,
    )

    elapsed = time.time() - t0
    best_config_map: Dict[str, str] = {}
    total_mem = 0.0
    total_l2 = 0.0
    comp_count = 0
    for i, exp_key in enumerate(experts):
        k_star = int(chosen_idx[i])
        cfg_name = configs[k_star]
        best_config_map[exp_key] = cfg_name
        total_mem += float(cost_np[i, k_star])
        total_l2 += float(loss_np[i, k_star])
        if extract_bit_and_rank(cfg_name)[1] > min_avail_rank:
            comp_count += 1

    summary = {
        "total_loss": total_l2,
        "total_memory_mb": total_mem,
        "budget_mb": float(target_budget),
        "compensated_experts": comp_count,
        "compensated_fraction": comp_count / max(1, N),
        "final_manifold_violation": history[-1]["violation"] if history else 0.0,
        "elapsed_sec": elapsed,
        "history": history,
    }

    if verbose:
        _print_summary(best_config_map, summary, N)

    return best_config_map, summary


def _decode_multi_budget_dp(
    log_probs: np.ndarray,
    loss_mat: np.ndarray,
    cost_vec: np.ndarray,
    comp_indicator: np.ndarray,
    target_budget: float,
    max_comp_count: float,
    resolution: int = 2000,
) -> np.ndarray:
    """Decode discrete assignment under both memory budget and max compensated count."""
    N, K = log_probs.shape
    max_comp_int = int(math.floor(max_comp_count + 1e-9))

    scores_base = log_probs - (loss_mat / (loss_mat.mean() + 1e-12))

    def solve_with_mu(mu: float) -> Tuple[np.ndarray, int]:
        penalized_scores = scores_base - mu * comp_indicator[None, :]
        idx = dp_solve_mckp(
            penalized_scores, cost_vec, target_budget, resolution=resolution
        )
        comp_used = int(np.sum(comp_indicator[idx]))
        return idx, comp_used

    idx0, used0 = solve_with_mu(0.0)
    if used0 <= max_comp_int:
        return idx0

    lo, hi = 0.0, 100.0
    best_idx = idx0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        idx_m, used_m = solve_with_mu(mid)
        if used_m <= max_comp_int:
            best_idx = idx_m
            hi = mid
        else:
            lo = mid

    comp_used = int(np.sum(comp_indicator[best_idx]))
    cost_2d = np.broadcast_to(
        cost_vec[None, :] if cost_vec.ndim == 1 else cost_vec, (N, K)
    )
    if comp_used > max_comp_int:
        uncomp_indices = np.where(comp_indicator == 0)[0]
        if len(uncomp_indices) > 0:
            comp_experts = [
                i for i in range(N) if comp_indicator[best_idx[i]] > 0.5
            ]
            candidates = []
            for i in comp_experts:
                cur_k = int(best_idx[i])
                cur_c = float(cost_2d[i, cur_k])
                cur_l = float(loss_mat[i, cur_k])
                best_alt_k = -1
                best_alt_delta = float("inf")
                for uk in uncomp_indices.tolist():
                    if float(cost_2d[i, uk]) <= cur_c + 1e-9:
                        delta = float(loss_mat[i, uk]) - cur_l
                        if delta < best_alt_delta:
                            best_alt_delta = delta
                            best_alt_k = uk
                if best_alt_k >= 0:
                    candidates.append((best_alt_delta, i, best_alt_k))
            candidates.sort(key=lambda x: x[0])
            need_drop = comp_used - max_comp_int
            for _, i, alt_k in candidates[:need_drop]:
                best_idx[i] = alt_k

    return best_idx


def _local_feasible_polish(
    chosen_idx: np.ndarray,
    loss_np: np.ndarray,
    costs_np: np.ndarray,
    target_budget: float,
    comp_np: Optional[np.ndarray] = None,
    max_comp_count: Optional[float] = None,
    layer_groups: Optional[List[np.ndarray]] = None,
    interaction_strength: float = 0.0,
) -> np.ndarray:
    """Fast O(N K) greedy upgrade pass to utilize any remaining budget slack from DP grid rounding."""
    N, K = loss_np.shape
    cost_2d = np.broadcast_to(
        costs_np[None, :] if costs_np.ndim == 1 else costs_np, (N, K)
    )
    idx = np.asarray(chosen_idx, dtype=np.int64).copy()
    max_comp_int = (
        int(math.floor(max_comp_count + 1e-9))
        if max_comp_count is not None
        else N
    )

    cur_mem = float(sum(cost_2d[i, idx[i]] for i in range(N)))
    cur_comp = (
        int(sum(comp_np[idx[i]] for i in range(N))) if comp_np is not None else 0
    )

    layer_of = np.zeros(N, dtype=np.int64)
    layer_size = np.ones(N, dtype=np.float64)
    layer_err_sum = {}
    if layer_groups is not None and interaction_strength > 0:
        for lid, l_arr in enumerate(layer_groups):
            s = float(sum(loss_np[i, idx[i]] for i in l_arr))
            layer_err_sum[lid] = s
            for i in l_arr:
                layer_of[i] = lid
                layer_size[i] = float(len(l_arr))

    rem_mem = target_budget - cur_mem
    if rem_mem > 1e-6:
        upgrades = []
        for i in range(N):
            k_cur = idx[i]
            c_cur = cost_2d[i, k_cur]
            l_cur = loss_np[i, k_cur]
            for k in range(K):
                dc = cost_2d[i, k] - c_cur
                dl = l_cur - loss_np[i, k]
                if 0.0 <= dc <= rem_mem and dl > 1e-9:
                    if layer_groups is not None and interaction_strength > 0:
                        lid = layer_of[i]
                        L_sum = layer_err_sum[lid]
                        dl_eff = dl + interaction_strength * (
                            2.0 * L_sum * dl - dl * dl
                        ) / layer_size[i]
                    else:
                        dl_eff = dl
                    ratio = dl_eff / max(dc, 1e-6)
                    upgrades.append((ratio, i, k))
        upgrades.sort(key=lambda x: x[0], reverse=True)
        for _, i, k_new in upgrades:
            k_cur = idx[i]
            if k_cur == k_new:
                continue
            dc = cost_2d[i, k_new] - cost_2d[i, k_cur]
            dl = loss_np[i, k_cur] - loss_np[i, k_new]
            if dl <= 0 or cur_mem + dc > target_budget + 1e-7:
                continue
            if comp_np is not None:
                d_comp = int(comp_np[k_new]) - int(comp_np[k_cur])
                if cur_comp + d_comp > max_comp_int:
                    continue
                cur_comp += d_comp
            if layer_groups is not None and interaction_strength > 0:
                layer_err_sum[layer_of[i]] -= dl
            idx[i] = k_new
            cur_mem += dc

    return idx


def _print_summary(best_config_map: Dict[str, str], summary: dict, N: int) -> None:
    print(f"\n[RCO] Optimization complete in {summary['elapsed_sec']:.2f}s")
    print(f"Total weighted loss: {summary['total_loss']:.4f}")
    print(
        f"Total memory used:   {summary['total_memory_mb']:.4f} MB "
        f"(budget: {summary['budget_mb']:.2f} MB)"
    )
    print(
        f"Compensated experts: {summary['compensated_experts']}/{N} "
        f"({summary['compensated_fraction'] * 100:.2f}%)"
    )
    print(f"Manifold violation:  {summary['final_manifold_violation']:.3e}")

    bit_counts = defaultdict(int)
    cfg_counts = defaultdict(int)
    for cfg in best_config_map.values():
        b, _ = extract_bit_and_rank(cfg)
        bit_counts[f"{b}bit"] += 1
        cfg_counts[cfg] += 1

    print("\nBit-width statistics:")
    bit_table = [
        (bit, count, f"{count / N * 100:.2f}%")
        for bit, count in sorted(bit_counts.items())
    ]
    print(tabulate(bit_table, headers=["Bits", "Count", "Percentage"], tablefmt="grid"))

    print("\nConfiguration breakdown:")
    cfg_table = [
        (cfg, count, f"{count / N * 100:.2f}%")
        for cfg, count in sorted(cfg_counts.items(), key=lambda x: -x[1])
    ]
    print(
        tabulate(
            cfg_table, headers=["Config", "Count", "Percentage"], tablefmt="grid"
        )
    )


def compare_granularity_freedom(
    experts: List[str],
    configs: List[str],
    loss_mat: np.ndarray,
    cost_vec: np.ndarray,
    freq_vec: np.ndarray,
    target_budget: float,
    granularities: Optional[List[str]] = None,
    num_experts_per_layer: int = 64,
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compare Per-3D-Tensor (`tensor` baseline) against coarser (`layer`, `vllm_tensor`)
    and finer (`expert`, `expert_tensor`) bit/rank freedom under the exact same MB budget.
    """
    from RCO_solver.model_adapters import aggregate_problem_to_granularity

    first_key = str(experts[0]) if len(experts) > 0 else ""
    is_input_3d_tensor = bool(
        re.match(r"L(\d+)_(gate_exps|up_exps|down_exps)$", first_key)
    )
    if granularities is None:
        if is_input_3d_tensor:
            granularities = ["layer", "vllm_tensor", "tensor"]
        else:
            granularities = ["layer", "vllm_tensor", "tensor", "expert", "expert_tensor"]

    results_by_gran: Dict[str, dict] = {}
    for gran in granularities:
        units_g, cfgs_g, loss_g, cost_g, _, meta_g = aggregate_problem_to_granularity(
            experts=experts,
            configs=configs,
            loss_mat=loss_mat,
            cost_vec=cost_vec,
            freq_vec=freq_vec,
            granularity=gran,
            num_experts_per_layer=num_experts_per_layer,
        )
        cfg_map_g, sum_g = rco_optimize_table(
            experts=units_g,
            configs=cfgs_g,
            loss_mat=loss_g,
            cost_vec=cost_g,
            target_budget=target_budget,
            steps=steps,
            lr=lr,
            seed=seed,
            verbose=False,
        )
        results_by_gran[gran] = {
            "num_units": len(units_g),
            "total_loss": float(sum_g["total_loss"]),
            "total_memory_mb": float(sum_g["total_memory_mb"]),
            "budget_mb": float(target_budget),
            "compensated_fraction": float(sum_g["compensated_fraction"]),
            "config_map": cfg_map_g,
            "meta": meta_g,
        }

    base_loss = results_by_gran.get(
        "tensor", next(iter(results_by_gran.values()))
    )["total_loss"]
    for gran, info in results_by_gran.items():
        info["rel_diff_vs_tensor_pct"] = (
            (info["total_loss"] - base_loss) / max(abs(base_loss), 1e-12) * 100.0
        )

    if verbose:
        rows = []
        gguf_compat = {
            "layer": "Yes (100% Native 3D)",
            "vllm_tensor": "Yes (100% Native 3D)",
            "tensor": "Yes (100% Native 3D - Baseline)",
            "expert": "Requires rank-padding / custom kernel",
            "expert_tensor": "Requires rank-padding / custom kernel",
        }
        for gran in granularities:
            info = results_by_gran[gran]
            rows.append(
                (
                    gran,
                    info["num_units"],
                    f"{info['total_loss']:.4f}",
                    f"{info['rel_diff_vs_tensor_pct']:+.2f}%",
                    f"{info['total_memory_mb']:.2f} / {info['budget_mb']:.2f}",
                    f"{info['compensated_fraction'] * 100:.1f}%",
                    gguf_compat.get(gran, ""),
                )
            )
        print("\n[Granularity Freedom Ablation vs Per-3D-Tensor Baseline]")
        print(
            tabulate(
                rows,
                headers=[
                    "Granularity",
                    "Units (N)",
                    "Total Distortion",
                    "vs 3D-Tensor",
                    "Memory (MB)",
                    "Comp %",
                    "llama.cpp GGUF",
                ],
                tablefmt="grid",
            )
        )

    return results_by_gran


def main():
    parser = argparse.ArgumentParser(
        description="RCO (Riemannian Constrained Optimization) Solver for BR-MoE"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["proxy", "int3_codesign", "e2e_kl"],
        default="proxy",
        help="Optimization mode: 'proxy' (single memory budget), 'int3_codesign' "
        "(joint memory + low-rank compensator sparsity for brmoe_int3_vllm), or "
        "'e2e_kl' (end-to-end calibration KL search on a HuggingFace MoE model)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/qwen1.5-moe",
        help="HuggingFace model path (used when --mode e2e_kl)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/wikitext-2-raw-v1",
        help="Calibration dataset directory containing .parquet files (used when --mode e2e_kl)",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=16,
        help="Number of calibration sequences for --mode e2e_kl",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Calibration sequence length for --mode e2e_kl",
    )
    parser.add_argument(
        "--l2_file",
        type=str,
        default="./model_statistics/Qwen/Qwen_expt_impact.json",
    )
    parser.add_argument(
        "--freq_file",
        type=str,
        default="./model_statistics/Qwen/Qwen_expt_freq.json",
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="./model_statistics/Qwen/Qwen_memory_usage.json",
    )
    parser.add_argument(
        "--model_preset",
        type=str,
        default=None,
        help="Built-in MoE model preset or HF repo ID or path to config.json "
        "(e.g., 'Qwen/Qwen3.6-35B-A3B', 'qwen3.6-35b-a3b', 'qwen3-30b-a3b', "
        "'deepseek-v3', 'glm-4.5-air', 'olmoe-1b-7b'). Automatically computes "
        "exact expert memory costs and supports prior-guided RCO search without pre-collected JSONs.",
    )
    parser.add_argument(
        "--avg_bits",
        type=float,
        default=None,
        help="Optional target average bits-per-weight (e.g. 3.15) to automatically compute --budget in MB",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensor", "vllm_tensor", "layer", "expert", "expert_tensor"],
        default="tensor",
        help="Decision granularity: 'tensor' (Per-3D-Tensor L{l}_{gate,up,down}_exps baseline for llama.cpp/vLLM, default), "
        "'vllm_tensor' (L{l}_{w13,w2}_exps), 'layer' (L{l}_experts), 'expert' (per-expert L{l}_E{e}), "
        "or 'expert_tensor' (L{l}_E{e}.{gate,up,down}_proj)",
    )
    parser.add_argument(
        "--compare_expert_freedom",
        action="store_true",
        help="Run an ablation comparing Per-3D-Tensor ('tensor' baseline) against 'layer', 'vllm_tensor', 'expert', and 'expert_tensor' under the exact same MB budget",
    )
    parser.add_argument(
        "--tensor_output_file",
        type=str,
        default=None,
        help="Optional path to save the structured Per-3D-Tensor allocation JSON (for llama.cpp GGUF & vLLM)",
    )
    parser.add_argument(
        "--output_file", type=str, default="best_config_rco.json"
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--min_rank", type=int, default=0)
    parser.add_argument("--max_rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--gumbel_samples",
        type=int,
        default=0,
        help="0 for exact Riemannian gradient on continuous manifold; >=1 for Gumbel-STE + DP",
    )
    parser.add_argument(
        "--interaction_strength",
        type=float,
        default=0.0,
        help="Weight of non-decomposable intra-layer expert coupling term",
    )
    parser.add_argument(
        "--max_compensated_fraction",
        type=float,
        default=None,
        help="Second simultaneous manifold constraint: max fraction of experts with rank > 0",
    )
    parser.add_argument(
        "--only_3bit",
        action="store_true",
        help="Restrict bit-width to 3-bit only (for pure INT3 vLLM kernel compatibility)",
    )
    parser.add_argument("--use_slack", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    from RCO_solver.model_adapters import (
        aggregate_problem_to_granularity,
        build_tensor_allocation_and_expanded_config,
        compute_budget_from_avg_bits,
        compute_expert_memory_table,
        resolve_moe_model_spec,
        synthesize_problem_tensors_for_model,
    )

    spec = None
    if args.model_preset:
        spec = resolve_moe_model_spec(args.model_preset)
        print(
            f"[Model Adapter] Resolved '{args.model_preset}' -> {spec.model_id} "
            f"(layers={spec.num_hidden_layers}, moe_layers={spec.num_moe_layers}, "
            f"experts={spec.num_experts}, top_k={spec.num_experts_per_tok}, "
            f"H={spec.hidden_size}, I={spec.moe_intermediate_size}, "
            f"hybrid_linear_attn={spec.is_hybrid_linear_attn})"
        )

    if args.memory_file and os.path.exists(args.memory_file) and args.model_preset is None:
        with open(args.memory_file, "r", encoding="utf-8") as f:
            memory_usage = json.load(f)
    else:
        target_spec = spec if spec is not None else resolve_moe_model_spec(args.model_path)
        spec = target_spec
        memory_usage = compute_expert_memory_table(target_spec)

    if args.budget is None:
        if args.avg_bits is not None and spec is not None:
            args.budget = compute_budget_from_avg_bits(spec, avg_bits=args.avg_bits)
        elif spec is not None:
            default_b = 3.10 if args.only_3bit else 3.20
            args.budget = compute_budget_from_avg_bits(spec, avg_bits=default_b)
        else:
            args.budget = 20728.5

    allowed_bits = [3] if args.only_3bit else None

    if args.mode == "e2e_kl":
        import glob
        import pandas as pd
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from RCO_solver.e2e_ste import run_e2e_kl_search

        prior_logits = None
        num_exp_per_layer = spec.num_experts if spec is not None else 64
        gran_meta = {"granularity": args.granularity, "layer_to_experts": {}}
        if os.path.exists(args.l2_file) and os.path.exists(args.freq_file) and args.model_preset is None:
            with open(args.l2_file, "r", encoding="utf-8") as f:
                l2_results = json.load(f)
            expert_freq = load_expert_freq(args.freq_file)
            experts_raw, configs, loss_mat, cost_vec, freq_vec = prepare_problem_tensors(
                l2_results=l2_results,
                memory_usage=memory_usage,
                expert_freq=expert_freq,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                allowed_bits=allowed_bits,
            )
            _, configs, loss_mat_g, cost_vec, _, gran_meta = aggregate_problem_to_granularity(
                experts=experts_raw,
                configs=configs,
                loss_mat=loss_mat,
                cost_vec=cost_vec,
                freq_vec=freq_vec,
                granularity=args.granularity,
                num_experts_per_layer=num_exp_per_layer,
            )
            norm_loss = loss_mat_g / (loss_mat_g.mean() + 1e-12)
            p_np = -np.log(np.clip(norm_loss, 1e-8, None))
            p_np = p_np - p_np.mean(axis=-1, keepdims=True)
            prior_logits = torch.tensor(p_np, dtype=torch.float64)
        else:
            configs = []
            for cfg in memory_usage.keys():
                bit, rank = extract_bit_and_rank(cfg)
                if allowed_bits is not None and bit not in allowed_bits:
                    continue
                if rank < args.min_rank or (args.max_rank is not None and rank > args.max_rank):
                    continue
                configs.append(cfg)
            cost_vec = np.array([float(memory_usage[c]) for c in configs], dtype=np.float64)
            if spec is not None and args.granularity != "expert":
                experts_raw = [
                    f"L{l}_E{e}"
                    for l in spec.moe_layer_indices
                    for e in range(spec.num_experts)
                ]
                dummy_loss = np.ones((len(experts_raw), len(configs)), dtype=np.float64)
                dummy_freq = np.ones(len(experts_raw), dtype=np.float64)
                _, configs, _, cost_vec, _, gran_meta = aggregate_problem_to_granularity(
                    experts=experts_raw,
                    configs=configs,
                    loss_mat=dummy_loss,
                    cost_vec=cost_vec,
                    freq_vec=dummy_freq,
                    granularity=args.granularity,
                    num_experts_per_layer=num_exp_per_layer,
                )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_cfg = transformers.AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        archs = getattr(hf_cfg, "architectures", None) or []
        auto_cls = AutoModelForCausalLM
        if archs and hasattr(transformers, archs[0]):
            auto_cls = getattr(transformers, archs[0])
        model = auto_cls.from_pretrained(
            args.model_path, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)

        parquet_files = glob.glob(os.path.join(args.dataset_path, "*.parquet"))
        texts = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            col = "text" if "text" in df.columns else df.columns[0]
            texts.extend([t for t in df[col].dropna().tolist() if len(str(t).strip()) > 32])
        selected = texts[: args.num_calib_samples]
        calib_inputs = [
            tokenizer(
                [t],
                padding="max_length",
                max_length=args.seq_len,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            for t in selected
        ]

        raw_configs, _ = run_e2e_kl_search(
            model=model,
            calib_inputs=calib_inputs,
            option_names=configs,
            option_costs=torch.tensor(cost_vec, dtype=torch.float64, device=device),
            target_budget=args.budget,
            steps=args.steps,
            lr=args.lr,
            prior_logits=prior_logits,
            cpu_offload=True,
            granularity=args.granularity,
        )
        best_configs, tensor_alloc = build_tensor_allocation_and_expanded_config(
            raw_configs, gran_meta
        )
        if args.granularity != "expert" or args.tensor_output_file:
            tensor_out = args.tensor_output_file or (
                os.path.splitext(args.output_file)[0] + "_tensor_allocation.json"
            )
            with open(tensor_out, "w", encoding="utf-8") as tf:
                json.dump(tensor_alloc, tf, indent=2)
            print(f"Saved Per-3D-Tensor allocation (GGUF & vLLM) to: {tensor_out}")
    else:
        max_comp_frac = args.max_compensated_fraction
        if args.mode == "int3_codesign" and max_comp_frac is None:
            max_comp_frac = 0.25

        if args.model_preset is not None and not (
            args.l2_file != "./model_statistics/Qwen/Qwen_expt_impact.json"
            and os.path.exists(args.l2_file)
        ):
            ext_freq = (
                load_expert_freq(args.freq_file)
                if (
                    args.freq_file != "./model_statistics/Qwen/Qwen_expt_freq.json"
                    and os.path.exists(args.freq_file)
                )
                else None
            )
            experts, configs, loss_mat, cost_vec, freq_vec = synthesize_problem_tensors_for_model(
                spec=spec,
                memory_usage=memory_usage,
                expert_freq=ext_freq,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                allowed_bits=allowed_bits,
                seed=args.seed,
            )
        else:
            with open(args.l2_file, "r", encoding="utf-8") as f:
                l2_results = json.load(f)
            expert_freq = load_expert_freq(args.freq_file)
            experts, configs, loss_mat, cost_vec, freq_vec = prepare_problem_tensors(
                l2_results=l2_results,
                memory_usage=memory_usage,
                expert_freq=expert_freq,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                allowed_bits=allowed_bits,
            )

        num_exp_per_layer = spec.num_experts if spec is not None else 64
        if args.compare_expert_freedom:
            compare_granularity_freedom(
                experts=experts,
                configs=configs,
                loss_mat=loss_mat,
                cost_vec=cost_vec,
                freq_vec=freq_vec,
                target_budget=args.budget,
                num_experts_per_layer=num_exp_per_layer,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
                verbose=True,
            )

        units, configs, loss_mat_g, cost_mat_g, _, gran_meta = aggregate_problem_to_granularity(
            experts=experts,
            configs=configs,
            loss_mat=loss_mat,
            cost_vec=cost_vec,
            freq_vec=freq_vec,
            granularity=args.granularity,
            num_experts_per_layer=num_exp_per_layer,
        )
        if args.granularity != "expert":
            print(
                f"[Granularity={args.granularity}] Aggregated {len(experts)} input units -> "
                f"{len(units)} 3D-Tensor decision units (2D cost matrix shape={cost_mat_g.shape})"
            )

        raw_configs, _ = rco_optimize_table(
            experts=units,
            configs=configs,
            loss_mat=loss_mat_g,
            cost_vec=cost_mat_g,
            target_budget=args.budget,
            steps=args.steps,
            lr=args.lr,
            gumbel_samples=args.gumbel_samples,
            use_slack=args.use_slack,
            interaction_strength=args.interaction_strength,
            max_compensated_fraction=max_comp_frac,
            seed=args.seed,
            verbose=True,
        )
        best_configs, tensor_alloc = build_tensor_allocation_and_expanded_config(
            raw_configs, gran_meta
        )
        if args.granularity != "expert" or args.tensor_output_file:
            tensor_out = args.tensor_output_file or (
                os.path.splitext(args.output_file)[0] + "_tensor_allocation.json"
            )
            with open(tensor_out, "w", encoding="utf-8") as tf:
                json.dump(tensor_alloc, tf, indent=2)
            print(f"Saved Per-3D-Tensor allocation (GGUF & vLLM) to: {tensor_out}")

    # When granularity is Per-3D-Tensor (`tensor`, `vllm_tensor`, `layer`), save `raw_configs`
    # (`L{l}_{gate,up,down}_exps`) directly to `output_file` so `BR_compress_auto.py` preserves
    # distinct `(bit, rank)` per 3D projection tensor.
    primary_out = raw_configs if args.granularity in ("tensor", "vllm_tensor", "layer") else best_configs
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(primary_out, f, indent=4)
    print(f"\nSaved RCO optimal configuration ({args.granularity}) to: {args.output_file}")


if __name__ == "__main__":
    main()
