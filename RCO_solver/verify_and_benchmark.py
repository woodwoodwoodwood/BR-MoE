#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verification & Benchmark Suite for BR-MoE RCO_solver.

Tests:
  1. Riemannian Manifold Geometry Axioms:
     - Normal vector closed form vs numerical Jacobian (Proposition 2)
     - Monotone binary-search retraction & multi-constraint Newton retraction (< 1e-8)
     - Tangent projection orthogonality (<g_proj, n> ~ 0, Proposition 5)
     - Vector transport orthogonality (Appendix A.5)
     - Inequality slack manifold (Appendix A.7) & Multi-budget manifold (Appendix A.8)
  2. Streaming STE Backward Identity (`streaming_ste_forward_backward_np` & `_StreamingSTELinearFn`):
     - Exact numerical equivalence of dL/dp_k = <G_W, W^{(k)}>_F against naive
       full-materialization weight interpolation and central finite differences
  3. Real BR-MoE Model Statistics Benchmarks (Qwen1.5-MoE, DeepSeek-MoE, Mixtral-8x7B):
     - Original Buggy ILP vs Lagrangian Penalty vs RCO Single-Budget vs RCO Gumbel-STE
       vs RCO Multi-Budget (`brmoe_int3_vllm` hardware co-design).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Tuple
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
    budget_normal,
    expected_cost,
    project_gradient,
    retraction,
    vector_transport,
)
from RCO_solver.e2e_ste import streaming_ste_forward_backward_np
from RCO_solver.rco_search import (
     load_expert_freq,
    prepare_problem_tensors,
    rco_optimize_table,
)


def test_manifold_axioms():
    print("=" * 80)
    print("1. Testing Riemannian Manifold Axioms (arXiv:2605.00649 Props 2, 3, 4, 5 & App A)")
    print("=" * 80)
    rng = np.random.default_rng(0)
    N, K = 32, 13
    costs = np.linspace(4.0, 20.0, K, dtype=np.float64)
    target = float(N * 11.3)

    alpha = rng.standard_normal((N, K))

    # 1) Proposition 2: Closed-form normal vs central finite-difference gradient
    closed_normal = budget_normal(alpha, costs)
    fd_normal = np.zeros_like(alpha)
    eps = 1e-6
    for i in range(4):  # spot-check first 4 rows (4 * 13 = 52 partials)
        for k in range(K):
            orig_val = alpha[i, k]
            alpha[i, k] = orig_val + eps
            c_plus = expected_cost(alpha[i : i + 1], costs)
            alpha[i, k] = orig_val - eps
            c_minus = expected_cost(alpha[i : i + 1], costs)
            alpha[i, k] = orig_val
            fd_normal[i, k] = (c_plus - c_minus) / (2.0 * eps)
    normal_diff = float(np.max(np.abs(closed_normal[:4] - fd_normal[:4])))
    assert normal_diff < 1e-8, f"Normal mismatch: {normal_diff}"

    # 2) Proposition 3: Monotone retraction precision
    alpha_ret = alpha.copy()
    retraction(alpha_ret, costs, target, tol=1e-11)
    ret_err = abs(expected_cost(alpha_ret, costs) - target)
    assert ret_err < 1e-8, f"Retraction error too large: {ret_err}"

    # 3) Proposition 5: Tangent projection orthogonality
    g = rng.standard_normal((N, K))
    n_new = budget_normal(alpha_ret, costs)
    g_tan = project_gradient(g, n_new)
    ortho_err = abs(float(np.sum(g_tan * n_new))) / (float(np.linalg.norm(n_new)) + 1e-12)
    assert ortho_err < 1e-14, f"Tangent orthogonality error: {ortho_err}"

    # 4) Appendix A.5: Vector transport orthogonality
    m_trans = vector_transport(g, n_new)
    trans_err = abs(float(np.sum(m_trans * n_new))) / (float(np.linalg.norm(n_new)) + 1e-12)
    assert trans_err < 1e-14, f"Vector transport orthogonality error: {trans_err}"

    # 5) Appendix A.7: Inequality slack manifold
    ineq_man = InequalityBudgetManifold(costs=costs, target_budget=target, tol=1e-9)
    alpha_ineq = rng.standard_normal((N, K))
    ineq_man.initialize(alpha_ineq)
    ineq_viol = ineq_man.current_violation(alpha_ineq)
    assert ineq_viol < 1e-8, f"Inequality manifold violation: {ineq_viol}"

    # 6) Appendix A.8: Multi-constraint manifold (Memory + Compensator sparsity)
    comp_ind = np.array(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    multi_man = MultiBudgetManifold(
        cost_list=[costs, comp_ind],
        target_budgets=[target, float(N * 0.25)],
        tol=1e-9,
    )
    alpha_m = rng.standard_normal((N, K))
    multi_man.initialize(alpha_m)
    multi_viol = multi_man.current_violation(alpha_m)
    assert multi_viol < 1e-7, f"Multi-budget retraction violation: {multi_viol}"

    print(f"  [PASS] Prop. 2 Closed-form normal vs Finite-Diff max err : {normal_diff:.2e}")
    print(f"  [PASS] Prop. 3 Monotone retraction |C(alpha) - B|        : {ret_err:.2e}")
    print(f"  [PASS] Prop. 5 Tangent projection |<g_tan, n>|/||n||     : {ortho_err:.2e}")
    print(f"  [PASS] App. A.5 Vector transport |<m_trans, n>|/||n||    : {trans_err:.2e}")
    print(f"  [PASS] App. A.7 Inequality slack manifold violation      : {ineq_viol:.2e}")
    print(f"  [PASS] App. A.8 Multi-Budget Newton retraction max err   : {multi_viol:.2e}\n")


def test_streaming_ste_identity():
    print("=" * 80)
    print("2. Testing Streaming STE Identity: dL/dp_k = <(dL/dy)^T x, W_hat^{(k)}>_F")
    print("=" * 80)
    rng = np.random.default_rng(7)
    M_tokens, D_in, D_out, K = 8, 32, 24, 5
    W_base = rng.standard_normal((D_out, D_in))
    cand_weights = [W_base + 0.1 * k * rng.standard_normal((D_out, D_in)) for k in range(K)]
    x = rng.standard_normal((M_tokens, D_in))
    logits = rng.standard_normal(K)
    k_star = 2
    tau = 0.7

    # Forward value at k_star, loss L = 0.5 * ||y - target||_F^2
    target_y = rng.standard_normal((M_tokens, D_out))
    y_stream, _, _ = streaming_ste_forward_backward_np(
        x, np.zeros((M_tokens, D_out)), logits, k_star, cand_weights, tau=tau
    )
    grad_out = y_stream - target_y
    y_stream, gx_stream, galpha_stream = streaming_ste_forward_backward_np(
        x, grad_out, logits, k_star, cand_weights, tau=tau
    )

    # Naive full-materialization weight interpolation: W(p_ste) = sum_k p_ste[k] W^{(k)}
    shifted = (logits - np.max(logits)) / tau
    p_soft = np.exp(shifted) / np.sum(np.exp(shifted))
    z_oh = np.eye(K)[k_star]
    p_ste = (z_oh - p_soft) + p_soft  # forward value = z_oh
    W_interp = sum(p_ste[k] * cand_weights[k] for k in range(K))
    y_naive = x @ W_interp.T
    gx_naive = grad_out @ W_interp

    # Naive per-option forward activation computation: <grad_out, x @ W^{(k)}.T>_F
    gp_naive = np.array(
        [float(np.sum(grad_out * (x @ cand_weights[k].T))) for k in range(K)],
        dtype=np.float64,
    )
    galpha_naive = (p_soft / tau) * (gp_naive - float(np.sum(p_soft * gp_naive)))

    # Central finite difference on logits for the straight-through surrogate
    galpha_fd = np.zeros(K, dtype=np.float64)
    eps = 1e-5
    for k in range(K):
        l_plus = logits.copy()
        l_plus[k] += eps
        p_plus = np.exp((l_plus - np.max(l_plus)) / tau)
        p_plus /= np.sum(p_plus)
        W_plus = W_interp + sum((p_plus[j] - p_soft[j]) * cand_weights[j] for j in range(K))
        loss_plus = 0.5 * float(np.sum((x @ W_plus.T - target_y) ** 2))

        l_minus = logits.copy()
        l_minus[k] -= eps
        p_minus = np.exp((l_minus - np.max(l_minus)) / tau)
        p_minus /= np.sum(p_minus)
        W_minus = W_interp + sum((p_minus[j] - p_soft[j]) * cand_weights[j] for j in range(K))
        loss_minus = 0.5 * float(np.sum((x @ W_minus.T - target_y) ** 2))

        galpha_fd[k] = (loss_plus - loss_minus) / (2.0 * eps)

    fwd_diff = float(np.max(np.abs(y_stream - y_naive)))
    gx_diff = float(np.max(np.abs(gx_stream - gx_naive)))
    galpha_exact_diff = float(np.max(np.abs(galpha_stream - galpha_naive)))
    galpha_fd_diff = float(np.max(np.abs(galpha_stream - galpha_fd)))

    assert fwd_diff < 1e-12, f"fwd_diff={fwd_diff}"
    assert gx_diff < 1e-12, f"gx_diff={gx_diff}"
    assert galpha_exact_diff < 1e-12, f"galpha_exact_diff={galpha_exact_diff}"
    assert galpha_fd_diff < 1e-7, f"galpha_fd_diff={galpha_fd_diff}"

    print(f"  [PASS] Streaming STE vs Naive Forward max diff           : {fwd_diff:.2e}")
    print(f"  [PASS] Streaming STE vs Naive dL/dx max diff             : {gx_diff:.2e}")
    print(f"  [PASS] Streaming STE vs Naive dL/d(alpha) exact diff     : {galpha_exact_diff:.2e}")
    print(f"  [PASS] Streaming STE vs Finite-Diff dL/d(alpha) max diff : {galpha_fd_diff:.2e}\n")


def _lagrangian_penalty_baseline(
    loss_mat: np.ndarray,
    cost_vec: np.ndarray,
    target_budget: float,
    lam: float = 1e-2,
    steps: int = 200,
    lr: float = 0.1,
) -> Tuple[float, float, float]:
    """Standard Euclidean Adam with quadratic budget penalty (as compared in RCO Fig. 2)."""
    N, K = loss_mat.shape
    alpha = np.zeros((N, K), dtype=np.float64)
    opt = NumpyAdam([alpha], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p = _softmax_np(alpha, axis=-1)
        c_exp = float(np.sum(p * cost_vec[None, :]))
        # d/dp of (obj + lam * (c_exp - B)^2)
        grad_p = loss_mat + 2.0 * lam * (c_exp - target_budget) * cost_vec[None, :]
        exp_gp = np.sum(p * grad_p, axis=-1, keepdims=True)
        grad_alpha = p * (grad_p - exp_gp)
        opt.set_grad(alpha, grad_alpha)
        opt.step()

    p = _softmax_np(alpha, axis=-1)
    cont_viol = abs(float(np.sum(p * cost_vec[None, :])) - target_budget)
    idx = np.argmax(p, axis=-1)
    disc_mem = float(np.sum(cost_vec[idx]))
    disc_loss = float(np.sum(loss_mat[np.arange(N), idx]))
    return disc_loss, disc_mem, cont_viol


def benchmark_real_brmoe_models():
    print("=" * 80)
    print("3. Benchmarking on Real BR-MoE Statistics (Qwen1.5-MoE, DeepSeek-MoE, Mixtral)")
    print("=" * 80)

    datasets = [
        (
            "Qwen1.5-MoE-A2.7B",
            os.path.join(_ROOT, "model_statistics/Qwen/Qwen_expt_impact.json"),
            os.path.join(_ROOT, "model_statistics/Qwen/Qwen_expt_freq.json"),
            os.path.join(_ROOT, "model_statistics/Qwen/Qwen_memory_usage.json"),
            20728.5,
        ),
        (
            "DeepSeek-MoE-16B",
            os.path.join(_ROOT, "model_statistics/DeepSeek/DeepSeek_expt_impact.json"),
            os.path.join(_ROOT, "model_statistics/DeepSeek/DeepSeek_expt_freq.json"),
            os.path.join(_ROOT, "model_statistics/DeepSeek/DeepSeek_expt_memory.json"),
            5448.5,
        ),
        (
            "Mixtral-8x7B",
            os.path.join(_ROOT, "model_statistics/Mixtral/Mixtral_expt_impact.json"),
            os.path.join(_ROOT, "model_statistics/Mixtral/Mixtral_expt_freq.json"),
            os.path.join(_ROOT, "model_statistics/Mixtral/Mixtral_memory_usage.json"),
            19008.0,
        ),
    ]

    summary_rows = []
    for model_name, l2_path, freq_path, mem_path, budget in datasets:
        with open(l2_path, "r", encoding="utf-8") as f:
            l2_data = json.load(f)
        with open(mem_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        freq_data = load_expert_freq(freq_path)

        experts, configs, loss_mat, cost_vec, _ = prepare_problem_tensors(
            l2_data, mem_data, freq_data, min_rank=0, max_rank=256
        )
        N = len(experts)

        # (A) Original Buggy ILP behavior (assigned lowest memory config to all experts)
        min_cost_k = int(np.argmin(cost_vec))
        buggy_loss = float(np.sum(loss_mat[:, min_cost_k]))
        buggy_mem = float(N * cost_vec[min_cost_k])

        # (B) Lagrangian Penalty Baseline (Euclidean Adam + quadratic penalty)
        lag_loss, lag_mem, lag_viol = _lagrangian_penalty_baseline(
            loss_mat, cost_vec, budget, lam=1e-2, steps=200
        )

        # (C) RCO Single-Budget Manifold
        _, rco_info = rco_optimize_table(
            experts,
            configs,
            loss_mat,
            cost_vec,
            target_budget=budget,
            steps=200,
            lr=0.15,
            verbose=False,
        )

        # (D) RCO Gumbel-STE + DP (Algorithm 1)
        _, rco_ste_info = rco_optimize_table(
            experts,
            configs,
            loss_mat,
            cost_vec,
            target_budget=budget,
            steps=30,
            lr=0.15,
            gumbel_samples=1,
            dp_resolution=1000,
            verbose=False,
        )

        # (E) RCO Multi-Budget Co-Design (Budget + <= 20% experts with rank > 0 for vLLM INT3)
        _, rco_co_info = rco_optimize_table(
            experts,
            configs,
            loss_mat,
            cost_vec,
            target_budget=budget,
            steps=200,
            lr=0.15,
            max_compensated_fraction=0.20,
            verbose=False,
        )

        summary_rows.append(
            [
                model_name,
                "Original Buggy ILP",
                f"{buggy_loss:.2f}",
                f"{buggy_mem:.1f} / {budget:.1f}",
                "100.0%",
                "N/A",
            ]
        )
        summary_rows.append(
            [
                model_name,
                "Lagrangian Penalty (Euclidean)",
                f"{lag_loss:.2f}",
                f"{lag_mem:.1f} / {budget:.1f}",
                "N/A",
                f"{lag_viol:.2e}",
            ]
        )
        summary_rows.append(
            [
                model_name,
                "RCO Single-Budget Manifold",
                f"{rco_info['total_loss']:.2f}",
                f"{rco_info['total_memory_mb']:.1f} / {budget:.1f}",
                f"{rco_info['compensated_fraction']*100:.1f}%",
                f"{rco_info['final_manifold_violation']:.2e}",
            ]
        )
        summary_rows.append(
            [
                model_name,
                "RCO Gumbel-STE + DP (Alg. 1)",
                f"{rco_ste_info['total_loss']:.2f}",
                f"{rco_ste_info['total_memory_mb']:.1f} / {budget:.1f}",
                f"{rco_ste_info['compensated_fraction']*100:.1f}%",
                f"{rco_ste_info['final_manifold_violation']:.2e}",
            ]
        )
        summary_rows.append(
            [
                model_name,
                "RCO Multi-Budget (<=20% rank>0)",
                f"{rco_co_info['total_loss']:.2f}",
                f"{rco_co_info['total_memory_mb']:.1f} / {budget:.1f}",
                f"{rco_co_info['compensated_fraction']*100:.1f}%",
                f"{rco_co_info['final_manifold_violation']:.2e}",
            ]
        )

    headers = [
        "Model",
        "Method",
        "Weighted L2 Loss (↓)",
        "Memory Used / Budget (MB)",
        "Rank>0 Ratio",
        "|C(α) - B|",
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    test_manifold_axioms()
    test_streaming_ste_identity()
    benchmark_real_brmoe_models()
