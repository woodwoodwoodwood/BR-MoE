"""RCO_solver: Riemannian Constrained Optimization for BR-MoE.

Implements the method from:
"Model Compression with Exact Budget Constraints via Riemannian Manifolds"
(arXiv:2605.00649, Helcig & Alistarh, 2026) adapted for BR-MoE's joint
(bit-width, low-rank compensator rank, expert pruning) optimization space.
"""

from .manifold import (
    BudgetManifold,
    InequalityBudgetManifold,
    MultiBudgetManifold,
    budget_normal,
    dp_solve_mckp,
    expected_cost,
    project_gradient,
    retraction,
    topk_solve_binary,
    vector_transport,
)
from .e2e_ste import (
    RCOInterpolatedLinear,
    reconstruct_brmoe_weight,
    run_e2e_kl_search,
    wrap_moe_model_for_rco,
)
from .rco_search import (
    load_expert_freq,
    prepare_problem_tensors,
    rco_optimize_table,
)

__all__ = [
    "BudgetManifold",
    "InequalityBudgetManifold",
    "MultiBudgetManifold",
    "RCOInterpolatedLinear",
    "budget_normal",
    "dp_solve_mckp",
    "expected_cost",
    "load_expert_freq",
    "prepare_problem_tensors",
    "project_gradient",
    "rco_optimize_table",
    "reconstruct_brmoe_weight",
    "retraction",
    "run_e2e_kl_search",
    "topk_solve_binary",
    "vector_transport",
    "wrap_moe_model_for_rco",
]
