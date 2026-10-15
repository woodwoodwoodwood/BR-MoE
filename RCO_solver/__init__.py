"""RCO Solver package for BR-MoE.

Riemannian Constrained Optimization for mixed-precision quantization,
low-rank compensator rank allocation, and expert pruning under exact
memory and latency/sparsity constraints.

Reference:
    Michael Helcig, Dan Alistarh (2026).
    "Model Compression with Exact Budget Constraints via Riemannian Manifolds"
    https://arxiv.org/abs/2605.00649
"""

from RCO_solver.manifold import (
    BudgetManifold,
    InequalityBudgetManifold,
    MultiBudgetManifold,
    NumpyAdam,
    budget_normal,
    dp_solve_mckp,
    expected_cost,
    project_gradient,
    retraction,
     topk_solve_binary,
    vector_transport,
)
from RCO_solver.model_adapters import (
    BUILTIN_MOE_PRESETS,
    MoEModelSpec,
    compute_budget_from_avg_bits,
    compute_expert_memory_table,
    parse_moe_spec_from_config_dict,
    resolve_moe_model_spec,
    synthesize_problem_tensors_for_model,
)
from RCO_solver.rco_search import (
    load_expert_freq,
    prepare_problem_tensors,
    rco_optimize_table,
)

__all__ = [
    "BUILTIN_MOE_PRESETS",
    "BudgetManifold",
    "InequalityBudgetManifold",
    "MoEModelSpec",
    "MultiBudgetManifold",
    "NumpyAdam",
    "budget_normal",
    "compute_budget_from_avg_bits",
    "compute_expert_memory_table",
    "dp_solve_mckp",
    "expected_cost",
    "load_expert_freq",
    "parse_moe_spec_from_config_dict",
    "prepare_problem_tensors",
    "project_gradient",
    "rco_optimize_table",
    "resolve_moe_model_spec",
    "retraction",
    "synthesize_problem_tensors_for_model",
    "topk_solve_binary",
    "vector_transport",
]
