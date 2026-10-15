"""Riemannian Budget Manifold primitives and discrete DP solvers for BR-MoE.

Reference:
    Michael Helcig, Dan Alistarh (2026).
    "Model Compression with Exact Budget Constraints via Riemannian Manifolds"
    https://arxiv.org/abs/2605.00649

Core geometry in logit space alpha in R^{N x K} with p_i = softmax(alpha_i):
  1. Expected cost:
       C(alpha) = sum_{i=1}^N w_i <p_i, c_i> = B
  2. Closed-form Euclidean normal (Proposition 2):
       (nabla C(alpha))_{ik} = w_i * p_{ik} * (c_{ik} - E_{p_i}[c_i])
  3. Tangent projection (Proposition 5 & 6):
       P_T(g) = g - (<g, n> / ||n||^2) * n
     Eliminates the constraint-normal component of any STE gradient bias.
  4. Monotone retraction along cost vector (Proposition 3 & 4):
       d/dt C(alpha + t * c) = sum_i w_i Var_{p_i(t)}[c_i] > 0
     Unique root t* found to machine precision via bisection / Newton's method.
  5. Vector transport by projection (Appendix A.5):
       T_{alpha -> alpha'}(m) = m - (<m, n'> / ||n'||^2) * n'
     Keeps Adam's first moment strictly inside the rotated tangent space.
  6. Extensions:
     - Inequality constraint C(alpha) + s^2 = B via slack variable s (Appendix A.7)
     - Multiple simultaneous constraints C_j(alpha) = b_j via q x q Gram projection
       and covariance-Jacobian Newton retraction (Section 2 & Appendix A.8)

Supports both PyTorch (`torch.Tensor`) on CPU/GPU and pure NumPy (`np.ndarray`)
when running offline optimization on machines without PyTorch installed.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


ArrayLike = Any  # Union[torch.Tensor, np.ndarray]


def _is_torch(x: Any) -> bool:
    return HAS_TORCH and isinstance(x, torch.Tensor)


def _to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax(x: ArrayLike, dim: int = -1) -> ArrayLike:
    if _is_torch(x):
        return torch.softmax(x, dim=dim)
    return _softmax_np(np.asarray(x, dtype=np.float64), axis=dim)


class NumpyAdam:
    """Lightweight Adam optimizer matching torch.optim.Adam for NumPy arrays."""

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = params
        self.param_groups = [{"lr": lr, "betas": betas, "eps": eps}]
        self.state: dict = {
            id(p): {
                "step": 0,
                "exp_avg": np.zeros_like(p, dtype=np.float64),
                "exp_avg_sq": np.zeros_like(p, dtype=np.float64),
                "grad": np.zeros_like(p, dtype=np.float64),
            }
            for p in params
        }

    def zero_grad(self) -> None:
        for p in self.params:
            self.state[id(p)]["grad"].fill(0.0)

    def set_grad(self, p: np.ndarray, g: np.ndarray) -> None:
        self.state[id(p)]["grad"][...] = g

    def get_grad(self, p: np.ndarray) -> np.ndarray:
        return self.state[id(p)]["grad"]

    def step(self) -> None:
        pg = self.param_groups[0]
        lr = pg["lr"]
        beta1, beta2 = pg["betas"]
        eps = pg["eps"]
        for p in self.params:
            st = self.state[id(p)]
            st["step"] += 1
            t = st["step"]
            g = st["grad"]
            m = st["exp_avg"]
            v = st["exp_avg_sq"]
            m[...] = beta1 * m + (1.0 - beta1) * g
            v[...] = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            p[...] = p - lr * m_hat / (np.sqrt(v_hat) + eps)


def _prepare_costs_and_weights_np(
    alpha: np.ndarray,
    costs: Any,
    weights: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    c = _to_numpy(costs).astype(np.float64)
    if c.ndim == 1:
        c = c[None, :]  # [1, K]
    if weights is None:
        w = np.ones((alpha.shape[0], 1), dtype=np.float64)
    else:
        w = _to_numpy(weights).astype(np.float64).reshape(-1, 1)
    return c, w


def _prepare_costs_and_weights(
    alpha: ArrayLike,
    costs: ArrayLike,
    weights: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Broadcast costs to [1, K] or [N, K] and weights to [N, 1]."""
    if _is_torch(alpha):
        c = costs.to(device=alpha.device, dtype=alpha.dtype) if _is_torch(costs) else torch.tensor(costs, device=alpha.device, dtype=alpha.dtype)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if weights is None:
            w = torch.ones((alpha.size(0), 1), device=alpha.device, dtype=alpha.dtype)
        else:
            wt = weights.to(device=alpha.device, dtype=alpha.dtype) if _is_torch(weights) else torch.tensor(weights, device=alpha.device, dtype=alpha.dtype)
            w = wt.view(-1, 1)
        return c, w
    return _prepare_costs_and_weights_np(alpha, costs, weights)


def expected_cost(
    alpha: ArrayLike,
    costs: ArrayLike,
    weights: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Compute expected cost C(alpha) = sum_i w_i * sum_k p_{ik} c_{ik}."""
    if _is_torch(alpha):
        c, w = _prepare_costs_and_weights(alpha, costs, weights)
        probs = torch.softmax(alpha, dim=-1)
        group_expected = (probs * c).sum(dim=-1, keepdim=True)
        return (w * group_expected).sum()
    c, w = _prepare_costs_and_weights_np(alpha, costs, weights)
    probs = _softmax_np(alpha, axis=-1)
    group_expected = np.sum(probs * c, axis=-1, keepdims=True)
    return float(np.sum(w * group_expected))


def budget_normal(
    alpha: ArrayLike,
    costs: ArrayLike,
    weights: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Closed-form normal vector nabla C(alpha) (Proposition 2).

    (nabla C)_{ik} = w_i * p_{ik} * (c_{ik} - E_{p_i}[c_i])
    """
    if _is_torch(alpha):
        c, w = _prepare_costs_and_weights(alpha, costs, weights)
        probs = torch.softmax(alpha, dim=-1)
        expected_c = (probs * c).sum(dim=-1, keepdim=True)
        return w * probs * (c - expected_c)
    c, w = _prepare_costs_and_weights_np(alpha, costs, weights)
    probs = _softmax_np(alpha, axis=-1)
    expected_c = np.sum(probs * c, axis=-1, keepdims=True)
    return w * probs * (c - expected_c)


def project_onto_tangent(vec: ArrayLike, normal: ArrayLike) -> ArrayLike:
    """Return the orthogonal projection of `vec` onto the tangent hyperplane n^perp:
        P_T(vec) = vec - (<vec, n> / ||n||^2) * n
    """
    if _is_torch(vec):
        nf = normal.flatten()
        vf = vec.flatten()
        denom = (nf @ nf).item()
        if denom < 1e-20:
            return vec.clone()
        coeff = (vf @ nf).item() / denom
        return vec - coeff * normal
    nf = np.asarray(normal, dtype=np.float64).ravel()
    vf = np.asarray(vec, dtype=np.float64).ravel()
    denom = float(np.dot(nf, nf))
    if denom < 1e-20:
        return np.array(vec, dtype=np.float64, copy=True)
    coeff = float(np.dot(vf, nf)) / denom
    return np.asarray(vec, dtype=np.float64) - coeff * np.asarray(normal, dtype=np.float64)


def project_gradient(
    alpha_or_grad: ArrayLike,
    costs_or_normal: ArrayLike,
    weights: Optional[ArrayLike] = None,
    grad_buf: Optional[np.ndarray] = None,
) -> Any:
    """Project gradient onto the tangent space T_{alpha} M (Proposition 5 & 6).

    Supports two calling conventions:
      1. Functional: `project_gradient(g, n)` where `costs_or_normal` has the same
         2D shape as `alpha_or_grad` -> returns projected tensor `P_T(g)`.
      2. In-place on `alpha`: `project_gradient(alpha, costs, weights)` -> modifies
         `alpha.grad` (or `grad_buf`) in-place and returns `(coeff, raw_norm, proj_norm)`.
    """
    # Calling convention 1: (g, n) with matching 2D shape
    if (
        hasattr(alpha_or_grad, "shape")
        and hasattr(costs_or_normal, "shape")
        and alpha_or_grad.shape == costs_or_normal.shape
        and len(alpha_or_grad.shape) >= 2
        and weights is None
        and grad_buf is None
        and (not _is_torch(alpha_or_grad) or alpha_or_grad.grad is None)
    ):
        return project_onto_tangent(alpha_or_grad, costs_or_normal)

    # Calling convention 2: in-place projection of alpha's gradient
    alpha = alpha_or_grad
    costs = costs_or_normal
    if _is_torch(alpha):
        with torch.no_grad():
            if alpha.grad is None:
                return 0.0, 0.0, 0.0
            n = budget_normal(alpha, costs, weights)
            nf = n.flatten()
            gf = alpha.grad.flatten()
            raw_norm = gf.norm().item()
            denom = (nf @ nf).item()
            if denom < 1e-20:
                return 0.0, raw_norm, raw_norm
            coeff = (gf @ nf).item() / denom
            alpha.grad.sub_(coeff * n)
            proj_norm = alpha.grad.norm().item()
            return coeff, raw_norm, proj_norm

    if grad_buf is None:
        return 0.0, 0.0, 0.0
    n = budget_normal(alpha, costs, weights)
    nf = n.ravel()
    gf = grad_buf.ravel()
    raw_norm = float(np.linalg.norm(gf))
    denom = float(np.dot(nf, nf))
    if denom < 1e-20:
        return 0.0, raw_norm, raw_norm
    coeff = float(np.dot(gf, nf)) / denom
    grad_buf -= coeff * n
    proj_norm = float(np.linalg.norm(grad_buf))
    return coeff, raw_norm, proj_norm


def retraction(
    alpha: ArrayLike,
    costs: ArrayLike,
    target_budget: float,
    weights: Optional[ArrayLike] = None,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> ArrayLike:
    """Monotonic binary-search retraction onto M = {C(alpha) = B} (Proposition 3).

    Shifts alpha <- alpha + t* * c where C(alpha + t* * c) = target_budget.
    Modifies alpha in-place AND returns alpha.
    """
    if _is_torch(alpha):
        with torch.no_grad():
            c, w = _prepare_costs_and_weights(alpha, costs, weights)

            def eval_cost(shift: float) -> float:
                p = torch.softmax(alpha + shift * c, dim=-1)
                return (w * (p * c).sum(dim=-1, keepdim=True)).sum().item()

            min_possible = (w * c.min(dim=-1, keepdim=True).values).sum().item()
            max_possible = (w * c.max(dim=-1, keepdim=True).values).sum().item()
            if target_budget <= min_possible + tol:
                alpha.add_(-50.0 * c)
                return alpha
            if target_budget >= max_possible - tol:
                alpha.add_(50.0 * c)
                return alpha

            cur = eval_cost(0.0)
            if abs(cur - target_budget) <= tol:
                return alpha

            lo, hi = -1.0, 1.0
            for _ in range(40):
                c_lo = eval_cost(lo)
                c_hi = eval_cost(hi)
                if c_lo <= target_budget <= c_hi:
                    break
                if c_lo > target_budget:
                    hi = lo
                    lo *= 2.0
                elif c_hi < target_budget:
                    lo = hi
                    hi *= 2.0

            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                val = eval_cost(mid)
                if abs(val - target_budget) <= tol or (hi - lo) < 1e-13:
                    lo = hi = mid
                    break
                if val > target_budget:
                    hi = mid
                else:
                    lo = mid

            t_star = 0.5 * (lo + hi)
            alpha.add_(t_star * c)
            return alpha

    # NumPy path
    c, w = _prepare_costs_and_weights_np(alpha, costs, weights)

    def eval_cost_np(shift: float) -> float:
        p = _softmax_np(alpha + shift * c, axis=-1)
        return float(np.sum(w * np.sum(p * c, axis=-1, keepdims=True)))

    min_possible = float(np.sum(w * np.min(c, axis=-1, keepdims=True)))
    max_possible = float(np.sum(w * np.max(c, axis=-1, keepdims=True)))
    if target_budget <= min_possible + tol:
        alpha += -50.0 * c
        return alpha
    if target_budget >= max_possible - tol:
        alpha += 50.0 * c
        return alpha

    cur = eval_cost_np(0.0)
    if abs(cur - target_budget) <= tol:
        return alpha

    lo, hi = -1.0, 1.0
    for _ in range(40):
        c_lo = eval_cost_np(lo)
        c_hi = eval_cost_np(hi)
        if c_lo <= target_budget <= c_hi:
            break
        if c_lo > target_budget:
            hi = lo
            lo *= 2.0
        elif c_hi < target_budget:
            lo = hi
            hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = eval_cost_np(mid)
        if abs(val - target_budget) <= tol or (hi - lo) < 1e-13:
            lo = hi = mid
            break
        if val > target_budget:
            hi = mid
        else:
            lo = mid

    t_star = 0.5 * (lo + hi)
    alpha += t_star * c
    return alpha


def vector_transport(
    optimizer_or_vec: Any,
    alpha_or_normal: ArrayLike,
    costs: Optional[ArrayLike] = None,
    weights: Optional[ArrayLike] = None,
) -> Any:
    """Vector transport by projection onto the new tangent space (Appendix A.5).

    Supports two calling conventions:
      1. Functional: `vector_transport(m, n_new)` -> returns projected vector `P_T(m)`.
      2. Optimizer state update: `vector_transport(optimizer, alpha, costs, weights)` ->
         projects Adam's `exp_avg` in-place.
    """
    if costs is None:
        return project_onto_tangent(optimizer_or_vec, alpha_or_normal)

    optimizer = optimizer_or_vec
    alpha = alpha_or_normal
    n = budget_normal(alpha, costs, weights)

    if _is_torch(alpha):
        with torch.no_grad():
            nf = n.flatten()
            nsq = (nf @ nf).item()
            if nsq < 1e-20:
                return None
            state = optimizer.state.get(alpha)
            if state and "exp_avg" in state:
                m = state["exp_avg"]
                coeff = (m.flatten() @ nf).item() / nsq
                m.sub_(coeff * n)
            return None

    nf = n.ravel()
    nsq = float(np.dot(nf, nf))
    if nsq < 1e-20:
        return None
    state = optimizer.state.get(id(alpha))
    if state and "exp_avg" in state:
        m = state["exp_avg"]
        coeff = float(np.dot(m.ravel(), nf)) / nsq
        m -= coeff * n
    return None


class BudgetManifold:
    """Single equality constraint manifold M = {alpha : C(alpha) = B}."""

    def __init__(
        self,
        costs: ArrayLike,
        target_budget: float,
        weights: Optional[ArrayLike] = None,
        tol: float = 1e-8,
    ):
        self.costs = costs
        self.target_budget = float(target_budget)
        self.weights = weights
        self.tol = tol

    def current_cost(self, alpha: ArrayLike) -> float:
        if _is_torch(alpha):
            with torch.no_grad():
                return float(expected_cost(alpha, self.costs, self.weights).item())
        return float(expected_cost(alpha, self.costs, self.weights))

    def current_violation(self, alpha: ArrayLike) -> float:
        return abs(self.current_cost(alpha) - self.target_budget)

    def initialize(self, alpha: ArrayLike) -> float:
        """Project initial logits alpha_0 onto the budget manifold M."""
        retraction(
            alpha,
            self.costs,
            self.target_budget,
            weights=self.weights,
            tol=self.tol,
        )
        return self.current_cost(alpha)

    def step(
        self,
        alpha: ArrayLike,
        optimizer: Any,
    ) -> dict:
        """Perform one complete RCO step wrapping Adam:
        1. Tangent projection of alpha's gradient
        2. Adam update
        3. Binary-search retraction onto M
        4. Vector transport of Adam's first moment exp_avg
        """
        grad_buf = None if _is_torch(alpha) else optimizer.get_grad(alpha)
        coeff, raw_norm, proj_norm = project_gradient(
            alpha, self.costs, self.weights, grad_buf=grad_buf
        )
        optimizer.step()
        cost_before = self.current_cost(alpha)
        retraction(
            alpha,
            self.costs,
            self.target_budget,
            weights=self.weights,
            tol=self.tol,
        )
        cost_after = self.current_cost(alpha)
        vector_transport(optimizer, alpha, self.costs, self.weights)
        return {
            "proj_coeff": coeff,
            "raw_grad_norm": raw_norm,
            "proj_grad_norm": proj_norm,
            "cost_before_retract": cost_before,
            "cost_after_retract": cost_after,
            "violation": abs(cost_after - self.target_budget),
        }


class InequalityBudgetManifold:
    """Inequality constraint C(alpha) <= B via slack variable s (Appendix A.7).

    Defines augmented manifold hat{M} = {(alpha, s) : C(alpha) + s^2 = B}.
    """

    def __init__(
        self,
        costs: ArrayLike,
        target_budget: float,
        weights: Optional[ArrayLike] = None,
        tol: float = 1e-8,
    ):
        self.costs = costs
        self.target_budget = float(target_budget)
        self.weights = weights
        self.tol = tol
        self.s = 0.0
        self.m_s = 0.0
        self.v_s = 0.0
        self.step_count = 0

    def current_cost(self, alpha: ArrayLike) -> float:
        if _is_torch(alpha):
            with torch.no_grad():
                return float(expected_cost(alpha, self.costs, self.weights).item())
        return float(expected_cost(alpha, self.costs, self.weights))

    def current_violation(self, alpha: ArrayLike) -> float:
        return abs((self.current_cost(alpha) + self.s ** 2) - self.target_budget)

    def initialize(self, alpha: ArrayLike) -> float:
        cur = self.current_cost(alpha)
        if cur > self.target_budget:
            retraction(
                alpha, self.costs, self.target_budget, self.weights, tol=self.tol
            )
            cur = self.current_cost(alpha)
            self.s = 0.0
        else:
            self.s = float(np.sqrt(max(0.0, self.target_budget - cur)))
        return cur

    def step(
        self,
        alpha: ArrayLike,
        optimizer: Any,
    ) -> dict:
        """Augmented tangent projection, step, and retraction (Appendix A.7)."""
        self.step_count += 1
        n_alpha = budget_normal(alpha, self.costs, self.weights)

        if _is_torch(alpha):
            with torch.no_grad():
                nf = n_alpha.flatten()
                n_norm_sq = (nf @ nf).item() + 4.0 * (self.s ** 2)
                raw_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
                coeff = 0.0
                grad_s = 0.0
                if alpha.grad is not None and n_norm_sq > 1e-20:
                    dot = (alpha.grad.flatten() @ nf).item()
                    coeff = dot / n_norm_sq
                    alpha.grad.sub_(coeff * n_alpha)
                    grad_s = -coeff * (2.0 * self.s)
                proj_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
                optimizer.step()
        else:
            grad_buf = optimizer.get_grad(alpha)
            nf = n_alpha.ravel()
            n_norm_sq = float(np.dot(nf, nf)) + 4.0 * (self.s ** 2)
            raw_norm = float(np.linalg.norm(grad_buf))
            coeff = 0.0
            grad_s = 0.0
            if n_norm_sq > 1e-20:
                dot = float(np.dot(grad_buf.ravel(), nf))
                coeff = dot / n_norm_sq
                grad_buf -= coeff * n_alpha
                grad_s = -coeff * (2.0 * self.s)
            proj_norm = float(np.linalg.norm(grad_buf))
            optimizer.step()

        pg = optimizer.param_groups[0]
        lr = pg.get("lr", 0.1)
        beta1, beta2 = pg.get("betas", (0.9, 0.999))
        eps = pg.get("eps", 1e-8)
        self.m_s = beta1 * self.m_s + (1.0 - beta1) * grad_s
        self.v_s = beta2 * self.v_s + (1.0 - beta2) * (grad_s ** 2)
        m_hat = self.m_s / (1.0 - beta1 ** self.step_count)
        v_hat = self.v_s / (1.0 - beta2 ** self.step_count)
        self.s = self.s - lr * m_hat / (np.sqrt(v_hat) + eps)

        cost_before = self.current_cost(alpha)
        if cost_before >= self.target_budget:
            retraction(
                alpha, self.costs, self.target_budget, self.weights, tol=self.tol
            )
            cost_after = self.current_cost(alpha)
            self.s = 0.0
        else:
            cost_after = cost_before
            self.s = float(np.sqrt(max(0.0, self.target_budget - cost_after)))

        n_new = budget_normal(alpha, self.costs, self.weights)
        if _is_torch(alpha):
            with torch.no_grad():
                nf_new = n_new.flatten()
                denom_new = (nf_new @ nf_new).item() + 4.0 * (self.s ** 2)
                state = optimizer.state.get(alpha)
                if state and "exp_avg" in state and denom_new > 1e-20:
                    m_alpha = state["exp_avg"]
                    dot_m = (m_alpha.flatten() @ nf_new).item() + self.m_s * (2.0 * self.s)
                    c_m = dot_m / denom_new
                    m_alpha.sub_(c_m * n_new)
                    self.m_s -= c_m * (2.0 * self.s)
        else:
            nf_new = n_new.ravel()
            denom_new = float(np.dot(nf_new, nf_new)) + 4.0 * (self.s ** 2)
            state = optimizer.state.get(id(alpha))
            if state and "exp_avg" in state and denom_new > 1e-20:
                m_alpha = state["exp_avg"]
                dot_m = float(np.dot(m_alpha.ravel(), nf_new)) + self.m_s * (2.0 * self.s)
                c_m = dot_m / denom_new
                m_alpha -= c_m * n_new
                self.m_s -= c_m * (2.0 * self.s)

        return {
            "proj_coeff": coeff,
            "raw_grad_norm": raw_norm,
            "proj_grad_norm": proj_norm,
            "cost_before_retract": cost_before,
            "cost_after_retract": cost_after,
            "slack_s": self.s,
            "violation": max(0.0, cost_after - self.target_budget),
        }


class MultiBudgetManifold:
    """Simultaneous equality constraints M = cap_{j=1}^q {C_j(alpha) = b_j} (Appendix A.8)."""

    def __init__(
        self,
        cost_list: List[ArrayLike],
        target_budgets: List[float],
        weight_list: Optional[List[Optional[ArrayLike]]] = None,
        tol: float = 1e-7,
        max_newton_iter: int = 35,
    ):
        assert len(cost_list) == len(target_budgets) and len(cost_list) >= 1
        self.q = len(cost_list)
        self.cost_list = cost_list
        self.target_budgets = [float(b) for b in target_budgets]
        self.weight_list = (
            weight_list if weight_list is not None else [None] * self.q
        )
        self.tol = tol
        self.max_newton_iter = max_newton_iter

    def current_costs(self, alpha: ArrayLike) -> List[float]:
        if _is_torch(alpha):
            with torch.no_grad():
                return [
                    float(expected_cost(alpha, self.cost_list[j], self.weight_list[j]).item())
                    for j in range(self.q)
                ]
        return [
            float(expected_cost(alpha, self.cost_list[j], self.weight_list[j]))
            for j in range(self.q)
        ]

    def current_violation(self, alpha: ArrayLike) -> float:
        cs = self.current_costs(alpha)
        return max(abs(cs[j] - self.target_budgets[j]) for j in range(self.q))

    def project_vec_inplace(self, alpha: ArrayLike, vec: ArrayLike) -> None:
        """Project vec onto the intersection of tangent spaces:
        vec <- vec - N (N^T N)^{-1} N^T vec
        """
        alpha_np = _to_numpy(alpha)
        cols = [
            _to_numpy(budget_normal(alpha_np, self.cost_list[j], self.weight_list[j])).ravel()
            for j in range(self.q)
        ]
        N_mat = np.stack(cols, axis=-1)  # [NK, q]
        vf = _to_numpy(vec).ravel()
        gram = N_mat.T @ N_mat + 1e-12 * np.eye(self.q)
        rhs = N_mat.T @ vf
        coeffs = np.linalg.solve(gram, rhs)
        correction = (N_mat @ coeffs).reshape(alpha_np.shape)
        if _is_torch(vec):
            with torch.no_grad():
                vec.sub_(torch.tensor(correction, device=vec.device, dtype=vec.dtype))
        else:
            vec -= correction

    def retract(self, alpha: ArrayLike) -> List[float]:
        """Newton retraction on the q-dimensional shift t in R^q (Appendix A.8)."""
        if self.q == 1:
            retraction(
                alpha,
                self.cost_list[0],
                self.target_budgets[0],
                weights=self.weight_list[0],
                tol=self.tol,
            )
            return self.current_costs(alpha)

        alpha_np = _to_numpy(alpha).copy()
        cs, ws = [], []
        for j in range(self.q):
            c, w = _prepare_costs_and_weights_np(alpha_np, self.cost_list[j], self.weight_list[j])
            cs.append(c)
            ws.append(w)

        dirs = [ws[l] * cs[l] for l in range(self.q)]
        b_vec = np.array(self.target_budgets, dtype=np.float64)
        t = np.zeros(self.q, dtype=np.float64)

        def eval_residuals_and_jacobian(t_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            shifted = alpha_np.copy()
            for l in range(self.q):
                shifted = shifted + t_vec[l] * dirs[l]
            probs = _softmax_np(shifted, axis=-1)

            res = np.zeros(self.q, dtype=np.float64)
            centered = []
            for j in range(self.q):
                ej = np.sum(probs * cs[j], axis=-1, keepdims=True)
                res[j] = float(np.sum(ws[j] * ej)) - b_vec[j]
                centered.append(ws[j] * (cs[j] - ej))

            J = np.zeros((self.q, self.q), dtype=np.float64)
            for j in range(self.q):
                for l in range(j, self.q):
                    val = float(np.sum(probs * centered[j] * centered[l]))
                    J[j, l] = val
                    J[l, j] = val
            return res, J

        for _ in range(self.max_newton_iter):
            res, J = eval_residuals_and_jacobian(t)
            if float(np.max(np.abs(res))) <= self.tol:
                break
            reg = 1e-10 * np.eye(self.q)
            delta = np.linalg.solve(J + reg, -res)
            step_size = 1.0
            cur_norm = float(np.linalg.norm(res))
            for _ls in range(12):
                t_cand = t + step_size * delta
                res_cand, _ = eval_residuals_and_jacobian(t_cand)
                if float(np.linalg.norm(res_cand)) < cur_norm or step_size < 1e-3:
                    t = t_cand
                    break
                step_size *= 0.5
            else:
                t = t + step_size * delta

        total_shift = np.zeros_like(alpha_np)
        for l in range(self.q):
            total_shift = total_shift + t[l] * dirs[l]

        if _is_torch(alpha):
            with torch.no_grad():
                alpha.add_(torch.tensor(total_shift, device=alpha.device, dtype=alpha.dtype))
        else:
            alpha += total_shift

        return self.current_costs(alpha)

    def initialize(self, alpha: ArrayLike) -> List[float]:
        return self.retract(alpha)

    def step(
        self,
        alpha: ArrayLike,
        optimizer: Any,
    ) -> dict:
        if _is_torch(alpha):
            with torch.no_grad():
                raw_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
                if alpha.grad is not None:
                    self.project_vec_inplace(alpha, alpha.grad)
                proj_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
                optimizer.step()
                costs_before = self.current_costs(alpha)
                costs_after = self.retract(alpha)
                state = optimizer.state.get(alpha)
                if state and "exp_avg" in state:
                    self.project_vec_inplace(alpha, state["exp_avg"])
        else:
            grad_buf = optimizer.get_grad(alpha)
            raw_norm = float(np.linalg.norm(grad_buf))
            self.project_vec_inplace(alpha, grad_buf)
            proj_norm = float(np.linalg.norm(grad_buf))
            optimizer.step()
            costs_before = self.current_costs(alpha)
            costs_after = self.retract(alpha)
            state = optimizer.state.get(id(alpha))
            if state and "exp_avg" in state:
                self.project_vec_inplace(alpha, state["exp_avg"])

        max_viol = max(
            abs(costs_after[j] - self.target_budgets[j]) for j in range(self.q)
        )
        return {
            "raw_grad_norm": raw_norm,
            "proj_grad_norm": proj_norm,
            "costs_before_retract": costs_before,
            "costs_after_retract": costs_after,
            "violation": max_viol,
        }


# ---------------------------------------------------------------------------
# Budget-Constrained Discrete Solvers (Section 3.1, Eq. 4)
# ---------------------------------------------------------------------------


def dp_solve_mckp(
    scores: ArrayLike,
    costs: ArrayLike,
    target_budget: float,
    weights: Optional[ArrayLike] = None,
    resolution: int = 2000,
    exact_equality: bool = False,
) -> ArrayLike:
    """Solve the Multiple-Choice Knapsack Problem (Eq. 4 in RCO paper)."""
    scores_np = _to_numpy(scores).astype(np.float64)
    N, K = scores_np.shape
    c_np, w_np = _prepare_costs_and_weights_np(scores_np, costs, weights)
    full_costs = np.broadcast_to(w_np * c_np, (N, K)).copy()

    min_per_group = full_costs.min(axis=1, keepdims=True)
    min_total = float(min_per_group.sum())

    if target_budget <= min_total + 1e-9:
        idx = full_costs.argmin(axis=1).astype(np.int64)
        if _is_torch(scores):
            return torch.tensor(idx, dtype=torch.long, device=scores.device)
        return idx

    rel_budget = float(target_budget) - min_total
    rel_costs = full_costs - min_per_group

    # Adaptively bound DP table size N * B_int so large MoE models (e.g. N=10240 in Qwen3.6-35B-A3B)
    # run in ~1s while vectorized float refinement afterwards packs the exact continuous budget.
    eff_resolution = int(resolution)
    if N * eff_resolution > 20_000_000:
        eff_resolution = max(2000, int(20_000_000 // max(N, 1)))

    scale = eff_resolution / max(rel_budget, 1e-9)
    int_costs = np.rint(rel_costs * scale).astype(np.int64)
    B_int = int(round(rel_budget * scale))

    neg_inf = -1e30
    dp = np.full(B_int + 1, neg_inf, dtype=np.float64)
    dp[0] = 0.0
    backtrack = np.zeros((N, B_int + 1), dtype=np.int16)

    for i in range(N):
        new_dp = np.full(B_int + 1, neg_inf, dtype=np.float64)
        ci = int_costs[i]
        si = scores_np[i]
        for k in range(K):
            ck = int(ci[k])
            sk = float(si[k])
            if ck == 0:
                cand = dp + sk
                mask = cand > new_dp
                new_dp[mask] = cand[mask]
                backtrack[i, mask] = k
            elif ck <= B_int:
                cand = dp[:-ck] + sk
                sub_new = new_dp[ck:]
                mask = cand > sub_new
                sub_new[mask] = cand[mask]
                backtrack[i, ck:][mask] = k
        dp = new_dp

    if exact_equality:
        valid_js = np.where(dp > neg_inf * 0.5)[0]
        if len(valid_js) == 0:
            best_j = 0
        else:
            best_j = int(valid_js[np.argmin(np.abs(valid_js - B_int))])
    else:
        best_j = int(np.argmax(dp))

    chosen = np.zeros(N, dtype=np.int64)
    cur_j = best_j
    for i in range(N - 1, -1, -1):
        k_star = int(backtrack[i, cur_j])
        chosen[i] = k_star
        cur_j = max(0, cur_j - int(int_costs[i, k_star]))

    # Vectorized post-repair if discretization slightly exceeded exact float budget
    row_idx = np.arange(N)
    total_actual = float(full_costs[row_idx, chosen].sum())
    while total_actual > target_budget + 1e-6:
        cur_c = full_costs[row_idx, chosen][:, None]
        cur_s = scores_np[row_idx, chosen][:, None]
        saving = cur_c - full_costs
        valid = saving > 1e-9
        if not np.any(valid):
            break
        score_drop = np.maximum(0.0, cur_s - scores_np)
        ratio = np.where(valid, score_drop / np.maximum(saving, 1e-12), np.inf)
        flat_best = int(np.argmin(ratio))
        best_i, best_k = divmod(flat_best, K)
        if not np.isfinite(ratio[best_i, best_k]):
            break
        total_actual -= float(saving[best_i, best_k])
        chosen[best_i] = best_k

    # Vectorized greedy upgrade if discretization left unused float budget
    if not exact_equality:
        while True:
            rem_budget = float(target_budget) - total_actual
            if rem_budget <= 1e-6:
                break
            cur_c = full_costs[row_idx, chosen][:, None]
            cur_s = scores_np[row_idx, chosen][:, None]
            extra_c = full_costs - cur_c
            gain_s = scores_np - cur_s
            valid = (extra_c > 1e-9) & (extra_c <= rem_budget + 1e-9) & (gain_s > 0.0)
            if not np.any(valid):
                break
            density = np.where(valid, gain_s / np.maximum(extra_c, 1e-12), -np.inf)
            flat_best = int(np.argmax(density))
            best_i, best_k = divmod(flat_best, K)
            if not np.isfinite(density[best_i, best_k]) or density[best_i, best_k] <= 0.0:
                break
            total_actual += float(extra_c[best_i, best_k])
            chosen[best_i] = best_k

    if _is_torch(scores):
        return torch.tensor(chosen, dtype=torch.long, device=scores.device)
    return chosen


def topk_solve_binary(
    scores: ArrayLike,
    budget_count: int,
) -> ArrayLike:
    """Exact O(N log N) solver for binary options with costs (0, 1) (Section 3.1)."""
    scores_np = _to_numpy(scores)
    assert scores_np.ndim == 2 and scores_np.shape[1] == 2
    N = scores_np.shape[0]
    k = max(0, min(N, int(round(budget_count))))
    diff = scores_np[:, 1] - scores_np[:, 0]
    chosen = np.zeros(N, dtype=np.int64)
    if k > 0:
        top_idx = np.argsort(diff)[-k:]
        chosen[top_idx] = 1
    if _is_torch(scores):
        return torch.tensor(chosen, dtype=torch.long, device=scores.device)
    return chosen
