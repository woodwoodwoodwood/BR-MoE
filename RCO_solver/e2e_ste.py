"""End-to-End Gumbel-STE + Memory-Efficient Weight Assembly for BR-MoE.

Implements the exact-gradient / STE backward identity:
    dL / d(p_tilde_{ik}) = < (dL / dy)^T x,  W_hat_i^{(k)} >_F
which avoids materializing all K candidate (bit, rank) weight matrices in GPU
memory during the forward pass:
  - Forward pass only uses the single DP-selected option k* = z_i^*.
  - Backward pass computes G_W = (dL / dy)^T x once per routed expert projection
    and takes Frobenius inner products with each candidate option k = 1..K.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False


def parse_config_name(config_name: str) -> Tuple[int, int]:
    """Parse BR-MoE option string like '3bit_rank32' or 'prune' -> (bits, rank)."""
    if config_name.lower() == "prune" or config_name.startswith("0bit"):
        return 0, 0
    m = re.match(r"(\d+)bit_rank(\d+)", config_name)
    if not m:
        raise ValueError(f"Unrecognized BR-MoE config string: {config_name}")
    return int(m.group(1)), int(m.group(2))


def streaming_ste_forward_backward_np(
    x: np.ndarray,
    grad_out: np.ndarray,
    logits: np.ndarray,
    k_star: int,
    cand_weights: List[np.ndarray],
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference NumPy implementation of the Streaming STE Forward & Backward Identity:
        Forward : y = x @ W^{(k*)}.T
        Backward:
          grad_x     = grad_out @ W^{(k*)}
          G_W        = grad_out.T @ x
          grad_p[k]  = <G_W, W^{(k)}>_F
          grad_alpha = (p_soft / tau) * (grad_p - <p_soft, grad_p>)
    """
    W_active = cand_weights[k_star]
    y = x @ W_active.T
    grad_x = grad_out @ W_active
    G_W = grad_out.T @ x  # Single GEMM regardless of K
    K = len(cand_weights)
    grad_p = np.array([float(np.sum(G_W * cand_weights[k])) for k in range(K)], dtype=np.float64)

    shifted = (logits - np.max(logits)) / tau
    p_soft = np.exp(shifted) / np.sum(np.exp(shifted))
    expected_gp = float(np.sum(p_soft * grad_p))
    grad_alpha = (p_soft / tau) * (grad_p - expected_gp)
    return y, grad_x, grad_alpha


if HAS_TORCH:

    @torch.no_grad()
    def reconstruct_brmoe_weight(
        W_orig: torch.Tensor,
        nbits: int,
        rank: int,
        group_size: int = 128,
        iters: int = 10,
        compensator_dtype: str = "int3",
        compensator_quant_gs: int = 64,
        axis: int = 1,
    ) -> torch.Tensor:
        """Reconstruct dequantized weight W_hat = W_q_dequant + U @ V for a given (nbits, rank).

        Matches `BRMoELinear.compress` in `BR-MoE/core/quantize.py`.
        """
        if nbits == 0:
            return torch.zeros_like(W_orig)

        from BR_MoE.core.quantize import Quantizer
        from BR_MoE.core.utils import (
            compensator_dequantize,
            quantize_full_to_int3,
            quantize_full_to_int4,
        )

        device = W_orig.device
        orig_dtype = W_orig.dtype
        W_unquant = W_orig.to(device=device, dtype=orig_dtype)
        U, V = None, None

        eff_iters = 0 if rank == 0 else iters
        for i in range(eff_iters + 1):
            W_cur = W_unquant if i == 0 else (W_unquant - (U @ V))
            W_q, meta = Quantizer.quantize(
                W_cur,
                nbits=nbits,
                channel_wise=True,
                group_size=group_size,
                optimize=True,
                round_zero=(nbits == 4),
                axis=axis,
                bitpack=False,
                compute_dtype=orig_dtype,
                device=str(device),
            )
            if rank == 0:
                break
            W_q_deq = Quantizer.dequantize(W_q, meta).to(device)
            residual = W_unquant.float() - W_q_deq.float()
            U_svd, S, V_svd = torch.svd_lowrank(residual, q=rank)
            sqrt_S = torch.diag(torch.sqrt(S))
            U = (U_svd @ sqrt_S).to(device=device, dtype=orig_dtype)
            V = (sqrt_S @ V_svd.T).to(device=device, dtype=orig_dtype)

        W_hat = Quantizer.dequantize(W_q, meta).to(device=device, dtype=orig_dtype)
        if rank > 0 and U is not None and V is not None:
            if compensator_dtype == "int3":
                uv_q = (
                    quantize_full_to_int3(U, compensator_quant_gs),
                    quantize_full_to_int3(V, compensator_quant_gs),
                )
                U_deq, V_deq = compensator_dequantize(
                    uv_q, meta["shape"], rank, compensator_quant_gs, "int3"
                )
                W_hat = W_hat + (
                    U_deq.to(device, orig_dtype) @ V_deq.to(device, orig_dtype)
                )
            elif compensator_dtype == "int4":
                uv_q = (
                    quantize_full_to_int4(U, compensator_quant_gs),
                    quantize_full_to_int4(V, compensator_quant_gs),
                )
                U_deq, V_deq = compensator_dequantize(
                    uv_q, meta["shape"], rank, compensator_quant_gs, "int4"
                )
                W_hat = W_hat + (
                    U_deq.to(device, orig_dtype) @ V_deq.to(device, orig_dtype)
                )
            else:
                W_hat = W_hat + (U @ V)

        return W_hat

    class _StreamingSTELinearFn(torch.autograd.Function):
        """Custom autograd function for Gumbel-STE over K candidate weight matrices."""

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            p_ste: torch.Tensor,
            k_star: int,
            bias: Optional[torch.Tensor],
            weight_getter: Callable[[int, torch.device, torch.dtype], torch.Tensor],
        ) -> torch.Tensor:
            ctx.save_for_backward(x, p_ste, bias)
            ctx.k_star = int(k_star)
            ctx.weight_getter = weight_getter

            W_active = weight_getter(ctx.k_star, x.device, x.dtype)
            out = F.linear(x, W_active, bias)
            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, p_ste, bias = ctx.saved_tensors
            k_star = ctx.k_star
            weight_getter = ctx.weight_getter
            K = p_ste.numel()

            grad_x = None
            grad_p = None
            grad_bias = None

            W_active = weight_getter(k_star, grad_output.device, grad_output.dtype)
            if ctx.needs_input_grad[0]:
                grad_x = torch.matmul(grad_output, W_active)

            if ctx.needs_input_grad[1]:
                go_2d = grad_output.reshape(-1, grad_output.size(-1)).float()
                x_2d = x.reshape(-1, x.size(-1)).float()
                G_W = torch.matmul(go_2d.t(), x_2d)  # [D_out, D_in]
                grad_p = torch.zeros_like(p_ste, dtype=torch.float32)
                for k in range(K):
                    W_k = weight_getter(k, G_W.device, torch.float32)
                    grad_p[k] = (G_W * W_k).sum()
                grad_p = grad_p.to(p_ste.dtype)

            if bias is not None and ctx.needs_input_grad[3]:
                grad_bias = grad_output.reshape(-1, grad_output.size(-1)).sum(dim=0)

            return grad_x, grad_p, None, grad_bias, None

    class RCOInterpolatedLinear(nn.Module):
        """Drop-in replacement for an expert nn.Linear layer during RCO search."""

        def __init__(
            self,
            orig_linear: nn.Linear,
            group_idx: int,
            option_names: List[str],
            shared_search_state: dict,
            cpu_offload: bool = True,
            precomputed_weights: Optional[List[torch.Tensor]] = None,
        ):
            super().__init__()
            self.in_features = orig_linear.in_features
            self.out_features = orig_linear.out_features
            self.group_idx = group_idx
            self.option_names = option_names
            self.K = len(option_names)
            self.shared_state = shared_search_state
            self.cpu_offload = cpu_offload

            self.orig_device = orig_linear.weight.device
            self.orig_dtype = orig_linear.weight.dtype
            self.bias = orig_linear.bias

            self._candidates: List[Optional[torch.Tensor]] = [None] * self.K
            if precomputed_weights is not None:
                for k, w in enumerate(precomputed_weights):
                    self.set_candidate_weight(k, w)
            else:
                self._orig_weight = orig_linear.weight.detach().cpu()

        def set_candidate_weight(self, k: int, w: torch.Tensor) -> None:
            if self.cpu_offload:
                w_cpu = w.detach().to(dtype=self.orig_dtype, device="cpu").contiguous()
                if torch.cuda.is_available():
                    try:
                        w_cpu = w_cpu.pin_memory()
                    except Exception:
                        pass
                self._candidates[k] = w_cpu
            else:
                self._candidates[k] = w.detach().to(
                    dtype=self.orig_dtype, device=self.orig_device
                ).contiguous()

        def get_candidate_weight(
            self, k: int, device: torch.device, dtype: torch.dtype
        ) -> torch.Tensor:
            w = self._candidates[k]
            if w is None:
                nbits, rank = parse_config_name(self.option_names[k])
                w_rec = reconstruct_brmoe_weight(
                    self._orig_weight.to(device),
                    nbits=nbits,
                    rank=rank,
                    group_size=self.shared_state.get("group_size", 128),
                    iters=self.shared_state.get("iters", 10),
                )
                self.set_candidate_weight(k, w_rec)
                w = self._candidates[k]
            return w.to(device=device, dtype=dtype, non_blocking=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            p_ste_all = self.shared_state["p_ste"]
            k_star_all = self.shared_state["k_star"]
            p_ste_i = p_ste_all[self.group_idx]
            k_star_i = int(k_star_all[self.group_idx].item())
            return _StreamingSTELinearFn.apply(
                x, p_ste_i, k_star_i, self.bias, self.get_candidate_weight
            )

    def wrap_moe_model_for_rco(
        model: nn.Module,
        option_names: List[str],
        shared_search_state: dict,
        cpu_offload: bool = True,
    ) -> List[str]:
        """Replace expert linear projections in a MoE model with RCOInterpolatedLinear."""
        expert_key_to_group: Dict[str, int] = {}
        expert_keys: List[str] = []

        replacements = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            m_qwen = re.search(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$",
                name,
            )
            m_mix = re.search(
                r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)$",
                name,
            )
            m = m_qwen or m_mix
            if not m:
                continue

            layer_idx = int(m.group(1))
            expert_idx = int(m.group(2))
            exp_key = f"L{layer_idx}_E{expert_idx}"
            if exp_key not in expert_key_to_group:
                expert_key_to_group[exp_key] = len(expert_keys)
                expert_keys.append(exp_key)

            parent_name, attr_name = name.rsplit(".", 1)
            parent_mod = model.get_submodule(parent_name)
            replacements.append(
                (parent_mod, attr_name, module, expert_key_to_group[exp_key])
            )

        for parent_mod, attr_name, orig_linear, group_idx in replacements:
            wrapped = RCOInterpolatedLinear(
                orig_linear=orig_linear,
                group_idx=group_idx,
                option_names=option_names,
                shared_search_state=shared_search_state,
                cpu_offload=cpu_offload,
            )
            setattr(parent_mod, attr_name, wrapped)

        return expert_keys

    def run_e2e_kl_search(
        model: nn.Module,
        calib_inputs: List[torch.Tensor],
        option_names: List[str],
        option_costs: torch.Tensor,
        target_budget: float,
        steps: int = 100,
        lr: float = 0.05,
        tau_init: float = 1.0,
        tau_min: float = 0.05,
        prior_logits: Optional[torch.Tensor] = None,
        cpu_offload: bool = True,
        dp_resolution: int = 2000,
    ) -> Tuple[Dict[str, str], dict]:
        """End-to-End Algorithm 1 (arXiv:2605.00649) on a MoE model using KL divergence."""
        from RCO_solver.manifold import BudgetManifold, dp_solve_mckp

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        ref_logits_cache = []
        with torch.no_grad():
            for inp in calib_inputs:
                out = model(inp)
                logits = out.logits if hasattr(out, "logits") else out
                ref_logits_cache.append(logits.detach().cpu())

        shared_state: dict = {"group_size": 128, "iters": 10}
        expert_keys = wrap_moe_model_for_rco(
            model, option_names, shared_state, cpu_offload=cpu_offload
        )
        N = len(expert_keys)
        K = len(option_names)
        if N == 0:
            raise ValueError("No MoE experts found in `model` to wrap for RCO!")

        dtype = torch.float64
        device = option_costs.device
        costs = option_costs.to(dtype=dtype, device=device)

        if prior_logits is not None:
            alpha = prior_logits.clone().detach().to(dtype=dtype, device=device)
        else:
            alpha = torch.zeros((N, K), dtype=dtype, device=device)
        alpha.requires_grad_(True)

        manifold = BudgetManifold(costs=costs, target_budget=target_budget, tol=1e-8)
        manifold.initialize(alpha)
        optimizer = torch.optim.Adam([alpha], lr=lr)

        history = []
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            tau = max(tau_min, tau_init * ((tau_min / tau_init) ** progress))

            optimizer.zero_grad()

            u = torch.rand_like(alpha).clamp(min=1e-15, max=1.0 - 1e-15)
            gumbel = -torch.log(-torch.log(u))
            noisy_logits = (alpha + gumbel) / tau
            soft_p = torch.softmax(noisy_logits, dim=-1)

            with torch.no_grad():
                k_star = dp_solve_mckp(
                    noisy_logits, costs, target_budget, resolution=dp_resolution
                )
                z_onehot = F.one_hot(k_star, num_classes=K).to(dtype)

            p_ste = (z_onehot - soft_p).detach() + soft_p
            shared_state["p_ste"] = p_ste
            shared_state["k_star"] = k_star

            batch_idx = step % len(calib_inputs)
            inp = calib_inputs[batch_idx]
            ref_logits = ref_logits_cache[batch_idx].to(inp.device)

            out = model(inp)
            q_logits = out.logits if hasattr(out, "logits") else out

            log_p_ref = F.log_softmax(ref_logits.float(), dim=-1)
            log_p_q = F.log_softmax(q_logits.float(), dim=-1)
            p_ref = log_p_ref.exp()
            kl_loss = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1).mean()

            kl_loss.backward()
            info = manifold.step(alpha, optimizer)

            history.append(
                {
                    "step": step,
                    "kl_loss": float(kl_loss.item()),
                    "violation": info["violation"],
                }
            )

        with torch.no_grad():
            final_log_probs = torch.log_softmax(alpha, dim=-1)
            final_idx = dp_solve_mckp(
                final_log_probs, costs, target_budget, resolution=dp_resolution
            )

        best_config = {
            expert_keys[i]: option_names[int(final_idx[i].item())] for i in range(N)
        }
        return best_config, {
            "history": history,
            "final_violation": history[-1]["violation"],
        }

else:
    reconstruct_brmoe_weight = None  # type: ignore
    RCOInterpolatedLinear = None  # type: ignore
    wrap_moe_model_for_rco = None  # type: ignore
    run_e2e_kl_search = None  # type: ignore
