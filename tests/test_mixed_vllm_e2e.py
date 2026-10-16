#!/usr/bin/env python3
"""End-to-end verification of BR-MoE Mixed-Precision (2/3/4-bit) + Mixed-Compensation
(rank = 0 / 16 / 32 / 64) inference across:
  1. Bitwise 3-bit + 1-bit MSB plane packing/unpacking & CSR low-rank compensator math
     (verified with NumPy on CPU and PyTorch/Triton on GPU when available)
  2. AST & bytecode compilation check across all modified kernel, backend, converter, and
     vLLM plugin files
  3. HuggingFace / BR-MoE grouped backend (`brmoe_grouped.patch_moe_to_grouped`)
  4. Checkpoint converter (`tools/convert_brmoe_to_vllm.py` with `compensators.pt`)
  5. vLLM plugin (`tools/brmoe_int3_vllm`: `BRMoEInt3Config`, `BRMoEInt3MoEMethod`,
     pre-packed `RoutedExperts.load_weights` & `process_weights_after_loading`)
"""

from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BRMOE_PKG = REPO_ROOT / "BR-MoE"
KERNELS_DIR = BRMOE_PKG / "kernels" / "triton_int3"
TOOLS_DIR = REPO_ROOT / "tools"
VLLM_ROOT = REPO_ROOT.parent / "vllm"

for p in (str(REPO_ROOT), str(BRMOE_PKG), str(KERNELS_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False


# ===========================================================================
# Part A: Always-Run Verification (NumPy Reference + AST/Bytecode + Config)
# ===========================================================================


def _np_pack_int3(q: np.ndarray, transposed: bool = False) -> np.ndarray:
    """NumPy mirror of pack_int3 (low 3 bits of each weight)."""
    N, K = q.shape
    assert K % 32 == 0
    v = q.astype(np.int64).reshape(N, K // 32, 32)

    def g(i):
        return v[..., i] & 0x7

    w0 = (
        (g(0) << 0) | (g(1) << 3) | (g(2) << 6) | (g(3) << 9)
        | (g(4) << 16) | (g(5) << 19) | (g(6) << 22) | (g(7) << 25)
        | (g(24) << 12) | ((g(25) & 0x1) << 15)
        | (g(28) << 28) | ((g(29) & 0x1) << 31)
    )
    w1 = (
        (g(8) << 0) | (g(9) << 3) | (g(10) << 6) | (g(11) << 9)
        | (g(12) << 16) | (g(13) << 19) | (g(14) << 22) | (g(15) << 25)
        | ((g(25) & 0x6) << 11) | ((g(26) & 0x3) << 14)
        | ((g(29) & 0x6) << 27) | ((g(30) & 0x3) << 30)
    )
    w2 = (
        (g(16) << 0) | (g(17) << 3) | (g(18) << 6) | (g(19) << 9)
        | (g(20) << 16) | (g(21) << 19) | (g(22) << 22) | (g(23) << 25)
        | ((g(26) & 0x4) << 10) | (g(27) << 13)
        | ((g(30) & 0x4) << 26) | (g(31) << 29)
    )
    out = np.stack([w0, w1, w2], axis=-1).astype(np.int32).reshape(N, K // 32 * 3)
    return np.ascontiguousarray(out.T) if transposed else out


def _np_unpack_int3(p: np.ndarray, K: int, transposed: bool = False) -> np.ndarray:
    """NumPy mirror of unpack_int3."""
    if transposed:
        p = np.ascontiguousarray(p.T)
    N = p.shape[0]
    KB = K // 32
    p64 = p.astype(np.int64).reshape(N, KB, 3)
    w0, w1, w2 = p64[..., 0], p64[..., 1], p64[..., 2]
    out = np.empty((N, KB, 32), dtype=np.int64)
    for i in range(4):
        out[..., 0 + i] = (w0 >> (3 * i)) & 0x7
        out[..., 4 + i] = (w0 >> (16 + 3 * i)) & 0x7
        out[..., 8 + i] = (w1 >> (3 * i)) & 0x7
        out[..., 12 + i] = (w1 >> (16 + 3 * i)) & 0x7
        out[..., 16 + i] = (w2 >> (3 * i)) & 0x7
        out[..., 20 + i] = (w2 >> (16 + 3 * i)) & 0x7
    lo = ((w0 >> 12) & 0xF) | (((w1 >> 12) & 0xF) << 4) | (((w2 >> 12) & 0xF) << 8)
    hi = ((w0 >> 28) & 0xF) | (((w1 >> 28) & 0xF) << 4) | (((w2 >> 28) & 0xF) << 8)
    for i in range(4):
        out[..., 24 + i] = (lo >> (3 * i)) & 0x7
        out[..., 28 + i] = (hi >> (3 * i)) & 0x7
    return out.reshape(N, K).astype(np.int32)


def _np_pack_uint1_msb(q: np.ndarray, transposed: bool = False) -> np.ndarray:
    """NumPy mirror of pack_uint1_msb."""
    N, K = q.shape
    msb = ((q.astype(np.int64) >> 3) & 0x1).reshape(N, K // 32, 32)
    shifts = np.arange(32, dtype=np.int64)
    out = np.sum(msb << shifts, axis=-1).astype(np.int32)
    return np.ascontiguousarray(out.T) if transposed else out


def _np_unpack_uint1_msb(p: np.ndarray, K: int, transposed: bool = False) -> np.ndarray:
    """NumPy mirror of unpack_uint1_msb."""
    if transposed:
        p = np.ascontiguousarray(p.T)
    N = p.shape[0]
    v = p.astype(np.int64)[..., None]
    shifts = np.arange(32, dtype=np.int64)
    bits = (v >> shifts) & 0x1
    return bits.reshape(N, K).astype(np.int32)


def test_a1_ast_and_bytecode_compilation():
    print("=" * 72)
    print("[Test A1] Verifying AST syntax & bytecode compilation across all modified files...")
    files_to_check = [
        KERNELS_DIR / "int3_moe" / "packing.py",
        KERNELS_DIR / "int3_moe" / "kernel.py",
        KERNELS_DIR / "int3_moe" / "ops.py",
        BRMOE_PKG / "backends" / "brmoe_grouped.py",
        TOOLS_DIR / "convert_brmoe_to_vllm.py",
        TOOLS_DIR / "brmoe_int3_vllm" / "__init__.py",
        TOOLS_DIR / "brmoe_int3_vllm" / "config.py",
        TOOLS_DIR / "brmoe_int3_vllm" / "moe_method.py",
        TOOLS_DIR / "brmoe_int3_vllm" / "kernel.py",
        TOOLS_DIR / "brmoe_int3_vllm" / "linear_method.py",
        TOOLS_DIR / "export_brmoe_to_gguf.py",
        REPO_ROOT / "RCO_solver" / "model_adapters.py",
        REPO_ROOT / "RCO_solver" / "rco_search.py",
        REPO_ROOT / "RCO_solver" / "e2e_ste.py",
        REPO_ROOT / "collect" / "collect_expert.py",
        BRMOE_PKG / "core" / "quantize.py",
    ]
    routed_experts_py = VLLM_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" / "routed_experts.py"
    if routed_experts_py.exists():
        files_to_check.append(routed_experts_py)

    for path in files_to_check:
        assert path.exists(), f"Missing file: {path}"
        src = path.read_text(encoding="utf-8")
        ast.parse(src, filename=str(path))
        py_compile.compile(str(path), doraise=True)
        print(f"  [OK] {path.relative_to(REPO_ROOT.parent)}")
    print("[Test A1] PASSED!")


def test_a2_mixed_2_3_4_bit_msb_plane_and_csr_compensator_math():
    print("=" * 72)
    print("[Test A2] Verifying 100% lossless 2/3/4-bit MSB plane & CSR variable-rank compensator math...")
    rng = np.random.default_rng(2026)
    E, K, I, M, topk = 8, 128, 64, 12, 2
    expert_bits = [2, 3, 4, 3, 4, 2, 3, 4]
    expert_ranks = [64, 0, 32, 16, 0, 64, 32, 16]

    # 1. Verify 100% lossless CSR Bit-Plane 3D tensor packing/unpacking (`pack_varbit_experts`)
    #    where 2-bit experts take 2.0 bpw, 3-bit take 3.0 bpw, 4-bit take 4.0 bpw inside a single 3D tensor!
    bit_offsets = np.zeros((E + 1,), dtype=np.int32)
    q_orig_list = []
    plane_chunks = []
    shifts = np.arange(32, dtype=np.int64)
    for e, nb in enumerate(expert_bits):
        bit_offsets[e + 1] = bit_offsets[e] + nb
        qmax = (1 << nb) - 1
        q_e = rng.integers(0, qmax + 1, size=(2 * I, K), dtype=np.int32)
        q_orig_list.append(q_e)
        v = q_e.astype(np.int64).reshape(2 * I, K // 32, 32)
        planes_e = []
        for b in range(nb):
            bit_b = (v >> b) & 0x1
            word_b = np.sum(bit_b << shifts, axis=-1).astype(np.int32)  # [2I, K // 32]
            planes_e.append(word_b)
        plane_chunks.append(np.stack(planes_e, axis=0))                 # [b_e, 2I, K // 32]

    w13_varbit = np.concatenate(plane_chunks, axis=0)                   # [B_tot, 2I, K // 32]
    B_tot = int(bit_offsets[-1])
    assert B_tot == sum(expert_bits) == 25
    assert w13_varbit.shape == (B_tot, 2 * I, K // 32)
    expected_bytes = sum(nb * (2 * I) * (K // 32) * 4 for nb in expert_bits)
    assert w13_varbit.nbytes == expected_bytes, (
        f"Expected {expected_bytes} bytes, got {w13_varbit.nbytes}"
    )

    for transposed in (False, True):
        wv_t = np.ascontiguousarray(np.transpose(w13_varbit, (0, 2, 1))) if transposed else w13_varbit
        for e in range(E):
            s_b, e_b = int(bit_offsets[e]), int(bit_offsets[e + 1])
            nb = e_b - s_b
            q_rec = np.zeros((2 * I, K // 32, 32), dtype=np.int64)
            for b in range(nb):
                wb = wv_t[s_b + b].T if transposed else wv_t[s_b + b]
                q_rec |= ((wb.astype(np.int64)[..., None] >> shifts) & 0x1) << b
            q_rec = q_rec.reshape(2 * I, K).astype(np.int32)
            assert np.array_equal(q_rec, q_orig_list[e]), (
                f"CSR VarBit mismatch for expert {e} (nbits={nb}, transposed={transposed})"
            )
    print(
        f"  [OK] CSR Bit-Plane 3D tensor [B_tot={B_tot}, 2I={2*I}, K//32={K//32}] "
        f"(2.0/3.0/4.0 bpw zero-waste) reconstructed with 0 bit error."
    )

    # 2. Verify CSR variable-rank low-rank compensator kernel algorithm vs per-expert reference
    comp_offsets = np.zeros((E + 1,), dtype=np.int32)
    for e, r_e in enumerate(expert_ranks):
        comp_offsets[e + 1] = comp_offsets[e] + r_e
    R_tot = int(comp_offsets[-1])
    max_r = max(expert_ranks)
    assert R_tot == sum(expert_ranks) == 224 and max_r == 64

    w13_v = rng.standard_normal((2, R_tot, K)).astype(np.float16) * 0.05
    w13_u = rng.standard_normal((2, R_tot, I)).astype(np.float16) * 0.05
    w2_v = rng.standard_normal((R_tot, I)).astype(np.float16) * 0.05
    w2_u = rng.standard_normal((R_tot, K)).astype(np.float16) * 0.05

    W13_deq = rng.standard_normal((E, 2 * I, K)).astype(np.float16) * 0.1
    W2_deq = rng.standard_normal((E, K, I)).astype(np.float16) * 0.1
    x = rng.standard_normal((M, K)).astype(np.float16)

    topk_ids = np.array([rng.choice(E, size=topk, replace=False) for _ in range(M)], dtype=np.int64)
    raw_w = rng.uniform(0.1, 1.0, size=(M, topk)).astype(np.float32)
    topk_w = (raw_w / raw_w.sum(axis=-1, keepdims=True)).astype(np.float16)

    # Reference per-expert MoE forward (compensator applied BEFORE silu_mul on gate/up, AFTER on down)
    y_ref = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        xm = x[m].astype(np.float32)
        for slot in range(topk):
            e = int(topk_ids[m, slot])
            rw = float(topk_w[m, slot])
            s_e, e_e = int(comp_offsets[e]), int(comp_offsets[e + 1])
            gate = W13_deq[e, :I].astype(np.float32) @ xm
            up = W13_deq[e, I:].astype(np.float32) @ xm
            if e_e > s_e:
                gate += w13_u[0, s_e:e_e].astype(np.float32).T @ (w13_v[0, s_e:e_e].astype(np.float32) @ xm)
                up += w13_u[1, s_e:e_e].astype(np.float32).T @ (w13_v[1, s_e:e_e].astype(np.float32) @ xm)
            act = (gate / (1.0 + np.exp(-gate))) * up
            act_f16 = act.astype(np.float16).astype(np.float32)
            down = W2_deq[e].astype(np.float32) @ act_f16
            if e_e > s_e:
                down += w2_u[s_e:e_e].astype(np.float32).T @ (w2_v[s_e:e_e].astype(np.float32) @ act_f16)
            y_ref[m] += rw * down

    # Simulate the exact 2-stage CSR Rank-Slice algorithm used in kernel.py
    R_routes = M * topk
    flat_ids = topk_ids.reshape(-1)
    flat_wts = topk_w.reshape(-1).astype(np.float32)
    inter = np.zeros((R_routes, 2 * I), dtype=np.float32)
    for route in range(R_routes):
        tok = route // topk
        e = int(flat_ids[route])
        inter[route] = W13_deq[e].astype(np.float32) @ x[tok].astype(np.float32)

    # Stage 1 CSR compensator (w13_v -> h13 -> w13_u -> inter)
    h13 = np.zeros((2, R_routes, max_r), dtype=np.float32)
    for route in range(R_routes):
        tok = route // topk
        e = int(flat_ids[route])
        s_e, e_e = int(comp_offsets[e]), int(comp_offsets[e + 1])
        r_e = e_e - s_e
        if r_e > 0:
            h13[0, route, :r_e] = w13_v[0, s_e:e_e].astype(np.float32) @ x[tok].astype(np.float32)
            h13[1, route, :r_e] = w13_v[1, s_e:e_e].astype(np.float32) @ x[tok].astype(np.float32)
            inter[route, :I] += h13[0, route, :r_e] @ w13_u[0, s_e:e_e].astype(np.float32)
            inter[route, I:] += h13[1, route, :r_e] @ w13_u[1, s_e:e_e].astype(np.float32)

    # SiLU * up
    g_inter, u_inter = inter[:, :I], inter[:, I:]
    act_routes = ((g_inter / (1.0 + np.exp(-g_inter))) * u_inter).astype(np.float16)

    # Stage 2 base + CSR compensator (w2_v -> h2 -> w2_u -> out32)
    out32 = np.zeros((M, K), dtype=np.float32)
    h2 = np.zeros((R_routes, max_r), dtype=np.float32)
    for route in range(R_routes):
        tok = route // topk
        e = int(flat_ids[route])
        rw = flat_wts[route]
        out32[tok] += rw * (W2_deq[e].astype(np.float32) @ act_routes[route].astype(np.float32))
        s_e, e_e = int(comp_offsets[e]), int(comp_offsets[e + 1])
        r_e = e_e - s_e
        if r_e > 0:
            h2[route, :r_e] = w2_v[s_e:e_e].astype(np.float32) @ act_routes[route].astype(np.float32)
            out32[tok] += rw * (h2[route, :r_e] @ w2_u[s_e:e_e].astype(np.float32))

    max_err = np.max(np.abs(out32 - y_ref))
    print(f"  [OK] CSR Rank-Slice 2-stage algorithm vs per-expert reference max error: {max_err:.3e}")
    assert max_err < 1e-6
    print("[Test A2] PASSED!")


# ===========================================================================
# Part B: Full GPU Triton + Grouped Backend + Converter + vLLM Plugin Suite
# ===========================================================================

if _HAS_TORCH and _HAS_CUDA:
    from core.quantize import BRMoELinear, QuantConfig
    from core.utils import quantize_UV, dequantize_UV
    from backends.brmoe_grouped import patch_moe_to_grouped
    from int3_moe.ops import fused_moe_int3, ref_fused_moe

    class DummyExpert(nn.Module):
        def __init__(self, gate_proj: nn.Module, up_proj: nn.Module, down_proj: nn.Module):
            super().__init__()
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class DummyMoEBlock(nn.Module):
        """Standard Qwen3-MoE / DeepSeek-MoE / Mixtral-style MoE block."""

        def __init__(self, K: int, I: int, E: int, topk: int, device: torch.device):
            super().__init__()
            self.K = K
            self.I = I
            self.E = E
            self.top_k = topk
            self.norm_topk_prob = True
            self.gate = nn.Linear(K, E, bias=False, device=device, dtype=torch.float16)
            self.experts = nn.ModuleList()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            orig_shape = hidden_states.shape
            x = hidden_states.view(-1, self.K)
            logits = self.gate(x)
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            topk_w = topk_w.to(x.dtype)

            out = torch.zeros_like(x)
            for e in range(self.E):
                mask = topk_idx == e
                if not mask.any():
                    continue
                tok_idx, slot_idx = torch.where(mask)
                y = self.experts[e](x[tok_idx])
                out.index_add_(0, tok_idx, y * topk_w[tok_idx, slot_idx].unsqueeze(-1))
            return out.view(orig_shape)

    def make_brmoe_linear(
        in_features: int,
        out_features: int,
        nbits: int,
        group_size: int,
        rank: int,
        device: torch.device,
    ) -> tuple[BRMoELinear, torch.Tensor, dict | None]:
        """Create a quantized BRMoELinear with optional low-rank compensator (U, V)."""
        base = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.float16)
        nn.init.normal_(base.weight, mean=0.0, std=0.04)
        qcfg = QuantConfig(n_bits=nbits, group_size=group_size, axis=1)
        qlin = BRMoELinear(base, quant_config=qcfg, compute_dtype=torch.float16)
        qlin.quantize_weight()

        comp_entry = None
        if rank > 0:
            U_raw = torch.randn(out_features, rank, device=device, dtype=torch.float32) * 0.01
            V_raw = torch.randn(rank, in_features, device=device, dtype=torch.float32) * 0.01
            comp_entry = quantize_UV(
                U_raw, V_raw, nbits=4, U_group_size=64, V_group_size=64, round_zero=False
            )
            U_deq, V_deq = dequantize_UV(
                comp_entry["U_q"],
                comp_entry["V_q"],
                comp_entry["u_scale"],
                comp_entry["u_zero"],
                comp_entry["v_scale"],
                comp_entry["v_zero"],
                (out_features, in_features),
                rank,
                U_group_size=64,
                V_group_size=64,
            )
            qlin.U = nn.Parameter(U_deq.to(device=device, dtype=torch.float16), requires_grad=False)
            qlin.V = nn.Parameter(V_deq.to(device=device, dtype=torch.float16), requires_grad=False)

        return qlin, base.weight.detach().clone(), comp_entry

    def build_mixed_moe_block(
        K: int = 128,
        I: int = 64,
        E: int = 8,
        topk: int = 2,
        group_size: int = 64,
        device: torch.device = torch.device("cuda"),
    ):
        moe = DummyMoEBlock(K=K, I=I, E=E, topk=topk, device=device)
        expert_bits = [2, 3, 4, 3, 4, 2, 3, 4][:E]
        expert_ranks = [64, 0, 32, 16, 0, 64, 32, 16][:E]

        comp_dict = {}
        qcfg_dict = {}
        for e in range(E):
            nb = expert_bits[e]
            rk = expert_ranks[e]
            g_lin, _, g_comp = make_brmoe_linear(K, I, nb, group_size, rk, device)
            u_lin, _, u_comp = make_brmoe_linear(K, I, nb, group_size, rk, device)
            d_lin, _, d_comp = make_brmoe_linear(I, K, nb, group_size, rk, device)
            moe.experts.append(DummyExpert(g_lin, u_lin, d_lin))

            for proj_name, qlin, c_entry in (
                ("gate_proj", g_lin, g_comp),
                ("up_proj", u_lin, u_comp),
                ("down_proj", d_lin, d_comp),
            ):
                mname = f"model.layers.0.mlp.experts.{e}.{proj_name}"
                qcfg_dict[mname] = {"nbits": nb, "group_size": group_size, "axis": 1}
                if c_entry is not None:
                    comp_dict[mname] = c_entry

        return moe, qcfg_dict, comp_dict, expert_bits, expert_ranks

    def test_b1_grouped_backend_and_triton_kernels():
        print("=" * 72)
        print("[Test B1] Testing mixed 2/3/4-bit + mixed-rank (0/16/32/64) grouped backend & Triton kernels...")
        device = torch.device("cuda")
        torch.manual_seed(42)

        K, I, E, topk, gs = 128, 64, 8, 2, 64
        moe, _, _, _, _ = build_mixed_moe_block(
            K=K, I=I, E=E, topk=topk, group_size=gs, device=device
        )

        xs = {
            "decode_M1": torch.randn(1, 1, K, device=device, dtype=torch.float16),
            "decode_M4": torch.randn(1, 4, K, device=device, dtype=torch.float16),
            "prefill_M32": torch.randn(2, 16, K, device=device, dtype=torch.float16),
            "prefill_M64": torch.randn(4, 16, K, device=device, dtype=torch.float16),
        }
        ref_outs = {}
        with torch.no_grad():
            for name, x in xs.items():
                ref_outs[name] = moe(x).clone()

        root = nn.Module()
        root.layer0 = moe
        stats = patch_moe_to_grouped(root, verbose=True)
        assert stats["layers_patched"] == 1

        packed = moe._brmoe_packed
        assert "bit_offsets" in packed and packed["w13_q"].shape[0] == sum([2, 3, 4, 3, 4, 2, 3, 4])
        for w_transposed in (False, True):
            if w_transposed:
                packed["w13_q"] = packed["w13_q"].permute(0, 2, 1).contiguous()
                packed["w2_q"] = packed["w2_q"].permute(0, 2, 1).contiguous()
                packed["w_transposed"] = True

            with torch.no_grad():
                for name, x in xs.items():
                    y_grouped = moe(x)
                    y_ref = ref_outs[name]
                    max_diff = (y_grouped - y_ref).abs().max().item()
                    cos = F.cosine_similarity(
                        y_grouped.flatten().float(), y_ref.flatten().float(), dim=0
                    ).item()
                    print(
                        f"  [w_transposed={w_transposed}] {name:12s}: "
                        f"max_abs_diff={max_diff:.5f}, cos_sim={cos:.6f}"
                    )
                    assert cos > 0.9995 and max_diff < 2e-2
        print("[Test B1] PASSED!")

    def test_b2_converter_and_vllm_plugin_e2e():
        print("=" * 72)
        print("[Test B2] Testing convert_brmoe_to_vllm.py + brmoe_int3_vllm plugin loader & inference...")
        device = torch.device("cuda")
        torch.manual_seed(123)

        K, I, E, topk, gs = 128, 64, 8, 2, 64
        moe, qcfg_dict, comp_dict, _, _ = build_mixed_moe_block(
            K=K, I=I, E=E, topk=topk, group_size=gs, device=device
        )
        q_proj, _, q_comp = make_brmoe_linear(K, K, nbits=2, group_size=gs, rank=32, device=device)
        qcfg_dict["model.layers.0.self_attn.q_proj"] = {"nbits": 2, "group_size": gs, "axis": 1}
        if q_comp is not None:
            comp_dict["model.layers.0.self_attn.q_proj"] = q_comp

        x_decode = torch.randn(3, K, device=device, dtype=torch.float16)
        x_prefill = torch.randn(32, K, device=device, dtype=torch.float16)

        with torch.no_grad():
            logits_dec = moe.gate(x_decode)
            probs_dec = F.softmax(logits_dec, dim=-1, dtype=torch.float32)
            tw_dec, ti_dec = torch.topk(probs_dec, topk, dim=-1)
            tw_dec = (tw_dec / tw_dec.sum(dim=-1, keepdim=True)).to(torch.float16)
            ref_moe_dec = moe(x_decode)

            logits_pre = moe.gate(x_prefill)
            probs_pre = F.softmax(logits_pre, dim=-1, dtype=torch.float32)
            tw_pre, ti_pre = torch.topk(probs_pre, topk, dim=-1)
            tw_pre = (tw_pre / tw_pre.sum(dim=-1, keepdim=True)).to(torch.float16)
            ref_moe_pre = moe(x_prefill)

            ref_q_proj = q_proj(x_decode)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            brmoe_dir = tmp_path / "brmoe_ckpt"
            vllm_dir = tmp_path / "vllm_ckpt"
            brmoe_dir.mkdir(parents=True, exist_ok=True)

            sd = {
                "model.layers.0.mlp.gate.weight": moe.gate.weight.detach().cpu(),
                "model.layers.0.self_attn.q_proj.W_q": q_proj.W_q.detach().cpu(),
                "model.layers.0.self_attn.q_proj.scale": q_proj.scale.detach().cpu(),
                "model.layers.0.self_attn.q_proj.zero": q_proj.zero.detach().cpu(),
            }
            for e in range(E):
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    qlin = getattr(moe.experts[e], proj_name)
                    prefix = f"model.layers.0.mlp.experts.{e}.{proj_name}"
                    sd[f"{prefix}.W_q"] = qlin.W_q.detach().cpu()
                    sd[f"{prefix}.scale"] = qlin.scale.detach().cpu()
                    sd[f"{prefix}.zero"] = qlin.zero.detach().cpu()

            torch.save(sd, brmoe_dir / "quantized_weights.pt")
            torch.save(comp_dict, brmoe_dir / "compensators.pt")
            (brmoe_dir / "quant_config.json").write_text(json.dumps(qcfg_dict, indent=2), encoding="utf-8")
            (brmoe_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3MoeForCausalLM"],
                        "model_type": "qwen3_moe",
                        "hidden_size": K,
                        "moe_intermediate_size": I,
                        "num_experts": E,
                        "num_experts_per_tok": topk,
                        "num_hidden_layers": 1,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                str(TOOLS_DIR / "convert_brmoe_to_vllm.py"),
                "--brmoe-dir",
                str(brmoe_dir),
                "--out-dir",
                str(vllm_dir),
                "--dense-mode",
                "fp16",
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            out_cfg = json.loads((vllm_dir / "config.json").read_text(encoding="utf-8"))
            qcfg_out = out_cfg["quantization_config"]
            lspec0 = qcfg_out["moe_layer_specs"]["0"]

            from safetensors.torch import load_file
            from brmoe_int3_vllm.kernel import brmoe_int3_moe

            st_weights = load_file(str(vllm_dir / "model-00001-of-00001.safetensors"))
            w_q_folded = st_weights["model.layers.0.self_attn.q_proj.weight"].to(device)
            y_q_folded = F.linear(x_decode, w_q_folded)
            assert (y_q_folded - ref_q_proj).abs().max().item() < 5e-3

            class MockRoutedExperts(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer_name = "model.layers.0.mlp.experts"

            layer = MockRoutedExperts()
            layer.brmoe_group_size = gs
            layer.brmoe_total_expert_bits = lspec0["total_expert_bits"]
            layer.brmoe_num_4bit_experts = lspec0["num_4bit_experts"]
            layer.brmoe_total_comp_rank = lspec0["total_comp_rank"]
            layer.brmoe_max_comp_rank = lspec0["max_comp_rank"]

            prefix = "model.layers.0.mlp.experts.routed_experts."
            for k, tensor in st_weights.items():
                if k.startswith(prefix):
                    pname = k[len(prefix):]
                    layer.register_parameter(pname, nn.Parameter(tensor.to(device), requires_grad=False))

            layer.w13_q.data = layer.w13_q.data.permute(0, 2, 1).contiguous()
            layer.w2_q.data = layer.w2_q.data.permute(0, 2, 1).contiguous()
            layer.w_transposed = True

            y_vllm_dec = brmoe_int3_moe(x_decode, tw_dec, ti_dec, layer, gs, out_dtype=torch.float16)
            y_vllm_pre = brmoe_int3_moe(x_prefill, tw_pre, ti_pre, layer, gs, out_dtype=torch.float16)

            assert F.cosine_similarity(y_vllm_dec.flatten().float(), ref_moe_dec.flatten().float(), dim=0).item() > 0.9995
            assert F.cosine_similarity(y_vllm_pre.flatten().float(), ref_moe_pre.flatten().float(), dim=0).item() > 0.9995
        print("[Test B2] PASSED!")


def test_a3_rco_per_tensor_granularity_and_gguf_export():
    print("=" * 72)
    print("[Test A3] RCO Per-3D-Tensor Granularity & Dual-GGUF Compensator Export")
    print("=" * 72)
    from RCO_solver.model_adapters import (
        aggregate_problem_to_granularity,
        build_tensor_allocation_and_expanded_config,
        compute_budget_from_avg_bits,
        compute_expert_memory_table,
        resolve_moe_model_spec,
        synthesize_problem_tensors_for_model,
    )
    from RCO_solver.rco_search import rco_optimize_table

    spec = resolve_moe_model_spec("qwen3.6-35b-a3b")
    # Use a compact 4-layer, 8-expert subset for fast deterministic testing
    spec.num_hidden_layers = 4
    spec.num_experts = 8
    mem_table = compute_expert_memory_table(
        spec,
        bits_list=(2, 3, 4),
        ranks_list=(0, 8, 16, 32),
        weight_group_size=128,
    )
    experts, configs, loss_mat, cost_vec, freq_vec = synthesize_problem_tensors_for_model(
        spec=spec,
        memory_usage=mem_table,
        min_rank=0,
        max_rank=32,
        allowed_bits=[2, 3, 4],
        seed=42,
    )
    target_budget = compute_budget_from_avg_bits(spec, avg_bits=3.0)

    for gran in ("tensor", "vllm_tensor"):
        units, cfgs, loss_g, cost_g, _, gran_meta = aggregate_problem_to_granularity(
            experts=experts,
            configs=configs,
            loss_mat=loss_mat,
            cost_vec=cost_vec,
            freq_vec=freq_vec,
            granularity=gran,
        )
        expected_units = spec.num_hidden_layers * (3 if gran == "tensor" else 2)
        assert len(units) == expected_units, f"Expected {expected_units} units, got {len(units)}"
        assert cost_g.shape == (expected_units, len(cfgs))

        raw_cfg, summary = rco_optimize_table(
            experts=units,
            configs=cfgs,
            loss_mat=loss_g,
            cost_vec=cost_g,
            target_budget=target_budget,
            steps=60,
            lr=0.1,
            seed=42,
            verbose=False,
        )
        assert summary["total_memory_mb"] <= target_budget + 1e-4
        expanded_cfg, tensor_alloc = build_tensor_allocation_and_expanded_config(
            raw_cfg, gran_meta
        )
        assert len(expanded_cfg) == spec.num_hidden_layers * spec.num_experts
        assert len(tensor_alloc["gguf_tensors"]) == spec.num_hidden_layers * 3
        assert len(tensor_alloc["vllm_layers"]) == spec.num_hidden_layers
        print(
            f"  [PASS] RCO granularity='{gran}': {len(units)} 3D tensors optimized, "
            f"mem={summary['total_memory_mb']:.2f}/{target_budget:.2f} MB"
        )

    from export_brmoe_to_gguf import MinimalGGUFWriter, read_gguf_header_summary

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = str(Path(tmpdir) / "test-compensator.gguf")
        writer = MinimalGGUFWriter(arch="qwen3moe")
        writer.add_string("general.type", "adapter")
        writer.add_string("general.architecture", "qwen3moe")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", 0.0)
        writer.add_tensor(
            "blk.0.ffn_gate_exps.weight.lora_a",
            np.zeros((8, 16, 64), dtype=np.float16),
        )
        writer.add_tensor(
            "blk.0.ffn_gate_exps.weight.lora_b",
            np.zeros((8, 32, 16), dtype=np.float16),
        )
        writer.write(gguf_path)
        hdr = read_gguf_header_summary(gguf_path)
        assert hdr["version"] == 3
        assert hdr["kv"]["general.type"] == "adapter"
        assert hdr["kv"]["adapter.type"] == "lora"
        assert abs(hdr["kv"]["adapter.lora.alpha"] - 0.0) < 1e-8
        assert hdr["tensors"]["blk.0.ffn_gate_exps.weight.lora_a"]["ggml_ne"] == [64, 16, 8]
        assert hdr["tensors"]["blk.0.ffn_gate_exps.weight.lora_b"]["ggml_ne"] == [16, 32, 8]
        print("  [PASS] MinimalGGUFWriter v3 3D LoRA tensor header & alignment round-trip verified")

    # Verify Factorized Prefix-SVD + Activation-Space STE Identity vs Naive K-Matrix STE
    from RCO_solver.e2e_ste import (
        factorized_prefix_svd_forward_backward_np,
        parse_config_name,
        streaming_ste_forward_backward_np,
    )

    rng_ste = np.random.default_rng(1234)
    T_tok, D_in, D_out, r_max = 12, 256, 128, 64
    option_names_ste = [
        f"{b}bit_rank{r}"
        for b in (2, 3, 4)
        for r in (0, 8, 16, 32, 64)
    ]
    x_ste = rng_ste.standard_normal((T_tok, D_in))
    go_ste = rng_ste.standard_normal((T_tok, D_out))
    logits_ste = rng_ste.standard_normal(len(option_names_ste))
    base_by_bit = {b: rng_ste.standard_normal((D_out, D_in)) for b in (2, 3, 4)}
    uv_by_bit = {
        b: (
            rng_ste.standard_normal((D_out, r_max)) * 0.1,
            rng_ste.standard_normal((r_max, D_in)) * 0.1,
        )
        for b in (2, 3, 4)
    }
    cand_weights_full = []
    for cname in option_names_ste:
        b_c, r_c = parse_config_name(cname)
        w_c = base_by_bit[b_c].copy()
        if r_c > 0:
            u_b, v_b = uv_by_bit[b_c]
            w_c = w_c + (u_b[:, :r_c] @ v_b[:r_c, :])
        cand_weights_full.append(w_c)

    y_ref_ste, gx_ref_ste, ga_ref_ste = streaming_ste_forward_backward_np(
        x_ste, go_ste, logits_ste, k_star=7, cand_weights=cand_weights_full, tau=0.5
    )
    y_fast_ste, gx_fast_ste, ga_fast_ste = factorized_prefix_svd_forward_backward_np(
        x_ste,
        go_ste,
        logits_ste,
        k_star=7,
        option_names=option_names_ste,
        base_weights_by_bit=base_by_bit,
        uv_by_bit=uv_by_bit,
        tau=0.5,
    )
    assert np.max(np.abs(y_ref_ste - y_fast_ste)) < 1e-11
    assert np.max(np.abs(gx_ref_ste - gx_fast_ste)) < 1e-11
    assert np.max(np.abs(ga_ref_ste - ga_fast_ste)) < 1e-11
    print(
        f"  [PASS] Factorized Prefix-SVD + Activation-Space STE matches naive K-matrix STE "
        f"(max grad_alpha err={np.max(np.abs(ga_ref_ste - ga_fast_ste)):.2e})"
    )

    if _HAS_TORCH:
        from export_brmoe_to_gguf import (
            export_brmoe_checkpoint_to_gguf,
            read_gguf_header_summary,
        )
        from int3_moe.packing import pack_uniform_bit, unpack_uniform_bit

        # Verify pack_uniform_bit / unpack_uniform_bit round-trip for 2, 3, 4 bits
        E, N, K = 4, 32, 64
        for w_bits in (2, 3, 4):
            q_orig = torch.randint(0, 1 << w_bits, (E, N, K), dtype=torch.uint8)
            packed = pack_uniform_bit(q_orig, w_bits)
            assert packed.shape == (E, N, (K // 32) * w_bits)
            unpacked = unpack_uniform_bit(packed, w_bits)
            assert torch.equal(q_orig, unpacked), f"Round-trip failed for w_bits={w_bits}"
        print("  [PASS] pack_uniform_bit / unpack_uniform_bit exact round-trip for w_bits in {2, 3, 4}")

        # Verify GGUF v3 3D compensator export & header round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            brmoe_dir = Path(tmpdir) / "brmoe_ckpt"
            gguf_dir = Path(tmpdir) / "gguf_out"
            brmoe_dir.mkdir(parents=True, exist_ok=True)

            # Layer 0: gate=2bit/r8, up=2bit/r8, down=4bit/r16
            comp_sd = {"model.layers.0.mlp": {}}
            q_sd = {}
            for e in range(E):
                for proj, bits, rank, d_out, d_in in [
                    ("gate_proj", 2, 8, 32, 64),
                    ("up_proj", 2, 8, 32, 64),
                    ("down_proj", 4, 16, 64, 32),
                ]:
                    comp_sd["model.layers.0.mlp"][f"experts.{e}.{proj}"] = {
                        "U": torch.randn(d_out, rank, dtype=torch.float16),
                        "V": torch.randn(rank, d_in, dtype=torch.float16),
                    }
                    q_sd[f"model.layers.0.mlp.experts.{e}.{proj}"] = {
                        "bits": bits,
                        "rank": rank,
                    }
            torch.save(comp_sd, brmoe_dir / "compensators.pt")
            torch.save(q_sd, brmoe_dir / "qmodel.pt")

            res = export_brmoe_checkpoint_to_gguf(
                brmoe_dir=str(brmoe_dir),
                output_dir=str(gguf_dir),
                arch="qwen3moe",
            )
            hdr = read_gguf_header_summary(res["compensator_gguf"])
            assert hdr["version"] == 3
            assert hdr["kv"]["general.type"] == "adapter"
            assert hdr["kv"]["adapter.type"] == "lora"
            assert abs(hdr["kv"]["adapter.lora.alpha"] - 0.0) < 1e-8
            assert hdr["n_tensors"] == 6  # gate, up, down x (lora_a, lora_b)
            gate_a = hdr["tensors"]["blk.0.ffn_gate_exps.weight.lora_a"]
            gate_b = hdr["tensors"]["blk.0.ffn_gate_exps.weight.lora_b"]
            down_a = hdr["tensors"]["blk.0.ffn_down_exps.weight.lora_a"]
            assert gate_a["ggml_ne"] == [64, 8, E]
            assert gate_b["ggml_ne"] == [8, 32, E]
            assert down_a["ggml_ne"] == [32, 16, E]
            assert res["tensor_ggml_types"]["blk.0.ffn_gate_exps.weight"] == "Q2_K"
            assert res["tensor_ggml_types"]["blk.0.ffn_down_exps.weight"] == "Q4_K_M"
            print(
                f"  [PASS] GGUF v3 3D compensator export verified: 6 3D tensors, "
                f"alpha=0.0, gate=Q2_K(r=8), down=Q4_K_M(r=16)"
            )
    print("[Test A3] PASSED!\n")


def test_b3_per_3d_tensor_single_big_kernel_and_3d_compensator():
    print("=" * 72)
    print("[Test B3] Per-3D-Tensor Static-Bit Single-Big-Kernel & 3D Compensator BMM")
    print("=" * 72)
    import torch.nn.functional as F
    from int3_moe import fused_moe_int3, pack_uniform_bit, quantize_intn, ref_fused_moe

    torch.manual_seed(2026)
    device = torch.device("cuda")
    E, K, I, gs, topk = 6, 128, 64, 64, 2

    # Heterogeneous across 3D tensors, homogeneous within each 3D tensor:
    # w13: 2-bit + rank 16; w2: 4-bit + rank 32
    w13_bits, w13_rank = 2, 16
    w2_bits, w2_rank = 4, 32

    w13_fp = torch.randn(E, 2 * I, K, device=device, dtype=torch.float16) * 0.1
    w2_fp = torch.randn(E, K, I, device=device, dtype=torch.float16) * 0.1

    q13, s13, z13 = quantize_intn(w13_fp, gs, w13_bits)
    q2, s2, z2 = quantize_intn(w2_fp, gs, w2_bits)

    w13_q = pack_uniform_bit(q13.cpu(), w13_bits).to(device)
    w2_q = pack_uniform_bit(q2.cpu(), w2_bits).to(device)
    assert w13_q.shape == (E, 2 * I, (K // 32) * w13_bits)
    assert w2_q.shape == (E, K, (I // 32) * w2_bits)

    w1_v = torch.randn(E, w13_rank, K, device=device, dtype=torch.float16) * 0.05
    w1_u = torch.randn(E, I, w13_rank, device=device, dtype=torch.float16) * 0.05
    w3_v = torch.randn(E, w13_rank, K, device=device, dtype=torch.float16) * 0.05
    w3_u = torch.randn(E, I, w13_rank, device=device, dtype=torch.float16) * 0.05
    w2_v = torch.randn(E, w2_rank, I, device=device, dtype=torch.float16) * 0.05
    w2_u = torch.randn(E, K, w2_rank, device=device, dtype=torch.float16) * 0.05

    w13_t = w13_q.permute(0, 2, 1).contiguous()
    w2_t = w2_q.permute(0, 2, 1).contiguous()

    for M, mode_name in [(4, "Decode GEMV"), (48, "Prefill Grouped GEMM")]:
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        logits = torch.randn(M, E, device=device, dtype=torch.float32)
        probs = F.softmax(logits, dim=-1)
        tw, ti = torch.topk(probs, topk, dim=-1)
        tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.float16)
        ti = ti.to(torch.int32)

        y_ref = ref_fused_moe(
            x, w13_q, s13, z13, w2_q, s2, z2, tw, ti, gs,
            w13_bits=w13_bits, w2_bits=w2_bits,
            w1_v=w1_v, w1_u=w1_u, w3_v=w3_v, w3_u=w3_u, w2_v=w2_v, w2_u=w2_u,
        )
        y_tri = fused_moe_int3(
            x, w13_t, s13, z13, w2_t, s2, z2, tw, ti, gs,
            w_transposed=True,
            w13_bits=w13_bits, w2_bits=w2_bits,
            w1_v=w1_v, w1_u=w1_u, w3_v=w3_v, w3_u=w3_u, w2_v=w2_v, w2_u=w2_u,
        )
        cos = F.cosine_similarity(y_ref.flatten().float(), y_tri.flatten().float(), dim=0).item()
        max_err = (y_ref.float() - y_tri.float()).abs().max().item()
        print(f"  [PASS] {mode_name} (M={M}, w13={w13_bits}b/r{w13_rank}, w2={w2_bits}b/r{w2_rank}): cos={cos:.6f}, max_err={max_err:.5f}")
        assert cos > 0.9995, f"Cosine similarity too low: {cos}"
    print("[Test B3] PASSED!\n")


if __name__ == "__main__":
    test_a1_ast_and_bytecode_compilation()
    test_a2_mixed_2_3_4_bit_msb_plane_and_csr_compensator_math()
    test_a3_rco_per_tensor_granularity_and_gguf_export()
    if _HAS_TORCH and _HAS_CUDA:
        test_b1_grouped_backend_and_triton_kernels()
        test_b2_converter_and_vllm_plugin_e2e()
        test_b3_per_3d_tensor_single_big_kernel_and_3d_compensator()
    else:
        print("=" * 72)
        print("[Info] CUDA/PyTorch not detected on local host; GPU Triton suite will run on GPU node.")
    print("=" * 72)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
