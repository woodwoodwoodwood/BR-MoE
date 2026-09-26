"""Grouped W3A16 GEMM with a full BK tile and one expert per CTA.

The checkpoint remains densely packed INT3 in K-major order. Both quantization
operations round to FP16 before the Tensor Core multiply. W13 writes FP16;
W2 retains FP32 until routing reduction, matching the Triton prefill path.
"""
import torch
import triton
import triton.language as tl

from .align_triton import moe_align_block_size_triton
from .kernel import silu_mul


@triton.jit
def _grouped_int3_tc(
    A, W, S, Z, STI, EID, META, RW, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GS: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GATHER: tl.constexpr, ATOMIC: tl.constexpr, HAS_ZERO: tl.constexpr,
):
    tile = tl.program_id(0)
    post = tl.load(META)
    if tile * BM >= post:
        return
    expert = tl.load(EID + tile)
    row = tile * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    tok = tl.load(STI + row, row < post, other=M)
    valid = (row < post) & (tok >= 0) & (tok < M)
    arow = tok if GATHER else row
    group = tl.arange(0, BK // 32)
    lane = tl.arange(0, 32)
    rk = tl.arange(0, BK)
    word = (lane // 8)[None, :, None]
    shift = ((lane % 4) * 3 + ((lane % 8) // 4) * 16)[None, :, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, tl.cdiv(K, BK)):
        kg = base * BK + group * 32
        mask = (kg[:, None] < K) & (col[None, :] < N)
        ptr = W + expert * (K // 32 * 3 * N) + (kg[:, None] // 32 * 3) * N + col[None, :]
        w0 = tl.load(ptr, mask, other=0)
        w1 = tl.load(ptr + N, mask, other=0)
        w2 = tl.load(ptr + 2 * N, mask, other=0)
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        bits = tl.where(word == 0, w0[:, None, :],
                        tl.where(word == 1, w1[:, None, :],
                                 tl.where(word == 2, w2[:, None, :], w3[:, None, :])))
        q = ((bits >> shift) & 7).to(tl.float16)
        sp = expert * (K // GS * N) + (kg[:, None] // GS) * N + col[None, :]
        scale = tl.load(S + sp, mask, other=0)
        if HAS_ZERO:
            zero = tl.load(Z + sp, mask, other=0)
            q = (q - zero[:, None, :]).to(tl.float16)
        else:
            q = (q - 4).to(tl.float16)
        b = tl.reshape((q * scale[:, None, :]).to(tl.float16), (BK, BN))
        kk = base * BK + rk
        a = tl.load(A + arow[:, None] * SX0 + kk[None, :] * SX1,
                    valid[:, None] & (kk[None, :] < K), other=0)
        acc = tl.dot(a, b, acc)
    mask_out = valid[:, None] & (col[None, :] < N)
    if ATOMIC:
        rw = tl.load(RW + row, valid, other=0).to(tl.float32)
        tl.atomic_add(C + tok[:, None] * N + col[None, :],
                      acc * rw[:, None], mask_out, sem='relaxed')
    else:
        tl.store(C + row[:, None] * N + col[None, :], acc, mask_out)


def grouped_int3_tc(a, w, scales, zeros, sti, eid, meta, out, *, m,
                    group_size, gather, route_weights=None,
                    block_m=64, block_n=64, block_k=128, num_warps=4, num_stages=3):
    """Launch with sorted rows padded to block_m and masked K/N tails."""
    e, kp, n = w.shape
    k = kp // 3 * 32
    assert kp % 3 == 0 and a.shape[1] == k
    assert a.dtype == scales.dtype == torch.float16 and w.dtype == torch.int32
    assert scales.shape == (e, k // group_size, n) and k % group_size == 0
    assert group_size % 32 == 0 and block_k >= 32 and block_k & (block_k - 1) == 0
    assert w.is_contiguous() and scales.is_contiguous()
    assert zeros is None or (zeros.shape == scales.shape and zeros.is_contiguous() and zeros.dtype == scales.dtype)
    assert route_weights is None or out.dtype == torch.float32
    _grouped_int3_tc[(triton.cdiv(sti.numel(), block_m), triton.cdiv(n, block_n))](
        a, w, scales, zeros if zeros is not None else scales, sti, eid, meta,
        route_weights if route_weights is not None else a, out,
        M=m, N=n, K=k, GS=group_size, SX0=a.stride(0), SX1=a.stride(1),
        BM=block_m, BN=block_n, BK=block_k,
        GATHER=gather, ATOMIC=route_weights is not None, HAS_ZERO=zeros is not None,
        num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return out


_WORKSPACE = {}


@triton.jit
def _reduce_routes_tc(SORTED, ROUTE_POS, RW, OUT,
                      K: tl.constexpr, TOPK: tl.constexpr,
                      RT: tl.constexpr, BN: tl.constexpr):
    """One writer per token/output; keep W2 and weighted accumulation in FP32."""
    token = tl.program_id(0)
    r = tl.arange(0, RT)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = tl.load(ROUTE_POS + token * TOPK + r, r < TOPK, other=0)
    rw = tl.load(RW + token * TOPK + r, r < TOPK, other=0).to(tl.float32)
    value = tl.load(SORTED + pos[:, None] * K + n[None, :],
                    (r[:, None] < TOPK) & (rw[:, None] != 0) & (n[None, :] < K),
                    other=0).to(tl.float32)
    result = tl.sum(value * rw[:, None], axis=0)
    tl.store(OUT + token * K + n, result, n < K)


@torch.no_grad()
def fused_moe_int3_tc(x, weights, ids, packed, *, block_m=64, block_n=64,
                       block_k=128, num_warps=4, num_stages=3,
                       reduce_topk=False, fast_align=False, out_dtype=torch.float16):
    """Complete routed MoE, with optional deterministic top-k reduction."""
    assert packed.get('w_transposed') and packed.get('layout', 'int3') == 'int3'
    m, k = x.shape
    e, _, two_i = packed['w13_q'].shape
    i, topk = two_i // 2, ids.shape[1]
    tw = weights.reshape(-1).float().contiguous()
    sti, eid, meta, buf = moe_align_block_size_triton(
        ids, e, block_m, flat_values=tw, return_route_positions=reduce_topk,
        histogram=fast_align, scatter_warps=8 if fast_align else 4)
    rows = buf['max_post']
    key = (m, k, i, rows, str(x.device))
    if key not in _WORKSPACE:
        _WORKSPACE[key] = dict(inter=torch.empty(rows, two_i, dtype=torch.float16, device=x.device),
                               act=torch.empty(rows, i, dtype=torch.float16, device=x.device))
    ws = _WORKSPACE[key]
    kw = dict(m=m, group_size=packed['group_size'], block_m=block_m, block_n=block_n,
              block_k=block_k, num_warps=num_warps, num_stages=num_stages)
    grouped_int3_tc(x, packed['w13_q'], packed['s13'], packed.get('z13'),
                    sti, eid, meta, ws['inter'], gather=True, **kw)
    silu_mul(ws['inter'], ws['act'], meta, i, rows)
    if reduce_topk:
        if 'sorted32' not in ws:
            ws['sorted32'] = torch.empty(rows, k, dtype=torch.float32, device=x.device)
        grouped_int3_tc(ws['act'], packed['w2_q'], packed['s2'], packed.get('z2'),
                        sti, eid, meta, ws['sorted32'], gather=False, **kw)
        out = torch.empty(m, k, dtype=out_dtype, device=x.device)
        _reduce_routes_tc[(m, triton.cdiv(k, 256))](
            ws['sorted32'], buf['route_pos'], tw, out, K=k, TOPK=topk,
            RT=triton.next_power_of_2(topk), BN=256, num_warps=4, enable_fp_fusion=False)
        return out
    if 'out32' not in ws:
        ws['out32'] = torch.empty(m, k, dtype=torch.float32, device=x.device)
    ws['out32'].zero_()
    grouped_int3_tc(ws['act'], packed['w2_q'], packed['s2'], packed.get('z2'),
                    sti, eid, meta, ws['out32'], gather=False, route_weights=buf['sv'], **kw)
    return ws['out32'].clone() if out_dtype == torch.float32 else ws['out32'].to(out_dtype)
