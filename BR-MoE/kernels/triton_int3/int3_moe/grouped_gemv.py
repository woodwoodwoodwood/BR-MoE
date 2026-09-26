"""Experimental small-batch W3A16 MoE, sharing dequantization across routes.

One GPU sort forms groups of at most rows routes to the same expert. Neither
sorting nor launch geometry reads GPU scalars on the host. The intermediate
stays in original route order, including when routing changes during replay.
This module is opt-in; production dispatch is not changed by importing it.
"""
import torch
import triton
import triton.language as tl

from .kernel import silu_mul_routes


@triton.jit
def _group_routes(IDS, SORTED, STARTS, EXPERTS, COUNT,
                  TOTAL: tl.constexpr, E: tl.constexpr,
                  ROWS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    e = tl.load(IDS + i, i < TOTAL, other=E).to(tl.int32)
    key = tl.where((i < TOTAL) & (e >= 0) & (e < E), e * TOTAL + i, E * TOTAL)
    key = tl.sort(key, descending=False)
    expert = key // TOTAL
    prev = tl.gather(expert, tl.maximum(i - 1, 0), axis=0)
    first = (i == 0) | (expert != prev)
    start = tl.associative_scan(tl.where(first, i, 0), 0, _max_int)
    leader = ((i - start) % ROWS == 0) & (expert < E) & (i < TOTAL)
    group = tl.cumsum(leader.to(tl.int32), 0) - 1
    tl.store(SORTED + i, key, i < TOTAL)
    tl.store(STARTS + group, i, leader)
    tl.store(EXPERTS + group, expert, leader)
    tl.store(COUNT, tl.sum(leader.to(tl.int32), 0))


@triton.jit
def _max_int(a, b):
    return tl.maximum(a, b)


@triton.jit
def _grouped_tile(A, W, S, Z, SORTED, STARTS, EXPERTS, COUNT, RW, OUT,
                  PID, NID, SPLIT, NSPLITS: tl.constexpr,
                  TOTAL: tl.constexpr, TOPK: tl.constexpr,
                  K: tl.constexpr, N: tl.constexpr, GS: tl.constexpr,
                  W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                  ADD: tl.constexpr, ROWS: tl.constexpr,
                  BN: tl.constexpr, GROUPS: tl.constexpr):
    pid = PID
    if pid >= tl.load(COUNT):
        return
    expert = tl.load(EXPERTS + pid)
    start = tl.load(STARTS + pid)
    r = tl.arange(0, ROWS)
    key = tl.load(SORTED + start + r, start + r < TOTAL, other=-1)
    valid = (start + r < TOTAL) & (key // TOTAL == expert)
    route = key % TOTAL
    token = route // TOPK
    ar = route if ADD else token
    n = NID * BN + tl.arange(0, BN)
    group = tl.arange(0, GROUPS)
    k = tl.arange(0, 32)
    STEP: tl.constexpr = GROUPS * 32
    per = tl.cdiv(tl.cdiv(K, STEP), NSPLITS) * STEP
    kb = SPLIT * per
    ke = tl.minimum(kb + per, K)
    acc = tl.full((ROWS, BN), 0, tl.float32)
    for kk in range(kb, ke, STEP):
        gk = kk + group * 32
        mask = (gk[:, None] < ke) & (n[None, :] < N)
        if W_T:
            base = W + expert * (K // 32 * 3 * N) + (gk[:, None] // 32 * 3) * N + n[None, :]
            step = N
        else:
            base = W + expert * (K // 32 * 3 * N) + n[None, :] * (K // 32 * 3) + gk[:, None] // 32 * 3
            step = 1
        w0 = tl.load(base, mask, other=0)
        w1 = tl.load(base + step, mask, other=0)
        w2 = tl.load(base + 2 * step, mask, other=0)
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        wid = (k // 8)[None, :, None]
        w = tl.where(wid == 1, w1[:, None, :], tl.where(wid == 2, w2[:, None, :],
                     tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
        shift = ((k % 4) * 3 + ((k % 8) // 4) * 16)[None, :, None]
        q = (w >> shift) & 7
        sz = expert * (K // GS * N) + (gk[:, None] // GS) * N + n[None, :]
        scale = tl.load(S + sz, mask, other=0)
        if HAS_ZERO:
            zero = tl.load(Z + sz, mask, other=0)
            b = (q.to(tl.float16) - zero[:, None, :]) * scale[:, None, :]
        else:
            b = (q - 4).to(tl.float16) * scale[:, None, :]
        ak = gk[:, None] + k[None, :]
        a = tl.load(A + ar[:, None, None] * K + ak[None, :, :],
                    valid[:, None, None] & (ak[None, :, :] < ke), other=0)
        prod = a[:, :, :, None].to(tl.float32) * b[None, :, :, :].to(tl.float32)
        acc += tl.sum(tl.sum(prod, 2), 1)
    if ADD:
        rw = tl.load(RW + route, valid, other=0).to(tl.float32)
        tl.atomic_add(OUT + token[:, None] * N + n[None, :], acc * rw[:, None],
                      valid[:, None] & (n[None, :] < N), sem="relaxed")
    else:
        tl.atomic_add(OUT + route[:, None] * N + n[None, :], acc,
                      valid[:, None] & (n[None, :] < N), sem="relaxed")


@triton.jit
def _grouped_gemv(A, W, S, Z, SORTED, STARTS, EXPERTS, COUNT, RW, OUT,
                  TOTAL: tl.constexpr, TOPK: tl.constexpr,
                  K: tl.constexpr, N: tl.constexpr, GS: tl.constexpr,
                  W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                  ADD: tl.constexpr, ROWS: tl.constexpr,
                  BN: tl.constexpr, GROUPS: tl.constexpr, KSPLIT: tl.constexpr):
    count = tl.load(COUNT)
    tiles_n = tl.cdiv(N, BN)
    for tile in range(tl.program_id(0), count * tiles_n * KSPLIT, tl.num_programs(0)):
        pid = tile % count
        nid = (tile // count) % tiles_n
        split = tile // (count * tiles_n)
        _grouped_tile(A, W, S, Z, SORTED, STARTS, EXPERTS, COUNT, RW, OUT,
                      pid, nid, split, KSPLIT,
                      TOTAL, TOPK, K, N, GS, W_T, HAS_ZERO, ADD, ROWS, BN, GROUPS)


_WORKSPACE = {}


def grouped_gemv_config(m):
    """5090 settings measured on full-model decode routes (M=4..16)."""
    if not 4 <= m <= 16:
        raise ValueError('grouped GEMV policy is only measured for M=4..16')
    return dict(rows=4 if m <= 4 else 8, block_n=128 if m <= 8 else 256,
                groups=1, ksplit=16 if m <= 4 else 8, num_warps=4)


@torch.no_grad()
def fused_moe_grouped_gemv(x, weights, ids, packed, *, rows=4, block_n=32,
                            groups=2, ksplit=4, num_warps=4,
                            num_stages=1, out_dtype=None):
    """Complete MoE: grouping, zeroing, both GEMVs, activation, reduction/cast.

    rows/block_n/groups must be powers of two. Split-K uses FP32 atomics;
    group membership is rebuilt on every invocation and every graph replay.
    The workspace has the same single-stream ownership as int3_moe.ops.
    """
    assert x.is_contiguous() and ids.is_contiguous() and weights.is_contiguous()
    assert x.dtype == torch.float16 and ids.shape == weights.shape
    assert packed.get('layout', 'int3') == 'int3'
    assert all(v > 0 and v & (v - 1) == 0 for v in (rows, block_n, groups))
    m, k = x.shape
    total, topk = ids.numel(), ids.shape[1]
    assert total > 0 and ksplit > 0
    e, _, two_i = packed['s13'].shape
    i = two_i // 2
    key = (m, k, i, total, str(x.device))
    if key not in _WORKSPACE:
        _WORKSPACE[key] = dict(
            sorted=torch.empty(total, dtype=torch.int32, device=x.device),
            starts=torch.empty(total, dtype=torch.int32, device=x.device),
            experts=torch.empty(total, dtype=torch.int32, device=x.device),
            count=torch.empty((), dtype=torch.int32, device=x.device),
            inter=torch.empty(total, two_i, dtype=torch.float32, device=x.device),
            act=torch.empty(total, i, dtype=torch.float16, device=x.device),
            out=torch.empty(m, k, dtype=torch.float32, device=x.device))
    ws = _WORKSPACE[key]
    _group_routes[(1,)](ids, ws['sorted'], ws['starts'], ws['experts'], ws['count'],
                        TOTAL=total, E=e, ROWS=rows, BLOCK=triton.next_power_of_2(total),
                        num_warps=4)
    ws['inter'].zero_()
    ws['out'].zero_()

    def launch(a, w, s, z, out, reduction, n, add, splits):
        _grouped_gemv[(min(total * triton.cdiv(n, block_n) * splits,
                              torch.cuda.get_device_properties(x.device).multi_processor_count * 4),)](
            a, w, s, z if z is not None else s, ws['sorted'], ws['starts'],
            ws['experts'], ws['count'], weights, out,
            TOTAL=total, TOPK=topk, K=reduction, N=n, GS=packed['group_size'],
            W_T=packed.get('w_transposed', False), HAS_ZERO=z is not None,
            ADD=add, ROWS=rows, BN=block_n, GROUPS=groups, KSPLIT=splits,
            num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)

    launch(x, packed['w13_q'], packed['s13'], packed.get('z13'), ws['inter'],
           k, two_i, False, ksplit)
    silu_mul_routes(ws['inter'], ws['act'], i, total)
    launch(ws['act'], packed['w2_q'], packed['s2'], packed.get('z2'), ws['out'],
           i, k, True, max(1, ksplit // 2))
    return ws['out'].to(out_dtype or x.dtype)
