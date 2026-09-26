"""Small-M W3A16 GEMV: half2 decoding, split-K stores and fused reduction.

Weights remain packed INT3. W13 keeps the original GEMV FP32 boundary before
SiLU, and W2 keeps FP32 until the final weighted reduction and output cast.
The plugin enables calibrated A100 configurations; scalar decoding and routed
row grouping remain explicit benchmark controls. Cached scratch buffers require
ordered use on one stream, as in the existing MoE workspace implementation.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _gemv_partials(A, W, S, Z, IDS, ROUTE_W, C,
                      K: tl.constexpr, N: tl.constexpr, TOP_K: tl.constexpr,
                      SW_E: tl.constexpr, SW_N: tl.constexpr, SS_E: tl.constexpr,
                      SS_K: tl.constexpr, SZ_E: tl.constexpr, SZ_K: tl.constexpr,
                      GS: tl.constexpr, W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                      ADD: tl.constexpr, ROUTES: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS: tl.constexpr):
    """Each split writes a disjoint output, including empty K splits."""
    route = tl.program_id(0)
    token = route // TOP_K
    expert = tl.load(IDS + route)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ni = n < N
    # ---- split-K: 本 program 负责的 k 区间 [kb, ke) ----
    ks = tl.program_id(2)
    nks = tl.num_programs(2)
    STEP: tl.constexpr = 32 * GROUPS
    per = tl.cdiv(tl.cdiv(K, STEP), nks) * STEP
    kb = ks * per
    ke = tl.minimum(kb + per, K)
    group = tl.arange(0, GROUPS)[:, None]
    k = tl.arange(0, 32)[None, :]
    wid = (k // 8)[:, :, None]
    ii = k % 8
    shift = ((ii % 4) * 3 + (ii // 4) * 16)[:, :, None]
    acc = tl.full((BLOCK_N,), 0, tl.float32)
    for kk in range(kb, ke, 32 * GROUPS):
        valid_group = kk + group * 32 < ke
        valid_weight = valid_group & ni[None, :]
        if W_T:
            base = W + expert * SW_E + ((kk // 32 + group) * 3 * N) + n[None, :]
            step = N
        else:
            base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 3
            step = 1
        w0 = tl.load(base, valid_weight, other=0)
        w1 = tl.load(base + step, valid_weight, other=0)
        w2 = tl.load(base + 2 * step, valid_weight, other=0)
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        w = tl.where(wid == 1, w1[:, None, :],
                     tl.where(wid == 2, w2[:, None, :],
                              tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
        q = (w >> shift) & 7
        sc = tl.load(S + expert * SS_E + ((kk + group * 32) // GS) * SS_K + n[None, :],
                     valid_weight, other=0)
        if HAS_ZERO:
            zc = tl.load(Z + expert * SZ_E + ((kk + group * 32) // GS) * SZ_K + n[None, :],
                         valid_weight, other=0)
            b = (q.to(tl.float16) - zc[:, None, :]) * sc[:, None, :]
        else:
            b = (q - 4).to(tl.float16) * sc[:, None, :]
        a = tl.load(A + (route if ADD else token) * SX0 + (kk + group * 32 + k) * SX1,
                    valid_group, other=0)
        acc += tl.sum(tl.sum(a[:, :, None].to(tl.float32) * b.to(tl.float32), axis=1), axis=0)
    tl.store(C + (ks * ROUTES + route) * N + n, acc, ni)


@triton.jit
def _gemv_partials_half2(A, W, S, Z, IDS, ROUTE_W, C,
                      K: tl.constexpr, N: tl.constexpr, TOP_K: tl.constexpr,
                      SW_E: tl.constexpr, SW_N: tl.constexpr, SS_E: tl.constexpr,
                      SS_K: tl.constexpr, SZ_E: tl.constexpr, SZ_K: tl.constexpr,
                      GS: tl.constexpr, W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                      ADD: tl.constexpr, ROUTES: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS: tl.constexpr):
    """Each split writes a disjoint output, including empty K splits."""
    route = tl.program_id(0)
    token = route // TOP_K
    expert = tl.load(IDS + route)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ni = n < N
    # ---- split-K: 本 program 负责的 k 区间 [kb, ke) ----
    ks = tl.program_id(2)
    nks = tl.num_programs(2)
    STEP: tl.constexpr = 32 * GROUPS
    per = tl.cdiv(tl.cdiv(K, STEP), nks) * STEP
    kb = ks * per
    ke = tl.minimum(kb + per, K)
    group = tl.arange(0, GROUPS)[:, None]
    pair = tl.arange(0, 16)[None, :]
    k = (pair // 4) * 8 + pair % 4
    wid = (pair // 4)[:, :, None]
    shift = ((pair % 4) * 3)[:, :, None]
    acc = tl.full((BLOCK_N,), 0, tl.float32)
    for kk in range(kb, ke, 32 * GROUPS):
        valid_group = kk + group * 32 < ke
        valid_weight = valid_group & ni[None, :]
        if W_T:
            base = W + expert * SW_E + ((kk // 32 + group) * 3 * N) + n[None, :]
            step = N
        else:
            base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 3
            step = 1
        w0 = tl.load(base, valid_weight, other=0)
        w1 = tl.load(base + step, valid_weight, other=0)
        w2 = tl.load(base + 2 * step, valid_weight, other=0)
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        w = tl.where(wid == 1, w1[:, None, :],
                     tl.where(wid == 2, w2[:, None, :],
                              tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
        sc = tl.load(S + expert * SS_E + ((kk + group * 32) // GS) * SS_K + n[None, :],
                     valid_weight, other=0)
        if HAS_ZERO:
            zc = tl.load(Z + expert * SZ_E + ((kk + group * 32) // GS) * SZ_K + n[None, :],
                         valid_weight, other=0)
        else:
            zc = tl.full((GROUPS, BLOCK_N), 4, tl.float16)
        zb = zc.to(tl.uint16, bitcast=True).to(tl.uint32) * 65537
        sb = sc.to(tl.uint16, bitcast=True).to(tl.uint32) * 65537
        b0, b1 = tl.inline_asm_elementwise(
            asm="""{
                .reg .b32 q, magic;
                and.b32 q, $2, 0x00070007;
                or.b32 q, q, 0x64006400;
                mov.b32 magic, 0x64006400;
                sub.f16x2 q, q, magic;
                sub.f16x2 q, q, $3;
                mul.f16x2 q, q, $4;
                mov.b32 {$0, $1}, q;
            }""",
            constraints="=h,=h,r,r,r", args=[w >> shift, zb[:, None, :], sb[:, None, :]],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=1)
        a0 = tl.load(A + (route if ADD else token) * SX0 + (kk + group * 32 + k) * SX1,
                     valid_group, other=0).to(tl.float32)
        a1 = tl.load(A + (route if ADD else token) * SX0 + (kk + group * 32 + k + 4) * SX1,
                     valid_group, other=0).to(tl.float32)
        product = a0[:, :, None] * b0.to(tl.float32) + a1[:, :, None] * b1.to(tl.float32)
        acc += tl.sum(tl.sum(product, axis=1), axis=0)
    tl.store(C + (ks * ROUTES + route) * N + n, acc, ni)


@triton.jit
def _gemv_partials_rows(A, W, S, Z, SORTED, STARTS, EXPERTS, COUNT, C,
                      K: tl.constexpr, N: tl.constexpr, TOP_K: tl.constexpr,
                      SW_E: tl.constexpr, SW_N: tl.constexpr, SS_E: tl.constexpr,
                      SS_K: tl.constexpr, SZ_E: tl.constexpr, SZ_K: tl.constexpr,
                      GS: tl.constexpr, W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                      ADD: tl.constexpr, ROWS: tl.constexpr, GROUPED: tl.constexpr, ROUTES: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS: tl.constexpr):
    """Each split writes a disjoint output, including empty K splits."""
    pid = tl.program_id(0)
    row = tl.arange(0, ROWS)
    if GROUPED:
        if pid >= tl.load(COUNT):
            return
        expert = tl.load(EXPERTS + pid)
        start = tl.load(STARTS + pid)
        key = tl.load(SORTED + start + row, start + row < ROUTES, other=-1)
        valid = (start + row < ROUTES) & (key // ROUTES == expert)
        route = key % ROUTES
    else:
        expert = 0
        route = pid * ROWS + row
        valid = route < ROUTES
    token = route // TOP_K
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ni = n < N
    # ---- split-K: 本 program 负责的 k 区间 [kb, ke) ----
    ks = tl.program_id(2)
    nks = tl.num_programs(2)
    STEP: tl.constexpr = 32 * GROUPS
    per = tl.cdiv(tl.cdiv(K, STEP), nks) * STEP
    kb = ks * per
    ke = tl.minimum(kb + per, K)
    group = tl.arange(0, GROUPS)[:, None]
    pair = tl.arange(0, 16)[None, :]
    k = (pair // 4) * 8 + pair % 4
    wid = (pair // 4)[:, :, None]
    shift = ((pair % 4) * 3)[:, :, None]
    acc = tl.full((ROWS, BLOCK_N), 0, tl.float32)
    for kk in range(kb, ke, 32 * GROUPS):
        valid_group = kk + group * 32 < ke
        valid_weight = valid_group & ni[None, :]
        if W_T:
            base = W + expert * SW_E + ((kk // 32 + group) * 3 * N) + n[None, :]
            step = N
        else:
            base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 3
            step = 1
        w0 = tl.load(base, valid_weight, other=0)
        w1 = tl.load(base + step, valid_weight, other=0)
        w2 = tl.load(base + 2 * step, valid_weight, other=0)
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        w = tl.where(wid == 1, w1[:, None, :],
                     tl.where(wid == 2, w2[:, None, :],
                              tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
        sc = tl.load(S + expert * SS_E + ((kk + group * 32) // GS) * SS_K + n[None, :],
                     valid_weight, other=0)
        if HAS_ZERO:
            zc = tl.load(Z + expert * SZ_E + ((kk + group * 32) // GS) * SZ_K + n[None, :],
                         valid_weight, other=0)
        else:
            zc = tl.full((GROUPS, BLOCK_N), 4, tl.float16)
        zb = zc.to(tl.uint16, bitcast=True).to(tl.uint32) * 65537
        sb = sc.to(tl.uint16, bitcast=True).to(tl.uint32) * 65537
        b0, b1 = tl.inline_asm_elementwise(
            asm="""{
                .reg .b32 q, magic;
                and.b32 q, $2, 0x00070007;
                or.b32 q, q, 0x64006400;
                mov.b32 magic, 0x64006400;
                sub.f16x2 q, q, magic;
                sub.f16x2 q, q, $3;
                mul.f16x2 q, q, $4;
                mov.b32 {$0, $1}, q;
            }""",
            constraints="=h,=h,r,r,r", args=[w >> shift, zb[:, None, :], sb[:, None, :]],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=1)
        ar = route if ADD else token
        ak = kk + group * 32 + k
        a0 = tl.load(A + ar[:, None, None] * SX0 + ak[None, :, :] * SX1,
                     valid[:, None, None] & valid_group[None, :, :], other=0).to(tl.float32)
        a1 = tl.load(A + ar[:, None, None] * SX0 + (ak[None, :, :] + 4) * SX1,
                     valid[:, None, None] & valid_group[None, :, :], other=0).to(tl.float32)
        product = a0[:, :, :, None] * b0[None, :, :, :].to(tl.float32) + a1[:, :, :, None] * b1[None, :, :, :].to(tl.float32)
        acc += tl.sum(tl.sum(product, axis=2), axis=1)
    tl.store(C + (ks * ROUTES + route[:, None]) * N + n[None, :], acc, valid[:, None] & ni[None, :])


@triton.jit
def _reduce_linear(P, O, N: tl.constexpr, M: tl.constexpr,
                   SPLITS: tl.constexpr, RS: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    s = tl.arange(0, RS)
    v = tl.load(P + (s[:, None] * M + row) * N + n[None, :],
                (s[:, None] < SPLITS) & (n[None, :] < N), other=0)
    tl.store(O + row * N + n, tl.sum(v, axis=0), n < N)


@triton.jit
def _reduce_silu(P, O, I: tl.constexpr, R: tl.constexpr,
                 SPLITS: tl.constexpr, RS: tl.constexpr, BN: tl.constexpr):
    route = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    s = tl.arange(0, RS)
    off = (s[:, None] * R + route) * (2 * I) + n[None, :]
    mask = (s[:, None] < SPLITS) & (n[None, :] < I)
    g = tl.sum(tl.load(P + off, mask, other=0), axis=0)
    u = tl.sum(tl.load(P + off + I, mask, other=0), axis=0)
    act = g * tl.sigmoid(g) * u
    tl.store(O + route * I + n, act, n < I)


@triton.jit
def _reduce_weighted(P, RW, O, N: tl.constexpr, M: tl.constexpr,
                     TOPK: tl.constexpr, SPLITS: tl.constexpr,
                     RR: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, RR)
    s = r // TOPK
    slot = r % TOPK
    rw = tl.load(RW + row * TOPK + slot, s < SPLITS, other=0)
    val = tl.load(P + (s[:, None] * M * TOPK + row * TOPK + slot[:, None]) * N + n[None, :],
                  (s[:, None] < SPLITS) & (n[None, :] < N), other=0)
    tl.store(O + row * N + n, tl.sum(val * rw[:, None], axis=0), n < N)


_WORKSPACE = {}


@triton.jit
def _sanitize_kernel(IDS, RW, OIDS, ORW, TOTAL: tl.constexpr, TOPK: tl.constexpr,
                      IS0: tl.constexpr, IS1: tl.constexpr,
                      WS0: tl.constexpr, WS1: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    expert = tl.load(IDS + (r // TOPK) * IS0 + (r % TOPK) * IS1, r < TOTAL, other=0)
    weight = tl.load(RW + (r // TOPK) * WS0 + (r % TOPK) * WS1, r < TOTAL, other=0)
    tl.store(OIDS + r, tl.maximum(expert, 0), r < TOTAL)
    tl.store(ORW + r, tl.where(expert < 0, 0, weight), r < TOTAL)


def sanitize_routing(ids, weights):
    """One kernel for negative-ID masking, int64 conversion and contiguous output."""
    assert ids.ndim == weights.ndim == 2 and ids.shape == weights.shape
    out_ids = torch.empty(ids.shape, dtype=torch.int64, device=ids.device)
    out_weights = torch.empty(weights.shape, dtype=weights.dtype, device=weights.device)
    _sanitize_kernel[(1,)](ids, weights, out_ids, out_weights, TOTAL=ids.numel(), TOPK=ids.shape[1],
                           IS0=ids.stride(0), IS1=ids.stride(1), WS0=weights.stride(0), WS1=weights.stride(1),
                           BLOCK=triton.next_power_of_2(ids.numel()), num_warps=4)
    return out_ids, out_weights


def _workspace(shape, device, dtype=torch.float32):
    key = (tuple(shape), str(device), dtype)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.empty(shape, device=device, dtype=dtype)
    return _WORKSPACE[key]


def partials(a, w, s, z, ids, topk, gs, *, gather_route=False,
             block_n=64, groups=4, splits=16, warps=2, out=None, half2=False, rows=1, grouping=None):
    e, kp, n = w.shape
    k = kp // 3 * 32
    assert kp % 3 == 0 and k == a.shape[1] and k % gs == 0
    assert w.is_contiguous() and s.is_contiguous() and (z is None or z.is_contiguous())
    assert w.dtype == torch.int32 and s.dtype == a.dtype == torch.float16
    assert s.shape == (e, k // gs, n) and gs % 32 == 0
    assert z is None or (z.shape == s.shape and z.dtype == s.dtype)
    assert groups > 0 and splits > 0 and ids.is_contiguous()
    r = ids.numel()
    if out is None:
        out = _workspace((splits, r, n), a.device)
    zero = z if z is not None else s
    kernel = _gemv_partials_half2 if half2 else _gemv_partials
    kw = {}
    group_args = [ids, a]
    grid_r = r
    if rows > 1:
        kernel = _gemv_partials_rows
        assert half2
        kw = dict(ROWS=rows, GROUPED=grouping is not None)
        if grouping is not None:
            group_args = list(grouping)
        else:
            assert e == 1 and topk == 1
            group_args = [ids, ids, ids, ids]
            grid_r = triton.cdiv(r, rows)
    kernel[(grid_r, triton.cdiv(n, block_n), splits)](
        a, w, s, zero, *group_args, out,
        K=k, N=n, TOP_K=topk, SW_E=w.stride(0), SW_N=1,
        SS_E=s.stride(0), SS_K=s.stride(1), SZ_E=zero.stride(0), SZ_K=zero.stride(1),
        GS=gs, W_T=True, HAS_ZERO=z is not None, ADD=gather_route, ROUTES=r,
        SX0=a.stride(0), SX1=a.stride(1), BLOCK_N=block_n, GROUPS=groups,
        num_warps=warps, num_stages=1, **kw)
    return out


@torch.no_grad()
def linear_gemv(x, w, s, z, gs, *, block_n=64, groups=4, splits=16, warps=2, half2=False, rows=1):
    m, k = x.shape
    n = w.shape[-1]
    key = ('ids', m, str(x.device))
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.zeros(m, dtype=torch.int32, device=x.device)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    p = partials(x, w[None], s[None], None if z is None else z[None], _WORKSPACE[key], 1, gs,
                 block_n=block_n, groups=groups, splits=splits, warps=warps,
                 out=out if splits == 1 else None, half2=half2, rows=rows)
    if splits > 1:
        _reduce_linear[(m, triton.cdiv(n, 128))](p, out, N=n, M=m, SPLITS=splits,
                                               RS=triton.next_power_of_2(splits), BN=128, num_warps=4)
    return out


@torch.no_grad()
def fused_moe_gemv(x, weights, ids, packed, *, block_n=64, groups=4,
                   splits=8, warps=2, out_dtype=torch.float16, half2=False, rows=1):
    assert packed.get('w_transposed') and packed.get('layout', 'int3') == 'int3'
    m, k = x.shape
    topk = ids.shape[1]
    i = packed['w13_q'].shape[-1] // 2
    ids = ids.reshape(-1).contiguous()
    rw = weights.reshape(-1).float().contiguous()
    grouping = None
    if rows > 1:
        from .grouped_gemv import _group_routes
        key = ('grouping', ids.numel(), rows, str(x.device))
        if key not in _WORKSPACE:
            _WORKSPACE[key] = tuple(torch.empty(ids.numel() if j < 3 else 1, device=x.device, dtype=torch.int32) for j in range(4))
        grouping = _WORKSPACE[key]
        _group_routes[(1,)](ids, *grouping, TOTAL=ids.numel(), E=packed['w13_q'].shape[0],
                            ROWS=rows, BLOCK=triton.next_power_of_2(ids.numel()), num_warps=4)
    p = partials(x, packed['w13_q'], packed['s13'], packed.get('z13'), ids, topk, packed['group_size'],
                 block_n=block_n, groups=groups, splits=splits, warps=warps, half2=half2, rows=rows, grouping=grouping)
    act = _workspace((ids.numel(), i), x.device, torch.float16)
    _reduce_silu[(ids.numel(), triton.cdiv(i, 128))](
        p, act, I=i, R=ids.numel(), SPLITS=splits, RS=triton.next_power_of_2(splits), BN=128, num_warps=4)
    ks2 = max(1, splits // 2)
    p2 = partials(act, packed['w2_q'], packed['s2'], packed.get('z2'), ids, topk, packed['group_size'],
                  gather_route=True, block_n=block_n, groups=groups, splits=ks2, warps=warps, half2=half2, rows=rows, grouping=grouping)
    out = torch.empty((m, k), device=x.device, dtype=out_dtype)
    _reduce_weighted[(m, triton.cdiv(k, 128))](
        p2, rw, out, N=k, M=m, TOPK=topk, SPLITS=ks2,
        RR=triton.next_power_of_2(topk * ks2), BN=128, num_warps=4)
    return out
