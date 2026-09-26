"""Fused GPU glue for the existing Marlin INT3 MoE matrix multiplies.

Only active padded expert blocks are processed. All row counts remain on the
device, including CUDA Graph replay. The CUDA matrix multiply implementation,
packing, and FP16 intermediate boundary are unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_tokens(X, STI, META, SORTED, M: tl.constexpr, K: tl.constexpr,
                    SX0: tl.constexpr, SX1: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    first = tl.program_id(0) * BM
    post = tl.load(META)
    if first >= post:
        return
    r = first + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    token = tl.load(STI + r, r < post, other=M)
    valid = (r < post) & (token >= 0) & (token < M)
    x = tl.load(X + token[:, None] * SX0 + n[None, :] * SX1,
                valid[:, None] & (n[None, :] < K), other=0)
    # Active padding rows must be initialized: the CUDA tile reads all 16 rows.
    tl.store(SORTED + r[:, None] * K + n[None, :], x,
              (r[:, None] < post) & (n[None, :] < K))


@triton.jit
def _silu_active(INTER, STI, META, ACT, M: tl.constexpr, I: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    first = tl.program_id(0) * BM
    post = tl.load(META)
    if first >= post:
        return
    r = first + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    token = tl.load(STI + r, r < post, other=M)
    valid = (r < post) & (token >= 0) & (token < M)
    mask = valid[:, None] & (n[None, :] < I)
    g = tl.load(INTER + r[:, None] * (2 * I) + n[None, :], mask, other=0).to(tl.float32)
    u = tl.load(INTER + r[:, None] * (2 * I) + I + n[None, :], mask, other=0).to(tl.float32)
    a = (g * tl.sigmoid(g)) * u
    tl.store(ACT + r[:, None] * I + n[None, :], a,
              (r[:, None] < post) & (n[None, :] < I))


@triton.jit
def _scatter_weighted(SORTED, STI, RW, META, OUT,
                       M: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    if row >= tl.load(META):
        return
    token = tl.load(STI + row)
    if token < 0 or token >= M:
        return
    rw = tl.load(RW + row).to(tl.float32)
    if rw == 0:
        return
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    val = tl.load(SORTED + row * K + n, n < K, other=0).to(tl.float32)
    tl.atomic_add(OUT + token * K + n, val * rw, n < K, sem="relaxed")


_WORKSPACE = {}


@triton.jit
def _reduce_routes(SORTED, ROUTE_POS, RW, OUT,
                     K: tl.constexpr, TOPK: tl.constexpr,
                     RT: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    r = tl.arange(0, RT)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = tl.load(ROUTE_POS + token * TOPK + r, r < TOPK, other=0)
    rw = tl.load(RW + token * TOPK + r, r < TOPK, other=0).to(tl.float32)
    value = tl.load(SORTED + pos[:, None] * K + n[None, :],
                    (r[:, None] < TOPK) & (rw[:, None] != 0) & (n[None, :] < K),
                    other=0).to(tl.float32)
    result = tl.sum(value * rw[:, None], axis=0)
    # The store converts to the final dtype; one writer per output element.
    tl.store(OUT + token * K + n, result, n < K)


@torch.no_grad()
def fused_moe_cuda_ops(x, weights, ids, pk, ext, *, out_dtype=torch.float16,
                        ksplit=1, ksplit2=None, cfg=None, reduce_topk=True,
                        tile_m=16, fast_align=False):
    """FP16 intermediate boundaries, with matching align / CUDA M tiles."""
    from int3_moe.align_triton import moe_align_block_size_triton
    m, k = x.shape
    e = pk['B13_1'].shape[0]
    two_i = pk['B13_1'].shape[2]
    i = two_i // 2
    flat_weights = weights.reshape(-1).contiguous()
    sti, eid, meta, buf = moe_align_block_size_triton(
        ids, e, tile_m, flat_values=flat_weights, return_route_positions=reduce_topk,
        histogram=fast_align, scatter_warps=8 if fast_align else 4)
    blocks = eid.numel()
    rows = blocks * tile_m
    key = (m, k, i, rows, str(x.device))
    if key not in _WORKSPACE:
        _WORKSPACE[key] = dict(
            a=torch.empty(rows, k, dtype=torch.float16, device=x.device),
            inter=torch.empty(rows, two_i, dtype=torch.float16, device=x.device),
            act=torch.empty(rows, i, dtype=torch.float16, device=x.device),
            sorted_out=torch.empty(rows, k, dtype=torch.float16, device=x.device),
            out=torch.empty(m, k, dtype=torch.float32, device=x.device))
    ws = _WORKSPACE[key]
    _gather_tokens[(blocks, triton.cdiv(k, 128))](
        x, sti, meta, ws['a'], M=m, K=k, SX0=x.stride(0), SX1=x.stride(1),
        BM=tile_m, BN=128, num_warps=4)
    ks1 = max(1, int(ksplit))
    ks2 = ks1 if ksplit2 is None else max(1, int(ksplit2))
    cfg_kw = dict(thread_n=cfg[0], thread_k=cfg[1], stages=cfg[2]) if cfg else {}
    if tile_m != 16:
        assert getattr(ext, 'supports_moe_thread_m', False), 'rebuild CUDA extension for tile_m > 16'
        cfg_kw['thread_m'] = tile_m
    kw1, kw2 = {}, {}
    inter, sorted_out = ws['inter'], ws['sorted_out']
    if ks1 > 1:
        if 'inter32' not in ws:
            ws['inter32'] = torch.empty(rows, two_i, dtype=torch.float32, device=x.device)
        inter = ws['inter32']
        inter.zero_()
        kw1 = dict(C32=inter, k_splits=ks1)
    if ks2 > 1:
        if 'sorted32' not in ws:
            ws['sorted32'] = torch.empty(rows, k, dtype=torch.float32, device=x.device)
        sorted_out = ws['sorted32']
        sorted_out.zero_()
        kw2 = dict(C32=sorted_out, k_splits=ks2)
    ext.mul_3bit_moe(ws['a'], pk['B13_1'], pk['B13_2'], ws['inter'],
                     pk['s13'], pk['z13'], eid, meta[:1], blocks, **kw1, **cfg_kw)
    _silu_active[(blocks, triton.cdiv(i, 128))](
        inter, sti, meta, ws['act'], M=m, I=i, BM=tile_m, BN=128,
        num_warps=4, enable_fp_fusion=False)
    ext.mul_3bit_moe(ws['act'], pk['B2_1'], pk['B2_2'], ws['sorted_out'],
                     pk['s2'], pk['z2'], eid, meta[:1], blocks, **kw2, **cfg_kw)
    if reduce_topk:
        # Allocate a result per invocation, as the original dtype conversion did.
        # Graph capture then owns distinct output tensors for each layer.
        out = torch.empty(m, k, dtype=out_dtype, device=x.device)
        _reduce_routes[(m, triton.cdiv(k, 256))](
            sorted_out, buf['route_pos'], flat_weights, out, K=k, TOPK=ids.shape[1],
            RT=triton.next_power_of_2(ids.shape[1]), BN=256,
            num_warps=4, enable_fp_fusion=False)
        return out
    ws['out'].zero_()
    _scatter_weighted[(rows, triton.cdiv(k, 256))](
        sorted_out, sti, buf['sv'], meta, ws['out'], M=m, K=k, BN=256, num_warps=4)
    return ws['out'].to(out_dtype)
