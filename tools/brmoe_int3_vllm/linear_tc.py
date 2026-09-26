"""Single-weight W3A16 Tensor Core kernel for attention/shared projections.

Weights remain in the checkpoint's K-major packed INT3 layout. A CTA decodes
one BK x BN tile into registers and shares it across BM input rows. There is
no persistent FP16 weight copy and no routed-expert metadata.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear_int3_tc(
    A, W, S, Z, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GS: tl.constexpr, SX0: tl.constexpr, SX1: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.arange(0, BK // 32)
    lane = tl.arange(0, 32)
    rk = tl.arange(0, BK)
    word = (lane // 8)[None, :, None]
    shift = ((lane % 4) * 3 + ((lane % 8) // 4) * 16)[None, :, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, tl.cdiv(K, BK)):
        kg = base * BK + group * 32
        mask = (kg[:, None] < K) & (col[None, :] < N)
        ptr = W + (kg[:, None] // 32 * 3) * N + col[None, :]
        w0 = tl.load(ptr, mask, other=0)
        w1 = tl.load(ptr + N, mask, other=0)
        w2 = tl.load(ptr + 2 * N, mask, other=0)
        # Recover the fourth logical word from the physical words' spill bits.
        lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
        hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
        w3 = lo | (hi << 16)
        bits = tl.where(
            word == 0, w0[:, None, :],
            tl.where(word == 1, w1[:, None, :],
                     tl.where(word == 2, w2[:, None, :], w3[:, None, :])),
        )
        q = ((bits >> shift) & 7).to(tl.float16)
        scale = tl.load(S + (kg[:, None] // GS) * N + col[None, :], mask, other=0)
        zero = tl.load(Z + (kg[:, None] // GS) * N + col[None, :], mask, other=0)
        # Preserve the original FP16 subtraction and multiplication boundaries.
        b = ((q - zero[:, None, :]).to(tl.float16) * scale[:, None, :]).to(tl.float16)
        b = tl.reshape(b, (BK, BN))
        kk = base * BK + rk
        a = tl.load(
            A + row[:, None] * SX0 + kk[None, :] * SX1,
            (row[:, None] < M) & (kk[None, :] < K), other=0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(C + row[:, None] * N + col[None, :], acc.to(tl.float16),
             (row[:, None] < M) & (col[None, :] < N))


def int3_linear_tc(
    x, qweight, scales, zeros, group_size, *,
    block_m=32, block_n=64, block_k=128, num_stages=3, num_warps=4,
):
    """Packed linear projection with strided x and masked M/N/K tails."""
    m, k = x.shape
    n = scales.shape[1]
    assert x.dtype == scales.dtype == zeros.dtype == torch.float16
    assert qweight.dtype == torch.int32
    assert group_size > 0 and group_size % 32 == 0 and k % 32 == 0 and k % group_size == 0
    assert scales.shape == zeros.shape == (k // group_size, n)
    assert qweight.shape == (k // 32 * 3, n)
    assert qweight.is_contiguous() and scales.is_contiguous() and zeros.is_contiguous()
    assert block_k >= 32 and block_k & (block_k - 1) == 0
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    bm = min(block_m, max(16, triton.next_power_of_2(m)))
    _linear_int3_tc[(triton.cdiv(m, bm), triton.cdiv(n, block_n))](
        x, qweight, scales, zeros, out,
        M=m, N=n, K=k, GS=group_size, SX0=x.stride(0), SX1=x.stride(1),
        BM=bm, BN=block_n, BK=block_k,
        num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False,
    )
    return out
