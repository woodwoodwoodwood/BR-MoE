"""GGUF K-Quant (`Q2_K`, `Q3_K`, `Q4_K`, `Q8_0`) Vectorized Quantizer & Binary Block Packer.

Strictly aligns BR-MoE's quantization, residual SVD compensation, sensitivity profiling,
and RCO budget manifold with `llama.cpp` (`ggml-common.h` and `ggml-quants.c`):

  1. `Q2_K` (`GGML_TYPE_Q2_K = 10`, `block_q2_K` = 84 bytes / 256 weights = 2.6250 bpw):
     - Super-block of 256 weights split into 16 sub-blocks of 16 weights.
     - Asymmetric 2-bit quants `q in {0, 1, 2, 3}`.
     - Sub-block scales & mins double-quantized to 4-bit unsigned integers `ls, lm in {0..15}`
       with FP16 super-block headers `(d, dmin)`.
     - Reconstruction: `w_hat = d * ls * q - dmin * lm`.

  2. `Q3_K` (`GGML_TYPE_Q3_K = 11`, `block_q3_K` = 110 bytes / 256 weights = 3.4375 bpw):
     - Super-block of 256 weights split into 16 sub-blocks of 16 weights.
     - Symmetric signed 3-bit quants `q in {-4, -3, -2, -1, 0, 1, 2, 3}` (stored unsigned `q + 4`
       across `qs[64]` low 2 bits and `hmask[32]` high 1 bit).
     - Sub-block signed scales double-quantized to 6-bit signed integers `ls in {-32..31}`
       (stored unsigned `ls + 32` in `scales[12]`) with FP16 super-block header `d`.
     - Reconstruction: `w_hat = d * ls * q`.

  3. `Q4_K` (`GGML_TYPE_Q4_K = 12`, `block_q4_K` = 144 bytes / 256 weights = 4.5000 bpw):
     - Super-block of 256 weights split into 8 sub-blocks of 32 weights.
     - Asymmetric 4-bit quants `q in {0..15}`.
     - Sub-block scales & mins double-quantized to 6-bit unsigned integers `ls, lm in {0..63}`
       packed via `get_scale_min_k4` into `scales[12]`, with FP16 super-block headers `(d, dmin)`.
     - Reconstruction: `w_hat = d * ls * q - dmin * lm`.

  4. `Q8_0` (`GGML_TYPE_Q8_0 = 8`, `block_q8_0` = 34 bytes / 32 weights = 8.5000 bpw):
     - Block of 32 weights with symmetric signed 8-bit quants `q in {-127..127}` and FP16 `d`.
     - Reconstruction: `w_hat = d * q`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

# ============================================================================
# ggml-common.h Constants & Exact Storage Metrics
# ============================================================================
QK_K: int = 256
QK8_0: int = 32
K_SCALE_SIZE: int = 12

GGML_TYPE_F32: int = 0
GGML_TYPE_F16: int = 1
GGML_TYPE_Q8_0: int = 8
GGML_TYPE_Q2_K: int = 10
GGML_TYPE_Q3_K: int = 11
GGML_TYPE_Q4_K: int = 12

GGML_TYPE_TO_NAME: Dict[int, str] = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
}

BIT_TO_GGML_TYPE: Dict[int, int] = {
    2: GGML_TYPE_Q2_K,
    3: GGML_TYPE_Q3_K,
    4: GGML_TYPE_Q4_K,
    8: GGML_TYPE_Q8_0,
    16: GGML_TYPE_F16,
    32: GGML_TYPE_F32,
}

BIT_TO_GGUF_NAME: Dict[int, str] = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    8: "Q8_0",
    16: "F16",
}

# Exact block sizes in bytes from ggml-common.h
GGUF_BLOCK_ELEMENTS: Dict[int, int] = {
    2: QK_K,   # 256
    3: QK_K,   # 256
    4: QK_K,   # 256
    8: QK8_0,  # 32
    16: 1,
    32: 1,
}

GGUF_BLOCK_BYTES: Dict[int, int] = {
    2: 84,   # sizeof(block_q2_K) = 16 + 64 + 2 + 2
    3: 110,  # sizeof(block_q3_K) = 32 + 64 + 12 + 2
    4: 144,  # sizeof(block_q4_K) = 2 + 2 + 12 + 128
    8: 34,   # sizeof(block_q8_0) = 2 + 32
    16: 2,   # sizeof(ggml_half)
    32: 4,   # sizeof(float)
}

# Exact effective bits-per-weight (bpw) in GGUF K-Quant format
GGUF_EFFECTIVE_BPW: Dict[int, float] = {
    b: (8.0 * float(GGUF_BLOCK_BYTES[b])) / float(GGUF_BLOCK_ELEMENTS[b])
    for b in GGUF_BLOCK_BYTES
}


def gguf_bits_per_weight(nbits: int) -> float:
    """Return exact GGUF effective bits-per-weight (including super-block & sub-block scales/mins)."""
    if nbits <= 0:
        return 0.0
    if nbits in GGUF_EFFECTIVE_BPW:
        return GGUF_EFFECTIVE_BPW[nbits]
    return float(nbits) + 0.5


def gguf_tensor_nbytes(num_elements: int, nbits: int) -> int:
    """Return exact byte count of a GGUF tensor with `num_elements` quantized to `nbits`."""
    if nbits <= 0 or num_elements <= 0:
        return 0
    blk_elems = GGUF_BLOCK_ELEMENTS[nbits]
    blk_bytes = GGUF_BLOCK_BYTES[nbits]
    n_blocks = (int(num_elements) + blk_elems - 1) // blk_elems
    return n_blocks * blk_bytes


# ============================================================================
# Padding Helpers for Arbitrary Shapes (Preserves Sub-Block Min/Max)
# ============================================================================
def _pad_last_dim_np(x: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    orig_shape = x.shape
    x2d = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, orig_shape[-1])
    k = x2d.shape[1]
    rem = k % block_size
    if rem == 0:
        return x2d, orig_shape, k
    pad_len = block_size - rem
    # Pad with the last column value so sub-block [min, max] is never widened by artificial zeros
    last_col = x2d[:, -1:]
    padded = np.concatenate([x2d, np.repeat(last_col, pad_len, axis=1)], axis=1)
    return padded, orig_shape, k


# ============================================================================
# NumPy Reference & Exact Binary Block Packers / Unpackers (block_q2_K/q3_K/q4_K/q8_0)
# ============================================================================
def pack_q2_k_blocks_np(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Quantize and pack a float array into `block_q2_K` binary blocks (`[N_sb, 84]` uint8).

    Layout of `block_q2_K` (84 bytes):
      - `scales[16]` (bytes 0..15)  : `ls | (lm << 4)` (4-bit scale, 4-bit min for 16 sub-blocks)
      - `qs[64]`     (bytes 16..79) : 256 2-bit unsigned quants interleaved by 32-element lanes
      - `d`          (bytes 80..81) : FP16 super-block scale
      - `dmin`       (bytes 82..83) : FP16 super-block min scale
    """
    x2d, orig_shape, orig_k = _pad_last_dim_np(x, QK_K)
    sb = x2d.reshape(-1, 16, 16)  # [N_sb, 16, 16]
    n_sb = sb.shape[0]

    w_min = np.min(sb, axis=-1)               # [N_sb, 16]
    w_max = np.max(sb, axis=-1)               # [N_sb, 16]
    mins = np.maximum(0.0, -w_min)            # [N_sb, 16], non-negative magnitude
    scales = (w_max + mins) / 3.0             # [N_sb, 16]

    # 1-step least-squares scale refinement (matching make_qkx2_quants with nmax=3)
    shifted = sb + mins[:, :, None]
    inv_s0 = np.where(scales > 1e-12, 1.0 / np.maximum(scales, 1e-12), 0.0)
    q0 = np.clip(np.rint(shifted * inv_s0[:, :, None]), 0.0, 3.0)
    num = np.sum(shifted * q0, axis=-1)
    den = np.sum(q0 * q0, axis=-1)
    scales = np.where(den > 0.0, np.maximum(0.0, num / np.maximum(den, 1e-12)), scales)

    max_scale = np.max(scales, axis=-1)       # [N_sb]
    max_min = np.max(mins, axis=-1)           # [N_sb]

    d_fp16 = (max_scale / 15.0).astype(np.float16)
    dmin_fp16 = (max_min / 15.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    dmin_f32 = dmin_fp16.astype(np.float32)

    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    inv_dmin = np.where(dmin_f32 > 0.0, 1.0 / np.maximum(dmin_f32, 1e-30), 0.0)

    ls = np.clip(np.rint(scales * inv_d[:, None]), 0.0, 15.0).astype(np.uint8)    # [N_sb, 16]
    lm = np.clip(np.rint(mins * inv_dmin[:, None]), 0.0, 15.0).astype(np.uint8)   # [N_sb, 16]

    dl = d_f32[:, None] * ls.astype(np.float32)     # [N_sb, 16]
    ml = dmin_f32[:, None] * lm.astype(np.float32)  # [N_sb, 16]
    inv_dl = np.where(dl > 0.0, 1.0 / np.maximum(dl, 1e-30), 0.0)

    q = np.clip(
        np.rint((sb + ml[:, :, None]) * inv_dl[:, :, None]), 0.0, 3.0
    ).astype(np.uint8)  # [N_sb, 16, 16]

    blocks = np.zeros((n_sb, 84), dtype=np.uint8)
    # 1. scales[16]
    blocks[:, 0:16] = (ls & 0x0F) | ((lm & 0x0F) << 4)
    # 2. qs[64]: [N_sb, 2, 4, 32] interleaved shifts 0, 2, 4, 6
    q_2_4_32 = q.reshape(n_sb, 2, 4, 32)
    qs = (
        (q_2_4_32[:, :, 0, :] & 0x03)
        | ((q_2_4_32[:, :, 1, :] & 0x03) << 2)
        | ((q_2_4_32[:, :, 2, :] & 0x03) << 4)
        | ((q_2_4_32[:, :, 3, :] & 0x03) << 6)
    ).reshape(n_sb, 64)
    blocks[:, 16:80] = qs
    # 3. d (bytes 80..81) and dmin (bytes 82..83)
    blocks[:, 80:82] = d_fp16.view(np.uint8).reshape(n_sb, 2)
    blocks[:, 82:84] = dmin_fp16.view(np.uint8).reshape(n_sb, 2)
    return blocks, orig_shape, orig_k


def unpack_q2_k_blocks_np(
    blocks: np.ndarray,
    orig_shape: Optional[Tuple[int, ...]] = None,
    orig_k: Optional[int] = None,
) -> np.ndarray:
    """Dequantize `block_q2_K` (`[N_sb, 84]` uint8) matching `dequantize_row_q2_K`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 84)
    n_sb = blocks.shape[0]

    sc = blocks[:, 0:16]
    ls = (sc & 0x0F).astype(np.float32)         # [N_sb, 16]
    lm = ((sc >> 4) & 0x0F).astype(np.float32)  # [N_sb, 16]

    qs = blocks[:, 16:80].reshape(n_sb, 2, 32)
    q_2_4_32 = np.stack(
        [
            (qs >> 0) & 0x03,
            (qs >> 2) & 0x03,
            (qs >> 4) & 0x03,
            (qs >> 6) & 0x03,
        ],
        axis=2,
    )  # [N_sb, 2, 4, 32]
    q = q_2_4_32.reshape(n_sb, 16, 16).astype(np.float32)

    d_f32 = np.ascontiguousarray(blocks[:, 80:82]).view(np.float16).reshape(n_sb, 1).astype(np.float32)
    dmin_f32 = np.ascontiguousarray(blocks[:, 82:84]).view(np.float16).reshape(n_sb, 1).astype(np.float32)

    dl = d_f32 * ls     # [N_sb, 16]
    ml = dmin_f32 * lm  # [N_sb, 16]
    out = dl[:, :, None] * q - ml[:, :, None]  # [N_sb, 16, 16]

    if orig_shape is not None:
        padded_k = ((orig_shape[-1] + QK_K - 1) // QK_K) * QK_K
        valid_k = orig_k if orig_k is not None else orig_shape[-1]
        out = out.reshape(-1, padded_k)[:, :valid_k].reshape(orig_shape)
    return out


def pack_q3_k_blocks_np(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Quantize and pack a float array into `block_q3_K` binary blocks (`[N_sb, 110]` uint8).

    Layout of `block_q3_K` (110 bytes):
      - `hmask[32]`  (bytes 0..31)   : high bit (bit 2) of 256 unsigned 3-bit quants `q_u = q + 4`
      - `qs[64]`     (bytes 32..95)  : low 2 bits (bits 0..1) of 256 unsigned 3-bit quants
      - `scales[12]` (bytes 96..107) : 16 6-bit unsigned sub-block scales `u = ls_signed + 32`
      - `d`          (bytes 108..109): FP16 super-block scale
    """
    x2d, orig_shape, orig_k = _pad_last_dim_np(x, QK_K)
    sb = x2d.reshape(-1, 16, 16)  # [N_sb, 16, 16]
    n_sb = sb.shape[0]

    # Signed sub-block scale matching make_q3_quants in ggml-quants.c
    w_min = np.min(sb, axis=-1)  # [N_sb, 16]
    w_max = np.max(sb, axis=-1)  # [N_sb, 16]
    signed_amax = np.where(np.abs(w_min) > w_max, w_min, w_max)
    iscale0 = np.where(np.abs(signed_amax) > 1e-12, -4.0 / np.where(np.abs(signed_amax) > 1e-12, signed_amax, 1.0), 0.0)
    q0 = np.clip(np.rint(sb * iscale0[:, :, None]), -4.0, 3.0)
    num = np.sum(sb * q0, axis=-1)
    den = np.sum(q0 * q0, axis=-1)
    scales = np.where(den > 0.0, num / np.maximum(den, 1e-12), -signed_amax / 4.0)  # [N_sb, 16]

    # Find sub-block j* with maximum |scales[j]| per super-block
    abs_idx = np.argmax(np.abs(scales), axis=-1)  # [N_sb]
    max_scale = scales[np.arange(n_sb), abs_idx]  # [N_sb], signed!

    iscale_sb = np.where(np.abs(max_scale) > 1e-12, -32.0 / np.where(np.abs(max_scale) > 1e-12, max_scale, 1.0), 0.0)
    d_fp16 = np.where(np.abs(iscale_sb) > 0.0, 1.0 / np.where(np.abs(iscale_sb) > 0.0, iscale_sb, 1.0), 0.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)

    ls_signed = np.clip(np.rint(iscale_sb[:, None] * scales), -32.0, 31.0).astype(np.int8)  # [N_sb, 16]
    u_scales = (ls_signed.astype(np.int16) + 32).astype(np.uint8)                           # [N_sb, 16] in 0..63

    dl = d_f32[:, None] * ls_signed.astype(np.float32)  # [N_sb, 16]
    inv_dl = np.where(np.abs(dl) > 0.0, 1.0 / np.where(np.abs(dl) > 0.0, dl, 1.0), 0.0)
    q_signed = np.clip(np.rint(sb * inv_dl[:, :, None]), -4.0, 3.0).astype(np.int8)
    q_u = (q_signed + 4).astype(np.uint8)  # [N_sb, 16, 16] in 0..7

    blocks = np.zeros((n_sb, 110), dtype=np.uint8)

    # 1. hmask[32]: bit 2 of q_u packed across 8 groups of 32
    q_u_8_32 = q_u.reshape(n_sb, 8, 32)
    hmask = np.zeros((n_sb, 32), dtype=np.uint8)
    for b in range(8):
        hmask |= (((q_u_8_32[:, b, :] >> 2) & 0x01) << b).astype(np.uint8)
    blocks[:, 0:32] = hmask

    # 2. qs[64]: low 2 bits of q_u packed in [N_sb, 2, 4, 32]
    q_low = (q_u & 0x03).reshape(n_sb, 2, 4, 32)
    qs = (
        q_low[:, :, 0, :]
        | (q_low[:, :, 1, :] << 2)
        | (q_low[:, :, 2, :] << 4)
        | (q_low[:, :, 3, :] << 6)
    ).reshape(n_sb, 64)
    blocks[:, 32:96] = qs

    # 3. scales[12]: 16 6-bit unsigned scales u_scales
    sc_bytes = np.zeros((n_sb, 12), dtype=np.uint8)
    sc_bytes[:, 0:8] = (u_scales[:, 0:8] & 0x0F) | ((u_scales[:, 8:16] & 0x0F) << 4)
    sc_bytes[:, 8:12] = (
        ((u_scales[:, 0:4] >> 4) & 0x03)
        | (((u_scales[:, 4:8] >> 4) & 0x03) << 2)
        | (((u_scales[:, 8:12] >> 4) & 0x03) << 4)
        | (((u_scales[:, 12:16] >> 4) & 0x03) << 6)
    )
    blocks[:, 96:108] = sc_bytes

    # 4. d (bytes 108..109)
    blocks[:, 108:110] = d_fp16.view(np.uint8).reshape(n_sb, 2)
    return blocks, orig_shape, orig_k


def unpack_q3_k_blocks_np(
    blocks: np.ndarray,
    orig_shape: Optional[Tuple[int, ...]] = None,
    orig_k: Optional[int] = None,
) -> np.ndarray:
    """Dequantize `block_q3_K` (`[N_sb, 110]` uint8) matching `dequantize_row_q3_K`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 110)
    n_sb = blocks.shape[0]

    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96].reshape(n_sb, 2, 32)
    sc_bytes = blocks[:, 96:108]
    d_f32 = np.ascontiguousarray(blocks[:, 108:110]).view(np.float16).reshape(n_sb, 1).astype(np.float32)

    # Unpack 16 6-bit unsigned scales u_scales -> ls_signed = u_scales - 32
    u_scales = np.zeros((n_sb, 16), dtype=np.int16)
    u_scales[:, 0:8] = sc_bytes[:, 0:8] & 0x0F
    u_scales[:, 8:16] = (sc_bytes[:, 0:8] >> 4) & 0x0F
    u_scales[:, 0:4] |= ((sc_bytes[:, 8:12] >> 0) & 0x03) << 4
    u_scales[:, 4:8] |= ((sc_bytes[:, 8:12] >> 2) & 0x03) << 4
    u_scales[:, 8:12] |= ((sc_bytes[:, 8:12] >> 4) & 0x03) << 4
    u_scales[:, 12:16] |= ((sc_bytes[:, 8:12] >> 6) & 0x03) << 4
    ls_signed = (u_scales - 32).astype(np.float32)  # [N_sb, 16]

    # Unpack low 2 bits from qs and high bit from hmask
    q_low_2_4_32 = np.stack(
        [
            (qs >> 0) & 0x03,
            (qs >> 2) & 0x03,
            (qs >> 4) & 0x03,
            (qs >> 6) & 0x03,
        ],
        axis=2,
    ).reshape(n_sb, 8, 32)
    q_high_8_32 = np.stack(
        [((hmask >> b) & 0x01) << 2 for b in range(8)],
        axis=1,
    )
    q_signed = ((q_low_2_4_32 | q_high_8_32).astype(np.int8) - 4).reshape(n_sb, 16, 16).astype(np.float32)

    dl = d_f32 * ls_signed  # [N_sb, 16]
    out = dl[:, :, None] * q_signed

    if orig_shape is not None:
        padded_k = ((orig_shape[-1] + QK_K - 1) // QK_K) * QK_K
        valid_k = orig_k if orig_k is not None else orig_shape[-1]
        out = out.reshape(-1, padded_k)[:, :valid_k].reshape(orig_shape)
    return out


def pack_q4_k_blocks_np(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Quantize and pack a float array into `block_q4_K` binary blocks (`[N_sb, 144]` uint8).

    Layout of `block_q4_K` (144 bytes):
      - `d`          (bytes 0..1)    : FP16 super-block scale
      - `dmin`       (bytes 2..3)    : FP16 super-block min scale
      - `scales[12]` (bytes 4..15)   : 8 6-bit scales `ls` and 8 6-bit mins `lm` (`get_scale_min_k4`)
      - `qs[128]`    (bytes 16..143) : 256 4-bit unsigned quants `q in {0..15}`
    """
    x2d, orig_shape, orig_k = _pad_last_dim_np(x, QK_K)
    sb = x2d.reshape(-1, 8, 32)  # [N_sb, 8, 32]
    n_sb = sb.shape[0]

    w_min = np.min(sb, axis=-1)          # [N_sb, 8]
    w_max = np.max(sb, axis=-1)          # [N_sb, 8]
    mins = np.maximum(0.0, -w_min)       # [N_sb, 8]
    scales = (w_max + mins) / 15.0       # [N_sb, 8]

    # 1-step least-squares scale refinement (matching make_qkx2_quants with nmax=15)
    shifted = sb + mins[:, :, None]
    inv_s0 = np.where(scales > 1e-12, 1.0 / np.maximum(scales, 1e-12), 0.0)
    q0 = np.clip(np.rint(shifted * inv_s0[:, :, None]), 0.0, 15.0)
    num = np.sum(shifted * q0, axis=-1)
    den = np.sum(q0 * q0, axis=-1)
    scales = np.where(den > 0.0, np.maximum(0.0, num / np.maximum(den, 1e-12)), scales)

    max_scale = np.max(scales, axis=-1)  # [N_sb]
    max_min = np.max(mins, axis=-1)      # [N_sb]

    d_fp16 = (max_scale / 63.0).astype(np.float16)
    dmin_fp16 = (max_min / 63.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    dmin_f32 = dmin_fp16.astype(np.float32)

    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    inv_dmin = np.where(dmin_f32 > 0.0, 1.0 / np.maximum(dmin_f32, 1e-30), 0.0)

    ls = np.clip(np.rint(scales * inv_d[:, None]), 0.0, 63.0).astype(np.uint8)    # [N_sb, 8]
    lm = np.clip(np.rint(mins * inv_dmin[:, None]), 0.0, 63.0).astype(np.uint8)   # [N_sb, 8]

    dl = d_f32[:, None] * ls.astype(np.float32)     # [N_sb, 8]
    ml = dmin_f32[:, None] * lm.astype(np.float32)  # [N_sb, 8]
    inv_dl = np.where(dl > 0.0, 1.0 / np.maximum(dl, 1e-30), 0.0)

    q = np.clip(
        np.rint((sb + ml[:, :, None]) * inv_dl[:, :, None]), 0.0, 15.0
    ).astype(np.uint8)  # [N_sb, 8, 32]

    blocks = np.zeros((n_sb, 144), dtype=np.uint8)
    # 1. d (bytes 0..1) and dmin (bytes 2..3)
    blocks[:, 0:2] = d_fp16.view(np.uint8).reshape(n_sb, 2)
    blocks[:, 2:4] = dmin_fp16.view(np.uint8).reshape(n_sb, 2)

    # 2. scales[12] (bytes 4..15) matching get_scale_min_k4 in ggml-quants.c
    sc_bytes = np.zeros((n_sb, 12), dtype=np.uint8)
    sc_bytes[:, 0:4] = (ls[:, 0:4] & 0x3F) | (((ls[:, 4:8] >> 4) & 0x03) << 6)
    sc_bytes[:, 4:8] = (lm[:, 0:4] & 0x3F) | (((lm[:, 4:8] >> 4) & 0x03) << 6)
    sc_bytes[:, 8:12] = (ls[:, 4:8] & 0x0F) | ((lm[:, 4:8] & 0x0F) << 4)
    blocks[:, 4:16] = sc_bytes

    # 3. qs[128] (bytes 16..143): 4 pairs of 32-element sub-blocks
    q_4_2_32 = q.reshape(n_sb, 4, 2, 32)
    qs = ((q_4_2_32[:, :, 0, :] & 0x0F) | ((q_4_2_32[:, :, 1, :] & 0x0F) << 4)).reshape(n_sb, 128)
    blocks[:, 16:144] = qs
    return blocks, orig_shape, orig_k


def unpack_q4_k_blocks_np(
    blocks: np.ndarray,
    orig_shape: Optional[Tuple[int, ...]] = None,
    orig_k: Optional[int] = None,
) -> np.ndarray:
    """Dequantize `block_q4_K` (`[N_sb, 144]` uint8) matching `dequantize_row_q4_K`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 144)
    n_sb = blocks.shape[0]

    d_f32 = np.ascontiguousarray(blocks[:, 0:2]).view(np.float16).reshape(n_sb, 1).astype(np.float32)
    dmin_f32 = np.ascontiguousarray(blocks[:, 2:4]).view(np.float16).reshape(n_sb, 1).astype(np.float32)
    sc_bytes = blocks[:, 4:16]
    qs = blocks[:, 16:144].reshape(n_sb, 4, 32)

    ls = np.zeros((n_sb, 8), dtype=np.float32)
    lm = np.zeros((n_sb, 8), dtype=np.float32)
    ls[:, 0:4] = sc_bytes[:, 0:4] & 0x3F
    lm[:, 0:4] = sc_bytes[:, 4:8] & 0x3F
    ls[:, 4:8] = (sc_bytes[:, 8:12] & 0x0F) | (((sc_bytes[:, 0:4] >> 6) & 0x03) << 4)
    lm[:, 4:8] = ((sc_bytes[:, 8:12] >> 4) & 0x0F) | (((sc_bytes[:, 4:8] >> 6) & 0x03) << 4)

    q_4_2_32 = np.stack([qs & 0x0F, (qs >> 4) & 0x0F], axis=2)  # [N_sb, 4, 2, 32]
    q = q_4_2_32.reshape(n_sb, 8, 32).astype(np.float32)

    dl = d_f32 * ls     # [N_sb, 8]
    ml = dmin_f32 * lm  # [N_sb, 8]
    out = dl[:, :, None] * q - ml[:, :, None]  # [N_sb, 8, 32]

    if orig_shape is not None:
        padded_k = ((orig_shape[-1] + QK_K - 1) // QK_K) * QK_K
        valid_k = orig_k if orig_k is not None else orig_shape[-1]
        out = out.reshape(-1, padded_k)[:, :valid_k].reshape(orig_shape)
    return out


def pack_q8_0_blocks_np(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Quantize and pack a float array into `block_q8_0` binary blocks (`[N_b, 34]` uint8)."""
    x2d, orig_shape, orig_k = _pad_last_dim_np(x, QK8_0)
    sb = x2d.reshape(-1, QK8_0)
    n_b = sb.shape[0]

    amax = np.max(np.abs(sb), axis=-1)
    d_fp16 = (amax / 127.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    inv_d = np.where(d_f32 > 0.0, 1.0 / np.maximum(d_f32, 1e-30), 0.0)
    q = np.clip(np.rint(sb * inv_d[:, None]), -127.0, 127.0).astype(np.int8)

    blocks = np.zeros((n_b, 34), dtype=np.uint8)
    blocks[:, 0:2] = d_fp16.view(np.uint8).reshape(n_b, 2)
    blocks[:, 2:34] = q.view(np.uint8)
    return blocks, orig_shape, orig_k


def unpack_q8_0_blocks_np(
    blocks: np.ndarray,
    orig_shape: Optional[Tuple[int, ...]] = None,
    orig_k: Optional[int] = None,
) -> np.ndarray:
    """Dequantize `block_q8_0` (`[N_b, 34]` uint8) matching `dequantize_row_q8_0`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 34)
    n_b = blocks.shape[0]
    d_f32 = np.ascontiguousarray(blocks[:, 0:2]).view(np.float16).reshape(n_b, 1).astype(np.float32)
    q = blocks[:, 2:34].view(np.int8).astype(np.float32)
    out = d_f32 * q
    if orig_shape is not None:
        padded_k = ((orig_shape[-1] + QK8_0 - 1) // QK8_0) * QK8_0
        valid_k = orig_k if orig_k is not None else orig_shape[-1]
        out = out.reshape(-1, padded_k)[:, :valid_k].reshape(orig_shape)
    return out


def pack_gguf_blocks_np(x: np.ndarray, nbits: int) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Dispatch `nbits in {2, 3, 4, 8}` to the exact GGUF binary block packer."""
    if nbits == 2:
        return pack_q2_k_blocks_np(x)
    if nbits == 3:
        return pack_q3_k_blocks_np(x)
    if nbits == 4:
        return pack_q4_k_blocks_np(x)
    if nbits == 8:
        return pack_q8_0_blocks_np(x)
    raise ValueError(f"Unsupported GGUF bit-width: {nbits}")


def unpack_gguf_blocks_np(
    blocks: np.ndarray,
    nbits: int,
    orig_shape: Optional[Tuple[int, ...]] = None,
    orig_k: Optional[int] = None,
) -> np.ndarray:
    """Dispatch `nbits in {2, 3, 4, 8}` to the exact GGUF binary block unpacker."""
    if nbits == 2:
        return unpack_q2_k_blocks_np(blocks, orig_shape=orig_shape, orig_k=orig_k)
    if nbits == 3:
        return unpack_q3_k_blocks_np(blocks, orig_shape=orig_shape, orig_k=orig_k)
    if nbits == 4:
        return unpack_q4_k_blocks_np(blocks, orig_shape=orig_shape, orig_k=orig_k)
    if nbits == 8:
        return unpack_q8_0_blocks_np(blocks, orig_shape=orig_shape, orig_k=orig_k)
    raise ValueError(f"Unsupported GGUF bit-width: {nbits}")


def gguf_kquant_quant_dequant_np(x: np.ndarray, nbits: int) -> np.ndarray:
    """Exact NumPy GGUF K-Quant quantize-dequantize (`Q2_K`, `Q3_K`, `Q4_K`, `Q8_0`)."""
    if nbits <= 0:
        return np.zeros_like(x)
    blocks, orig_shape, orig_k = pack_gguf_blocks_np(x, nbits=nbits)
    return unpack_gguf_blocks_np(blocks, nbits=nbits, orig_shape=orig_shape, orig_k=orig_k).astype(x.dtype)


# ============================================================================
# High-Speed Vectorized PyTorch (GPU/CPU) GGUF K-Quant Quantizer
# ============================================================================
if HAS_TORCH:

    def _pad_last_dim_torch(
        W: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Size, int]:
        orig_shape = W.shape
        W2d = W.float().reshape(-1, orig_shape[-1])
        k = W2d.size(1)
        rem = k % block_size
        if rem == 0:
            return W2d, orig_shape, k
        pad_len = block_size - rem
        last_col = W2d[:, -1:].expand(-1, pad_len)
        padded = torch.cat([W2d, last_col], dim=1)
        return padded, orig_shape, k

    def _unpad_last_dim_torch(
        W2d: torch.Tensor, orig_shape: torch.Size, orig_k: int, orig_dtype: torch.dtype
    ) -> torch.Tensor:
        padded_k = ((orig_k + W2d.size(-1) - 1) // W2d.size(-1)) * W2d.size(-1)
        out = W2d.reshape(-1, padded_k)
        if padded_k != orig_k:
            out = out[:, :orig_k]
        return out.reshape(orig_shape).to(orig_dtype)

    @torch.no_grad()
    def quantize_dequantize_q2_k_torch(W: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch `Q2_K` quantize-dequantize matching `ggml-quants.c`."""
        orig_dtype = W.dtype
        W2d, orig_shape, orig_k = _pad_last_dim_torch(W, QK_K)
        sb = W2d.reshape(-1, 16, 16)  # [N_sb, 16, 16]

        w_min = sb.amin(dim=-1)
        w_max = sb.amax(dim=-1)
        mins = (-w_min).clamp(min=0.0)
        scales = (w_max + mins) / 3.0

        shifted = sb + mins.unsqueeze(-1)
        inv_s0 = torch.where(scales > 1e-12, 1.0 / scales.clamp(min=1e-12), torch.zeros_like(scales))
        q0 = torch.round(shifted * inv_s0.unsqueeze(-1)).clamp(0.0, 3.0)
        num = (shifted * q0).sum(dim=-1)
        den = (q0 * q0).sum(dim=-1)
        scales = torch.where(den > 0.0, (num / den.clamp(min=1e-12)).clamp(min=0.0), scales)

        max_scale = scales.amax(dim=-1)
        max_min = mins.amax(dim=-1)

        # Round super-block scales through FP16 exactly like ggml_fp32_to_fp16
        d_f32 = (max_scale / 15.0).to(torch.float16).float()
        dmin_f32 = (max_min / 15.0).to(torch.float16).float()

        inv_d = torch.where(d_f32 > 0.0, 1.0 / d_f32.clamp(min=1e-30), torch.zeros_like(d_f32))
        inv_dmin = torch.where(dmin_f32 > 0.0, 1.0 / dmin_f32.clamp(min=1e-30), torch.zeros_like(dmin_f32))

        ls = torch.round(scales * inv_d.unsqueeze(-1)).clamp(0.0, 15.0)
        lm = torch.round(mins * inv_dmin.unsqueeze(-1)).clamp(0.0, 15.0)

        dl = d_f32.unsqueeze(-1) * ls
        ml = dmin_f32.unsqueeze(-1) * lm
        inv_dl = torch.where(dl > 0.0, 1.0 / dl.clamp(min=1e-30), torch.zeros_like(dl))

        q = torch.round((sb + ml.unsqueeze(-1)) * inv_dl.unsqueeze(-1)).clamp(0.0, 3.0)
        out = dl.unsqueeze(-1) * q - ml.unsqueeze(-1)
        return _unpad_last_dim_torch(out.reshape(-1, QK_K), orig_shape, orig_k, orig_dtype)

    @torch.no_grad()
    def quantize_dequantize_q3_k_torch(W: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch `Q3_K` quantize-dequantize matching `ggml-quants.c`."""
        orig_dtype = W.dtype
        W2d, orig_shape, orig_k = _pad_last_dim_torch(W, QK_K)
        sb = W2d.reshape(-1, 16, 16)  # [N_sb, 16, 16]

        w_min = sb.amin(dim=-1)
        w_max = sb.amax(dim=-1)
        signed_amax = torch.where(w_min.abs() > w_max, w_min, w_max)
        safe_amax = torch.where(signed_amax.abs() > 1e-12, signed_amax, torch.ones_like(signed_amax))
        iscale0 = torch.where(signed_amax.abs() > 1e-12, -4.0 / safe_amax, torch.zeros_like(signed_amax))
        q0 = torch.round(sb * iscale0.unsqueeze(-1)).clamp(-4.0, 3.0)
        num = (sb * q0).sum(dim=-1)
        den = (q0 * q0).sum(dim=-1)
        scales = torch.where(den > 0.0, num / den.clamp(min=1e-12), -signed_amax / 4.0)

        abs_idx = scales.abs().argmax(dim=-1, keepdim=True)
        max_scale = torch.gather(scales, 1, abs_idx).squeeze(-1)  # [N_sb], signed

        safe_ms = torch.where(max_scale.abs() > 1e-12, max_scale, torch.ones_like(max_scale))
        iscale_sb = torch.where(max_scale.abs() > 1e-12, -32.0 / safe_ms, torch.zeros_like(max_scale))
        safe_isc = torch.where(iscale_sb.abs() > 0.0, iscale_sb, torch.ones_like(iscale_sb))
        d_f32 = torch.where(iscale_sb.abs() > 0.0, 1.0 / safe_isc, torch.zeros_like(iscale_sb)).to(torch.float16).float()

        ls_signed = torch.round(iscale_sb.unsqueeze(-1) * scales).clamp(-32.0, 31.0)
        dl = d_f32.unsqueeze(-1) * ls_signed
        safe_dl = torch.where(dl.abs() > 0.0, dl, torch.ones_like(dl))
        inv_dl = torch.where(dl.abs() > 0.0, 1.0 / safe_dl, torch.zeros_like(dl))

        q_signed = torch.round(sb * inv_dl.unsqueeze(-1)).clamp(-4.0, 3.0)
        out = dl.unsqueeze(-1) * q_signed
        return _unpad_last_dim_torch(out.reshape(-1, QK_K), orig_shape, orig_k, orig_dtype)

    @torch.no_grad()
    def quantize_dequantize_q4_k_torch(W: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch `Q4_K` quantize-dequantize matching `ggml-quants.c`."""
        orig_dtype = W.dtype
        W2d, orig_shape, orig_k = _pad_last_dim_torch(W, QK_K)
        sb = W2d.reshape(-1, 8, 32)  # [N_sb, 8, 32]

        w_min = sb.amin(dim=-1)
        w_max = sb.amax(dim=-1)
        mins = (-w_min).clamp(min=0.0)
        scales = (w_max + mins) / 15.0

        shifted = sb + mins.unsqueeze(-1)
        inv_s0 = torch.where(scales > 1e-12, 1.0 / scales.clamp(min=1e-12), torch.zeros_like(scales))
        q0 = torch.round(shifted * inv_s0.unsqueeze(-1)).clamp(0.0, 15.0)
        num = (shifted * q0).sum(dim=-1)
        den = (q0 * q0).sum(dim=-1)
        scales = torch.where(den > 0.0, (num / den.clamp(min=1e-12)).clamp(min=0.0), scales)

        max_scale = scales.amax(dim=-1)
        max_min = mins.amax(dim=-1)

        d_f32 = (max_scale / 63.0).to(torch.float16).float()
        dmin_f32 = (max_min / 63.0).to(torch.float16).float()

        inv_d = torch.where(d_f32 > 0.0, 1.0 / d_f32.clamp(min=1e-30), torch.zeros_like(d_f32))
        inv_dmin = torch.where(dmin_f32 > 0.0, 1.0 / dmin_f32.clamp(min=1e-30), torch.zeros_like(dmin_f32))

        ls = torch.round(scales * inv_d.unsqueeze(-1)).clamp(0.0, 63.0)
        lm = torch.round(mins * inv_dmin.unsqueeze(-1)).clamp(0.0, 63.0)

        dl = d_f32.unsqueeze(-1) * ls
        ml = dmin_f32.unsqueeze(-1) * lm
        inv_dl = torch.where(dl > 0.0, 1.0 / dl.clamp(min=1e-30), torch.zeros_like(dl))

        q = torch.round((sb + ml.unsqueeze(-1)) * inv_dl.unsqueeze(-1)).clamp(0.0, 15.0)
        out = dl.unsqueeze(-1) * q - ml.unsqueeze(-1)
        return _unpad_last_dim_torch(out.reshape(-1, QK_K), orig_shape, orig_k, orig_dtype)

    @torch.no_grad()
    def quantize_dequantize_q8_0_torch(W: torch.Tensor) -> torch.Tensor:
        """Vectorized PyTorch `Q8_0` quantize-dequantize matching `ggml-quants.c`."""
        orig_dtype = W.dtype
        W2d, orig_shape, orig_k = _pad_last_dim_torch(W, QK8_0)
        sb = W2d.reshape(-1, QK8_0)

        amax = sb.abs().amax(dim=-1)
        d_f32 = (amax / 127.0).to(torch.float16).float()
        inv_d = torch.where(d_f32 > 0.0, 1.0 / d_f32.clamp(min=1e-30), torch.zeros_like(d_f32))
        q = torch.round(sb * inv_d.unsqueeze(-1)).clamp(-127.0, 127.0)
        out = d_f32.unsqueeze(-1) * q
        return _unpad_last_dim_torch(out.reshape(-1, QK8_0), orig_shape, orig_k, orig_dtype)

    @torch.no_grad()
    def gguf_kquant_quant_dequant(
        W: torch.Tensor,
        nbits: int,
    ) -> torch.Tensor:
        """Vectorized GPU/CPU GGUF K-Quant quantize-dequantize (`Q2_K`, `Q3_K`, `Q4_K`, `Q8_0`)."""
        if nbits <= 0:
            return torch.zeros_like(W)
        if nbits == 2:
            return quantize_dequantize_q2_k_torch(W)
        if nbits == 3:
            return quantize_dequantize_q3_k_torch(W)
        if nbits == 4:
            return quantize_dequantize_q4_k_torch(W)
        if nbits == 8:
            return quantize_dequantize_q8_0_torch(W)
        if nbits >= 16:
            return W.to(torch.float16).to(W.dtype)
        raise ValueError(f"Unsupported GGUF K-Quant bit-width: {nbits}")

    @torch.no_grad()
    def pack_gguf_blocks_torch(
        W: torch.Tensor, nbits: int
    ) -> Tuple[torch.Tensor, dict]:
        """Pack a PyTorch tensor into a `uint8` tensor of GGUF K-Quant blocks + `meta` dict."""
        w_np = W.detach().cpu().float().numpy()
        blocks_np, orig_shape, orig_k = pack_gguf_blocks_np(w_np, nbits=nbits)
        W_q = torch.from_numpy(blocks_np).to(device=W.device, dtype=torch.uint8)
        packing_name = f"gguf_q{nbits}_k" if nbits in (2, 3, 4) else f"gguf_q{nbits}_0"
        meta = {
            "quant_format": "gguf",
            "nbits": int(nbits),
            "ggml_type": BIT_TO_GGML_TYPE[nbits],
            "gguf_type_name": BIT_TO_GGUF_NAME[nbits],
            "group_size": GGUF_BLOCK_ELEMENTS[nbits],
            "shape": W.shape,
            "orig_k": int(orig_k),
            "axis": 1,
            "packing": packing_name,
            "unpack_view_dtype": torch.uint8,
            "view_as_float": False,
            "compute_dtype": W.dtype if W.dtype.is_floating_point else torch.float16,
            "quant_scale": False,
            "quant_zero": False,
        }
        return W_q, meta

    @torch.no_grad()
    def unpack_gguf_blocks_torch(
        W_q: torch.Tensor, meta: dict, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Unpack a `uint8` tensor of GGUF K-Quant blocks back into a dequantized PyTorch tensor."""
        target_dtype = dtype or meta.get("compute_dtype", torch.float16)
        nbits = int(meta["nbits"])
        orig_shape = tuple(int(d) for d in meta["shape"])
        orig_k = int(meta.get("orig_k", orig_shape[-1]))
        blocks_np = W_q.detach().cpu().numpy()
        deq_np = unpack_gguf_blocks_np(
            blocks_np, nbits=nbits, orig_shape=orig_shape, orig_k=orig_k
        )
        return torch.from_numpy(deq_np).to(device=W_q.device, dtype=target_dtype)
