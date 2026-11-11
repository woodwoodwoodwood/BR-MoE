"""Triton decoding of the exact packed GGML IQ/K blocks used by RCO.

Encoding stays in the pinned upstream CPU codec. Decode a packed CUDA byte tensor
without materializing an FP32 host weight matrix. Tables come from that same
native library, rather than a second copy of the upstream codebooks.
"""
import ctypes as ct
from functools import lru_cache
import math

import numpy as np
import torch
import triton
import triton.language as tl

from .formats import get_format, tensor_nbytes
from .native import library

GPU_FORMATS = frozenset(('Q2_K', 'Q3_K', 'Q4_K', 'IQ2_XXS', 'IQ2_XS',
    'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_XS', 'IQ4_NL', 'F16', 'F32'))


@lru_cache(maxsize=64)
def _tables(type_id, device_index):
    lib = library()
    if not hasattr(lib, 'brmoe_quant_table'):
        raise RuntimeError('Rebuild tools/build_ggml_quant.py for GPU IQ codebooks')
    lib.brmoe_quant_table.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
    lib.brmoe_quant_table.restype = ct.c_void_p
    def get_table(which):
        size = ct.c_int()
        address = lib.brmoe_quant_table(which, ct.byref(size))
        if not address:
            return torch.zeros(1, dtype=torch.uint8, device=f'cuda:{device_index}')
        raw = np.ctypeslib.as_array((ct.c_uint8 * size.value).from_address(address)).copy()
        return torch.from_numpy(raw).to(f'cuda:{device_index}')
    return get_table(type_id), get_table(-1)


@triton.jit
def _u16(P, offset, mask):
    lo = tl.load(P + offset, mask, 0).to(tl.uint32)
    hi = tl.load(P + offset + 1, mask, 0).to(tl.uint32)
    return lo | (hi << 8)


@triton.jit
def _u32(P, offset, mask):
    return _u16(P, offset, mask) | (_u16(P, offset + 2, mask) << 16)


@triton.jit
def _half(P, offset, mask):
    return _u16(P, offset, mask).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _decode(P, O, GRID, SIGNS, N: tl.constexpr, TYPE: tl.constexpr,
            QK: tl.constexpr, QB: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    j = i % QK
    b = (i // QK) * QB
    if TYPE == 10:  # Q2_K
        d = _half(P, b + 80, mask)
        dm = _half(P, b + 82, mask)
        sc = tl.load(P + b + j // 16, mask, 0).to(tl.int32)
        q = tl.load(P + b + 16 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
        q = (q >> (2 * ((j % 128) // 32))) & 3
        value = (d * (sc & 15)) * q - dm * (sc >> 4)
    elif TYPE == 11:  # Q3_K
        d = _half(P, b + 108, mask)
        g = j // 16
        lo = tl.load(P + b + 96 + g % 8, mask, 0).to(tl.int32)
        hi = tl.load(P + b + 104 + g % 4, mask, 0).to(tl.int32)
        sc = ((lo >> (4 * (g // 8))) & 15) | (((hi >> (2 * (g // 4))) & 3) << 4)
        ql = tl.load(P + b + 32 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
        qh = tl.load(P + b + j % 32, mask, 0).to(tl.int32)
        q = ((ql >> (2 * ((j % 128) // 32))) & 3) - (1 - ((qh >> (j // 32)) & 1)) * 4
        value = (d * (sc - 32)) * q
    elif TYPE == 12:  # Q4_K
        d = _half(P, b, mask)
        dm = _half(P, b + 2, mask)
        g = j // 32
        a = tl.load(P + b + 4 + g % 4, mask, 0).to(tl.int32)
        c = tl.load(P + b + 8 + g % 4, mask, 0).to(tl.int32)
        ac = tl.load(P + b + 12 + g % 4, mask, 0).to(tl.int32)
        sc = tl.where(g < 4, a & 63, (ac & 15) | ((a >> 6) << 4))
        mn = tl.where(g < 4, c & 63, (ac >> 4) | ((c >> 6) << 4))
        q = tl.load(P + b + 16 + (j // 64) * 32 + j % 32, mask, 0).to(tl.int32)
        q = (q >> (4 * ((j // 32) % 2))) & 15
        value = (d * sc) * q - dm * mn
    elif TYPE == 16 or TYPE == 18:  # IQ2_XXS / IQ3_XXS
        d = _half(P, b, mask)
        if TYPE == 16:
            code = tl.load(P + b + 2 + (j // 32) * 8 + (j // 8) % 4, mask, 0).to(tl.int32)
            bits = _u32(P, b + 6 + (j // 32) * 8, mask)
            grid = tl.load(GRID + code * 8 + j % 8, mask, 0).to(tl.float32)
            scale = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.25
        else:
            code = tl.load(P + b + 2 + j // 4, mask, 0).to(tl.int32)
            bits = _u32(P, b + 66 + (j // 32) * 4, mask)
            grid = tl.load(GRID + code * 4 + j % 4, mask, 0).to(tl.float32)
            scale = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.5
        si = (bits >> (7 * ((j // 8) % 4))) & 127
        sb = tl.load(SIGNS + si, mask, 0).to(tl.int32)
        sign = 1 - 2 * ((sb >> (j % 8)) & 1)
        value = (scale * grid) * sign
    elif TYPE == 17:  # IQ2_XS
        d = _half(P, b, mask)
        q = _u16(P, b + 2 + (j // 8) * 2, mask)
        sc = tl.load(P + b + 66 + j // 32, mask, 0).to(tl.int32)
        sc = (sc >> (4 * ((j // 16) % 2))) & 15
        sb = tl.load(SIGNS + (q >> 9), mask, 0).to(tl.int32)
        sign = 1 - 2 * ((sb >> (j % 8)) & 1)
        grid = tl.load(GRID + (q & 511) * 8 + j % 8, mask, 0).to(tl.float32)
        value = ((d * (0.5 + sc) * 0.25) * grid) * sign
    elif TYPE == 22:  # IQ2_S
        d = _half(P, b, mask)
        q = tl.load(P + b + 2 + j // 8, mask, 0).to(tl.int32)
        qh = tl.load(P + b + 66 + j // 32, mask, 0).to(tl.int32)
        code = q | (((qh >> (2 * ((j // 8) % 4))) & 3) << 8)
        sb = tl.load(P + b + 34 + j // 8, mask, 0).to(tl.int32)
        sign = 1 - 2 * ((sb >> (j % 8)) & 1)
        sc = tl.load(P + b + 74 + j // 32, mask, 0).to(tl.int32)
        sc = (sc >> (4 * ((j // 16) % 2))) & 15
        grid = tl.load(GRID + code * 8 + j % 8, mask, 0).to(tl.float32)
        value = ((d * (0.5 + sc) * 0.25) * grid) * sign
    elif TYPE == 21:  # IQ3_S
        d = _half(P, b, mask)
        q = tl.load(P + b + 2 + j // 4, mask, 0).to(tl.int32)
        qh = tl.load(P + b + 66 + j // 32, mask, 0).to(tl.int32)
        code = q | (((qh >> ((j // 4) % 8)) & 1) << 8)
        sb = tl.load(P + b + 74 + j // 8, mask, 0).to(tl.int32)
        sign = 1 - 2 * ((sb >> (j % 8)) & 1)
        sc = tl.load(P + b + 106 + j // 64, mask, 0).to(tl.int32)
        sc = (sc >> (4 * ((j // 32) % 2))) & 15
        grid = tl.load(GRID + code * 4 + j % 4, mask, 0).to(tl.float32)
        value = ((d * (1 + 2 * sc)) * grid) * sign
    elif TYPE == 20:  # IQ4_NL
        d = _half(P, b, mask)
        q = tl.load(P + b + 2 + j % 16, mask, 0).to(tl.int32)
        code = (q >> (4 * (j // 16))) & 15
        grid = tl.load(GRID + code, mask, 0).to(tl.int8).to(tl.float32)
        value = d * grid
    elif TYPE == 23:  # IQ4_XS
        d = _half(P, b, mask)
        hi = _u16(P, b + 2, mask)
        lo = tl.load(P + b + 4 + j // 64, mask, 0).to(tl.int32)
        g = j // 32
        sc = (((lo >> (4 * (g % 2))) & 15) | (((hi >> (2 * g)) & 3) << 4)).to(tl.int32) - 32
        q = tl.load(P + b + 8 + g * 16 + j % 16, mask, 0).to(tl.int32)
        code = (q >> (4 * ((j % 32) // 16))) & 15
        grid = tl.load(GRID + code, mask, 0).to(tl.int8).to(tl.float32)
        value = (d * sc) * grid
    tl.store(O + i, value, mask)


def dequantize_cuda(packed, shape, quant_type, dtype=torch.float16):
    """Decode a contiguous CUDA uint8 buffer. No CPU fallback for supported types."""
    fmt = get_format(quant_type)
    if fmt.name not in GPU_FORMATS:
        raise ValueError(f'No GPU decoder for {fmt.name}')
    if not packed.is_cuda or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError('packed must be a contiguous CUDA uint8 tensor')
    if packed.numel() != tensor_nbytes(shape, fmt):
        raise ValueError('Packed bytes do not match shape and GGML format')
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('GPU decode supports float16, bfloat16 or float32')
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability(packed.device)[0] < 8:
        raise ValueError('Triton bfloat16 output requires compute capability 8.0 or newer')
    if fmt.name in ('F16', 'F32'):
        return packed.view(torch.float16 if fmt.name == 'F16' else torch.float32).reshape(shape).to(dtype)
    grid, signs = _tables(fmt.type_id, packed.device.index)
    out = torch.empty(shape, dtype=dtype, device=packed.device)
    n = math.prod(shape)
    _decode[(triton.cdiv(n, 256),)](packed, out, grid, signs, n,
        fmt.type_id, fmt.block_size, fmt.block_bytes, 256, enable_fp_fusion=False)
    return out
