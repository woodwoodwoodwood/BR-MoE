"""CPU llama.cpp codecs shared by search, compression and GGUF export.

Named GGML formats are encoded by upstream ggml_quantize_chunk. IQ2_XS/XXS
require input-channel importance; callers must supply calibration statistics or
explicitly supply uniform weights. No silent format substitution is performed.
"""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import sys
import threading

import numpy as np
from .formats import FORMATS, get_format, tensor_nbytes

REVISION = "ceefd16daa2093730bf3340315fa2b84b286ddbc"
_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def library():
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    default = Path.home() / ".cache/brmoe-ggml" / REVISION / "build" / ("libbrmoe_ggml_quant" + suffix)
    path = Path(os.environ.get("BRMOE_GGML_QUANT_LIB", str(default)))
    if not path.is_file():
        raise RuntimeError("Native GGML quantizer is missing. Run: python tools/build_ggml_quant.py "
                           "(or set BRMOE_GGML_QUANT_LIB to the built bridge)")
    lib = ct.CDLL(str(path))
    lib.ggml_blck_size.argtypes = [ct.c_int]
    lib.ggml_blck_size.restype = ct.c_int64
    lib.ggml_type_size.argtypes = [ct.c_int]
    lib.ggml_type_size.restype = ct.c_size_t
    lib.ggml_quantize_requires_imatrix.argtypes = [ct.c_int]
    lib.ggml_quantize_requires_imatrix.restype = ct.c_bool
    lib.ggml_quantize_chunk.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
    lib.ggml_quantize_chunk.restype = ct.c_size_t
    lib.brmoe_dequantize.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int64]
    lib.brmoe_dequantize.restype = ct.c_int
    for fmt in FORMATS.values():
        if lib.ggml_blck_size(fmt.type_id) != fmt.block_size or lib.ggml_type_size(fmt.type_id) != fmt.block_bytes:
            raise RuntimeError(f"GGML ABI/format mismatch for {fmt.name}: rebuild the pinned bridge")
    return lib


def quantize(x, quant_type, importance=None):
    fmt = get_format(quant_type)
    x = np.ascontiguousarray(x, dtype=np.float32)
    nbytes = tensor_nbytes(x.shape, fmt)
    if not np.isfinite(x).all():
        raise ValueError("Quantization input contains NaN or infinity")
    if fmt.name in ("F16", "F32"):
        return x.astype(np.float16 if fmt.name == "F16" else np.float32).view(np.uint8).reshape(-1).copy()
    lib = library()
    imp = None
    if importance is not None:
        imp = np.ascontiguousarray(importance, dtype=np.float32)
        if imp.shape != (x.shape[-1],) or not np.isfinite(imp).all() or (imp < 0).any() or not (imp > 0).any():
            raise ValueError(f"Importance must be a finite, nonnegative vector of width {x.shape[-1]} with positive mass")
        imp = np.maximum(imp, np.finfo(np.float32).tiny)
    if lib.ggml_quantize_requires_imatrix(fmt.type_id) and imp is None:
        raise ValueError(f"{fmt.name} requires input-channel importance; collect calibration statistics or explicitly use uniform importance")
    out = np.empty(nbytes, dtype=np.uint8)
    with _LOCK:
        size = lib.ggml_quantize_chunk(fmt.type_id, x.ctypes.data, out.ctypes.data,
                                       0, x.size // x.shape[-1], x.shape[-1],
                                       None if imp is None else imp.ctypes.data)
    if size != nbytes:
        raise RuntimeError(f"GGML returned {size} bytes, expected {nbytes} for {fmt.name}")
    return out


def dequantize(data, shape, quant_type):
    fmt = get_format(quant_type)
    data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    if data.nbytes != tensor_nbytes(shape, fmt):
        raise ValueError("Packed payload size does not match GGML tensor shape/type")
    if fmt.name in ("F16", "F32"):
        return data.view(np.float16 if fmt.name == "F16" else np.float32).astype(np.float32).reshape(shape)
    out = np.empty(shape, dtype=np.float32)
    if library().brmoe_dequantize(fmt.type_id, data.ctypes.data, out.ctypes.data, out.size):
        raise RuntimeError(f"No GGML decoder for {fmt.name}")
    return out


def quant_dequant_torch(weight, quant_type, importance=None):
    import torch
    if isinstance(importance, torch.Tensor):
        importance = importance.detach().float().cpu().numpy()
    packed = quantize(weight.detach().float().cpu().numpy(), quant_type, importance)
    return torch.from_numpy(dequantize(packed, tuple(weight.shape), quant_type)).to(weight.device, weight.dtype)
