"""GGML tensor types, not model-wide presets such as Q4_K_M.

Block sizes include all scales/codebook indices. Shapes must satisfy the actual
GGML row alignment; silently padding a row produces a different model contract.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class QuantFormat:
    name: str
    type_id: int
    block_size: int
    block_bytes: int
    bits: int
    requires_imatrix: bool = False


FORMATS = {f.name: f for f in (
    QuantFormat("F32", 0, 1, 4, 32),
    QuantFormat("F16", 1, 1, 2, 16),
    QuantFormat("Q4_0", 2, 32, 18, 4),
    QuantFormat("Q4_1", 3, 32, 20, 4),
    QuantFormat("Q5_0", 6, 32, 22, 5),
    QuantFormat("Q5_1", 7, 32, 24, 5),
    QuantFormat("Q8_0", 8, 32, 34, 8),
    QuantFormat("Q2_K", 10, 256, 84, 2),
    QuantFormat("Q3_K", 11, 256, 110, 3),
    QuantFormat("Q4_K", 12, 256, 144, 4),
    QuantFormat("Q5_K", 13, 256, 176, 5),
    QuantFormat("Q6_K", 14, 256, 210, 6),
    QuantFormat("IQ2_XXS", 16, 256, 66, 2, True),
    QuantFormat("IQ2_XS", 17, 256, 74, 2, True),
    QuantFormat("IQ3_XXS", 18, 256, 98, 3),
    QuantFormat("IQ4_NL", 20, 32, 18, 4),
    QuantFormat("IQ3_S", 21, 256, 110, 3),
    QuantFormat("IQ2_S", 22, 256, 82, 2),
    QuantFormat("IQ4_XS", 23, 256, 136, 4),
)}

BIT_DEFAULTS = {2: "Q2_K", 3: "Q3_K", 4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0", 16: "F16", 32: "F32"}


def get_format(value) -> QuantFormat:
    if isinstance(value, QuantFormat):
        return value
    name = BIT_DEFAULTS.get(value, value) if isinstance(value, int) else str(value).strip().upper()
    if name not in FORMATS:
        raise ValueError(f"Unsupported GGML tensor type {value!r}; choose from {', '.join(FORMATS)}. Model presets like Q4_K_M are not tensor types.")
    return FORMATS[name]


def parse_option(value):
    """Accept historical '3bit_rank16' and explicit 'IQ3_S_rank16' options."""
    value = str(value)
    if value.lower() in ("prune", "0bit_rank0"):
        return 0, 0
    m = re.fullmatch(r"(\d+)bit_rank(\d+)", value)
    if m:
        bits, rank = map(int, m.groups())
        get_format(bits)
        return bits, rank
    m = re.fullmatch(r"([A-Za-z0-9_]+)_rank(\d+)", value)
    if not m:
        raise ValueError(f"Invalid quantization option: {value!r}")
    return get_format(m[1]).name, int(m[2])


def tensor_nbytes(shape, quant_type) -> int:
    fmt = get_format(quant_type)
    shape = tuple(int(x) for x in shape)
    if not shape or any(x <= 0 for x in shape):
        raise ValueError(f"Invalid tensor shape {shape}")
    if shape[-1] % fmt.block_size:
        raise ValueError(f"{fmt.name} needs row width divisible by {fmt.block_size}, got {shape[-1]}")
    return math.prod(shape) // fmt.block_size * fmt.block_bytes
