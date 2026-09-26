"""A100 small-decode policy and rollback, calibrated on full-model requests."""
import os
from functools import lru_cache


def small_decode_enabled():
    mode = os.environ.get('BRMOE_SMALL_DECODE_BACKEND', 'auto')
    if mode not in ('auto', 'legacy'):
        raise ValueError(f'BRMOE_SMALL_DECODE_BACKEND must be auto or legacy, got {mode!r}')
    return mode == 'auto'


@lru_cache(maxsize=1)
def gemv_module():
    # Also works for callers that only initialized the linear quantization
    # method and have not loaded routed MoE weights yet.
    from .kernel import get_fused_moe_int3
    get_fused_moe_int3()
    from int3_moe import gemv_reduce
    return gemv_reduce
