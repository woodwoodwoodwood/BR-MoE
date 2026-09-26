"""Rollback switch for the A100 prefill policy validated on DeepSeek-MoE."""
import os


def prefill_enabled():
    mode = os.environ.get('BRMOE_PREFILL_BACKEND', 'auto')
    if mode not in ('auto', 'legacy'):
        raise ValueError(f'BRMOE_PREFILL_BACKEND must be auto or legacy, got {mode!r}')
    return mode == 'auto'
