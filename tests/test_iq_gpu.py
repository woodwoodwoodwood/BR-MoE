"""Exact native-block decoding and RCO-gradient parity on CUDA."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from brmoe_quant.native import quantize, dequantize
from RCO_solver.device_cache import DeviceCache


def test_device_cache_bound_and_lru():
    cache = DeviceCache(16)
    cache.put('a', (torch.ones(2),))
    cache.put('b', (torch.ones(2),))
    assert cache.get('a') is not None
    cache.put('c', (torch.ones(2),))
    assert cache.get('b') is None
    assert cache.bytes == 16
    cache.put('oversized', (torch.ones(5),))
    assert cache.bytes == 16
    cache.put('a', (torch.ones(1),))
    assert cache.bytes == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('qtype', ['Q2_K','Q3_K','Q4_K','IQ2_XXS','IQ2_XS',
    'IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS','F16','F32'])
def test_cuda_decode_native_parity(qtype):
    from brmoe_quant.gpu import dequantize_cuda
    rng = np.random.default_rng(9)
    for shape in [(3, 256), (37, 512), (2, 32) if qtype in ('IQ4_NL','F16','F32') else (1, 256)]:
        for source in [rng.normal(size=shape).astype(np.float32), np.zeros(shape, np.float32)]:
            raw = quantize(source, qtype, np.linspace(.01,2,shape[-1],dtype=np.float32))
            packed = torch.from_numpy(raw).cuda()
            expected = torch.from_numpy(dequantize(raw, shape, qtype)).cuda()
            dtypes = [torch.float32, torch.float16]
            if torch.cuda.get_device_capability()[0] >= 8:
                dtypes.append(torch.bfloat16)
            for dtype in dtypes:
                actual = dequantize_cuda(packed, shape, qtype, dtype)
                torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_rco_gpu_and_cpu_decode_gradients(monkeypatch):
    from RCO_solver.e2e_ste import RCOInterpolatedLinear
    torch.manual_seed(71)
    original = torch.nn.Linear(256, 32, bias=False, device='cuda',dtype=torch.float16)
    options = ['IQ2_XS_rank0','IQ2_XS_rank16','IQ3_S_rank0','IQ3_S_rank32','IQ4_NL_rank16','F16_rank0']
    shared = {'importance':{'projection': np.ones(256, dtype=np.float32)}}
    layer = RCOInterpolatedLinear(original,0,options,shared,module_name='projection')
    x0 = torch.randn(1,5,256,device='cuda',dtype=torch.float16)
    records = []
    for backend in ('cpu','gpu'):
        monkeypatch.setenv('BRMOE_RCO_DECODE_BACKEND',backend)
        x = x0.clone().requires_grad_()
        p = torch.ones(1,len(options),device='cuda',dtype=torch.float64,requires_grad=True)
        shared.update(p_ste=p,k_star=torch.tensor([1],device='cuda'))
        output = layer(x)
        output.float().square().sum().backward()
        records.append((output.detach(),x.grad,p.grad))
        # Independent scalar candidate assembly checks the vectorized rank gather.
        expected = torch.zeros(len(options), device='cuda', dtype=torch.float32)
        go = (2 * output.detach()).reshape(-1, 32)
        flat_x = x.detach().reshape(-1, 256)
        for b in layer.unique_bits:
            base, u, v = layer.get_bit_factors(b, x.device, x.dtype)
            bg = (go.float() * (flat_x @ base.T).float()).sum()
            cs = None if u is None else torch.cumsum(
                ((flat_x @ v.T).float() * (go @ u).float()).sum(0), 0)
            for k, (candidate, rank) in enumerate(layer.parsed_options):
                if candidate == b:
                    expected[k] = bg if rank == 0 else bg + cs[min(rank,cs.numel())-1]
        torch.testing.assert_close(p.grad[0],expected.double(),atol=0,rtol=0)
    for before, after in zip(*records):
        torch.testing.assert_close(before,after,atol=0,rtol=0)


@pytest.mark.parametrize('qtype', ['Q2_K','Q3_K','Q4_K','IQ2_XXS','IQ2_XS',
    'IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS'])
def test_parallel_native_encoder_matches_serial(qtype):
    rng = np.random.default_rng(33)
    source = rng.normal(size=(2, 19, 256)).astype(np.float32)
    importance = np.linspace(.02,3,256,dtype=np.float32)
    expected = quantize(source,qtype,importance,workers=1)
    actual = quantize(source,qtype,importance,workers=4)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('keep_original,keep_factors', [(0,0),(1,0),(0,1),(1,1)])
def test_rco_large_gpu_residency(monkeypatch,keep_original,keep_factors):
    from RCO_solver.e2e_ste import RCOInterpolatedLinear
    monkeypatch.setenv('BRMOE_RCO_KEEP_ORIGINAL_ON_GPU',str(keep_original))
    monkeypatch.setenv('BRMOE_RCO_KEEP_FACTORS_ON_GPU',str(keep_factors))
    original = torch.nn.Linear(256,32,bias=False,device='cuda',dtype=torch.float16)
    shared = {'p_ste':torch.ones(1,3,device='cuda',dtype=torch.float64,requires_grad=True),
              'k_star':torch.tensor([1],device='cuda')}
    layer = RCOInterpolatedLinear(original,0,
        ['IQ4_NL_rank0','IQ4_NL_rank8','F16_rank0'],shared,module_name='projection')
    assert layer._orig_weight.is_cuda == bool(keep_original)
    base,u,v = layer.get_bit_factors('IQ4_NL',original.weight.device,original.weight.dtype)
    assert all(t.is_cuda == bool(keep_factors) for t in layer._uv_cache['IQ4_NL'])
    raw = layer._packed_base_cache['IQ4_NL']
    expected = torch.from_numpy(dequantize(raw,(32,256),'IQ4_NL')).to('cuda',torch.float16)
    torch.testing.assert_close(base,expected,atol=0,rtol=0)
    if keep_factors:
        assert not any(key[-1]=='uv' for key in shared['device_cache'].entries)
    x = torch.randn(2,256,device='cuda',dtype=torch.float16,requires_grad=True)
    actual = layer(x)
    actual.float().square().sum().backward()
    assert torch.isfinite(actual).all() and torch.isfinite(shared['p_ste'].grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('qtype',['Q2_K','Q3_K','Q4_K','IQ2_XXS','IQ2_XS',
    'IQ2_S','IQ3_XXS','IQ3_S','IQ4_NL','IQ4_XS','F16','F32'])
def test_compressed_inference_gpu_decode(qtype):
    from examples.BR_compress_auto import br_moe_base_compress_config
    from BR_MoE.core.quantize import Quantizer
    source = np.random.default_rng(17).normal(size=(7,256)).astype(np.float32)
    raw = quantize(source,qtype,np.ones(256,dtype=np.float32))
    packed = torch.from_numpy(raw).cuda()
    meta = {'quant_format':'gguf','quant_type':qtype,'packing':'gguf_'+qtype.lower(),
            'shape':source.shape,'compute_dtype':torch.float16}
    actual = Quantizer.dequantize(packed,meta)
    expected = torch.from_numpy(dequantize(raw,source.shape,qtype)).to('cuda',torch.float16)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
