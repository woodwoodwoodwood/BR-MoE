"""Measure native CPU-to-CUDA versus exact GPU decode on real model weights."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from safetensors import safe_open
from brmoe_quant.native import quantize,dequantize
from brmoe_quant.gpu import dequantize_cuda


def timed(fn, repeats=10):
    fn(); torch.cuda.synchronize()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return statistics.median(times)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();path=Path(args.model)
    index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    rows=[]
    for name in ['model.layers.0.self_attn.q_proj.weight','model.layers.0.mlp.shared_expert.up_proj.weight','model.layers.0.mlp.experts.0.up_proj.weight']:
        with safe_open(str(path/index[name]),framework='pt',device='cpu') as stream:
            source=stream.get_tensor(name).float().numpy()
        for qtype in ['Q2_K','Q3_K','Q4_K','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_XS','IQ4_NL']:
            raw=quantize(source,qtype,np.ones(source.shape[-1],np.float32))
            cpu=torch.from_numpy(raw); packed=cpu.cuda()
            expected=torch.from_numpy(dequantize(raw,source.shape,qtype)).to('cuda',torch.float16)
            actual=dequantize_cuda(packed,source.shape,qtype)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            old=timed(lambda:torch.from_numpy(dequantize(raw,source.shape,qtype)).to('cuda',torch.float16))
            transfer=timed(lambda:dequantize_cuda(cpu.to('cuda'),source.shape,qtype))
            resident=timed(lambda:dequantize_cuda(packed,source.shape,qtype))
            row=dict(module=name,format=qtype,shape=list(source.shape),cpu_decode_and_transfer_ms=old,
                gpu_decode_and_transfer_ms=transfer,gpu_resident_decode_ms=resident,
                transfer_speedup=old/transfer,resident_speedup=old/resident)
            print(json.dumps(row),flush=True);rows.append(row)
            Path(args.out).write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
