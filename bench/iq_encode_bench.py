"""Check byte-identical parallel native encoding on one real expert projection."""
import json
from pathlib import Path
import sys
import time
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from safetensors import safe_open
from brmoe_quant.native import quantize

model=Path('/mnt/4090/data/jianglei/models/Qwen/qwen1.5-moe')
name='model.layers.0.mlp.experts.0.up_proj.weight'
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(str(model/index[name]),framework='pt',device='cpu') as stream:
    source=stream.get_tensor(name).float().numpy()
importance=np.ones(source.shape[-1],dtype=np.float32)
rows=[]
for qtype in ['Q2_K','Q3_K','Q4_K','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ4_XS','IQ4_NL']:
    baseline=quantize(source,qtype,importance,workers=1)
    timings={}
    for workers in (1,8):
        samples=[]
        for _ in range(3):
            start=time.perf_counter();actual=quantize(source,qtype,importance,workers=workers)
            samples.append(time.perf_counter()-start)
            np.testing.assert_array_equal(actual,baseline)
        timings[workers]=statistics.median(samples)
    row=dict(format=qtype,serial_seconds=timings[1],parallel_seconds=timings[8],speedup=timings[1]/timings[8])
    rows.append(row);print(json.dumps(row),flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2))
