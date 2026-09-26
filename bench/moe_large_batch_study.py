"""Full INT3 large-batch route capture and complete-MoE configuration study.

Capture/events are deliberately separate from the clean e2e benchmark. All
sampled tensors are real model inputs, after the last observed decode call.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools', 'bench', 'BR-MoE/kernels', 'BR-MoE/kernels/triton_int3'):
    sys.path.insert(0, str(ROOT/sub))
import torch
import triton
from full_int3_study import MODEL, TOKENIZER, graph_us
from moe_fuse_study import weights, e2e
from brmoe_int3_vllm.kernel import get_fused_moe_int3, get_moe_cuda_ext
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda


def collect(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import brmoe_int3_vllm.moe_method as method
    import brmoe_int3_vllm.linear_method as linear
    import vllm.model_executor.model_loader.base_loader as loader
    import marlin_int3_moe.fused_ops as fusion
    import int3_moe.align_triton as alignment
    from vllm_perf import make_prompt
    shapes = set(map(int, args.batch_sizes.split(',')))
    buffers, events = {}, {}
    current = [None]

    @contextlib.contextmanager
    def timed(key):
        if key not in events:
            events[key] = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        begin, end = events[key]
        begin.record()
        yield
        end.record()

    def wrap(fn, label):
        def call(*a, **kw):
            if current[0] is None: return fn(*a, **kw)
            name = label
            if label == 'matmul': name = 'w13' if a[3].shape[-1] == 2816 else 'w2'
            with timed((*current[0], name)): return fn(*a, **kw)
        return call

    class Launch:
        def __init__(self, jit, name): self.jit, self.name = jit, name
        def __getitem__(self, grid): return wrap(self.jit[grid], self.name)

    get_fused_moe_int3()
    ext = get_moe_cuda_ext()
    assert ext is not None
    ext.mul_3bit_moe = wrap(ext.mul_3bit_moe, 'matmul')
    alignment.moe_align_block_size_triton = wrap(alignment.moe_align_block_size_triton, 'align')
    for name, label in [('_gather_tokens','gather'),('_silu_active','activation'),('_reduce_routes','reduce')]:
        setattr(fusion, name, Launch(getattr(fusion, name), label))
    moe_original = method.brmoe_int3_moe

    def capture(x, tw, ids, layer, group_size, out_dtype=None):
        m = x.shape[0]
        if m not in shapes: return moe_original(x, tw, ids, layer, group_size, out_dtype)
        key = (layer.layer_name, m)
        if key not in buffers:
            buffers[key] = dict(x=torch.empty_like(x), weights=torch.empty_like(tw), ids=torch.empty_like(ids),
                                calls=torch.zeros((), dtype=torch.int32, device=x.device))
        for name, src in [('x',x),('weights',tw),('ids',ids)]: buffers[key][name].copy_(src)
        buffers[key]['calls'].add_(1)
        previous, current[0] = current[0], key
        try:
            with timed((*key, 'moe_total')): return moe_original(x, tw, ids, layer, group_size, out_dtype)
        finally: current[0] = previous
    method.brmoe_int3_moe = capture

    # Label attention and shared/dense projections without modifying their kernels.
    # Some attention constructors pass an empty prefix to their quant method.
    # Use actual model paths before weight processing to avoid profile-key collisions.
    process_original = loader.process_weights_after_loading
    def named(model, *a, **kw):
        for name, layer in model.named_modules(): layer._large_batch_name = name
        return process_original(model, *a, **kw)
    loader.process_weights_after_loading = named
    linear_names = {}
    create_original = linear.BRMoEInt3LinearMethod.process_weights_after_loading
    def remember(self, layer):
        create_original(self, layer)
        linear_names[layer.qweight.data_ptr()] = layer._large_batch_name
    linear.BRMoEInt3LinearMethod.process_weights_after_loading = remember
    linear_original = linear._brmoe_int3_linear_impl
    def timed_linear(x, w, s, z, gs):
        if x.shape[0] not in shapes: return linear_original(x,w,s,z,gs)
        with timed((linear_names[w.data_ptr()], x.shape[0], 'linear')):
            return linear_original(x,w,s,z,gs)
    linear._brmoe_int3_linear_impl = timed_linear
    llm = LLM(model=str(MODEL), tokenizer=str(TOKENIZER), tokenizer_mode='hf',
              quantization='brmoe_int3', dtype='float16', trust_remote_code=True,
              max_model_len=768, max_num_batched_tokens=2048, max_num_seqs=max(shapes),
              gpu_memory_utilization=.9, enable_prefix_caching=False, disable_log_stats=True)
    assert len(linear_names)>100 and len(set(linear_names.values()))==len(linear_names), linear_names
    assert all(linear_names.values()), 'every linear projection needs a unique nonempty path'
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompt = make_prompt(tok,128)
    samples, routes, stages = [], [], []
    for batch in sorted(shapes):
        with torch.inference_mode():
            for b in buffers.values(): b['calls'].zero_()
        llm.generate([prompt]*batch, SamplingParams(max_tokens=128,min_tokens=128,
                       temperature=0,ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        for (name,m), b in buffers.items():
            if m != batch or int(b['calls']) == 0: continue
            cpu = {k:b[k].cpu() for k in ('x','weights','ids')}
            samples.append(dict(batch=batch,m=m,layer=name,step='last_observed',**cpu))
            valid = (cpu['ids']>=0)&(cpu['weights']!=0)
            counts = torch.bincount(cpu['ids'][valid].long(),minlength=64)
            routes.append(dict(batch=batch,layer=name,counts=counts.tolist(),
                calls=int(b['calls']), active=int((counts>0).sum()),
                padded16=int(((counts+15)//16*16).sum()),valid_routes=int(valid.sum())))
        assert sum(s['m']==batch for s in samples)==27, 'decode batch was not captured in all 27 layers'
        for (name,m,stage), (begin,end) in events.items():
            if m==batch: stages.append(dict(batch=batch,layer=name,stage=stage,us=begin.elapsed_time(end)*1000))
        torch.save(samples,args.out/'real_inputs.pt')
        (args.out/'routes.json').write_text(json.dumps(routes,indent=2))
        (args.out/'stages.json').write_text(json.dumps(stages,indent=2))
        print('CAPTURE',batch,'layers=27',flush=True)


def micro(args):
    tri, ext = get_fused_moe_int3(), get_moe_cuda_ext()
    assert ext is not None
    data = torch.load(args.input,weights_only=True)
    names = sorted({s['layer'] for s in data})
    if args.quick: names=names[:1]
    packed = {name:weights(name) for name in names}
    records=[]
    for m in map(int,args.batch_sizes.split(',')):
        inputs=[{**s,**{key:s[key].cuda() for key in ('x','weights','ids')}}
                for s in data if s['m']==m and s['layer'] in names]
        assert len(inputs)==len(names)
        def one(s,config):
            p,pk=packed[s['layer']]
            if config=='baseline':
                os.environ['BRMOE_MOE_SMEM']='legacy'
            if config.startswith(('legacy_', 'rightsize_')):
                mode, config = config.split('_',1)
                os.environ['BRMOE_MOE_SMEM'] = mode
            if config.startswith('triton_'):
                bk,stages=map(int,config.split('_')[1:])
                return tri(s['x'],s['weights'],s['ids'],p,fast=True,gemv=False,block_k=bk,num_stages=stages)
            cfg=None if config=='baseline' else tuple(map(int,config.split('_')))
            return fused_moe_int3_cuda(s['x'],s['weights'],s['ids'],pk,ext,packed=p,
                                      fuse_ops=config!='baseline',gemv_max_m=0,cfg=cfg)
        reference=[one(s,'baseline').clone().float() for s in inputs]
        for config in args.configs.split(','):
            errors=[]
            for s,ref in zip(inputs,reference):
                got=one(s,config).float()
                rel=float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel<.005,(m,config,s['layer'],rel)
                errors.append(rel)
            def fn():
                for s in inputs: one(s,config)
            us=graph_us(fn,repeats=7)
            rec=dict(m=m,config=config,layers=len(inputs),us=us,max_relative_l2=max(errors),
                     smem_mode=os.environ.get('BRMOE_MOE_SMEM','legacy'))
            records.append(rec);print(json.dumps(rec),flush=True)
            (args.out/'micro.json').write_text(json.dumps(records,indent=2))
            if args.profile:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): fn()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay();torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.out/f'trace_m{m}_{config}.json'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['collect','micro','e2e'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--input',type=Path)
    ap.add_argument('--batch-sizes',default='32,64,128')
    ap.add_argument('--configs',default='baseline,128_128_4,128_128_3,128_128_5')
    ap.add_argument('--config',default='fused4')
    ap.add_argument('--cuda-cfg',default=None)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--profile',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',BRMOE_GROUPED_GEMV='0',
                      BRMOE_CUDA_FUSE='1' if args.phase=='collect' else '0')
    metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
                  smem_mode=os.environ.get('BRMOE_MOE_SMEM','legacy'),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    import hashlib
    get_fused_moe_int3()
    extension=get_moe_cuda_ext()
    assert extension is not None
    binary=Path(extension.__file__).resolve()
    metadata.update(extension=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (args.out/f'metadata_{args.phase}_{args.config}.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    dict(collect=collect,micro=micro,e2e=e2e)[args.phase](args)


if __name__=='__main__': main()
