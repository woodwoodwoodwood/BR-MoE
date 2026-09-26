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
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    from vllm_perf import make_prompt
    batches = sorted(set(map(int, args.batch_sizes.split(','))))
    prefill_only = getattr(args, 'prefill_only', False)
    shapes = {min(b * 128, 2048) for b in batches} if prefill_only else set(batches)
    buffers, events = {}, {}
    current = [None]
    shared_orders = {}
    order_original = SharedExperts._determine_shared_experts_order
    def record_order(self, hidden_states):
        order = order_original(self, hidden_states)
        name = getattr(self._layer, '_large_batch_name', type(self._layer).__name__)
        key = (name, hidden_states.shape[0], order.name)
        shared_orders[key] = shared_orders.get(key, 0) + 1
        return order
    SharedExperts._determine_shared_experts_order = record_order

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
    import int3_moe.grouped_tc as grouped_tc
    ext = get_moe_cuda_ext()
    assert ext is not None
    ext.mul_3bit_moe = wrap(ext.mul_3bit_moe, 'matmul')
    alignment.moe_align_block_size_triton = wrap(alignment.moe_align_block_size_triton, 'align')
    # The prefill fallback imports these functions into int3_moe.ops directly.
    # Instrument those aliases as well as the CUDA decode path above.
    import int3_moe.ops as tri_ops
    tri_ops.moe_align_block_size_triton = wrap(tri_ops.moe_align_block_size_triton, 'align')
    tri_gemm = tri_ops.int3_moe_gemm
    def timed_gemm(*a, **kw):
        return wrap(tri_gemm, 'w13' if kw.get('a_gather') else 'w2')(*a, **kw)
    tri_ops.int3_moe_gemm = timed_gemm
    tri_ops.silu_mul = wrap(tri_ops.silu_mul, 'activation')
    old_gemv = tri_ops.routed_int3_gemv
    def timed_gemv(*a, **kw):
        return wrap(old_gemv, 'w2' if kw.get('add') else 'w13')(*a, **kw)
    tri_ops.routed_int3_gemv = timed_gemv
    tri_ops.silu_mul_routes = wrap(tri_ops.silu_mul_routes, 'activation')
    import int3_moe.gemv_reduce as small_gemv
    small_partials = small_gemv.partials
    def timed_partials(*a, **kw):
        return wrap(small_partials, 'w2' if kw.get('gather_route') else 'w13')(*a, **kw)
    small_gemv.partials = timed_partials
    small_gemv.sanitize_routing = wrap(small_gemv.sanitize_routing, 'sanitize')
    small_gemv._reduce_silu = Launch(small_gemv._reduce_silu, 'activation')
    small_gemv._reduce_weighted = Launch(small_gemv._reduce_weighted, 'reduce')
    grouped_tc.moe_align_block_size_triton = wrap(grouped_tc.moe_align_block_size_triton, 'align')
    tc_original = grouped_tc.grouped_int3_tc
    def timed_tc(*a, **kw):
        return wrap(tc_original, 'w13' if kw.get('gather') else 'w2')(*a, **kw)
    grouped_tc.grouped_int3_tc = timed_tc
    grouped_tc.silu_mul = wrap(grouped_tc.silu_mul, 'activation')
    grouped_tc._reduce_routes_tc = Launch(grouped_tc._reduce_routes_tc, 'reduce')
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
    linear_names, linear_packed, linear_buffers, representatives = {}, {}, {}, {}
    save_linear = getattr(args, 'save_linear', False)
    create_original = linear.BRMoEInt3LinearMethod.process_weights_after_loading
    def remember(self, layer):
        create_original(self, layer)
        name = layer._large_batch_name
        linear_names[layer.qweight.data_ptr()] = name
        if save_linear:
            linear_packed[name] = (layer.qweight, layer.scales, layer.zeros, layer._br_int3['group_size'])
            representatives.setdefault(tuple(layer.scales.shape),name)
    linear.BRMoEInt3LinearMethod.process_weights_after_loading = remember
    linear_original = linear._brmoe_int3_linear_impl
    def timed_linear(x, w, s, z, gs):
        name,m = linear_names[w.data_ptr()],x.shape[0]
        selected_prefill = save_linear and m in (512,2048) and name in representatives.values()
        if m not in shapes and not selected_prefill: return linear_original(x,w,s,z,gs)
        # Prefill replay needs routed inputs and shared weights; retain just the
        # six representative linear inputs to keep the capture footprint small.
        if save_linear and (not prefill_only or name in representatives.values()):
            key=(name,m)
            if key not in linear_buffers:
                linear_buffers[key]=dict(x=torch.empty_like(x),calls=torch.zeros((),dtype=torch.int32,device=x.device))
            linear_buffers[key]['x'].copy_(x)
            linear_buffers[key]['calls'].add_(1)
        with timed((name,m,'linear')):
            return linear_original(x,w,s,z,gs)
    linear._brmoe_int3_linear_impl = timed_linear
    llm = LLM(model=str(MODEL), tokenizer=str(TOKENIZER), tokenizer_mode='hf',
              quantization='brmoe_int3', dtype='float16', trust_remote_code=True,
              max_model_len=768, max_num_batched_tokens=2048,
              max_num_seqs=512 if prefill_only else max(shapes),
              gpu_memory_utilization=.9, enable_prefix_caching=False, disable_log_stats=True)
    assert len(linear_names)>100 and len(set(linear_names.values()))==len(linear_names), linear_names
    assert all(linear_names.values()), 'every linear projection needs a unique nonempty path'
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompt = make_prompt(tok,128)
    samples, routes, stages, linear_samples = [], [], [], []
    for batch in batches:
        with torch.inference_mode():
            for b in buffers.values(): b['calls'].zero_()
            for b in linear_buffers.values(): b['calls'].zero_()
        output_len = 1 if prefill_only else 128
        llm.generate([prompt]*batch, SamplingParams(max_tokens=output_len,min_tokens=output_len,
                       temperature=0,ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        for (name,m), b in buffers.items():
            if (not prefill_only and m != batch) or int(b['calls']) == 0: continue
            cpu = {k:b[k].cpu() for k in ('x','weights','ids')}
            samples.append(dict(batch=batch,m=m,layer=name,step='last_observed',**cpu))
            valid = (cpu['ids']>=0)&(cpu['weights']!=0)
            counts = torch.bincount(cpu['ids'][valid].long(),minlength=64)
            routes.append(dict(batch=batch,m=m,layer=name,counts=counts.tolist(),
                calls=int(b['calls']), active=int((counts>0).sum()),
                padded16=int(((counts+15)//16*16).sum()),valid_routes=int(valid.sum())))
        expected_m = min(batch*128,2048) if prefill_only else batch
        assert sum(s['batch']==batch and s['m']==expected_m for s in samples)==27, 'requested shape was not captured in all 27 layers'
        for (name,m,stage), (begin,end) in events.items():
            if m==expected_m: stages.append(dict(batch=batch,m=m,layer=name,stage=stage,us=begin.elapsed_time(end)*1000))
        if save_linear:
            for (name,m),b in linear_buffers.items():
                if (m==batch or m in (512,2048)) and int(b['calls'])>0:
                    linear_samples.append(dict(batch=batch,m=m,layer=name,x=b['x'].cpu(),calls=int(b['calls'])))
            if not prefill_only:
                assert sum(s['batch']==batch and s['m']==batch for s in linear_samples)==112
            torch.save(linear_samples,args.out/'linear_inputs.pt')
        torch.save(samples,args.out/'real_inputs.pt')
        (args.out/'routes.json').write_text(json.dumps(routes,indent=2))
        (args.out/'stages.json').write_text(json.dumps(stages,indent=2))
        print('CAPTURE',batch,'layers=27',flush=True)
    (args.out/'shared_orders.json').write_text(json.dumps([
        dict(layer=name,m=m,order=order,python_calls=count)
        for (name,m,order),count in sorted(shared_orders.items())],indent=2))
    if save_linear:
        torch.save({name:tuple(v.detach().cpu() if isinstance(v,torch.Tensor) else v for v in values)
                    for name,values in linear_packed.items()},args.out/'linear_weights.pt')
        print('LINEAR CAPTURE',len(linear_samples),'inputs',len(linear_packed),'weights',flush=True)


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
    ap.add_argument('--save-linear',action='store_true')
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
