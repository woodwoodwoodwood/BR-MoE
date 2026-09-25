"""A100 full-INT3 real-route capture, CUDA-graph stage timing and MoE replay.

Instrumentation runs separately from latency measurements. External CUDA events
and device copies are captured into graphs, so replay updates real route data.
Only GPU kernel events are summed; nested full-MoE ranges are reported separately.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/709/data3/home/jianglei')
MODEL = BASE / 'models/brmoe-3bit-vllm-int3dense'
TOKENIZER = BASE / 'models/brmoe-tokfix'
INSTALLED = False


def configure_linear(policy):
    import brmoe_int3_vllm.linear_method as linear
    old_tiles, old_gemm = linear.pick_tiles, linear.int3_moe_gemm
    def tiles(K,N,gs,M=None):
        bm,bn,bk,slot=old_tiles(K,N,gs,M)
        if policy!='baseline':
            slot=bm
            if policy!='tile':
                requested=int(policy.split('_')[1])
                bk=next(k for k in (requested,64,32) if K%k==0)
        return bm,bn,bk,slot
    def gemm(*a,**kw):
        if policy not in ('baseline','tile'):
            kw['num_stages']=int(policy.split('_')[2])
        return old_gemm(*a,**kw)
    linear.pick_tiles,linear.int3_moe_gemm=tiles,gemm
    return old_tiles,old_gemm


def install():
    """Optional experiment-only hook, installed in each vLLM worker."""
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    import brmoe_int3_vllm.kernel as plugin
    original = plugin.get_fused_moe_int3()
    config = os.environ.get('FULL_INT3_CONFIG', 'baseline')
    if config != 'baseline':
        bk, stages = map(int, config.split('_'))

        def tuned(x, tw, ids, packed, **kw):
            if x.shape[0] > 4:
                kw.update(block_k=bk, num_stages=stages)
            return original(x, tw, ids, packed, **kw)
        plugin._FUSED = tuned
    linear_policy=os.environ.get('FULL_INT3_LINEAR','baseline')
    if linear_policy!='baseline':configure_linear(linear_policy)
    print(f'[full_int3] config={config} source={plugin.__file__}', flush=True)


def collect(args):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import torch
    import brmoe_int3_vllm.kernel as plugin
    plugin.get_fused_moe_int3()
    import int3_moe.ops as ops
    import brmoe_int3_vllm.moe_method as method
    import brmoe_int3_vllm.linear_method as linear
    from brmoe_int3_vllm.config import BRMoEInt3Config
    original_quant_method = BRMoEInt3Config.get_quant_method
    def named_quant_method(self, layer, prefix):
        layer._full_int3_study_name = prefix
        return original_quant_method(self, layer, prefix)
    BRMoEInt3Config.get_quant_method = named_quant_method

    shapes = {8, 32, 1024, 2048}
    buffers, events, packed_by_name = {}, {}, {}
    linear_buffers, linear_packed, linear_samples = {}, {}, []
    current = [None]

    @contextlib.contextmanager
    def timed(key):
        if key not in events:
            events[key] = [torch.cuda.Event(enable_timing=True, external=True)
                           for _ in range(2)]
        start, end = events[key]
        start.record()
        yield
        end.record()

    original_moe = method.brmoe_int3_moe

    def capture_moe(x, tw, ids, layer, group_size, out_dtype=None):
        m = x.shape[0]
        name = layer.layer_name
        if m not in shapes:
            return original_moe(x, tw, ids, layer, group_size, out_dtype)
        key = (name, m)
        packed_by_name[name] = plugin.build_packed(layer, group_size)
        if key not in buffers:
            buffers[key] = dict(x=torch.empty_like(x), weights=torch.empty_like(tw),
                                ids=torch.empty_like(ids), count=torch.zeros((), device=x.device))
        b = buffers[key]
        b['x'].copy_(x)
        b['weights'].copy_(tw)
        b['ids'].copy_(ids)
        b['count'].add_(1)
        old = current[0]
        current[0] = key
        with timed((*key, 'moe_total')):
            result = original_moe(x, tw, ids, layer, group_size, out_dtype)
        current[0] = old
        return result

    method.brmoe_int3_moe = capture_moe

    def stage(fn, label):
        def wrapped(*a, **kw):
            if current[0] is None:
                return fn(*a, **kw)
            tag = label
            if label == 'gemm':
                tag = 'w2' if kw.get('add') else 'w13'
            with timed((*current[0], tag)):
                return fn(*a, **kw)
        return wrapped

    ops.int3_moe_gemm = stage(ops.int3_moe_gemm, 'gemm')
    ops.moe_align_block_size_triton = stage(ops.moe_align_block_size_triton, 'align')
    ops.silu_mul = stage(ops.silu_mul, 'activation')
    original_linear = linear._brmoe_int3_linear_impl
    linear_names = {}
    original_create = linear.BRMoEInt3LinearMethod.process_weights_after_loading

    def remember_linear(self, layer):
        original_create(self, layer)
        linear_names[layer.qweight.data_ptr()] = layer._full_int3_study_name
    linear.BRMoEInt3LinearMethod.process_weights_after_loading = remember_linear

    def timed_linear(x, w, s, z, gs):
        m = x.shape[0]
        if m not in shapes:
            return original_linear(x, w, s, z, gs)
        name = linear_names.get(w.data_ptr(), f'linear_{w.shape[0]}_{w.shape[1]}_{w.data_ptr()}')
        linear_packed[name] = (w, s, z, gs)
        if m <= 32 or '.layers.0.' in name or '.layers.14.' in name:
            key = (name, m)
            if key not in linear_buffers:
                linear_buffers[key] = torch.empty_like(x)
            linear_buffers[key].copy_(x)
        with timed((name, m, 'linear_total')):
            return original_linear(x, w, s, z, gs)
    linear._brmoe_int3_linear_impl = timed_linear

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm_perf import make_prompt
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompt = make_prompt(tok, 128)
    assert len(tok.encode(prompt)) == 128
    llm = LLM(model=str(MODEL), tokenizer=str(TOKENIZER), tokenizer_mode='hf',
              quantization='brmoe_int3', dtype='float16', trust_remote_code=True,
              max_model_len=768, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.90, enforce_eager=False,
              enable_prefix_caching=False, disable_log_stats=True)
    print('[capture] loaded', len(packed_by_name), 'MoE layers,', len(linear_names), 'linear layers', flush=True)
    assert len(packed_by_name) == 27
    assert len(linear_names) > 100, 'expected attention and shared/dense INT3 projections'
    assert len(set(linear_names.values())) == len(linear_names), 'each linear layer needs a unique profile label'
    args.out.mkdir(parents=True, exist_ok=True)
    stage_rows, routes, samples = [], [], []
    saved_steps = {0, 1, 2, 16, 64, 127, 128}
    original_step = llm.llm_engine.step
    tracking = dict(batch=0, step=0, active=False)
    previous = {}

    def after_step():
        result = original_step()
        if not tracking['active']:
            return result
        torch.cuda.synchronize()
        bs, step = tracking['batch'], tracking['step']
        active_shapes = set()
        for key, b in buffers.items():
            count = int(b['count'].item())
            if count == previous.get(key):
                continue
            delta = count - previous.get(key, count)
            previous[key] = count
            name, m = key
            active_shapes.add(m)
            ids, tw = b['ids'].cpu(), b['weights'].cpu()
            valid = (ids >= 0) & (tw != 0)
            cnt = torch.bincount(ids[valid].long(), minlength=64)
            routes.append(dict(batch=bs, step=step, layer=name, m=m, invocations=delta,
                               counts=cnt.tolist(), active_experts=int((cnt > 0).sum()),
                               max_tokens_per_expert=int(cnt.max()),
                               padded16=int(((cnt+15)//16*16).sum()),
                               padded64=int(((cnt+63)//64*64).sum())))
            if step in saved_steps:
                samples.append(dict(batch=bs, step=step, layer=name, m=m,
                                    x=b['x'].cpu(), weights=tw, ids=ids))
        for (name, m, stage_name), (start, end) in events.items():
            if m in active_shapes:
                stage_rows.append(dict(batch=bs, step=step, layer=name, m=m,
                                       stage=stage_name, us=start.elapsed_time(end)*1000))
        if step in saved_steps:
            for (name, m), xbuf in linear_buffers.items():
                if m in active_shapes:
                    linear_samples.append(dict(batch=bs,step=step,layer=name,m=m,x=xbuf.cpu()))
        tracking['step'] += 1
        if step % 32 == 0:
            print(f'[capture] batch={bs} step={step} M={sorted(active_shapes)}', flush=True)
        return result

    llm.llm_engine.step = after_step
    output_tokens = {}
    for bs in (8, 32):
        llm.generate([prompt]*bs, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        previous.update({key:int(b['count'].item()) for key,b in buffers.items()})
        tracking.update(batch=bs, step=0, active=True)
        out = llm.generate([prompt]*bs, SamplingParams(max_tokens=128, temperature=0, ignore_eos=True), use_tqdm=False)
        tracking['active'] = False
        output_tokens[bs] = [o.outputs[0].token_ids for o in out]
        print(f'[capture] batch={bs} complete steps={tracking["step"]}', flush=True)
        torch.save(samples, args.out/'real_inputs.pt')
        torch.save(linear_samples, args.out/'linear_inputs.pt')
        (args.out/'routes.json').write_text(json.dumps(routes))
        (args.out/'stages.json').write_text(json.dumps(stage_rows))
        (args.out/'tokens.json').write_text(json.dumps(output_tokens))
        # CUPTI traces contain individual kernels inside CUDA graph replays.
        # Sum GPU events only, never CPU ranges with inclusive device totals.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt]*bs, SamplingParams(max_tokens=8,temperature=0,ignore_eos=True),use_tqdm=False)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.out/f'graph_trace_bs{bs}.json'))
        summary = {}
        for ev in prof.events():
            if str(ev.device_type) != 'DeviceType.CUDA':
                continue
            rec = summary.setdefault(ev.name,dict(us=0.,count=0))
            rec['us'] += ev.time_range.elapsed_us()
            rec['count'] += 1
        (args.out/f'gpu_kernels_bs{bs}.json').write_text(json.dumps(summary,indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out/'real_inputs.pt')
    torch.save(linear_samples, args.out/'linear_inputs.pt')
    torch.save({name:tuple(v.cpu() if isinstance(v,torch.Tensor) else v for v in vals)
                for name,vals in linear_packed.items()}, args.out/'linear_weights.pt')
    (args.out/'routes.json').write_text(json.dumps(routes))
    (args.out/'stages.json').write_text(json.dumps(stage_rows))
    (args.out/'tokens.json').write_text(json.dumps(output_tokens))
    (args.out/'capture_metadata.json').write_text(json.dumps(dict(
        model=str(MODEL), gpu=torch.cuda.get_device_name(), input=128, output=128,
        graph=True, instrumented=True, note='external CUDA events; snapshots/counter copies excluded from moe_total',
        linear_names=linear_names, samples=len(samples)), indent=2))
    print(f'[capture] wrote {len(samples)} input samples and {len(stage_rows)} stage timings', flush=True)


def load_packed(layer, device):
    import torch
    from safetensors import safe_open
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    # Runtime and checkpoint both carry the decoder layer number; expert wrapper
    # prefixes can differ between vLLM versions.
    import re
    number = re.search(r'layers\.(\d+)', layer).group(1)
    prefix = f'model.layers.{number}.mlp.experts.routed_experts.'
    result = dict(group_size=64, layout='int3', w_transposed=True)
    for key, suffix in [('w13_q','w13_q'),('s13','w13_s'),('z13','w13_z'),
                        ('w2_q','w2_q'),('s2','w2_s'),('z2','w2_z')]:
        full = prefix+suffix
        with safe_open(str(MODEL/index[full]), framework='pt', device='cpu') as f:
            t = f.get_tensor(full)
        if key.endswith('_q'):
            t = t.transpose(1,2).contiguous()
        result[key] = t.to(device)
    return result


def graph_us(fn, repeats=5):
    import torch
    for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    times = []
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(repeats):
        start.record()
        for _ in range(5): g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)*1000/5)
    return statistics.median(times)


def sweep(args):
    import torch
    import brmoe_int3_vllm.kernel as plugin
    fused = plugin.get_fused_moe_int3()
    samples = torch.load(args.out/'real_inputs.pt', weights_only=True)
    names = sorted({s['layer'] for s in samples})
    packed = {name:load_packed(name, 'cuda') for name in names}
    configs = ['baseline']+[f'{bk}_{st}' for bk in (32,64,128) for st in (1,2,3)]
    rows = []
    # Replay all 27 layers with their actual inputs/weights in sequence. This
    # avoids repeatedly timing a single expert matrix resident in L2.
    for bs in (8,32):
        for step in (0,2,16,64):
            group = [s for s in samples if s['batch']==bs and s['step']==step]
            if not group:
                continue
            group.sort(key=lambda s:int(s['layer'].split('layers.')[1].split('.')[0]))
            inputs = []
            for s in group:
                ids, weights = s['ids'].cuda(), s['weights'].cuda()
                neg = ids < 0
                ids = ids.masked_fill(neg, 0)
                weights = weights.masked_fill(neg, 0)
                inputs.append((s['x'].cuda(), weights, ids, packed[s['layer']]))
            refs = [fused(*a).clone() for a in inputs]
            for config in configs:
                kw = {} if config=='baseline' else dict(zip(('block_k','num_stages'),map(int,config.split('_'))))
                worst_l2, worst_max = 0., 0.
                for a, ref in zip(inputs,refs):
                    y = fused(*a, **kw)
                    diff = y.float()-ref.float()
                    l2 = float(diff.norm()/ref.float().norm().clamp_min(1e-8))
                    mx = float(diff.abs().max()/ref.float().abs().max().clamp_min(1e-8))
                    worst_l2, worst_max = max(worst_l2,l2),max(worst_max,mx)
                    assert torch.isfinite(y).all() and l2 < .003 and mx < .01, (bs,step,config,l2,mx)
                def fn():
                    for a in inputs: fused(*a, **kw)
                us = graph_us(fn)
                row = dict(batch=bs,step=step,m=group[0]['m'],layers=len(group),config=config,
                           moe_chain_us=us,relative_l2=worst_l2,normalized_max_error=worst_max)
                rows.append(row)
                print('[sweep]',json.dumps(row),flush=True)
                (args.out/'moe_sweep.json').write_text(json.dumps(rows,indent=2))
            del inputs,refs
    print('[sweep] complete',flush=True)


def linear_sweep(args):
    import torch
    import brmoe_int3_vllm.linear_method as linear
    packed_cpu=torch.load(args.out/'linear_weights.pt',weights_only=True)
    packed={k:tuple(v.cuda() if isinstance(v,torch.Tensor) else v for v in vals)
            for k,vals in packed_cpu.items()}
    samples=torch.load(args.out/'linear_inputs.pt',weights_only=True)
    original_tiles,original_gemm=linear.pick_tiles,linear.int3_moe_gemm
    rows=[]
    for bs in (8,32):
        for step in (0,16,64):
            group=[s for s in samples if s['batch']==bs and s['step']==step]
            if not group:continue
            inputs=[(s['x'].cuda(),*packed[s['layer']]) for s in group]
            linear.pick_tiles,linear.int3_moe_gemm=original_tiles,original_gemm
            refs=[linear._brmoe_int3_linear_impl(*a).clone() for a in inputs]
            for cfg in ('baseline','tile','tile_64_1','tile_64_3','tile_128_1','tile_128_3'):
                linear.pick_tiles,linear.int3_moe_gemm=original_tiles,original_gemm
                configure_linear(cfg)
                worst_l2,worst_max=0.,0.
                for a,ref in zip(inputs,refs):
                    y=linear._brmoe_int3_linear_impl(*a)
                    diff=y.float()-ref.float()
                    l2=float(diff.norm()/ref.float().norm().clamp_min(1e-8))
                    mx=float(diff.abs().max()/ref.float().abs().max().clamp_min(1e-8))
                    worst_l2,worst_max=max(worst_l2,l2),max(worst_max,mx)
                    assert torch.isfinite(y).all() and l2<.003 and mx<.01,(bs,step,cfg,l2,mx)
                def fn():
                    for a in inputs:linear._brmoe_int3_linear_impl(*a)
                us=graph_us(fn)
                row=dict(batch=bs,step=step,m=group[0]['m'],layers=len(group),config=cfg,
                         linear_chain_us=us,relative_l2=worst_l2,normalized_max_error=worst_max)
                rows.append(row)
                print('[linear_sweep]',json.dumps(row),flush=True)
                (args.out/'linear_sweep.json').write_text(json.dumps(rows,indent=2))
            del inputs,refs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode', choices=['collect','sweep','linear_sweep'])
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    {'collect':collect,'sweep':sweep,'linear_sweep':linear_sweep}[args.mode](args)


if __name__=='__main__':
    main()
