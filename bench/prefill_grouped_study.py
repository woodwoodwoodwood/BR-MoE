"""A100 real-prefill / decode grouped GEMM, complete MoE and clean e2e study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools', 'bench', 'BR-MoE/kernels', 'BR-MoE/kernels/triton_int3'):
    sys.path.insert(0, str(ROOT/sub))
import torch
import triton
from full_int3_study import graph_us
from moe_fuse_study import weights as load_weights
from brmoe_int3_vllm import linear_method as linear
from brmoe_int3_vllm.kernel import get_fused_moe_int3, get_moe_cuda_ext
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda
from int3_moe.grouped_tc import fused_moe_int3_tc


def parameters(config):
    kind, bm, bn, bk, stages, warps = config.split('_')
    return kind, dict(block_m=int(bm), block_n=int(bn), block_k=int(bk),
                       num_stages=int(stages), num_warps=int(warps))


def config_for_m(config, m):
    if config == 'mixed':
        return 'baseline' if m < 256 else ('cudafast_32_128_128_4_8' if m <= 512
                                          else 'fastreduce_128_128_64_3_4')
    return config


def routed(s, p, pk, ext, config):
    config = config_for_m(config, s['x'].shape[0])
    if config in ('previous','production'):
        from types import SimpleNamespace
        from brmoe_int3_vllm.kernel import brmoe_int3_moe
        layer=SimpleNamespace(w13_q=p['w13_q'],w13_s=p['s13'],w13_z=p['z13'],
                              w2_q=p['w2_q'],w2_s=p['s2'],w2_z=p['z2'],
                              brmoe_cuda_packed=pk,w_transposed=True)
        return brmoe_int3_moe(s['x'],s['weights'],s['ids'],layer,p['group_size'])
    if config == 'baseline':
        if s['x'].shape[0] <= 512:
            return fused_moe_int3_cuda(s['x'], s['weights'], s['ids'], pk, ext,
                                       packed=p, fuse_ops=True)
        return get_fused_moe_int3()(s['x'], s['weights'], s['ids'], p, fast=True)
    kind, kw = parameters(config)
    if kind in ('cuda', 'cudafast'):
        return fused_moe_int3_cuda(s['x'],s['weights'],s['ids'],pk,ext,packed=p,
            fuse_ops=True,gemv_max_m=0,tile_m=kw['block_m'],fast_align=kind=='cudafast',
            cfg=(kw['block_n'],kw['block_k'],kw['num_stages']))
    if kind == 'old':
        return get_fused_moe_int3()(s['x'], s['weights'], s['ids'], p,
                                    fast=True, gemv=False, **kw)
    assert kind in ('tc', 'reduce', 'fast', 'fastreduce'), kind
    return fused_moe_int3_tc(s['x'], s['weights'], s['ids'], p,
                             reduce_topk=kind in ('reduce','fastreduce'), fast_align=kind.startswith('fast'), **kw)


def sweep(args):
    import vllm._custom_ops
    import int3_linear_study as linear_study
    data = torch.load(args.input / 'real_inputs.pt', weights_only=True)
    if args.decode_input:
        data += torch.load(args.decode_input / 'real_inputs.pt', weights_only=True)
    # Keep the last observed prefill chunk per layer/M; no fabricated routing.
    chosen = {(s['m'], s['layer']): s for s in data}
    names = sorted({s['layer'] for s in data})
    if args.quick:
        names = names[:1]
    packed = {name: load_weights(name) for name in names}
    shared = torch.load(args.input / 'linear_weights.pt', weights_only=True)
    shared = {name: tuple(v.cuda() if isinstance(v, torch.Tensor) else v for v in values)
              for name, values in shared.items() if '.shared_experts.' in name}
    ext = get_moe_cuda_ext()
    assert ext is not None
    records = []
    for m in map(int, args.m_values.split(',')):
        samples = [{**chosen[m, name], **{key: chosen[m, name][key].cuda()
                    for key in ('x', 'weights', 'ids')}} for name in names]
        def one(s, config, complete=False):
            value = routed(s, *packed[s['layer']], ext, config)
            if not complete:
                return value
            prefix = s['layer'].removesuffix('.experts') + '.shared_experts.'
            a = linear._brmoe_int3_linear_impl(s['x'], *shared[prefix + 'gate_up_proj'])
            act = torch.empty((m, a.shape[1] // 2), device='cuda', dtype=torch.float16)
            torch.ops._C.silu_and_mul(act, a)
            return value + linear._brmoe_int3_linear_impl(act, *shared[prefix + 'down_proj'])
        linear_study.install('previous')
        refs = {scope: [one(s, 'baseline', scope == 'complete').clone().float()
                       for s in samples] for scope in ('routed', 'complete')}
        for spec in args.configs.split(','):
            config, linear_config = spec.split('+') if '+' in spec else (spec,args.linear_config)
            if spec in ('previous','production'): linear_config=spec
            linear_study.install(linear_config)
            for scope in ('routed', 'complete'):
                errors = []
                for s, ref in zip(samples, refs[scope]):
                    got = one(s, config, scope == 'complete').float()
                    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-8))
                    assert torch.isfinite(got).all() and rel < .003, (m, config, scope, s['layer'], rel)
                    errors.append(rel)
                def fn():
                    for s in samples:
                        one(s, config, scope == 'complete')
                rec = dict(m=m, config=spec, linear_config=linear_config, scope=scope, layers=len(samples),
                           us=graph_us(fn, repeats=args.repeats), max_relative_l2=max(errors))
                records.append(rec)
                print(json.dumps(rec), flush=True)
                (args.out / 'sweep.json').write_text(json.dumps(records, indent=2))
                if args.trace and scope == 'routed':
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): fn()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        graph.replay()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.out / f'trace_m{m}_{config}.json'))
        del samples, refs


def install(config, min_m):
    if config in ('baseline', 'fp16', 'previous', 'production'):
        return
    import brmoe_int3_vllm.moe_method as method
    import brmoe_int3_vllm.kernel as plugin
    original = method.brmoe_int3_moe
    def tuned(x, tw, ids, layer, gs, out_dtype=None):
        selected = config_for_m(config, x.shape[0])
        if x.shape[0] < min_m or selected == 'baseline':
            return original(x, tw, ids, layer, gs, out_dtype)
        kind, kw = parameters(selected)
        assert kind in ('tc', 'reduce', 'old', 'fast', 'fastreduce', 'cuda', 'cudafast')
        ids, tw = plugin._sanitize_routing(ids, tw)
        p = plugin.build_packed(layer, gs)
        if kind in ('cuda','cudafast'):
            return fused_moe_int3_cuda(x,tw,ids,layer.brmoe_cuda_packed,get_moe_cuda_ext(),
                packed=p,fuse_ops=True,gemv_max_m=0,tile_m=kw['block_m'],fast_align=kind=='cudafast',
                cfg=(kw['block_n'],kw['block_k'],kw['num_stages']),out_dtype=out_dtype or x.dtype)
        if kind == 'old':
            return get_fused_moe_int3()(x, tw, ids, p, fast=True, gemv=False,
                                        out_dtype=out_dtype or x.dtype, **kw)
        return fused_moe_int3_tc(x, tw, ids, p, reduce_topk=kind in ('reduce','fastreduce'),
                                 fast_align=kind.startswith('fast'),
                                 out_dtype=out_dtype or x.dtype, **kw)
    method.brmoe_int3_moe = tuned


def e2e(args):
    config, linear_config = args.config.split('+') if '+' in args.config else (args.config,args.linear_config)
    if config in ('previous','production'): linear_config=config
    install(config, args.min_m)
    sys.argv = ['a100_full_int3_study.py', '--case', 'fp16' if config == 'fp16' else linear_config,
                '--out', str(args.out), '--batch-sizes', args.batch_sizes, '--repeat', str(args.repeats)]
    runpy.run_path(str(ROOT / 'bench/a100_full_int3_study.py'), run_name='__main__')


def verify(args):
    """Independent dequantized reference, changing routes and poisoned graphs."""
    from full_int3_study import load_packed
    from int3_moe.packing import pack_int3, unpack_int3
    from int3_moe.grouped_tc import _WORKSPACE
    from int3_moe.align_triton import _BUF
    torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [('real', load_packed('model.layers.1.mlp.experts', 'cuda'))]
    for k, i, gs in ((192, 96, 32), (256, 128, 64), (256, 128, 128)):
        p = dict(group_size=gs, layout='int3', w_transposed=True)
        e = 7
        for tag, n, red in [('13', 2*i, k), ('2', k, i)]:
            q = torch.randint(0, 8, (e*n, red), dtype=torch.int32, device='cuda')
            p['w'+tag+'_q'] = pack_int3(q, transposed=True).reshape(red//32*3, e, n).permute(1,0,2).contiguous()
            p['s'+tag] = (torch.rand(e, red//gs, n, device='cuda')*.02+.005).half()
            p['z'+tag] = (torch.rand(e, red//gs, n, device='cuda')*5+.5).half()
        cases.append((f'tail_gs{gs}', p))
    records = []
    for name, p in cases:
        e, _, two_i = p['w13_q'].shape
        k, i, gs = p['w2_q'].shape[-1], two_i//2, p['group_size']
        topk = 6 if e == 64 else 3
        def dequant(tag, red, n):
            q = unpack_int3(p['w'+tag+'_q'].transpose(1,2).reshape(e*n,-1), red, transposed=False).reshape(e,n,red).half()
            z = p['z'+tag].repeat_interleave(gs,1).transpose(1,2)
            s = p['s'+tag].repeat_interleave(gs,1).transpose(1,2)
            return ((q-z).half()*s).half().float()
        w13, w2 = dequant('13',k,two_i), dequant('2',i,k)
        cuda_configs = [c for c in args.configs.split(',') if c.startswith('cuda')]
        pk = None
        if cuda_configs and gs == 64 and k%128==0 and i%128==0:
            from marlin_int3_moe.repack import repack_moe
            pn={**p,**{key:p[key].transpose(1,2).contiguous() for key in ('w13_q','w2_q')}}
            pk=repack_moe(pn)
        for m in map(int, args.m_values.split(',')):
            x = (torch.randn(m,k*2,device='cuda')*.1).half()[:,::2]
            tw = torch.empty(m,topk*2,device='cuda')[:,::2]
            ids = torch.arange(topk,device='cuda').repeat(m,1)
            tw.copy_(torch.softmax(torch.randn_like(tw),-1))
            for config in args.configs.split(','):
                kind, kw = parameters(config)
                is_cuda = kind in ('cuda','cudafast')
                if is_cuda and pk is None: continue
                assert kind in ('tc', 'reduce', 'fast', 'fastreduce', 'cuda', 'cudafast')
                def fn():
                    if is_cuda:
                        return routed(dict(x=x,weights=tw,ids=ids),p,pk,get_moe_cuda_ext(),config)
                    return fused_moe_int3_tc(x,tw,ids,p,reduce_topk=kind in ('reduce','fastreduce'),
                                              fast_align=kind.startswith('fast'),**kw)
                for _ in range(2): fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): result = fn()
                for pattern in ('hot','random','zero_weight','hot_again'):
                    x.copy_((torch.randn_like(x)*.1).half())
                    ids.copy_(torch.rand(m,e,device='cuda').argsort(1)[:,:topk] if pattern=='random'
                              else torch.arange(topk,device='cuda').repeat(m,1))
                    tw.copy_(torch.softmax(torch.randn_like(tw),-1))
                    if pattern=='zero_weight': tw.zero_()
                    for ws in _WORKSPACE.values():
                        for buf in ws.values(): buf.fill_(float('nan'))
                    if is_cuda:
                        from marlin_int3_moe.fused_ops import _WORKSPACE as cuda_ws
                        for ws in cuda_ws.values():
                            for buf in ws.values():buf.fill_(float('nan'))
                    for buf in _BUF.cache.values():
                        if 'route_pos' in buf: buf['route_pos'].fill_(-999)
                    result.fill_(float('nan'))
                    graph.replay()
                    got = result.clone().float()
                    gold = torch.zeros(m,k,device='cuda',dtype=torch.float32)
                    for expert in range(e):
                        row, route = torch.where(ids==expert)
                        if not row.numel(): continue
                        h = (x[row].float() @ w13[expert].T).half().float()
                        act = (h[:,:i]*torch.sigmoid(h[:,:i])*h[:,i:]).half().float()
                        val = act @ w2[expert].T
                        if is_cuda: val=val.half().float()
                        val = val * tw[row,route,None]
                        gold.index_add_(0,row,val)
                    gold = gold.half().float()
                    rel = float((got-gold).norm()/gold.norm().clamp_min(1e-8))
                    assert torch.isfinite(got).all() and rel<.003,(name,m,config,pattern,rel)
                    records.append(dict(shape=name,m=m,k=k,i=i,config=config,routing=pattern,relative_l2=rel))
                print('VERIFY',name,m,config,'PASS',flush=True)
                (args.out/'verify.json').write_text(json.dumps(records,indent=2))
        del w13,w2
    print('VERIFY PASS',len(records),max(r['relative_l2'] for r in records),flush=True)


def alignment(args):
    import int3_moe.align_triton as mod
    data = torch.load(args.input/'real_inputs.pt', weights_only=True)
    chosen = {(s['m'],s['layer']):s for s in data}
    records = []
    for m in map(int,args.m_values.split(',')):
        samples=[{**s, 'ids':s['ids'].cuda(), 'weights':s['weights'].cuda().reshape(-1).float()}
                 for (shape,_),s in chosen.items() if shape==m]
        refs=[]
        for s in samples:
            sti,eid,meta,buf=mod.moe_align_block_size_triton(s['ids'],64,64,flat_values=s['weights'],return_route_positions=True)
            post,blocks=map(int,meta.tolist())
            refs.append((sti[:post].clone(),eid[:blocks].clone(),meta.clone(),buf['sv'][:post].clone(),buf['route_pos'].clone()))
        for hist,warps in [(False,4),(False,8),(False,16),(True,4),(True,8),(True,16)]:
            def one(s):
                return mod.moe_align_block_size_triton(s['ids'],64,64,flat_values=s['weights'],
                            return_route_positions=True,histogram=hist,scatter_warps=warps)
            for s,ref in zip(samples,refs):
                sti,eid,meta,buf=one(s)
                assert all(torch.equal(a,b) for a,b in zip(
                    (sti[:len(ref[0])],eid[:len(ref[1])],meta,buf['sv'][:len(ref[3])],buf['route_pos']),ref))
            def fn():
                for s in samples:one(s)
            rec=dict(m=m,histogram=hist,scatter_warps=warps,layers=len(samples),us=graph_us(fn,repeats=7))
            records.append(rec);print(json.dumps(rec),flush=True)
            (args.out/'align.json').write_text(json.dumps(records,indent=2))
            if args.trace:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):fn()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay();torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.out/f'align_m{m}_hist{int(hist)}_w{warps}.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['sweep', 'e2e', 'verify', 'align'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--decode-input', type=Path)
    ap.add_argument('--m-values', default='512,1024,2048')
    ap.add_argument('--configs', default='baseline,tc_64_64_128_3_4,reduce_64_64_128_3_4')
    ap.add_argument('--config', default='baseline')
    ap.add_argument('--linear-config', default='previous')
    ap.add_argument('--min-m', type=int, default=513)
    ap.add_argument('--batch-sizes', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(BRMOE_LINEAR_BACKEND='auto', BRMOE_CUDA_FUSE='1',
                      BRMOE_GROUPED_GEMV='0', BRMOE_MOE_SMEM='legacy')
    assert torch.cuda.get_device_capability() == (8, 0)
    if os.environ.get('BRMOE_STUDY_CUDA_SO'):
        import importlib.util
        import brmoe_int3_vllm.kernel as plugin
        spec=importlib.util.spec_from_file_location('brmoe_moe_int3',os.environ['BRMOE_STUDY_CUDA_SO'])
        ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
        plugin._EXT,plugin._EXT_TRIED=ext,True
    files = [Path(__file__), ROOT / 'BR-MoE/kernels/triton_int3/int3_moe/grouped_tc.py',
             ROOT / 'BR-MoE/kernels/triton_int3/int3_moe/kernel.py',
             ROOT / 'tools/brmoe_int3_vllm/kernel.py', ROOT / 'tools/brmoe_int3_vllm/linear_method.py',
             ROOT / 'BR-MoE/kernels/triton_int3/int3_moe/align_triton.py']
    ext=get_moe_cuda_ext()
    meta = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                cuda_extension=str(ext.__file__),cuda_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.out / 'study_metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    dict(sweep=sweep, e2e=e2e, verify=verify, align=alignment)[args.phase](args)


if __name__ == '__main__':
    main()
