"""Current CUDA wrapper vs fused active-row glue; full MoE and full-INT3 e2e."""
import argparse
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
from full_int3_study import load_packed, graph_us, MODEL, TOKENIZER
from brmoe_int3_vllm.kernel import get_fused_moe_int3, get_moe_cuda_ext
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda
from marlin_int3_moe.repack import repack_moe
from int3_moe.grouped_gemv import fused_moe_grouped_gemv, grouped_gemv_config


def weights(name):
    p = load_packed(name, 'cuda')
    pn = {**p, **{key: p[key].transpose(1, 2).contiguous() for key in ('w13_q', 'w2_q')}}
    return p, repack_moe(pn)


def run_one(x, tw, ids, p, pk, ext, config, **kw):
    if config == 'grouped' and 4 <= x.shape[0] <= 16:
        return fused_moe_grouped_gemv(x, tw, ids, p, **grouped_gemv_config(x.shape[0]))
    return fused_moe_int3_cuda(x, tw, ids, pk, ext, packed=p,
                               gemv_max_m=2 if config.endswith('4') else None,
                               fuse_ops='atomic' if config == 'fused_atomic4' else config.startswith('fused'), **kw)


def micro(args, ext):
    data = torch.load(args.input, weights_only=True)
    names = sorted({s['layer'] for s in data})
    if args.quick: names = names[:1]
    all_weights = {name: weights(name) for name in names}
    results = []
    for m in (4, 8, 16, 32):
        inputs = [{**s, **{key: s[key].cuda() for key in ('x', 'weights', 'ids')}}
                  for s in data if s['m'] == m and s['layer'] in names]
        assert len(inputs) == len(names)
        if args.routing == 'random':
            torch.manual_seed(20260926 + m)
            for s in inputs:
                e = all_weights[s['layer']][0]['w13_q'].shape[0]
                s['ids'] = torch.rand(m, e, device='cuda').argsort(1)[:, :s['ids'].shape[1]].contiguous()
        def one(s, config):
            return run_one(s['x'], s['weights'], s['ids'], *all_weights[s['layer']], ext, config)
        reference = [one(s, 'baseline').clone().float() for s in inputs]
        for config in args.configs.split(','):
            errors = []
            for s, ref in zip(inputs, reference):
                got = one(s, config).float()
                rel = float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel < 0.005, (m, config, rel)
                errors.append(rel)
            def fn():
                for s in inputs: one(s, config)
            us = graph_us(fn, repeats=7)
            rec = dict(m=m, config=config, routing=args.routing, layers=len(inputs), us=us,
                       max_relative_l2=max(errors), active_experts=[int(s['ids'].unique().numel()) for s in inputs])
            results.append(rec)
            print(json.dumps(rec), flush=True)
            (args.out/'micro.json').write_text(json.dumps(results, indent=2))
            if args.profile:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): fn()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.out/f'trace_m{m}_{config}.json'))


def verify(args, ext):
    from marlin_int3_moe import fused_ops
    import int3_moe.align_triton as alignment
    import inspect
    print('alignment', alignment.__file__, inspect.signature(alignment.moe_align_block_size_triton), flush=True)
    print('fused weights contiguous in loaded bytecode:', 'contiguous' in
          fused_ops.fused_moe_cuda_ops.__wrapped__.__code__.co_names, flush=True)
    data = torch.load(args.input, weights_only=True)
    p, pk = weights(data[0]['layer'])
    tri = get_fused_moe_int3()
    torch.manual_seed(20260926)
    e, k, topk = p['w13_q'].shape[0], p['w2_q'].shape[-1], 6
    records = []
    for m in (1, 2, 4, 8, 16, 32, 64, 512):
        strided_x = m in (4, 64)
        x = (torch.randn(m, k * (2 if strided_x else 1), device='cuda') * .1).half()
        if strided_x: x = x[:, ::2]
        tw = torch.empty(m, topk*2, device='cuda')[:, ::2] if m == 8 else torch.empty(m, topk, device='cuda')
        tw.copy_(torch.softmax(torch.randn(m, topk, device='cuda'), -1))
        ids = torch.arange(topk, device='cuda').repeat(m, 1).contiguous()
        for splits in ((1, 1), (2, 2), (2, 1)) if m <= 32 else ((1, 1),):
            def fn(fuse):
                return fused_moe_int3_cuda(x, tw, ids, pk, ext, gemv_max_m=0,
                    fuse_ops=fuse, ksplit=splits[0], ksplit2=splits[1],
                    cfg=tuple(map(int,args.cuda_cfg.split(','))) if args.cuda_cfg else None)
            for _ in range(2): fn(True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): result = fn(True)
            for pattern in ('hot', 'random', 'zero_weight', 'hot_again'):
                ids.copy_(torch.randint(0, e, ids.shape, device='cuda') if pattern == 'random'
                          else torch.arange(topk, device='cuda').repeat(m, 1))
                tw.copy_(torch.softmax(torch.randn_like(tw), -1))
                if pattern == 'zero_weight': tw.zero_()
                # Poison cached intermediate/padding buffers before replay.
                for ws in fused_ops._WORKSPACE.values():
                    for key in ('a', 'inter', 'act', 'sorted_out'):
                        ws[key].fill_(float('nan'))
                for buf in alignment._BUF.cache.values():
                    if 'route_pos' in buf: buf['route_pos'].fill_(-999)
                graph.replay()
                got = result.clone().float()
                ref = fn(False).clone().float()
                rel = float((got-ref).norm()/ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel < .002, (m, splits, pattern, rel)
                tri_rel = None
                if splits == (1, 1) and m <= 32:
                    gold = tri(x.contiguous(), tw.contiguous(), ids, p, fast=True, gemv=True).float()
                    tri_rel = float((got-gold).norm()/gold.norm().clamp_min(1e-8))
                    if tri_rel >= .005:
                        contiguous = fused_moe_int3_cuda(x.contiguous(), tw.contiguous(), ids,
                            pk, ext, gemv_max_m=0, fuse_ops=True).float()
                        print('diagnostic contiguous fused vs triton:', float((contiguous-gold).norm()/gold.norm()),
                              'tw stride', tw.stride(), flush=True)
                    assert tri_rel < .005, (m, pattern, tri_rel)
                rec = dict(m=m, splits=splits, routing=pattern, strided_x=strided_x,
                           strided_weights=not tw.is_contiguous(), relative_l2=rel, triton_l2=tri_rel)
                records.append(rec)
                print(json.dumps(rec), flush=True)
    (args.out/'verify.json').write_text(json.dumps(records, indent=2))
    print('VERIFY PASS', len(records), flush=True)


def e2e(args):
    # Same clean full-model benchmark; patch only runtime MoE call configuration.
    import brmoe_int3_vllm.moe_method as method
    import brmoe_int3_vllm.kernel as plugin
    from vllm import LLM
    batches = getattr(args, 'batch_sizes', '4,8,16,32')
    cuda_cfg = getattr(args, 'cuda_cfg', None)
    if cuda_cfg:
        import marlin_int3_moe.moe_cuda as cuda_module
        cfg_original = cuda_module.fused_moe_int3_cuda
        cfg_tuple = tuple(map(int, cuda_cfg.split(',')))
        def configured(*a, **kw):
            kw['cfg'] = cfg_tuple
            return cfg_original(*a, **kw)
        cuda_module.fused_moe_int3_cuda = configured
    dispatch_calls = {}
    if args.config == 'production':
        os.environ.update(BRMOE_GROUPED_GEMV='1', BRMOE_CUDA_FUSE='1')
        assert 'use_fused_cuda' in plugin.brmoe_int3_moe.__wrapped__.__code__.co_varnames, 'stale plugin bytecode'
        import marlin_int3_moe.fused_ops as fusion
        fused_original = fusion.fused_moe_cuda_ops
        def recorded(x, *a, **kw):
            dispatch_calls[x.shape[0]] = dispatch_calls.get(x.shape[0], 0) + 1
            return fused_original(x, *a, **kw)
        fusion.fused_moe_cuda_ops = recorded
    elif args.config == 'grouped': os.environ['BRMOE_GROUPED_GEMV'] = '1'
    elif args.config.startswith('fused'):
        os.environ['BRMOE_CUDA_FUSE'] = 'atomic' if args.config == 'fused_atomic4' else '1'
        if args.config.endswith('4'):
            import marlin_int3_moe.moe_cuda as module
            original = module.fused_moe_int3_cuda
            def force(*a, **kw):
                kw['gemv_max_m'] = 2
                return original(*a, **kw)
            module.fused_moe_int3_cuda = force
    generated = []
    generate = LLM.generate
    def remember(self, *a, **kw):
        result = generate(self, *a, **kw)
        generated.append(result)
        return result
    LLM.generate = remember
    sys.argv = ['vllm_perf.py', '--model', str(MODEL), '--tokenizer', str(TOKENIZER),
                '--tag', 'full_int3_' + args.config, '--quantization', 'brmoe_int3',
                '--batch-sizes', batches, '--input-len', '128', '--output-len', '128',
                '--repeat', '3', '--enforce-eager', '0', '--max-num-batched-tokens', '2048',
                '--out', str(args.out / ('e2e_' + args.config + '.json'))]
    runpy.run_path(str(ROOT/'bench/vllm_perf.py'), run_name='__main__')
    tokens = [dict(batch=len(r), ids=[o.outputs[0].token_ids for o in r]) for r in generated
              if r and len(r[0].outputs[0].token_ids) == 128]
    (args.out/('tokens_' + args.config + '.json')).write_text(json.dumps(tokens))
    if args.config == 'production':
        assert all(dispatch_calls.get(int(m),0)>0 for m in batches.split(',')), dispatch_calls
        (args.out/'dispatch_calls.json').write_text(json.dumps(dispatch_calls, indent=2))
        print('production fused dispatch counts', dispatch_calls, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['micro', 'verify', 'e2e'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--input', type=Path, default=Path('/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/grouped_gemv/run_39061/real_inputs.pt'))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--routing', choices=['real','random'], default='real')
    ap.add_argument('--configs', default='baseline,grouped,fused_atomic4,fused4')
    ap.add_argument('--config', default='baseline')
    ap.add_argument('--batch-sizes', default='4,8,16,32')
    ap.add_argument('--cuda-cfg', default=None, help='thread_n,thread_k,stages')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', BRMOE_GROUPED_GEMV='0', BRMOE_CUDA_FUSE='0')
    (args.out/f'metadata_{args.phase}_{args.config}.json').write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),indent=2))
    print(torch.cuda.get_device_name(), args.phase, args.config, flush=True)
    if args.phase == 'e2e': e2e(args)
    else:
        get_fused_moe_int3()
        ext = get_moe_cuda_ext()
        assert ext is not None, 'CUDA baseline must be active'
        (verify if args.phase == 'verify' else micro)(args, ext)


if __name__ == '__main__': main()
