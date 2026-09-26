"""Opt-in grouped GEMV study against current CUDA MoE and full INT3 e2e.

micro uses captured real tensors; sweep numbers include the entire MoE. e2e
runs a clean engine without instrumentation. collect captures the last decode
step and saves it separately, never using instrumented times as latency data.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in ('tools', 'bench', 'BR-MoE/kernels', 'BR-MoE/kernels/triton_int3'):
    sys.path.insert(0, str(ROOT / p))
import torch
import triton
from full_int3_study import load_packed, graph_us, MODEL, TOKENIZER
from int3_moe.grouped_gemv import fused_moe_grouped_gemv


def options(s):
    r, n, g, ks, nw = map(int, s.split('_'))
    return dict(rows=r, block_n=n, groups=g, ksplit=ks, num_warps=nw)


def adaptive_config(m):
    return '4_128_1_16_4' if m <= 4 else ('8_128_1_8_4' if m <= 8 else '8_256_1_8_4')


def install(config):
    import brmoe_int3_vllm.moe_method as method
    import brmoe_int3_vllm.kernel as plugin
    original = method.brmoe_int3_moe
    def grouped(x, tw, ids, layer, group_size, out_dtype=None):
        if 4 <= x.shape[0] <= (16 if config == 'adaptive' else 32):
            ids, tw = plugin._sanitize_routing(ids, tw)
            return fused_moe_grouped_gemv(x, tw, ids, plugin.build_packed(layer, group_size),
                                          out_dtype=out_dtype,
                                          **options(adaptive_config(x.shape[0]) if config == 'adaptive' else config))
        return original(x, tw, ids, layer, group_size, out_dtype)
    method.brmoe_int3_moe = grouped
    print('[grouped_gemv] experimental dispatch config=' + config, flush=True)


def collect(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import brmoe_int3_vllm.moe_method as method
    from vllm_perf import make_prompt
    original = method.brmoe_int3_moe
    buffers = {}
    def capture(x, tw, ids, layer, group_size, out_dtype=None):
        if x.shape[0] in (4, 8, 16, 32):
            key = (layer.layer_name, x.shape[0])
            if key not in buffers:
                buffers[key] = dict(x=torch.empty_like(x), weights=torch.empty_like(tw),
                                    ids=torch.empty_like(ids))
            for name, src in (('x', x), ('weights', tw), ('ids', ids)):
                buffers[key][name].copy_(src)
        return original(x, tw, ids, layer, group_size, out_dtype)
    method.brmoe_int3_moe = capture
    llm = LLM(model=str(MODEL), tokenizer=str(TOKENIZER), tokenizer_mode='hf',
              trust_remote_code=True, dtype='float16', quantization='brmoe_int3',
              max_model_len=768, max_num_seqs=32, max_num_batched_tokens=2048,
              enable_prefix_caching=False, gpu_memory_utilization=0.90)
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    prompt = make_prompt(tok, 128)
    samples = []
    for batch in (4, 8, 16, 32):
        llm.generate([prompt] * batch, SamplingParams(max_tokens=128, min_tokens=128,
                      temperature=0, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        for (layer, m), b in buffers.items():
            if m == batch:
                samples.append(dict(batch=batch, m=m, step=127, layer=layer,
                                    **{k: v.cpu() for k, v in b.items()}))
        print('captured', batch, 'total samples', len(samples), flush=True)
    torch.save(samples, args.out / 'real_inputs.pt')


def micro(args):
    from brmoe_int3_vllm.kernel import get_moe_cuda_ext, get_fused_moe_int3
    from marlin_int3_moe.repack import repack_moe
    from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda
    tri, ext = get_fused_moe_int3(), get_moe_cuda_ext()
    assert ext is not None, 'baseline CUDA extension must be available'
    data = torch.load(args.input, weights_only=True)
    data = [s for s in data if s['m'] in (4, 8, 16, 32) and s['step'] == args.step]
    assert data, 'no matching real decode samples'
    names = sorted({s['layer'] for s in data})
    if args.quick:
        names = names[:1]
    packed, cuda = {}, {}
    for name in names:
        p = load_packed(name, 'cuda')
        packed[name] = p
        pn = {**p, 'w13_q': p['w13_q'].transpose(1, 2).contiguous(),
                    'w2_q': p['w2_q'].transpose(1, 2).contiguous()}
        cuda[name] = repack_moe(pn)
    configs = args.configs.split(',') if args.configs else [
        f'{r}_{n}_{g}_{ks}_{nw}' for r, n, g, ks, nw in itertools.product(
            (2, 4), (32, 64), (1, 2), (2, 4), (4,))]
    results = []
    for batch in sorted({s['m'] for s in data}):
        samples = [s for s in data if s['m'] == batch and s['layer'] in names]
        inputs = [{**s, **{k: s[k].cuda() for k in ('x', 'weights', 'ids')}} for s in samples]
        def one(s, config):
            p = packed[s['layer']]
            if config == 'baseline':
                return fused_moe_int3_cuda(s['x'], s['weights'], s['ids'],
                                          cuda[s['layer']], ext, packed=p)
            if config == 'routed':
                return tri(s['x'], s['weights'], s['ids'], p, fast=True, gemv=True)
            if config == 'adaptive' and batch > 16:
                return one(s, 'baseline')
            return fused_moe_grouped_gemv(s['x'], s['weights'], s['ids'], p,
                                         **options(adaptive_config(batch) if config == 'adaptive' else config))
        gold = [one(s, 'routed').clone().float() for s in inputs]
        for config in ['baseline', 'routed'] + configs:
            errors = []
            for s, ref in zip(inputs, gold):
                got = one(s, config).float()
                rel = float((got - ref).norm() / ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel < 0.005, (batch, config, s['layer'], rel)
                errors.append(rel)
            def run():
                for s in inputs:
                    one(s, config)
            us = graph_us(run, repeats=7)
            result = dict(batch=batch, config=config, layers=len(inputs), us=us,
                          max_relative_l2=max(errors),
                          active_experts=[int(s['ids'].unique().numel()) for s in inputs])
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.out / 'micro.json').write_text(json.dumps(results, indent=2))
            if args.profile and config != 'routed':
                trace = args.out / f'trace_bs{batch}_{config}.json'
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): run()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graph.replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace))


def verify(args):
    from int3_moe.packing import pack_int3
    torch.manual_seed(20260926)
    e, k, i, gs = 8, 256, 192, 64
    p = dict(group_size=gs, w_transposed=True, layout='int3')
    dense = {}
    for suffix, n, reduction in [('13', 2*i, k), ('2', k, i)]:
        q = torch.randint(0, 8, (e, n, reduction), device='cuda', dtype=torch.int32)
        s = (torch.rand(e, reduction//gs, n, device='cuda') * 0.02 + 0.005).half()
        z = (torch.rand_like(s) * 5 + 0.5).half()
        packed = pack_int3(q.reshape(e*n, reduction), transposed=False).reshape(e, n, -1)
        p['w'+suffix+'_q'] = packed.transpose(1, 2).contiguous()
        p['s'+suffix], p['z'+suffix] = s, z
        sb = s.repeat_interleave(gs, 1).transpose(1, 2)
        zb = z.repeat_interleave(gs, 1).transpose(1, 2)
        dense[suffix] = ((q.half() - zb).half() * sb).half().float()
    records = []
    for m in (1, 2, 4, 8, 16, 32):
        x = (torch.randn(m, k, device='cuda') * 0.1).half()
        ids = torch.empty(m, 3, dtype=torch.int64, device='cuda')
        w = torch.softmax(torch.randn(m, 3, device='cuda'), -1).half()
        ids.copy_(torch.randint(0, e, ids.shape, device='cuda'))
        for config in (args.configs.split(',') if args.configs else ['4_128_1_8_4']):
            fn = lambda: fused_moe_grouped_gemv(x, w, ids, p, **options(config))
            for _ in range(2): fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fn()
            for pattern in ('random', 'same', 'invalid', 'random_again'):
                if pattern == 'same':
                    ids.fill_(3)  # duplicates per token must also be safe
                elif pattern == 'invalid':
                    ids.fill_(-1)
                else:
                    ids.copy_(torch.randint(0, e, ids.shape, device='cuda'))
                graph.replay()
                got = result.clone().float()
                ref = torch.zeros_like(got)
                for token in range(m):
                    for j in range(3):
                        expert = int(ids[token, j])
                        if expert < 0: continue
                        inter = x[token].float() @ dense['13'][expert].T
                        act = (torch.nn.functional.silu(inter[:i]) * inter[i:]).half()
                        ref[token] += (act.float() @ dense['2'][expert].T) * w[token, j]
                rel = float((got - ref).norm() / ref.norm().clamp_min(1e-8))
                assert torch.isfinite(got).all() and rel < 0.002, (m, config, pattern, rel)
                records.append(dict(m=m, config=config, routing=pattern, relative_l2=rel))
    (args.out/'verify.json').write_text(json.dumps(records, indent=2))
    print('VERIFY PASS', len(records), 'cases; max relative L2',
          max(r['relative_l2'] for r in records), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['micro', 'collect', 'e2e', 'verify'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--input', type=Path, default=ROOT/'bench_results/full_int3_study/run_38936/real_inputs.pt')
    ap.add_argument('--step', type=int, default=16)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--configs', default='')
    ap.add_argument('--config', default='baseline')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['BRMOE_GROUPED_GEMV'] = '0'
    (args.out/f'metadata_{args.phase}_{args.config}.json').write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, triton=triton.__version__, source=str(ROOT),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), indent=2))
    print('GPU', torch.cuda.get_device_name(), 'phase', args.phase, flush=True)
    if args.phase == 'micro':
        micro(args)
    elif args.phase == 'collect':
        collect(args)
    elif args.phase == 'verify':
        verify(args)
    else:
        if args.config == 'production':
            assert torch.cuda.get_device_capability() == (12, 0)
            os.environ['BRMOE_GROUPED_GEMV'] = '1'
        elif args.config != 'baseline':
            install(args.config)
        # Retain returned Python objects only; serialize token IDs after timing.
        from vllm import LLM
        generated = []
        original_generate = LLM.generate
        def remember(self, *a, **kw):
            result = original_generate(self, *a, **kw)
            generated.append(result)
            return result
        LLM.generate = remember
        sys.argv = ['vllm_perf.py', '--model', str(MODEL), '--tokenizer', str(TOKENIZER),
                    '--tag', 'full_int3_' + args.config, '--quantization', 'brmoe_int3',
                    '--batch-sizes', '4,8,16,32', '--input-len', '128', '--output-len', '128',
                    '--repeat', '3', '--enforce-eager', '0', '--max-num-batched-tokens', '2048',
                    '--out', str(args.out / ('e2e_' + args.config + '.json'))]
        runpy.run_path(str(ROOT/'bench/vllm_perf.py'), run_name='__main__')
        tokens = [dict(batch=len(result), ids=[o.outputs[0].token_ids for o in result])
                  for result in generated if result and len(result[0].outputs[0].token_ids) == 128]
        (args.out/('tokens_' + args.config + '.json')).write_text(json.dumps(tokens))


if __name__ == '__main__':
    main()
