"""Same-device FP16 / full INT3 end-to-end comparison using the shared harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import int3_linear_study as study
import torch
import triton
from full_int3_study import MODEL, TOKENIZER

FP16 = Path('/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-base')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True,
                    help='fp16, unfused, fused, previous (pre-prefill optimization), production, or a linear study config')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--batch-sizes', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--model-fp16', type=Path, default=FP16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.get_device_capability() == (8, 0), 'A100 study requires sm_80'
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', BRMOE_GROUPED_GEMV='0',
                      BRMOE_MOE_SMEM='legacy',
                      BRMOE_CUDA_FUSE='0' if args.case in ('fp16', 'unfused') else '1')
    config = 'baseline' if args.case in ('fp16', 'unfused', 'fused') else args.case
    study.install(config)
    if args.case == 'production':
        assert study.linear._brmoe_int3_linear_impl is study.ORIGINAL_IMPL
    model = args.model_fp16 if args.case == 'fp16' else MODEL
    source_files = [Path(__file__).resolve(), ROOT / 'bench/vllm_perf.py',
                    ROOT / 'bench/int3_linear_study.py',
                    ROOT / 'tools/brmoe_int3_vllm/linear_method.py',
                    ROOT / 'tools/brmoe_int3_vllm/linear_tc.py',
                    ROOT / 'tools/brmoe_int3_vllm/kernel.py',
                    ROOT / 'tools/brmoe_int3_vllm/prefill.py',
                    ROOT / 'BR-MoE/kernels/triton_int3/int3_moe/grouped_tc.py',
                    ROOT / 'BR-MoE/kernels/triton_int3/int3_moe/align_triton.py']
    metadata = dict(case=args.case, model=str(model), tokenizer=str(TOKENIZER),
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    triton=triton.__version__, batch_sizes=args.batch_sizes,
                    repeat=args.repeat, source_root=str(ROOT),
                    env={k:os.environ.get(k) for k in
                         ('BRMOE_LINEAR_BACKEND', 'BRMOE_PREFILL_BACKEND', 'BRMOE_CUDA_FUSE',
                          'BRMOE_MOE_SMEM', 'BRMOE_GROUPED_GEMV')},
                    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in source_files})
    if args.case != 'fp16':
        study.get_fused_moe_int3()
        ext = study.get_moe_cuda_ext()
        assert ext is not None, 'sm_80 CUDA MoE extension must be active'
        p = Path(ext.__file__)
        metadata['cuda_extension'] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    from vllm import LLM
    generated = []
    original = LLM.generate
    def remember(self, *a, **kw):
        result = original(self, *a, **kw)
        if result and len(result[0].outputs[0].token_ids) == 128:
            generated.append(dict(batch=len(result), ids=[r.outputs[0].token_ids for r in result]))
        return result
    LLM.generate = remember
    sys.argv = ['vllm_perf.py', '--model', str(model), '--tokenizer', str(TOKENIZER),
                '--tag', 'a100_' + args.case, '--batch-sizes', args.batch_sizes,
                '--input-len', '128', '--output-len', '128', '--repeat', str(args.repeat),
                '--enforce-eager', '0', '--max-num-batched-tokens', '2048',
                '--out', str(args.out / 'e2e.json')]
    if args.case != 'fp16':
        sys.argv += ['--quantization', 'brmoe_int3']
    runpy.run_path(str(ROOT / 'bench/vllm_perf.py'), run_name='__main__')
    (args.out / 'tokens.json').write_text(json.dumps(generated))
    rows = json.loads((args.out / 'e2e.json').read_text())['rows']
    assert len(rows) == len(args.batch_sizes.split(',')) and all('error' not in r for r in rows), rows
    assert len(generated) == len(rows) * args.repeat, 'every measured decode must finish'
    print('E2E PASS', args.case, len(rows), 'batches', flush=True)


if __name__ == '__main__':
    main()
