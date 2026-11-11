"""Real-model IQ smoke check: selected projections, not whole-model quantization."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from brmoe_quant import FORMATS, tensor_nbytes
from brmoe_quant.native import quantize, dequantize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map='auto',
        max_memory={i: '21GiB' for i in range(torch.cuda.device_count())},
        attn_implementation='eager', local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = tokenizer('Explain how a telescope collects light from distant stars.', return_tensors='pt')
    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
    names = ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.shared_expert.up_proj']
    # Find an expert that actually receives this prompt, rather than fabricating importance.
    stats = {}
    handles = []
    def capture(name):
        def hook(module, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            if x.numel():
                stats[name] = x.square().mean(0).cpu().numpy()
        return hook
    for name, module in model.named_modules():
        if name in names or (name.startswith('model.layers.0.mlp.experts.') and name.endswith('.up_proj')):
            handles.append(module.register_forward_pre_hook(capture(name)))
    with torch.inference_mode():
        reference = model(**inputs, use_cache=False).logits.float().cpu()
    for handle in handles:
        handle.remove()
    names.append(next(n for n in stats if '.experts.' in n))
    results = []
    for name in names:
        layer = model.get_submodule(name)
        original = layer.weight.detach().clone()
        source = original.float().cpu().numpy()
        for qtype in [n for n in FORMATS if n.startswith('IQ')]:
            start = time.monotonic()
            packed = quantize(source, qtype, stats[name])
            restored = dequantize(packed, source.shape, qtype)
            assert packed.nbytes == tensor_nbytes(source.shape, qtype)
            base = torch.from_numpy(restored).to(original.device)
            residual = original.float() - base
            u, s, v = torch.svd_lowrank(residual, q=16, niter=2)
            for rank in (0, 16):
                candidate = base if rank == 0 else base + (u * s) @ v.T
                try:
                    with torch.inference_mode():
                        layer.weight.copy_(candidate.to(original.dtype))
                        actual = model(**inputs, use_cache=False).logits.float().cpu()
                    assert torch.isfinite(actual).all()
                    kl = F.kl_div(actual.log_softmax(-1), reference.softmax(-1), reduction='none').sum(-1).mean().item()
                    results.append(dict(module=name, quant_type=qtype, rank=rank, packed_bytes=packed.nbytes,
                        relative_weight_l2=float(torch.linalg.vector_norm(original.float()-candidate)/torch.linalg.vector_norm(original.float())),
                        next_token_kl=kl, elapsed_seconds=time.monotonic()-start))
                    print(json.dumps(results[-1]), flush=True)
                finally:
                    with torch.no_grad():
                        layer.weight.copy_(original)
            (out/'results.json').write_text(json.dumps({'scope':'one attention, one shared and one active routed projection; independent replacements',
                'model':args.model, 'device_map':{k:str(v) for k,v in model.hf_device_map.items()},'results':results}, indent=2))
    print('IQ_MODEL_SMOKE_PASSED', flush=True)

if __name__ == '__main__':
    main()
