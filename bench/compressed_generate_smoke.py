"""Compress the final RCO allocation and reload it for a 4090 generation check."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer
from examples.BR_compress_auto import load_hf_moe_model, build_auto_compress_config, UniversalMoEBRMoE

PROMPTS = ['The capital of France is', 'To boil water safely, first', 'Once upon a time, a little robot']
START = time.monotonic()


def report(stage, **values):
    values.update(stage=stage, elapsed_seconds=time.monotonic()-START,
        gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    print(json.dumps(values,ensure_ascii=False),flush=True)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_logits(model, tokenizer, prompt):
    inputs = tokenizer(prompt,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        logits = model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite logits in compressed-model prefill')
    return logits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--allocation',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4)
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint)
    manifest_path=Path(args.allocation)
    manifest=json.loads(manifest_path.read_text())
    expected_signature={'allocation_sha256':fingerprint(manifest_path),
        'importance_sha256':fingerprint(manifest_path.parent/manifest['imatrix_file']),
        'source_model':str(Path(args.model).resolve())}
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    marker=checkpoint/'COMPRESSION_COMPLETE.json'
    if marker.is_file():
        saved=json.loads(marker.read_text())
        if any(saved.get(k)!=v for k,v in expected_signature.items()):
            raise RuntimeError('Existing checkpoint belongs to a different allocation/model')
        report('reuse_checkpoint',checkpoint=str(checkpoint))
    else:
        if checkpoint.exists():
            raise RuntimeError('Checkpoint directory exists without completion marker; preserve it and choose a new output')
        build=checkpoint.with_name(checkpoint.name+'.build_'+os.environ.get('SLURM_JOB_ID',str(os.getpid())))
        build.mkdir(parents=True,exist_ok=False)
        report('load_source',model=args.model)
        model=load_hf_moe_model(args.model,dtype=torch.float16).eval()
        model.config._attn_implementation='eager'
        report('source_loaded')
        with np.load(manifest_path.parent/manifest['imatrix_file'],allow_pickle=False) as data:
            importance=dict(data)
        config=build_auto_compress_config(model,manifest,importance=importance,
            compensator_dtype='fp16',svd_iter=1)
        model.mix_moe_bit=True
        report('compress_start',units=len(manifest['units']))
        # The model stays on CPU initially; each projection is quantized on CUDA
        # and replaced with packed weights before proceeding to the next one.
        original_compress = UniversalMoEBRMoE.compress_model
        original_compress(model,config,device='cuda',compute_dtype=torch.float16)
        model.eval()
        report('compress_done')
        reference=last_logits(model,tokenizer,PROMPTS[0])
        torch.save(reference,build/'prefill_reference.pt')
        UniversalMoEBRMoE.save_compressed(model,str(build))
        tokenizer.save_pretrained(build)
        (build/'COMPRESSION_COMPLETE.json').write_text(json.dumps({**expected_signature,
            'allocation_payload_bytes':manifest['total_payload_bytes'],
            'compute_dtype':'float16','compensator_dtype':'fp16'},indent=2))
        build.rename(checkpoint)
        report('checkpoint_saved',checkpoint=str(checkpoint))
        del model,config,importance
        gc.collect();torch.cuda.empty_cache()
    report('reload_start')
    model=UniversalMoEBRMoE.from_compressed(str(checkpoint),device='cuda',
        compute_dtype=torch.float16,attn_implementation='eager').eval()
    model.config._attn_implementation='eager'
    reference=torch.load(checkpoint/'prefill_reference.pt',map_location='cpu',weights_only=True)
    actual=last_logits(model,tokenizer,PROMPTS[0])
    torch.testing.assert_close(actual,reference,atol=1e-2,rtol=1e-3)
    report('reload_verified',max_logit_difference=float((actual-reference).abs().max()))
    generations=[]
    for prompt in PROMPTS:
        inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
        torch.cuda.synchronize();before=time.monotonic()
        with torch.inference_mode():
            tokens=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        new_ids=tokens[0,inputs['input_ids'].shape[1]:].tolist()
        text=tokenizer.decode(new_ids,skip_special_tokens=True)
        result={'prompt':prompt,'generated_text':text,'new_token_ids':new_ids,
            'new_tokens':len(new_ids),'seconds':time.monotonic()-before}
        if not text.strip():raise RuntimeError(f'Empty generated text for {prompt!r}')
        generations.append(result);report('generation',**result)
        (output/'generations.json').write_text(json.dumps({'checkpoint':str(checkpoint),
            'model':args.model,'allocation':args.allocation,'generations':generations},ensure_ascii=False,indent=2))
    report('complete',checkpoint=str(checkpoint),prompts=len(generations))

if __name__=='__main__':main()
