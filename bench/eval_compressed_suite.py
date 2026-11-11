"""Evaluate a saved BR-MoE checkpoint with local, explicitly recorded protocols."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def preprocess_hs(text):
    return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')


def mmlu_question(doc):
    return doc['question'].strip() + '\n' + '\n'.join(
        f'{letter}. {choice}' for letter, choice in zip('ABCD', doc['choices'])) + '\nAnswer:'


def pair(tokenizer, context, continuation):
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    sequence = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or sequence[:len(prefix)] != prefix or len(sequence) <= len(prefix):
        raise ValueError('Invalid or unstable continuation token boundary')
    return sequence, len(prefix)


def verify_native_decode(torch, manifest):
    """Check the active GPU codec bit-for-bit against independent native decode."""
    import numpy as np
    from brmoe_quant.native import quantize, dequantize
    from brmoe_quant.gpu import dequantize_cuda
    allocation = json.loads((Path(manifest)/'best_tensor_allocation.json').read_text())
    formats = sorted({u['quant_type'] for u in allocation['units'].values()})
    source = np.random.default_rng(17).normal(size=(3,256)).astype(np.float32)
    importance = np.linspace(.01,2,256,dtype=np.float32)
    for qtype in formats:
        raw = quantize(source,qtype,importance)
        expected = torch.from_numpy(dequantize(raw,source.shape,qtype)).to('cuda',torch.float16)
        actual = dequantize_cuda(torch.from_numpy(raw).cuda(),source.shape,qtype,torch.float16)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return formats


class Evaluator:
    def __init__(self, model, tokenizer, torch):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.max_length = model.config.max_position_embeddings

    def likelihood(self, requests):
        torch = self.torch
        import torch.nn.functional as F
        encoded = [pair(self.tokenizer, *request) for request in requests]
        width = max(len(seq) - 1 for seq, _ in encoded)
        if width > self.max_length:
            raise ValueError(f'Input {width} exceeds model context {self.max_length}')
        ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, (seq, _) in enumerate(encoded):
            ids[i, :len(seq)-1] = torch.tensor(seq[:-1], device='cuda')
            mask[i, :len(seq)-1] = 1
        start = min(n - 1 for _, n in encoded)
        # Only candidate positions need vocabulary logits; no effect on attention.
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                logits_to_keep=width-start).logits
        result = []
        for i, (seq, n) in enumerate(encoded):
            selected = logits[i, n-1-start:len(seq)-1-start].float()
            target = torch.tensor(seq[n:], device='cuda')
            if not torch.isfinite(selected).all():
                raise RuntimeError('Non-finite continuation logits')
            loss = F.cross_entropy(selected, target, reduction='sum').item()
            result.append(dict(loglikelihood=-loss,
                               greedy=bool((selected.argmax(-1) == target).all()),
                               target_tokens=len(seq)-n))
        return result

    def mmlu(self, contexts):
        torch = self.torch
        sequences = [self.tokenizer.encode(x, add_special_tokens=False) for x in contexts]
        width = max(map(len, sequences))
        if width > self.max_length:
            raise ValueError(f'MMLU prompt {width} exceeds model context; no truncation allowed')
        candidates = []
        for context in contexts:
            ids = [pair(self.tokenizer, context, ' '+letter) for letter in 'ABCD']
            if any(len(seq)-n != 1 for seq, n in ids):
                raise ValueError('MMLU single-token fast path requires one token per answer')
            candidates.append([seq[-1] for seq, _ in ids])
        ids = torch.full((len(contexts), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, -len(seq):] = torch.tensor(seq, device='cuda')
            mask[i, -len(seq):] = 1
        positions = mask.cumsum(-1)-1
        positions.masked_fill_(mask == 0, 0)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask,
                position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite MMLU logits')
        logp = logits.log_softmax(-1)
        return [logp[i, token_ids].cpu().tolist() for i, token_ids in enumerate(candidates)]


def aggregate(task, records):
    n = len(records)
    if not n:
        return {}
    if task == 'wikitext2':
        tokens = sum(r['tokens'] for r in records)
        nll = sum(r['nll'] for r in records)
        return dict(ppl=math.exp(nll/tokens), nll=nll, scored_tokens=tokens,
                    token_accuracy=sum(r['correct_tokens'] for r in records)/tokens)
    if task == 'lambada_openai':
        return dict(acc=sum(r['correct'] for r in records)/n,
                    perplexity=math.exp(-sum(r['loglikelihood'] for r in records)/n))
    if task == 'hellaswag':
        return dict(acc=sum(r['prediction']==r['label'] for r in records)/n,
                    acc_norm=sum(r['prediction_norm']==r['label'] for r in records)/n)
    subjects = defaultdict(list)
    for r in records:
        subjects[r['subject']].append(r['prediction']==r['label'])
    by_subject = {s:dict(acc=sum(v)/len(v), total=len(v)) for s,v in subjects.items()}
    return dict(acc=sum(r['prediction']==r['label'] for r in records)/n,
                subject_macro_acc=sum(s['acc'] for s in by_subject.values())/len(by_subject),
                subjects=by_subject)


def evaluate(task, evaluator, args, manifest, run_metadata):
    import torch
    import torch.nn.functional as F
    output = args.out / task
    output.mkdir(parents=True, exist_ok=True)
    data_path = args.data / manifest[task]['path']
    assert digest(data_path) == manifest[task]['sha256']
    with data_path.open() as stream:
        docs = [json.loads(line) for line in stream if line.strip()]
    state = dict(run_metadata, task=task, status='running', dataset_sha256=digest(data_path),
                 num_fewshot=5 if task=='mmlu' else 0)
    if task == 'mmlu':
        dev_path = args.data / manifest['mmlu_dev']['path']
        assert digest(dev_path) == manifest['mmlu_dev']['sha256']
        devs = json.loads(dev_path.read_text())
        prefixes = {subject: f'The following are multiple choice questions (with answers) about {subject.replace("_", " ")}.\n\n' +
                    '\n\n'.join(mmlu_question(d)+' '+'ABCD'[int(d['answer'])] for d in dev)+'\n\n'
                    for subject, dev in devs.items()}
        state.update(fewshot_selection='first five subject dev examples',
                     aggregation='test-example weighted mean over 57 subjects',
                     dev_sha256=digest(dev_path), max_batch_tokens=args.max_batch_tokens)
    if task == 'wikitext2':
        text = '\n\n'.join(d['text'] for d in docs)
        token_ids = evaluator.tokenizer.encode(text, add_special_tokens=False)
        chunks = len(token_ids)//args.context_length
        docs = [token_ids[i*args.context_length:(i+1)*args.context_length] for i in range(chunks)]
        state.update(context_length=args.context_length, stride=args.context_length,
                     corpus_tokens=len(token_ids), dropped_tail_tokens=len(token_ids)%args.context_length,
                     protocol='raw test text joined with double newline; nonoverlapping full windows; no BOS')
    if args.limit:
        docs = docs[:args.limit]
    state['total'] = len(docs)
    path = output / 'predictions.jsonl'
    records = []
    if path.exists():
        old_state = json.loads((output/'progress.json').read_text())
        for key in ['dataset_sha256','evaluation_sha256','checkpoint_manifest_sha256','num_fewshot','total']:
            if old_state.get(key) != state.get(key):
                raise ValueError(f'Cannot resume with different {key}')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if [r['index'] for r in records] != list(range(len(records))):
            raise ValueError('Resume file has noncontiguous sample indices')
    started, initial = time.monotonic(), len(records)

    def progress(final=False):
        completed = len(records)
        elapsed = time.monotonic()-started
        state.update(completed=completed, status='complete' if final else 'running',
                     **aggregate(task, records), evaluation_seconds=elapsed,
                     resumed_records=initial, gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                     eta_seconds=elapsed/max(1,completed-initial)*(len(docs)-completed))
        save(output/'progress.json', state)
        public = {k:v for k,v in state.items() if k not in ('subjects','compression_manifest')}
        print(json.dumps(public), flush=True)
        if final:
            save(output/'results.json', state)
    progress()
    try:
        with path.open('a', buffering=1) as stream:
            while len(records) < len(docs):
                i = len(records)
                if task == 'hellaswag':
                    doc = docs[i]
                    context = preprocess_hs(doc['activity_label']+': '+doc['ctx_a']+' '+doc['ctx_b'].capitalize())
                    choices = [preprocess_hs(c) for c in doc['endings']]
                    scores = evaluator.likelihood([(context, ' '+c) for c in choices])
                    lls = [r['loglikelihood'] for r in scores]
                    norm = [ll/len(c) for ll,c in zip(lls,choices)]
                    new = [dict(index=i, label=int(doc['label']), loglikelihood=lls,
                        prediction=max(range(4),key=lls.__getitem__),
                        prediction_norm=max(range(4),key=norm.__getitem__))]
                elif task == 'lambada_openai':
                    batch = docs[i:i+args.batch_size]
                    requests = [(d['text'].rsplit(' ',1)[0], ' '+d['text'].rsplit(' ',1)[1]) for d in batch]
                    scores = evaluator.likelihood(requests)
                    new = [dict(index=i+j, loglikelihood=s['loglikelihood'],
                        correct=s['greedy'], target_tokens=s['target_tokens']) for j,s in enumerate(scores)]
                elif task == 'mmlu':
                    batch = docs[i:i+args.batch_size]
                    contexts = [prefixes[d['subject']]+mmlu_question(d) for d in batch]
                    lengths = [len(evaluator.tokenizer.encode(c,add_special_tokens=False)) for c in contexts]
                    while len(batch)>1 and max(lengths)*len(batch)>args.max_batch_tokens:
                        batch, contexts, lengths = batch[:-1], contexts[:-1], lengths[:-1]
                    scores = evaluator.mmlu(contexts)
                    if not records:
                        # Check optimized, left-padded batched scoring against full likelihood scoring.
                        independent = evaluator.likelihood([(contexts[0], ' '+c) for c in 'ABCD'])
                        reference = [r['loglikelihood'] for r in independent]
                        if max(abs(x-y) for x,y in zip(scores[0],reference)) > 0.05:
                            raise RuntimeError('MMLU batched scorer failed independent likelihood check')
                        state['scorer_check_max_difference'] = max(abs(x-y) for x,y in zip(scores[0],reference))
                    new = [dict(index=i+j, subject=d['subject'], label=int(d['answer']),
                         loglikelihood=s, prediction=max(range(4),key=s.__getitem__),
                         prompt_tokens=lengths[j]) for j,(d,s) in enumerate(zip(batch,scores))]
                else:
                    ids = torch.tensor([docs[i]],dtype=torch.long,device='cuda')
                    with torch.inference_mode():
                        logits = evaluator.model(input_ids=ids,use_cache=False).logits[0,:-1]
                    target = ids[0,1:]
                    nll = 0.0
                    for start in range(0,len(target),64):
                        value = logits[start:start+64].float()
                        if not torch.isfinite(value).all():
                            raise RuntimeError('Non-finite WikiText logits')
                        nll += F.cross_entropy(value,target[start:start+64],reduction='sum').item()
                    new = [dict(index=i,nll=nll,tokens=len(target),
                                correct_tokens=int((logits.argmax(-1)==target).sum()))]
                for row in new:
                    stream.write(json.dumps(row)+'\n')
                    records.append(row)
                if i == initial or len(records)//25 != i//25 or len(records) == len(docs):
                    progress()
        progress(final=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error),traceback=traceback.format_exc(),completed=len(records))
        save(output/'progress.json',state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tasks',required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--max-batch-tokens',type=int,default=4096)
    parser.add_argument('--context-length',type=int,default=2048)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'startup.json',dict(status='importing',tasks=args.tasks,job_id=os.environ.get('SLURM_JOB_ID')))
    import torch
    from transformers import AutoTokenizer
    from examples.BR_compress_auto import UniversalMoEBRMoE
    torch.manual_seed(42)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    verified_formats = verify_native_decode(torch,args.checkpoint.parent)
    print(json.dumps(dict(native_decode_bit_exact=verified_formats)),flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = UniversalMoEBRMoE.from_compressed(str(args.checkpoint),device='cuda',
                compute_dtype=torch.float16,attn_implementation='eager').eval()
    model.config._attn_implementation = 'eager'
    with torch.inference_mode():
        inputs = tokenizer('The capital of France is',return_tensors='pt').to('cuda')
        # Preserve the original generation check's full-prompt lm_head shape.
        # Cropping to one position selects a different FP16 GEMM kernel.
        actual = model(**inputs,use_cache=False).logits[:,-1].float().cpu()
    reference = torch.load(args.checkpoint/'prefill_reference.pt',map_location='cpu',weights_only=True)
    reference_error = dict(max_abs=(actual-reference).abs().max().item(),
        relative_l2=((actual-reference).norm()/reference.norm()).item(),
        same_argmax=bool(actual.argmax()==reference.argmax()))
    print(json.dumps(dict(reference_check=reference_error)),flush=True)
    # Different GPU architectures choose different FP16 GEMM implementations.
    # Use a vector norm and absolute bound instead of relative error near zeros;
    # independently require exact native block decoding and the same top token.
    if not (reference_error['same_argmax'] and reference_error['relative_l2'] <= 0.003
            and reference_error['max_abs'] <= 0.05):
        raise RuntimeError(f'Cross-GPU FP16 reference check failed: {reference_error}')
    metadata = dict(job_id=os.environ.get('SLURM_JOB_ID'),checkpoint=str(args.checkpoint),
        checkpoint_manifest_sha256=digest(args.checkpoint/'COMPRESSION_COMPLETE.json'),
        compression_manifest=json.loads((args.checkpoint/'COMPRESSION_COMPLETE.json').read_text()),
        evaluation_sha256=digest(__file__),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        reference_max_logit_difference=(actual-reference).abs().max().item(),
        reference_relative_l2=reference_error['relative_l2'],
        reference_same_argmax=reference_error['same_argmax'],
        reference_check_protocol='FP16 max_abs<=0.05; relative_l2<=0.003; same argmax; bit-exact codec check',
        native_decode_verified_formats=verified_formats,
        attention_implementation='eager',add_special_tokens=False,batch_size=args.batch_size)
    save(args.out/'startup.json',dict(metadata,status='model_verified',tasks=args.tasks))
    print(json.dumps(metadata),flush=True)
    evaluator = Evaluator(model,tokenizer,torch)
    manifest = json.loads((args.data/'manifest.json').read_text())
    for task in args.tasks.split(','):
        if task not in ['wikitext2','hellaswag','lambada_openai','mmlu']:
            raise ValueError(f'Unknown task {task}')
        evaluate(task,evaluator,args,manifest,metadata)


if __name__ == '__main__':
    main()
