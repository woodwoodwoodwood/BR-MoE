"""Zero-shot PIQA likelihood evaluation of an existing BR-MoE checkpoint.

Prompt, leading space and character-normalized accuracy match the repository's
lm-eval PIQA task. No chat template or answer generation is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from examples.BR_compress_auto import UniversalMoEBRMoE


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def score_choices(model, tokenizer, doc):
    context = f"Question: {doc['goal']}\nAnswer:"
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    choices = [doc['sol1'], doc['sol2']]
    sequences = [tokenizer.encode(context + ' ' + c, add_special_tokens=False)
                 for c in choices]
    for sequence in sequences:
        if sequence[:len(context_ids)] != context_ids:
            raise ValueError('Tokenizer changed the context/continuation boundary')
        if len(sequence) <= len(context_ids):
            raise ValueError('Empty candidate continuation')
        if len(sequence) > model.config.max_position_embeddings:
            raise ValueError('PIQA input exceeds the model context window')
    # Drop the final input token, retaining one logit for every target token.
    lengths = [len(s) - 1 for s in sequences]
    width = max(lengths)
    input_ids = torch.full((2, width), tokenizer.pad_token_id,
                           dtype=torch.long, device='cuda')
    attention_mask = torch.zeros_like(input_ids)
    for i, sequence in enumerate(sequences):
        input_ids[i, :lengths[i]] = torch.tensor(sequence[:-1], device='cuda')
        attention_mask[i, :lengths[i]] = 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False).logits
    scores = []
    for i, sequence in enumerate(sequences):
        start = len(context_ids) - 1
        candidate_logits = logits[i, start:lengths[i]].float()
        targets = torch.tensor(sequence[len(context_ids):], device='cuda')
        if not torch.isfinite(candidate_logits).all():
            raise RuntimeError('Non-finite candidate logits')
        score = -F.cross_entropy(candidate_logits, targets, reduction='sum').item()
        if not math.isfinite(score):
            raise RuntimeError('Non-finite candidate likelihood')
        scores.append(score)
    normalized = [s / len(c) for s, c in zip(scores, choices)]
    return context, scores, normalized, [len(s) - len(context_ids) for s in sequences]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / 'progress.json'
    started = time.monotonic()
    state = dict(status='loading', job_id=os.environ.get('SLURM_JOB_ID'),
                 checkpoint=str(Path(args.checkpoint).resolve()),
                 dataset=str(Path(args.dataset).resolve()), task='piqa',
                 split='validation', num_fewshot=0, completed=0,
                 prompt_template='Question: {goal}\nAnswer:',
                 target_delimiter=' ', normalization='candidate character count',
                 add_special_tokens=False, batch_size=2,
                 torch_version=torch.__version__)
    save_json(progress_path, state)
    print(json.dumps(state), flush=True)
    try:
        torch.manual_seed(42)
        torch.set_num_threads(4)
        dataset_path = Path(args.dataset)
        dataset = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        if args.limit is not None:
            if not 0 < args.limit <= len(dataset):
                raise ValueError('Invalid evaluation limit')
            dataset = dataset[:args.limit]
        if any(int(doc['label']) not in (0, 1) for doc in dataset):
            raise ValueError('PIQA validation requires labels 0 or 1')
        state['total'] = len(dataset)
        state['dataset_sha256'] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        marker = Path(args.checkpoint) / 'COMPRESSION_COMPLETE.json'
        state['compression_manifest'] = json.loads(marker.read_text())
        state['evaluation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = UniversalMoEBRMoE.from_compressed(
            args.checkpoint, device='cuda', compute_dtype=torch.float16,
            attn_implementation='eager').eval()
        model.config._attn_implementation = 'eager'
        # Verify the exact same logits used to validate the generation run.
        inputs = tokenizer('The capital of France is', return_tensors='pt').to('cuda')
        with torch.inference_mode():
            actual = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        reference = torch.load(Path(args.checkpoint) / 'prefill_reference.pt',
                               map_location='cpu', weights_only=True)
        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-3)
        state['reference_max_logit_difference'] = (actual - reference).abs().max().item()
        state['gpu'] = torch.cuda.get_device_name()
        state['status'] = 'running'
        save_json(progress_path, state)
        print(json.dumps(state), flush=True)
        evaluated_at = time.monotonic()
        correct = correct_norm = 0
        with (output / 'predictions.jsonl').open('x', buffering=1) as predictions:
            for index, doc in enumerate(dataset):
                _, scores, normalized, token_lengths = score_choices(model, tokenizer, doc)
                prediction = max(range(2), key=scores.__getitem__)
                prediction_norm = max(range(2), key=normalized.__getitem__)
                label = int(doc['label'])
                correct += prediction == label
                correct_norm += prediction_norm == label
                predictions.write(json.dumps(dict(index=index, goal=doc['goal'],
                    choices=[doc['sol1'], doc['sol2']], label=label,
                    loglikelihood=scores, normalized_loglikelihood=normalized,
                    continuation_tokens=token_lengths, prediction=prediction,
                    prediction_norm=prediction_norm), ensure_ascii=False) + '\n')
                completed = index + 1
                if completed == 1 or completed % 10 == 0 or completed == len(dataset):
                    elapsed = time.monotonic() - evaluated_at
                    state.update(completed=completed, acc=correct / completed,
                        acc_norm=correct_norm / completed,
                        elapsed_seconds=time.monotonic() - started,
                        evaluation_seconds=elapsed,
                        eta_seconds=elapsed / completed * (len(dataset) - completed),
                        gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
                    save_json(progress_path, state)
                    print(json.dumps(state), flush=True)
        state['status'] = 'complete'
        save_json(output / 'results.json', state)
        save_json(progress_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                     elapsed_seconds=time.monotonic() - started)
        save_json(progress_path, state)
        raise


if __name__ == '__main__':
    main()
