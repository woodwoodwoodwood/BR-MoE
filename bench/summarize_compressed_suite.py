"""Validate completed sample records and publish a suite summary."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def summarize(root):
    jobs = json.loads((root/'jobs.json').read_text())
    entries = {}
    for job in jobs['jobs']:
        for task in job['tasks']:
            base = root/f"run_{job['job_id']}"/task
            result = base/'results.json'
            progress = base/'progress.json'
            if result.exists():
                state = json.loads(result.read_text())
                with (base/'predictions.jsonl').open() as f:
                    records = [json.loads(line) for line in f]
                assert [r['index'] for r in records] == list(range(state['total']))
                assert state['completed'] == state['total']
                expected = dict(mmlu=14042,hellaswag=10042,lambada_openai=5153)
                if task in expected:
                    assert state['total'] == expected[task], 'Partial datasets cannot be reported as full results'
                else:
                    assert state['total'] == state['corpus_tokens']//state['context_length']
                if task in ['mmlu','hellaswag']:
                    accuracy = sum(r['prediction']==r['label'] for r in records)/len(records)
                    assert abs(accuracy-state['acc']) < 1e-12
                if task == 'hellaswag':
                    normalized = sum(r['prediction_norm']==r['label'] for r in records)/len(records)
                    assert abs(normalized-state['acc_norm']) < 1e-12
                if task == 'lambada_openai':
                    assert abs(sum(r['correct'] for r in records)/len(records)-state['acc']) < 1e-12
                entries[task] = dict(state, result_file=str(result))
            elif progress.exists():
                entries[task] = json.loads(progress.read_text())
            else:
                scheduler = job.get('scheduler_state')
                status = 'planned' if job.get('planned') else (
                    'failed_before_scoring' if scheduler=='finished_or_removed' else 'waiting_or_loading')
                entries[task] = dict(status=status,job_id=job['job_id'])
    piqa = root.parent/'piqa/run_56019/results.json'
    entries['piqa'] = dict(json.loads(piqa.read_text()),result_file=str(piqa),reused=True)
    manifests = {v['checkpoint_manifest_sha256'] for v in entries.values() if 'checkpoint_manifest_sha256' in v}
    assert len(manifests) <= 1
    markers = {json.dumps(v['compression_manifest'],sort_keys=True) for v in entries.values() if 'compression_manifest' in v}
    assert len(markers) <= 1
    complete = all(v.get('status') == 'complete' for v in entries.values())
    summary = dict(status='complete' if complete else 'incomplete',
                   updated_utc=datetime.now(timezone.utc).isoformat(),tasks=entries)
    temporary = root/'summary.json.tmp'
    temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(root/'summary.json')
    lines = ['# Qwen1.5-MoE RCO per-tensor 压缩模型评测','',
        'Checkpoint：`artifacts/runs/rco_e2e/run_55343/compressed`。PIQA 复用相同 checkpoint 已完成的完整评测；其他任务为 2026-11-11 新增。','',
        '| 任务 | 协议 | 最终指标 | 状态 / 进度 |','|---|---|---|---|']
    settings = dict(wikitext2='raw test；2048 token 非重叠完整窗口；无 BOS；末尾不足一窗丢弃',
        hellaswag='validation，0-shot；主指标 acc_norm',piqa='validation，0-shot',
        lambada_openai='test，0-shot；最后一个词的所有子 token 贪心匹配',mmlu='57 学科 test，5-shot；每学科前 5 道 dev 示例；按题数加权')
    for task in ['wikitext2','hellaswag','piqa','lambada_openai','mmlu']:
        state = entries[task]
        if state['status'] == 'complete':
            if task == 'wikitext2':
                value = f"PPL **{state['ppl']:.4f}**"
            elif task == 'hellaswag':
                value = f"acc_norm **{state['acc_norm']*100:.2f}%**；acc {state['acc']*100:.2f}%"
            elif task == 'piqa':
                value = f"acc **{state['acc']*100:.2f}%**；acc_norm {state['acc_norm']*100:.2f}%"
            else:
                value = f"acc **{state['acc']*100:.2f}%**"
            status = f"完成 {state['completed']}/{state['total']}"
        else:
            value = '待完成'
            status = f"{state['status']} {state.get('completed',0)}/{state.get('total','?')}，作业 {state['job_id']}"
        lines.append(f'| {task} | {settings[task]} | {value} | {status} |')
    lines += ['', '各任务逐题或逐窗口记录在相应 `run_<jobid>/<task>/predictions.jsonl` 中，配置、数据哈希、参考 logits 检查与硬件记录在同目录的 `results.json` / `progress.json` 中。',
              '', '进行中的成绩不作为最终结果。原模型同协议基线尚未测量，当前不能据此计算压缩前后的精度损失。']
    temporary = root/'summary.md.tmp'
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(root/'summary.md')
    print(json.dumps({k:dict(status=v['status'],completed=v.get('completed'),total=v.get('total')) for k,v in entries.items()}),flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    summarize(args.root.resolve())
