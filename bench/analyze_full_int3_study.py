"""Summarize stage events, real routing, MoE replay and clean E2E results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    out={}
    if (args.run/'stages.json').exists():
        rows=json.loads((args.run/'stages.json').read_text())
        groups=defaultdict(lambda:defaultdict(float))
        for r in rows:
            phase='decode' if r['m']==r['batch'] else 'prefill_or_mixed'
            stage=r['stage']
            if stage=='linear_total':
                stage='attention_linear' if 'self_attn' in r['layer'] else 'shared_or_dense_linear'
            groups[(r['batch'],phase,r['step'])][stage]+=r['us']/1000
        summary=defaultdict(lambda:defaultdict(list))
        for (bs,phase,step),vals in groups.items():
            for k,v in vals.items():summary[f'bs{bs}_{phase}'][k].append(v)
        out['stage_ms_per_step']={k:{s:statistics.median(v) for s,v in vals.items()} for k,vals in summary.items()}
    if (args.run/'routes.json').exists():
        rows=json.loads((args.run/'routes.json').read_text())
        for bs in (8,32):
            rs=[r for r in rows if r['batch']==bs and r['m']==bs]
            if rs:
                out[f'routing_bs{bs}']=dict(samples=len(rs),
                    mean_active_experts=statistics.mean(r['active_experts'] for r in rs),
                    mean_padding16_ratio=statistics.mean(r['padded16']/(bs*6) for r in rs),
                    max_tokens_per_expert= max(r['max_tokens_per_expert'] for r in rs))
    if (args.run/'moe_sweep.json').exists():
        out['moe_sweep']=json.loads((args.run/'moe_sweep.json').read_text())
    out['e2e']={p.stem.removeprefix('e2e_'):json.loads(p.read_text())['rows'] for p in args.run.glob('e2e_*.json')}
    (args.run/'summary.json').write_text(json.dumps(out,indent=2))
    compact={k:v for k,v in out.items() if k not in ('moe_sweep','e2e')}
    print(json.dumps(compact,indent=2))
    for cfg,rows in out['e2e'].items():
        print(cfg,[(r['batch'],round(r.get('e2e_ms',0),2),round(r.get('tpot_ms',0),3)) for r in rows])


if __name__=='__main__':main()
