"""Monitor submitted evaluations and validate their continuously updated summary."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from summarize_compressed_suite import summarize


def main(root):
    previous = None
    while True:
        jobs = json.loads((root/'jobs.json').read_text())
        text = subprocess.check_output(['squeue','-h','-u','jianglei','-o','%i|%T|%R'],text=True)
        active = {line.split('|')[0]:line.split('|')[1:] for line in text.splitlines()}
        for job in jobs['jobs']:
            job['scheduler_state'] = active.get(job['job_id'],['finished_or_removed'])[0]
        temporary = root/'jobs.json.tmp'
        temporary.write_text(json.dumps(jobs,indent=2)+'\n')
        temporary.replace(root/'jobs.json')
        complete = summarize(root)
        summary = json.loads((root/'summary.json').read_text())
        status = {task:(state['status'],state.get('completed')) for task,state in summary['tasks'].items()}
        if status != previous:
            print(datetime.now(timezone.utc).isoformat(),json.dumps(status),flush=True)
            previous = status
        if complete:
            print('All benchmark results validated and summarized.',flush=True)
            return
        if not any(j['job_id'] in active for j in jobs['jobs']):
            print('All jobs ended; inspect summary for incomplete or failed tasks.',flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    main(args.root.resolve())
