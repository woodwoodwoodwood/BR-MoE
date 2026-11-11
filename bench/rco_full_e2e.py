"""Run the full-model RCO CLI with progress, timing and optimizer snapshots."""
import json
import os
import resource
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from RCO_solver import full_model_search, e2e_ste, rco_search


def main():
    out = Path(os.environ['RCO_RUN_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initialization = {'count': 0, 'seconds': 0.0}
    original_factors = e2e_ste.RCOInterpolatedLinear.get_bit_factors
    def timed_factors(self, quant_type, device, dtype):
        fresh = (quant_type not in ('F16', 'F32') and quant_type not in self._packed_base_cache and quant_type not in self._w_base_cache)
        before = time.monotonic()
        result = original_factors(self, quant_type, device, dtype)
        if fresh:
            initialization['count'] += 1
            initialization['seconds'] += time.monotonic() - before
            if initialization['count'] <= 12 or initialization['count'] % 100 == 0:
                print(json.dumps({'stage':'candidate_initialization', 'module':self.module_name,
                    'quant_type':str(quant_type), 'last_seconds':time.monotonic()-before,
                    **initialization, 'wall_seconds':time.monotonic()-started,
                    'gpu_allocated_gib':torch.cuda.memory_allocated()/1024**3,
                    'host_peak_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}), flush=True)
        return result
    e2e_ste.RCOInterpolatedLinear.get_bit_factors = timed_factors
    def progress(stage, data):
        private = {'alpha', 'optimizer', 'unit_keys', 'option_names', 'k_star', 'history'}
        public = {k:v for k,v in data.items() if k not in private}
        public.update(stage=stage, wall_seconds=time.monotonic()-started,
            gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3,
            gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3,
            host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        if stage == 'step':
            step = data['step'] + 1
            public['completed_steps'] = step
            # Initial iterations include lazy quantization/SVD cache construction.
            durations = step_times[-5:] + [data['step_seconds']]
            public['recent_step_eta_hours'] = sum(durations)/len(durations)*(data['steps']-step)/3600
            step_times.append(data['step_seconds'])
            if step == 1 or step % 10 == 0 or step == data['steps']:
                snapshot = {'completed_steps':step, 'alpha':data['alpha'].detach().cpu(),
                    'optimizer':data['optimizer'].state_dict(), 'history':data['history'],
                    'unit_keys':data['unit_keys'], 'option_names':data['option_names'],
                    'torch_rng_state':torch.get_rng_state(), 'cuda_rng_states':torch.cuda.get_rng_state_all()}
                tmp = out/'search_state.tmp'
                torch.save(snapshot, tmp)
                os.replace(tmp, out/'search_state.pt')
                chosen = {name:data['option_names'][int(k)] for name,k in zip(data['unit_keys'], data['k_star'].tolist())}
                tmp = out/'sampled_config.tmp'
                tmp.write_text(json.dumps(chosen,indent=2))
                os.replace(tmp, out/'sampled_config.json')
        print(json.dumps(public), flush=True)
        with (out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(public)+'\n')
    step_times = []
    original_cli = full_model_search.run_cli
    full_model_search.run_cli = lambda args: original_cli(args, progress_callback=progress)
    print(json.dumps({'stage':'start','torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
        'arguments':sys.argv[1:],
        'placement':{name:os.environ.get(name) for name in (
            'BRMOE_RCO_KEEP_ORIGINAL_ON_GPU','BRMOE_RCO_KEEP_FACTORS_ON_GPU','BRMOE_RCO_GPU_CACHE_GB')}}), flush=True)
    rco_search.main()
    print(json.dumps({'stage':'complete','wall_seconds':time.monotonic()-started}),flush=True)

if __name__ == '__main__':
    main()
