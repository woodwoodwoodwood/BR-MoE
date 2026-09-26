#!/bin/bash
#SBATCH -J small-decode
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/small_decode_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/small_decode_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9 FLASHINFER_NVCC=$BASE/cuda-12.9/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-} BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/tmp/brmoe_small_pycache_${SLURM_JOB_ID}
unset PYTHONDONTWRITEBYTECODE
OUT=$REPO/bench_results/small_decode/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.sm,clocks.mem --format=csv
if [[ ${PHASE:-micro} == counters ]]; then
  # Use a writable profiler lock directory only after checking the node has no
  # other Nsight Compute instance (see Nsight Compute Profiling Guide FAQ).
  if pgrep -x ncu >/dev/null; then
    echo 'Another Nsight Compute instance is active; retry the counter job later.' >&2
    exit 1
  fi
  export TMPDIR=/tmp/brmoe_ncu_${SLURM_JOB_ID}
  mkdir -p "$TMPDIR"
  for config in ${CASES:-baseline}; do
    mkdir -p "$OUT/$config"
    "${NCU_BIN:-$CUDA_HOME/bin/ncu}" --nvtx --nvtx-include 'measure/' --target-processes all \
      --cache-control "${NCU_CACHE:-all}" --clock-control none --set basic \
      --section MemoryWorkloadAnalysis --section InstructionStats \
      --export "$OUT/$config/counters" --force-overwrite \
      "$PY" "$SRC/bench/small_decode_study.py" counters --config "$config" --out "$OUT/$config" ${ARGS:-}
  done
elif [[ ${PHASE:-micro} == e2e ]]; then
  for config in ${CASES:-baseline}; do
    "$PY" "$SRC/bench/small_decode_study.py" e2e --config "$config" --out "$OUT/$config" ${ARGS:-}
  done
else
  "$PY" "$SRC/bench/small_decode_study.py" "${PHASE:-micro}" --out "$OUT" ${ARGS:-}
fi
