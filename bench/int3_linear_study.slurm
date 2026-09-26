#!/bin/bash
#SBATCH -J int3-linear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/int3_linear_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/int3_linear_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9 FLASHINFER_NVCC=$BASE/cuda-12.9/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-} BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/tmp/brmoe_linear_pycache_${SLURM_JOB_ID}
# Fresh per-job cache prevents shared stale bytecode; subsequent phases reuse dependencies.
unset PYTHONDONTWRITEBYTECODE
OUT=$REPO/bench_results/int3_linear/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ${PHASE:-collect} == collect ]]; then
  "$PY" "$SRC/bench/int3_linear_study.py" collect --out "$OUT" ${ARGS:-}
  "$PY" "$SRC/bench/int3_linear_study.py" micro --input "$OUT" --out "$OUT" --prefill
elif [[ ${PHASE:-collect} == e2e_pair || ${PHASE:-collect} == profile_pair ]]; then
  STUDY_PHASE=${PHASE%_pair}
  for config in baseline ${CANDIDATE:-single_32_3}; do
    "$PY" "$SRC/bench/int3_linear_study.py" "$STUDY_PHASE" --out "$OUT/$config" --linear-config "$config" ${ARGS:-}
  done
elif [[ ${PHASE:-collect} == compare ]]; then
  for phase in micro moe verify; do
    "$PY" "$SRC/bench/int3_linear_study.py" "$phase" --out "$OUT" ${ARGS:-}
  done
else
  "$PY" "$SRC/bench/int3_linear_study.py" "$PHASE" --out "$OUT" ${ARGS:-}
fi
