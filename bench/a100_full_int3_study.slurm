#!/bin/bash
#SBATCH -J a100-full-int3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/a100_full_int3_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/a100_full_int3_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9 FLASHINFER_NVCC=$BASE/cuda-12.9/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-} BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/tmp/brmoe_a100_pycache_${SLURM_JOB_ID}
unset PYTHONDONTWRITEBYTECODE
OUT=$REPO/bench_results/a100_full_int3/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.sm,clocks.mem --format=csv
for config in ${CASES:-fp16 unfused fused hybrid64}; do
  "$PY" "$SRC/bench/a100_full_int3_study.py" --case "$config" --out "$OUT/$config" ${ARGS:-}
done
