#!/bin/bash
#SBATCH -J moe-fuse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/moe_fuse_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/moe_fuse_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9
export FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-}
export BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=$REPO/bench_results/moe_fuse/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
export PYTHONPYCACHEPREFIX=/tmp/brmoe_fuse_pycache_${SLURM_JOB_ID}
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ${PHASE:-micro} == all ]]; then
  "$PY" "$SRC/bench/moe_fuse_study.py" verify --out "$OUT"
  "$PY" "$SRC/bench/moe_fuse_study.py" micro --out "$OUT" \
    --input "$REPO/bench_results/grouped_gemv/run_39061/real_inputs.pt" --profile
fi
if [[ ${PHASE:-micro} == e2e_pair || ${PHASE:-micro} == all ]]; then
  for config in baseline fused4; do
    "$PY" "$SRC/bench/moe_fuse_study.py" e2e --out "$OUT" --config "$config"
  done
else
  "$PY" "$SRC/bench/moe_fuse_study.py" "${PHASE:-micro}" --out "$OUT" ${ARGS:-}
fi
