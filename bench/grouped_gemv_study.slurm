#!/bin/bash
#SBATCH -J grouped-gemv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/grouped_gemv_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/grouped_gemv_%j.err
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
OUT=$REPO/bench_results/grouped_gemv/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ${PHASE:-micro} == all ]]; then
  "$PY" "$SRC/bench/grouped_gemv_study.py" verify --out "$OUT" \
    --configs 4_128_1_16_4,8_128_1_8_4,8_256_1_8_4
  "$PY" "$SRC/bench/grouped_gemv_study.py" collect --out "$OUT"
  "$PY" "$SRC/bench/grouped_gemv_study.py" micro --out "$OUT" \
    --input "$OUT/real_inputs.pt" --step 127 --configs adaptive --profile
  for cfg in baseline adaptive; do
    "$PY" "$SRC/bench/grouped_gemv_study.py" e2e --out "$OUT" --config "$cfg"
  done
else
  "$PY" "$SRC/bench/grouped_gemv_study.py" "${PHASE:-micro}" --out "$OUT" ${ARGS:-}
fi
