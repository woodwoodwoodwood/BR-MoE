#!/bin/bash
#SBATCH -J full-int3-study
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/full_int3_study_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/full_int3_study_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
STUDY=$REPO/bench_results/full_int3_study
SRC=${SRC:-$STUDY/source}
PY=$BASE/miniconda3/envs/vllm5090/bin/python
export CUDA_HOME=$BASE/cuda-12.9
export FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-}
export BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=$STUDY/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ${PHASE:-all} == all || ${PHASE:-all} == collect ]]; then
  "$PY" "$SRC/bench/full_int3_study.py" collect --out "$OUT" 2>&1 | tee "$OUT/collect.log"
fi
INPUT=${INPUT:-$OUT}
if [[ ${PHASE:-all} == all || ${PHASE:-all} == sweep ]]; then
  "$PY" "$SRC/bench/full_int3_study.py" sweep --out "$INPUT" 2>&1 | tee "$OUT/sweep.log"
fi
if [[ ${PHASE:-all} == all || ${PHASE:-all} == e2e ]]; then
  for cfg in ${CONFIGS:-baseline 32_1 32_2 32_3 64_1 64_2 64_3 128_1 128_2 128_3}; do
    export FULL_INT3_CONFIG=$cfg
    "$PY" "$SRC/bench/vllm_perf.py" \
      --model "$BASE/models/brmoe-3bit-vllm-int3dense" \
      --tokenizer "$BASE/models/brmoe-tokfix" --tag "full_int3_${cfg}" \
      --quantization brmoe_int3 --batch-sizes 8,32 --input-len 128 --output-len 128 \
      --repeat 3 --enforce-eager 0 --max-num-batched-tokens 2048 \
      --out "$OUT/e2e_${cfg}.json" 2>&1 | tee "$OUT/e2e_${cfg}.log"
  done
fi
echo "[done] $OUT"
