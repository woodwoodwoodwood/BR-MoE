#!/bin/bash
#SBATCH -J moe-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/moe_large_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/moe_large_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9 FLASHINFER_NVCC=$BASE/cuda-12.9/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-} BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/tmp/brmoe_large_pycache_${SLURM_JOB_ID} PYTHONDONTWRITEBYTECODE=1
OUT=$REPO/bench_results/moe_large/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
if [[ ${PHASE:-collect} == collect ]]; then
  "$PY" "$SRC/bench/moe_large_batch_study.py" collect --out "$OUT"
  "$PY" "$SRC/bench/moe_large_batch_study.py" micro --input "$OUT/real_inputs.pt" --out "$OUT" --profile
elif [[ ${PHASE:-collect} == e2e_pair ]]; then
  for config in baseline fused4; do
    "$PY" "$SRC/bench/moe_large_batch_study.py" e2e --out "$OUT" --config "$config"
  done
elif [[ ${PHASE:-collect} == e2e_ab ]]; then
  # Two fresh engines on the same allocated GPU; the candidate .so stays isolated.
  : "${REFERENCE_SRC:?set REFERENCE_SRC for the comparison engine}"
  BRMOE_PKG=$REFERENCE_SRC/BR-MoE BRMOE_MOE_SMEM=legacy \
    "$PY" "$REFERENCE_SRC/bench/moe_large_batch_study.py" e2e --out "$OUT/reference" --config fused4
  BRMOE_MOE_SMEM=rightsize \
    "$PY" "$SRC/bench/moe_large_batch_study.py" e2e --out "$OUT/candidate" --config fused4 --cuda-cfg "${CANDIDATE_CFG:-128,128,3}"
elif [[ ${PHASE:-collect} == micro_ab ]]; then
  : "${REFERENCE_SRC:?set REFERENCE_SRC for the comparison process}"
  : "${INPUT:?set INPUT to captured real tensors}"
  BRMOE_PKG=$REFERENCE_SRC/BR-MoE BRMOE_MOE_SMEM=legacy \
    "$PY" "$REFERENCE_SRC/bench/moe_large_batch_study.py" micro --input "$INPUT" --out "$OUT/reference" --configs 128_128_4 --profile
  "$PY" "$SRC/bench/moe_large_batch_study.py" micro --input "$INPUT" --out "$OUT/candidate" --configs legacy_128_128_3,rightsize_128_128_3,rightsize_128_128_4 --profile
else
  "$PY" "$SRC/bench/moe_large_batch_study.py" "${PHASE}" --out "$OUT" ${ARGS:-}
fi
