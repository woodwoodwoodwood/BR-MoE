#!/bin/bash
#SBATCH --job-name=brmoe-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/eval_suite_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/eval_suite_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/vllm5090/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export BRMOE_QUANT_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
export TRITON_CACHE_DIR=$REPO/artifacts/runs/triton_iq_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
OUT=${SUITE_ROOT:?set SUITE_ROOT}/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
LOCALROOT=$(mktemp -d /tmp/jianglei_eval_${SLURM_JOB_ID}_XXXXXX)
trap 'rm -rf "$LOCALROOT"' EXIT
mkdir -p "$LOCALROOT/source"
ARCHIVE=$REPO/artifacts/runs/compressed_generate_source_20261026.tar
sha256sum "$ARCHIVE" > "$OUT/source_archive.sha256"
tar -xf "$ARCHIVE" -C "$LOCALROOT/source"
cp "${SUITE_EVAL_SCRIPT:?set frozen evaluator}" "$LOCALROOT/source/bench/eval_compressed_suite.py"
cp "$LOCALROOT/source/bench/eval_compressed_suite.py" "$OUT/eval_compressed_suite.py"
cd "$LOCALROOT/source"
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" bench/eval_compressed_suite.py \
  --checkpoint "$REPO/artifacts/runs/rco_e2e/run_55343/compressed" \
  --data "${SUITE_DATA:?set SUITE_DATA}" --out "$OUT" \
  --tasks "${SUITE_TASKS//+/,}" --batch-size "${SUITE_BATCH_SIZE:-4}" \
  --max-batch-tokens "${SUITE_MAX_BATCH_TOKENS:-4096}"
