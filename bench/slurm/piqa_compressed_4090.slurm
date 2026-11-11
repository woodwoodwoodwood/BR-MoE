#!/bin/bash
#SBATCH --job-name=brmoe-piqa
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/piqa_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/piqa_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/vllm5090/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 BRMOE_QUANT_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
export TRITON_CACHE_DIR=$REPO/artifacts/runs/triton_iq_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
OUT=$REPO/artifacts/runs/piqa/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
LOCALROOT=$(mktemp -d /tmp/jianglei_piqa_${SLURM_JOB_ID}_XXXXXX)
trap 'rm -rf "$LOCALROOT"' EXIT
mkdir -p "$LOCALROOT/source"
# Reuse the frozen model implementation from the successful generation run.
ARCHIVE=$REPO/artifacts/runs/compressed_generate_source_20261026.tar
sha256sum "$ARCHIVE" > "$OUT/source_archive.sha256"
tar -xf "$ARCHIVE" -C "$LOCALROOT/source"
cp "${PIQA_EVAL_SCRIPT:?set a frozen evaluation script}" "$LOCALROOT/source/bench/eval_compressed_piqa.py"
cp "$LOCALROOT/source/bench/eval_compressed_piqa.py" "$OUT/eval_compressed_piqa.py"
cd "$LOCALROOT/source"
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" bench/eval_compressed_piqa.py \
  --checkpoint "$REPO/artifacts/runs/rco_e2e/run_55343/compressed" \
  --dataset "${PIQA_DATASET:?set the exported PIQA validation JSONL}" \
  --out "$OUT"
