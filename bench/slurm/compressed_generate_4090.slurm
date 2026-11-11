#!/bin/bash
#SBATCH --job-name=brmoe-4090-gen
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/compressed_gen_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/compressed_gen_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/vllm5090/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 BRMOE_QUANT_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
export TRITON_CACHE_DIR=$REPO/artifacts/runs/triton_iq_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
OUT=$REPO/artifacts/runs/compressed_generate/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
LOCALROOT=$(mktemp -d /tmp/jianglei_brmoe_gen_${SLURM_JOB_ID}_XXXXXX)
trap 'rm -rf "$LOCALROOT"' EXIT
mkdir -p "$LOCALROOT/source"
tar -xf "${SMOKE_SOURCE_ARCHIVE:?set a frozen source archive}" -C "$LOCALROOT/source"
printf '%s\n' "$SMOKE_SOURCE_ARCHIVE" > "$OUT/source_archive.txt"
cd "$LOCALROOT/source"
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" -m pytest -xq tests/test_iq_gpu.py::test_compressed_inference_gpu_decode tests/test_full_tensor_iq.py::test_real_hf_checkpoint_pipeline
"$PY" bench/compressed_generate_smoke.py \
 --model /mnt/4090/data/jianglei/models/Qwen/qwen1.5-moe \
 --allocation "$REPO/artifacts/runs/rco_e2e/run_55343/best_tensor_allocation.json" \
 --checkpoint "$REPO/artifacts/runs/rco_e2e/run_55343/compressed" --out "$OUT"
