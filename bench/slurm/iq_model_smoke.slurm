#!/bin/bash
#SBATCH --job-name=brmoe-iq-smoke
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_smoke_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_smoke_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/auto-quarot/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
OUT=$REPO/artifacts/runs/iq_smoke/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
# Freeze source so later workspace edits cannot change this queued run.
SRC=$OUT/source
mkdir -p "$SRC"
tar -xf "$REPO/artifacts/runs/iq_smoke_source.tar" -C "$SRC"
cd "$SRC"
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" -m pytest -q tests/test_full_tensor_iq.py RCO_solver/verify_and_benchmark.py
"$PY" bench/iq_model_smoke.py --model /mnt/4090/data/jianglei/models/Qwen/qwen1.5-moe --out "$OUT"
