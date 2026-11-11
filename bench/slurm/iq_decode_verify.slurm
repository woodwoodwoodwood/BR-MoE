#!/bin/bash
#SBATCH --job-name=iq-gpu-verify
#SBATCH --partition=5090
#SBATCH --gres=gpu:5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_gpu_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_gpu_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=${RCO_PYTHON:-$BASE/miniconda3/envs/vllm5090/bin/python}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
export TRITON_CACHE_DIR=$REPO/artifacts/runs/triton_iq_cache
cd "$REPO"
hostname
"$PY" -m pytest -xq tests/test_iq_gpu.py tests/test_full_tensor_iq.py RCO_solver/verify_and_benchmark.py
"$PY" bench/iq_decode_bench.py --model /mnt/4090/data/jianglei/models/Qwen/qwen1.5-moe --out "$REPO/artifacts/runs/iq_gpu_${SLURM_JOB_ID}.json"
