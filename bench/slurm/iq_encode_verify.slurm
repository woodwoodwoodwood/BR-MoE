#!/bin/bash
#SBATCH --job-name=iq-encode-verify
#SBATCH --partition=prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_encode_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/iq_encode_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/auto-quarot/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
cd "$REPO"
"$PY" -m pytest -xq tests/test_iq_gpu.py -k 'parallel_native or device_cache_bound'
"$PY" bench/iq_encode_bench.py "$REPO/artifacts/runs/iq_encode_${SLURM_JOB_ID}.json"
