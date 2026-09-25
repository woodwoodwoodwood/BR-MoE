#!/bin/bash
#SBATCH --job-name=brmoe-vfy
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_%j.err
#
# 验证 brmoe_grouped 后端
#   sbatch -p a100 bench/verify_grouped.slurm --limit 2     # 快速
#   sbatch -p a100 bench/verify_grouped.slurm --limit 0     # 全部 27 层

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
ENV_PREFIX=${BASE}/miniconda3/envs/milo
PY=${ENV_PREFIX}/bin/python

WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY} -c "import torch; print('torch', torch.__version__)"

cd "${REPO}"
timeout 2400 ${PY} bench/verify_grouped_backend.py "$@"
echo "=== exit=$? (124=timeout) ==="
