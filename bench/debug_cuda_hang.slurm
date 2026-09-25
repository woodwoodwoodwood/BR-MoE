#!/bin/bash
#SBATCH --job-name=brmoe-debug
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_%j.err
#
# 定位 brmoe int3 CUDA kernel 在 M>1 时挂起的位置
#   sbatch bench/debug_cuda_hang.slurm

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/milo/bin/python

WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY} -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

cd "${REPO}"
timeout 240 ${PY} bench/debug_cuda_hang.py
rc=$?
echo ""
echo "=== exit=${rc} (124 = timeout, 说明在某一步挂住) ==="
