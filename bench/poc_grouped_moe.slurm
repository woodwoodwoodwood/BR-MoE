#!/bin/bash
#SBATCH --job-name=brmoe-poc
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/poc_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/poc_%j.err
#
# POC: grouped GEMM (一个 launch 算完所有专家) vs per-expert 循环
#   sbatch -p a100 bench/poc_grouped_moe.slurm
#   sbatch -p a100 bench/poc_grouped_moe.slurm --ms 1,16 --skip-brmoe

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
ENV_PREFIX=${BASE}/miniconda3/envs/milo
PY=${ENV_PREFIX}/bin/python

WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY} -c "import torch; print('torch', torch.__version__)"
${PY} -c "import triton; print('triton', triton.__version__)"

cd "${REPO}"
timeout 3000 ${PY} bench/poc_grouped_moe.py "$@"
rc=$?
echo "=== exit=${rc} (124=timeout) ==="
