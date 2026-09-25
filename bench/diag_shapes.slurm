#!/bin/bash
#SBATCH --job-name=brmoe-shapes
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/shapes_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/shapes_%j.err

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/milo/bin/python

WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${REPO}"
timeout 1500 ${PY} bench/diag_quant_shapes.py
echo "=== exit=$? ==="
