#!/bin/bash
#SBATCH -J dbg-probe
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_kernel_probe_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_kernel_probe_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
set -uo pipefail
PY=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090/bin/python
export PYTHONUNBUFFERED=1
cd /mnt/709/data3/home/jianglei/ada/BR-MoE || exit 1
echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
"${PY}" bench/debug_kernel_probe.py
echo "=== DONE rc=$? ==="
