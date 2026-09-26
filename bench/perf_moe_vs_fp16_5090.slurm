#!/bin/bash
#SBATCH -J moe-vs-fp16
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_vs_fp16_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_vs_fp16_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# 5090: kernel 级 int3 全路径 vs fp16 (同一条 grouped 流水线 + cuBLAS bmm 参照)
# 用法: sbatch -p 5090 bench/perf_moe_vs_fp16_5090.slurm

set -uo pipefail
PY=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090/bin/python
export PYTHONUNBUFFERED=1
ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
cd "${ROOT}" || exit 1

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
rc=0
for RT in zipf random; do
  echo "===== routing=${RT} ====="
  "${PY}" bench/verify_moe_cuda.py --perf-only --routing "${RT}" \
      --ms 1,2,4,8,16,32,64 --ksplits 1 --fp16-grouped 2>&1 || rc=1
done
echo "=== DONE rc=${rc} ==="
exit ${rc}
