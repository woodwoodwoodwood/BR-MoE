#!/bin/bash
#SBATCH -J lin-gemv
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_linear_gemv_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_linear_gemv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# linear GEMV+split-K 分支的数值验证 + 测速 (bench/verify_linear_gemv.py)
# 用法: sbatch -p a100 bench/verify_linear_gemv.slurm

set -uo pipefail

ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1

CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
cd "${ROOT}" || exit 1

JID=${SLURM_JOB_ID:-manual}
echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader

LOG=${ROOT}/bench_results/verify_linear_gemv_${JID}.log
echo "日志: ${LOG}"
if "${PY}" bench/verify_linear_gemv.py 2>&1 | tee "${LOG}"; then
  echo "-- 完成"; rc=0
else
  echo "!! 失败"; rc=1
fi
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
