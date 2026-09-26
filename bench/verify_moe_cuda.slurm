#!/bin/bash
#SBATCH -J moe-cuda
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_moe_cuda_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/verify_moe_cuda_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# int3 grouped MoE CUDA kernel 的数值验证 + 性能对比 (bench/verify_moe_cuda.py)
# 用法: sbatch -p 5090 bench/verify_moe_cuda.slurm   (或 -p a100)

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

LOG=${ROOT}/bench_results/verify_moe_cuda_${JID}.log
echo "日志: ${LOG}"
rc=0
# 先小形状冒烟 (E=8, 秒级), 过了再跑真实形状
if ! "${PY}" bench/verify_moe_cuda.py --small 2>&1 | tee "${LOG}"; then
  echo "!! 小形状冒烟失败, 中止"; exit 1
fi
if "${PY}" bench/verify_moe_cuda.py 2>&1 | tee -a "${LOG}"; then
  echo "-- 完成"
else
  echo "!! 真实形状失败"; rc=1
fi
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
