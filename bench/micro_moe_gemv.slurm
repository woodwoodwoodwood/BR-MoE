#!/bin/bash
#SBATCH -J moe-gemv
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_gemv_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_gemv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# GEMV 旋钮扫描: block_n x groups x num_warps x num_stages, 两种打包布局。
#
# 背景: GEMV 在 M=1 是 88 us @ 257 GB/s (13% 峰值), 离同效率目标 21.8 us 差 4 倍。
# 特征: grid 仅 ~528 program (<5/SM), 每 program 16 次串行 k 迭代, stages=1 无流水
# -> 延迟受限。ops.py 的 GEMV 分支四个旋钮全用默认值, 从未调过。
#
# 用法: sbatch -p a100 bench/micro_moe_gemv.slurm

set -uo pipefail

ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1

CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
RES=${ROOT}/bench_results
cd "${ROOT}" || exit 1

JID=${SLURM_JOB_ID:-manual}
MS=${MS:-1,2,4,8}

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ms=${MS}"
echo

LOG=${RES}/micro_moe_gemv_${JID}.log
echo "日志: ${LOG}"
if "${PY}" bench/micro_moe.py \
      --ms "${MS}" --layouts int3 --gemv-sweep --check --stage none \
      --peak-gbs 1935 2>&1 | tee "${LOG}"; then
  echo "-- 完成"
  rc=0
else
  echo "!! 失败"
  rc=1
fi

echo
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
