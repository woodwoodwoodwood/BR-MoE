#!/bin/bash
#SBATCH -J moe-tc5090
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_tc5090_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_tc5090_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# 5090 上 bs=16/32 的 TC (grouped GEMM) 路径扫描。
#
# 背景: GEMV 阈值扩展在 e2e 相关路由下失败 (38946: bs=16 回退 7.30->7.95ms),
# 所以 bs>=16 仍走 TC。而 TC 的 slot16/block_m16 自动配置是 A100 上标的
# (ops.py 233-239), 5090 的最优点可能不同 —— 这个扫描就是来回答这件事的。
#
# 用法: sbatch -p 5090 bench/micro_moe_tc5090.slurm

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
MS=${MS:-16,32}

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
echo "ms=${MS}  (TC 路径全阶段扫描, 5090 峰值 1792 GB/s)"

LOG=${RES}/micro_moe_tc5090_${JID}.log
echo "日志: ${LOG}"
if "${PY}" bench/micro_moe.py \
      --ms "${MS}" --layouts int3,int4d --check --quick \
      --peak-gbs 1792 2>&1 | tee "${LOG}"; then
  echo "-- 完成"; rc=0
else
  echo "!! 失败"; rc=1
fi
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
