#!/bin/bash
#SBATCH -J moe-cuda-smallm
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_cuda_smallm_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_cuda_smallm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# 小 M 补测: MoE CUDA kernel (多级流水) vs Triton 自动路径 (5090 上 M<=8 自动走 GEMV)
# 数值未修复 (verify_moe_cuda 未过), 所以用 --perf-only 只计时。
# 用法: sbatch -p 5090 bench/perf_moe_cuda_smallm.slurm

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

LOG=${ROOT}/bench_results/perf_moe_cuda_smallm_${JID}.log
echo "日志: ${LOG}"
rc=0
# 真实形状 E=64 K=2048 I=1408; M 扫描覆盖 GEMV 区间 (1-8) 与 TC 区间 (16-64)。
# 注意: "Triton-TC" 列在 M<=8 时实际是分派到的 GEMV 路径 (fused_moe_int3 自动选择)。
if "${PY}" bench/verify_moe_cuda.py --perf-only --ms 1,2,4,8,16,32,64 --ksplits 1,2,4,8 2>&1 | tee "${LOG}"; then
  echo "-- 完成"
else
  echo "!! 失败"; rc=1
fi
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
