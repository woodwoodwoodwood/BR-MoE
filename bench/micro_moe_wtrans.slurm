#!/bin/bash
#SBATCH -J moe-wtrans
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_wtrans_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_wtrans_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# GEMV × 权重布局 (w_transposed) 验证。
#
# 假设: _routed_int3_gemv 的 w 载入形状是 [GROUPS, BLOCK_N]。N-major (现况) 下
# 相邻 lane 间隔 Kpack*4=768B -> 每 4B 拉一个 32B sector -> 最多 8x 带宽浪费
# (实测 GEMV 只有 257 GB/s = 13% 峰值, 8x 放大后约 2 TB/s, 正好打满 —— 吻合)。
# K-major (transposed=True) 下 n 连续 -> 128B 合并, 预期接近带宽下界。
# 历史上"布局中性 (0.93~1.01x)"的结论只在 TC 路径测过, 对 GEMV 不成立。
#
# 用法: sbatch -p a100 bench/micro_moe_wtrans.slurm

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
echo "ms=${MS}  (GEMV 生产范围 M<=4, M=8 看外延)"
echo

LOG=${RES}/micro_moe_wtrans_${JID}.log
echo "日志: ${LOG}"
# --stage none: 跳过已知道答案的 A/B/C 参数扫描, 只跑基线 + 自动模式 + wtrans 对照
# --check: 每种布局的 GEMV 都和金标准 (反量化 + cuBLAS bmm) 对数值
if "${PY}" bench/micro_moe.py \
      --ms "${MS}" --layouts int3 --wtrans --check --stage none \
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
