#!/bin/bash
#SBATCH -J moe-ref
# !! 必须用 /mnt/709/... 全路径 (计算节点上 /home/jianglei 是另一个本地 home) !!
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_ref_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/micro_moe_ref_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# MoE kernel 的 cuBLAS 参照 (--ref bmm) 验证。
#
# 目的: 实测 (A100, 2026-09-26) 三个 layout (int3/int4/fp16) 的 kernel 耗时几乎
# 相同, 字节数却差 4.57 倍 -> 既不是访存受限也不是解包受限, 上限是 slot padding
# 造成的算力浪费。这一步用 torch.bmm (cuBLAS, 无 padding) 给出"有用工作量"的
# 耗时基准, 用来判定「反超 fp16」是否有结构性空间。
#
# 用法:
#   sbatch -p a100 bench/micro_moe_ref.slurm
#   (可选) MS=1,8,32 LAYOUTS=int3,int4,fp16 sbatch -p a100 bench/micro_moe_ref.slurm

set -uo pipefail

ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1

# 与 vllm_perf_compare.slurm 相同: 显式钉死 CUDA 12.9 (triton JIT 要用)
CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
RES=${ROOT}/bench_results
cd "${ROOT}" || exit 1

JID=${SLURM_JOB_ID:-manual}
MS=${MS:-1,8,32}
LAYOUTS=${LAYOUTS:-int3,int4,fp16}
PEAK_GBS=${PEAK_GBS:-1935}     # A100 80GB PCIe

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ms=${MS}  layouts=${LAYOUTS}  peak_gbs=${PEAK_GBS}"
echo

rc=0

# ---- decode: M = 1, 8, 32 (反超空间最明确的场景) ----
LOG=${RES}/micro_moe_ref_decode_${JID}.log
echo "########## decode: --ms ${MS} --ref bmm ##########"
echo "日志: ${LOG}"
if "${PY}" bench/micro_moe.py \
      --ms "${MS}" --layouts "${LAYOUTS}" --ref bmm --quick \
      --peak-gbs "${PEAK_GBS}" 2>&1 | tee "${LOG}"; then
  echo "-- decode 完成"
else
  echo "!! decode 失败"
  rc=1
fi

# ---- prefill: M = 512, 2048 (大 M, 张量核心主导的场景) ----
LOG=${RES}/micro_moe_ref_prefill_${JID}.log
echo
echo "########## prefill: --ms 512,2048 --ref bmm ##########"
echo "日志: ${LOG}"
if "${PY}" bench/micro_moe.py \
      --ms 512,2048 --layouts "${LAYOUTS}" --ref bmm --quick \
      --peak-gbs "${PEAK_GBS}" 2>&1 | tee "${LOG}"; then
  echo "-- prefill 完成"
else
  echo "!! prefill 失败"
  rc=1
fi

echo
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
