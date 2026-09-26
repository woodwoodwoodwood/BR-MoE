#!/bin/bash
#SBATCH -J moe-cuda-a100
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_cuda_a100_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_cuda_a100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# A100 (sm_80): MoE CUDA kernel vs fp16 (cuBLAS-ref) 性能对比。
# sm_80 扩展在作业内编译 (build_ext, 不覆盖 sm_120 的默认 .so), 产物带 _sm80 后缀,
# verify_moe_cuda.py / 插件会按 GPU 架构自动选库。
# 数值未修复 -> --perf-only 只计时。
# 用法: sbatch -p a100 bench/perf_moe_cuda_a100.slurm

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

# ---- 1) 编译 sm_80 扩展 (幂等: 产物已存在则跳过) ----
KDIR=${ROOT}/BR-MoE/kernels/marlin_int3_moe
SO=${KDIR}/brmoe_moe_int3_sm80.cpython-312-x86_64-linux-gnu.so
if [ ! -f "${SO}" ] || [ "${KDIR}/int3_moe_kernel.cu" -nt "${SO}" ]; then
  echo "[build] sm_80 扩展不存在, 开始编译 ..."
  cd "${KDIR}"
  BRMOE_CUDA_ARCH=sm_80 BRMOE_ABI=1 "${PY}" setup_moe.py build_ext \
      --build-lib build_sm80_lib --build-temp build_sm80_tmp || {
    echo "!! sm_80 编译失败"; exit 1
  }
  BUILT=$(find build_sm80_lib -name "brmoe_moe_int3*.so" | head -1)
  [ -n "${BUILT}" ] || { echo "!! 找不到编译产物"; exit 1; }
  cp "${BUILT}" "${SO}"
  rm -rf build_sm80_lib build_sm80_tmp
  cd "${ROOT}"
  cuobjdump --list-elf "${SO}" | grep -q sm_80 || { echo "!! 产物不含 sm_80"; exit 1; }
  echo "[build] -> ${SO} (sm_80 已确认)"
else
  echo "[build] 命中已有 sm_80 扩展: ${SO}"
fi

LOG=${ROOT}/bench_results/perf_moe_cuda_a100_${JID}.log
echo "日志: ${LOG}"
rc=0

# ---- 2) 生产 dispatch 口径 (M<=8 委托 GEMV) vs Triton vs fp16 cuBLAS ----
echo "===== [A] dispatch 口径 (M<=8 走 GEMV) =====" | tee "${LOG}"
"${PY}" bench/verify_moe_cuda.py --perf-only --ms 1,2,4,8,16,32,64 \
    --ksplits 1 --peak-gbs 1935 2>&1 | tee -a "${LOG}" || rc=1

# ---- 3) 纯 CUDA kernel 口径 (不委托 GEMV) ----
echo "===== [B] 纯 CUDA kernel (gemv-max-m 0) =====" | tee -a "${LOG}"
"${PY}" bench/verify_moe_cuda.py --perf-only --ms 1,2,4,8,16,32,64 \
    --ksplits 1 --peak-gbs 1935 --gemv-max-m 0 2>&1 | tee -a "${LOG}" || rc=1

echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
