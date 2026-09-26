#!/bin/bash
#SBATCH -J moe-zipf
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_zipf_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/perf_moe_zipf_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# 相关路由 (zipf) 下的路径重测: dispatch 自动口径 vs 纯 CUDA kernel
# 用法: sbatch -p 5090 bench/perf_moe_zipf.slurm  (或 -p a100)

set -uo pipefail
ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1
CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
cd "${ROOT}" || exit 1

# sm_80 扩展: A100 上且源码比产物新时作业内重编
KDIR=${ROOT}/BR-MoE/kernels/marlin_int3_moe
SO80=${KDIR}/brmoe_moe_int3_sm80.cpython-312-x86_64-linux-gnu.so
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100 && \
   [ "${KDIR}/int3_moe_kernel.cu" -nt "${SO80}" ]; then
  echo "[build] 重编 sm_80 ..."
  cd "${KDIR}"
  BRMOE_CUDA_ARCH=sm_80 BRMOE_ABI=1 "${PY}" setup_moe.py build_ext \
      --build-lib build_sm80_lib --build-temp build_sm80_tmp || exit 1
  cp "$(find build_sm80_lib -name 'brmoe_moe_int3*.so' | head -1)" "${SO80}"
  cd "${ROOT}"
fi

JID=${SLURM_JOB_ID:-manual}
echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# A100 峰值 1935 GB/s, 5090 1792 GB/s (测量口径沿用)
PEAK=1792
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100 && PEAK=1935

LOG=${ROOT}/bench_results/perf_moe_zipf_${JID}.log
rc=0
echo "===== [A] zipf 路由 + dispatch 自动 (gemv-max-m=None) =====" | tee "${LOG}"
"${PY}" bench/verify_moe_cuda.py --perf-only --routing zipf \
    --ms 1,2,4,8,16,32,64 --ksplits 1 --peak-gbs ${PEAK} 2>&1 | tee -a "${LOG}" || rc=1
echo "===== [B] zipf 路由 + 纯 CUDA kernel (gemv-max-m 0) =====" | tee -a "${LOG}"
"${PY}" bench/verify_moe_cuda.py --perf-only --routing zipf \
    --ms 1,2,4,8,16,32,64 --ksplits 1 --peak-gbs ${PEAK} --gemv-max-m 0 2>&1 | tee -a "${LOG}" || rc=1
echo "=== ${JID} DONE rc=${rc} ==="
exit ${rc}
