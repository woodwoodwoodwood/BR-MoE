#!/bin/bash
#SBATCH -J dbg-splitk
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_splitk_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/debug_splitk_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -uo pipefail
ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1
ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
cd "${ROOT}" || exit 1
echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# sm_80 扩展: A100 上且源码比产物新时在作业内重编
KDIR=${ROOT}/BR-MoE/kernels/marlin_int3_moe
SO80=${KDIR}/brmoe_moe_int3_sm80.cpython-312-x86_64-linux-gnu.so
SRC=${KDIR}/int3_moe_kernel.cu
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100 && \
   [ ! -f "${SO80}" ] || [ "${SRC}" -nt "${SO80}" ]; then
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100; then
    echo "[build] A100 且 sm_80 产物缺失/过期, 作业内编译 ..."
    export CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    cd "${KDIR}"
    BRMOE_CUDA_ARCH=sm_80 BRMOE_ABI=1 "${PY}" setup_moe.py build_ext \
        --build-lib build_sm80_lib --build-temp build_sm80_tmp || { echo "!! 编译失败"; exit 1; }
    cp "$(find build_sm80_lib -name 'brmoe_moe_int3*.so' | head -1)" "${SO80}"
    cd "${ROOT}"
    echo "[build] -> ${SO80}"
  fi
fi
rc=0
"${PY}" bench/debug_splitk.py || rc=1
"${PY}" bench/debug_splitk.py --n 256 || rc=1
"${PY}" bench/debug_splitk.py --e 2 || rc=1
echo "=== DONE rc=${rc} ==="
exit ${rc}
