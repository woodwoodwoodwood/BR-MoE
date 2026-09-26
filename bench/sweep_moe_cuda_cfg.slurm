#!/bin/bash
#SBATCH -J moe-cfg-sweep
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/sweep_moe_cfg_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/sweep_moe_cfg_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# MOE CUDA kernel tile/stages 扫描; 每配置独立子进程 (sm_120 防 illegal instruction 连坐)
# 用法: sbatch -p 5090 bench/sweep_moe_cuda_cfg.slurm  (或 -p a100)

set -uo pipefail
ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1
CUDA_HOME=/mnt/709/data3/home/jianglei/cuda-12.9   # 钉死: 登录节点泄漏的 CUDA_HOME (=/home/jianglei/...) 在计算节点不存在, 39044 因此没找到 nvcc
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE
cd "${ROOT}" || exit 1

# A100 且源码更新 -> 作业内重编 sm_80
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

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name --format=csv,noheader
for cfg in "128,128,4" "128,128,3" "128,128,5" "64,128,4" "128,64,4"; do
  "${PY}" bench/sweep_moe_cuda_cfg.py --cfg "${cfg}" || echo "cfg=${cfg} 进程失败(rc=$?)"
done
echo "=== DONE ==="
