#!/bin/bash
#SBATCH --job-name=vllm5090-build
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/framework/vllm_%j.log
#SBATCH --error=/mnt/709/data3/home/jianglei/framework/vllm_%j.err

set -euo pipefail

export CUDA_HOME=/mnt/709/data3/home/jianglei/cuda-12.9
ENV_PREFIX=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
export PATH=$ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16
export NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export VLLM_TARGET_DEVICE=cuda
# Let setuptools_scm derive the development version from the current main commit.
export PYTHONUNBUFFERED=1

EXT=/mnt/709/data3/home/jianglei/ext-deps
export DEEPGEMM_SRC_DIR=$EXT/DeepGEMM
export FLASH_KDA_SRC_DIR=$EXT/FlashKDA
export FLASH_MLA_SRC_DIR=$EXT/FlashMLA
export FMHA_SM100_SRC_DIR=$EXT/fmha_sm100
export QUTLASS_SRC_DIR=$EXT/qutlass
export VLLM_CUTLASS_SRC_DIR=$EXT/cutlass
export TML_FA4_SRC_DIR=$EXT/tml-fa4
export VLLM_FLASH_ATTN_SRC_DIR=$EXT/vllm-flash-attn-src
export TRITON_KERNELS_SRC_DIR=$EXT/triton-kernels-src/python/triton_kernels/triton_kernels

PYTHON=$ENV_PREFIX/bin/python
PIP=$ENV_PREFIX/bin/pip

cd /mnt/709/data3/home/jianglei/framework/vllm

export VLLM_VERSION_OVERRIDE="0.29.0.dev0+g$(git rev-parse --short=10 HEAD)"

echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
nvcc --version | tail -1

echo "=== Building vLLM ==="
$PIP install --no-cache-dir --no-build-isolation -e . 2>&1

echo "=== Verify ==="
$PYTHON -c "
import vllm
print(f'vLLM {vllm.__version__} built successfully')
"
echo "=== BUILD COMPLETE ==="
