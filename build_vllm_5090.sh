#!/bin/bash
# Build vLLM for RTX 5090 (SM 12.0) on ada-5090
# Submit with: sbatch build_vllm_5090.slurm

set -euo pipefail

export CUDA_HOME=/home/jianglei/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=32
export NVCC_THREADS=8
export VLLM_TARGET_DEVICE=cuda

PYTHON=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090/bin/python
PIP=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090/bin/pip

cd /home/jianglei/framework/vllm

echo "=== Environment ==="
nvcc --version | tail -1
$PYTHON -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}')"
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "=== Building vLLM ==="
$PIP install --no-cache-dir --no-build-isolation -e . 2>&1

echo "=== Verify ==="
$PYTHON -c "
import vllm
print(f'vLLM {vllm.__version__} built successfully')
"

