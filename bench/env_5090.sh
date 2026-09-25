#!/bin/bash
# 用法:  source bench/env_5090.sh
#
# 在 5090 节点上跑 vLLM 前**必须**先 source 这个, 否则会撞两个坑:
#
#   1) FlashInfer: RuntimeError: FlashInfer requires GPUs with sm75 or higher
#      原因不是 GPU (5090 明明是 sm_120), 而是 flashinfer 的 JIT
#      check_cuda_arch() 需要 CUDA 工具链来判断目标架构。缺 CUDA_HOME /
#      PATH 就会一律报这个错。VLLM_USE_FLASHINFER_SAMPLER=0 是第二道保险,
#      直接改用 PyTorch 原生采样器, 彻底绕开该 JIT 依赖 (对延迟测试无影响,
#      采样不是瓶颈)。
#
#   2) Triton / torch JIT 找不到 nvcc 或链接不到 CUDA 运行库。
#
# 注意: 用 source 而不是 bash —— 环境变量要留在当前 shell 里。

export BR_BASE=/mnt/709/data3/home/jianglei
export BR_REPO=${BR_BASE}/ada/BR-MoE
export BR_PY=${BR_BASE}/miniconda3/envs/vllm5090/bin/python
export BR_MODEL_INT3=${BR_MODEL_INT3:-${BR_BASE}/models/brmoe-3bit-vllm}
export BR_MODEL_INT3DENSE=${BR_MODEL_INT3DENSE:-${BR_BASE}/models/brmoe-3bit-vllm-int3dense}

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 单卡独占分配下 OMP 线程开多了反而互相抢占
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

export CUDA_HOME=${BR_BASE}/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}

# 绕开 flashinfer 采样的 JIT (见上)
export VLLM_USE_FLASHINFER_SAMPLER=0

# 插件的调试输出 (topk_ids shape / align 回退等)
export BRMOE_DEBUG=${BRMOE_DEBUG:-1}

echo "[env_5090] base=${BR_BASE}"
echo "[env_5090] python=${BR_PY}"
echo "[env_5090] CUDA_HOME=${CUDA_HOME}"
echo "[env_5090] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "[env_5090] CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置! 这个分配可能没拿到 GPU>}'"
