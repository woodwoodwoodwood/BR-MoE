#!/bin/bash
#SBATCH --job-name=brmoe-vllm-e2e
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_e2e_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_e2e_%j.err
#
# 验证 BR-MoE 的 3bit 权重在 vLLM 里跑通, 并与 BR-MoE 原生实现逐 token 对拍。
#
#   sbatch -p a100 bench/vllm_e2e.slurm
#
# 三段:
#   A. BR-MoE 原生 (PyTorch 后端)  —— 参考答案
#   B. vLLM + brmoe_int3 插件, --enforce-eager   —— 先求正确
#   C. vLLM + brmoe_int3 插件, CUDA Graph        —— 再求可用
#
# 依赖: vllm5090 环境里已 pip install -e tools/brmoe_int3_vllm

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
RESULT_DIR=${REPO}/bench_results
MODEL_3BIT=/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0
MODEL_VLLM=${BASE}/models/brmoe-3bit-vllm
PY_MILO=${BASE}/miniconda3/envs/milo/bin/python
PY_VLLM=${BASE}/miniconda3/envs/vllm5090/bin/python

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# vllm5090 环境自带的 CUDA 12.9 运行时
export CUDA_HOME=${BASE}/cuda-12.9
if [[ -d "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "=== node $(hostname) / $(date +%H:%M:%S) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "--- vllm5090 环境自检 ---"
${PY_VLLM} - <<'PY' 2>&1 | tail -12
import sys
try:
    import torch
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
    print("device_capability", torch.cuda.get_device_capability())
except Exception as e:
    print("torch 自检失败:", type(e).__name__, e); sys.exit(1)
try:
    import vllm
    print("vllm", vllm.__version__)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print("brmoe_int3 已注册:", "brmoe_int3" in QUANTIZATION_METHODS)
except Exception as e:
    print("vllm 自检失败:", type(e).__name__, e)
PY

mkdir -p "${RESULT_DIR}"

# ---------------------------------------------------------------- A. 参考实现
# 参考答案是确定性的 (贪心解码), 所以跑过一次就缓存到 ref_latest.json,
# 后续迭代直接复用, 省掉约 2 分钟的模型加载。
REF_STABLE="${RESULT_DIR}/ref_latest.json"
if [[ -s "${REF_STABLE}" ]]; then
  echo ""
  echo "############ A. 复用已有参考 ${REF_STABLE} ############"
  cp "${REF_STABLE}" "${RESULT_DIR}/ref_${SLURM_JOB_ID}.json"
else
  WORK=$(mktemp -d); ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
  echo ""
  echo "############ A. BR-MoE 原生 (PyTorch 后端) ############"
  PYTHONPATH="${WORK}:${PYTHONPATH:-}" timeout 2400 ${PY_MILO} \
      "${REPO}/bench/brmoe_ref_check.py" \
      --model-path "${MODEL_3BIT}" \
      --out "${RESULT_DIR}/ref_${SLURM_JOB_ID}.json" \
    || echo "!! A 失败"
  [[ -s "${RESULT_DIR}/ref_${SLURM_JOB_ID}.json" ]] \
    && cp "${RESULT_DIR}/ref_${SLURM_JOB_ID}.json" "${REF_STABLE}"
fi

# ---------------------------------------------------------------- B. vLLM eager
echo ""
echo "############ B. vLLM + brmoe_int3 (--enforce-eager) ############"
timeout 2400 ${PY_VLLM} "${REPO}/bench/vllm_plugin_check.py" \
    --model "${MODEL_VLLM}" \
    --out "${RESULT_DIR}/vllm_eager_${SLURM_JOB_ID}.json" \
    --enforce-eager 1 \
  || echo "!! B 失败"

# ---------------------------------------------------------------- C. vLLM graph
echo ""
echo "############ C. vLLM + brmoe_int3 (CUDA Graph) ############"
timeout 2400 ${PY_VLLM} "${REPO}/bench/vllm_plugin_check.py" \
    --model "${MODEL_VLLM}" \
    --out "${RESULT_DIR}/vllm_graph_${SLURM_JOB_ID}.json" \
    --enforce-eager 0 \
  || echo "!! C 失败"

# ---------------------------------------------------------------- 对拍
echo ""
echo "############ 对拍 ############"
${PY_VLLM} - "${RESULT_DIR}" "${SLURM_JOB_ID}" <<'PY'
import json, os, sys
rd, jid = sys.argv[1], sys.argv[2]

def load(name):
    p = os.path.join(rd, f"{name}_{jid}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

ref = load("ref")
eager = load("vllm_eager")
graph = load("vllm_graph")

for tag, r in (("A BR-MoE 原生", ref), ("B vLLM eager", eager),
               ("C vLLM graph", graph)):
    if r is None:
        print(f"{tag:<16} : 未产出")
        continue
    print(f"{tag:<16} : {r['token_ids']}")
    print(f"{'':<16}   text={r['text']!r}")

if ref is None:
    print("\n没有参考答案, 无法对拍")
    raise SystemExit(0)

rt = ref["token_ids"]
for tag, r in (("B vLLM eager", eager), ("C vLLM graph", graph)):
    if r is None:
        continue
    vt = r["token_ids"]
    n = min(len(rt), len(vt))
    same = sum(1 for i in range(n) if rt[i] == vt[i])
    exact = (rt == vt)
    print(f"\n{tag}: 逐位一致={exact}  前 {n} 个 token 命中 {same}/{n}")
    if not exact:
        for i in range(n):
            if rt[i] != vt[i]:
                print(f"  首个分歧 @token {i}: 参考={rt[i]} vLLM={vt[i]}"
                      f"  (参考上下文={rt[max(0,i-3):i+3]}"
                      f" vLLM上下文={vt[max(0,i-3):i+3]})")
                break
PY

echo ""
echo "=== DONE $(date +%H:%M:%S) ==="
