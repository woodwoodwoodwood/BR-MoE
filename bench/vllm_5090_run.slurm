#!/bin/bash
#SBATCH --job-name=brmoe-5090
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_5090_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_5090_%j.err
#
# 在 5090 (sm_120) 上跑 vLLM:
#   A) BR-MoE 3bit + brmoe_int3 插件 (--enforce-eager)
#   B) 同上但开 CUDA Graph
#   C) fp16 基线 (无量化)
#
# 为什么只能在 5090 上这样测:
#   milo 环境的 torch 2.5.1+cu121 没有 sm_120 内核, 所以 BR-MoE 原生实现跑不了 5090;
#   而 vllm5090 环境的 vLLM 就是为 sm_120 编的, 正好可用 (A100 那份要等 sm_80 重编)。
#
#   sbatch -p 5090 bench/vllm_5090_run.slurm

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
RESULT_DIR=${REPO}/bench_results
MODEL_VLLM=${BASE}/models/brmoe-3bit-vllm
MODEL_FP16=/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-base
PY_VLLM=${BASE}/miniconda3/envs/vllm5090/bin/python

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# 打印一次 topk_ids 的 shape/dtype/min/max, 便于定位 align 的入参契约
export BRMOE_DEBUG=1
export CUDA_HOME=${BASE}/cuda-12.9
if [[ -d "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "=== node $(hostname) / $(date +%H:%M:%S) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo "--- 环境自检 ---"
${PY_VLLM} - <<'PY' 2>&1 | tail -8
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("capability", torch.cuda.get_device_capability(), "|", torch.cuda.get_device_name(0))
import vllm
print("vllm", vllm.__version__)
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print("brmoe_int3 registered:", "brmoe_int3" in QUANTIZATION_METHODS)
PY

run_case () {   # tag model extra_args...
  local tag=$1 model=$2; shift 2
  local out="${RESULT_DIR}/v5090_${tag}_${SLURM_JOB_ID}.json"
  echo ""
  echo "################ CASE ${tag} ################"
  if [[ ! -e "${model}" ]]; then
    echo "!! 模型不存在: ${model} -> skip"
    return 0
  fi
  timeout 2400 ${PY_VLLM} "${REPO}/bench/vllm_plugin_check.py" \
      --model "${model}" --out "${out}" --tag "${tag}" "$@" \
    || echo "!! ${tag} 失败"
  [[ -s "${out}" ]] && ${PY_VLLM} -c "
import json;d=json.load(open('${out}'))
print('  text:', repr(d['text']))
print('  tokens:', d['token_ids'])
print('  gpu:', d['gpu'], '| mem used %.2f / %.2f GiB' % (d['memory'].get('used_gib',0), d['memory'].get('total_gib',0)))
" 2>/dev/null || true
}

# --- A/B: BR-MoE 3bit (8.75 GiB, 32 GB 卡上很宽裕) ---
run_case brmoe3bit_eager "${MODEL_VLLM}" --quantization brmoe_int3 --enforce-eager 1 --gpu-mem-util 0.75
run_case brmoe3bit_graph "${MODEL_VLLM}" --quantization brmoe_int3 --enforce-eager 0 --gpu-mem-util 0.75

# --- C: fp16 基线 ---
# 32 GB 的 5090 上必然失败: 权重 30.5 GiB, 而 vLLM 在
# process_weights_after_loading 里还要为 w13 再开一份 0.7 GiB 的交换副本,
# 合计 31.2 GiB + KV/工作区 > 31.4 GiB -> OOM (已实测)。
# 这个对照留给 A100 (等 sm_80 重编完成后)。
echo ""
echo "############ C. fp16 跳过 (5090 32GB 放不下 30.5 GiB 权重) ############"

echo ""
echo "############ 与 A100 参考的软对比 ############"
${PY_VLLM} - "${RESULT_DIR}" "${SLURM_JOB_ID}" <<'PY'
import json, os, sys
rd, jid = sys.argv[1], sys.argv[2]
ref_p = os.path.join(rd, "ref_latest.json")
ref = json.load(open(ref_p)) if os.path.exists(ref_p) else None
if ref:
    print("A100 BR-MoE 原生参考:", ref["token_ids"])

for tag in ("brmoe3bit_eager", "brmoe3bit_graph", "fp16"):
    p = os.path.join(rd, f"v5090_{tag}_{jid}.json")
    if not os.path.exists(p):
        print(f"{tag:<20}: 未产出")
        continue
    d = json.load(open(p))
    line = f"{tag:<20}: {d['token_ids']}"
    if ref:
        rt, vt = ref["token_ids"], d["token_ids"]
        n = min(len(rt), len(vt))
        hit = sum(1 for i in range(n) if rt[i] == vt[i])
        first = next((i for i in range(n) if rt[i] != vt[i]), None)
        line += f"   命中 {hit}/{n}" + (f" (首个分歧 @{first})" if first is not None else " 完全一致")
    print(line)
PY

echo ""
echo "=== DONE $(date +%H:%M:%S) ==="
