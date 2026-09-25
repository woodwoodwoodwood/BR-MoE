#!/bin/bash
# 在 5090 节点上跑 int3 插件的**正确性**校验, 并自动与 A100 的 BR-MoE 参考对拍。
#
# 为什么单独成脚本 (两边都踩过的坑, 在此固化):
#
# 1) CUDA_HOME 必须指向 12.9 —— 否则会得到一句完全误导人的报错。
#    flashinfer/jit/cpp_ext.py:49 的 get_cuda_home() 优先读 CUDA_HOME, 没有就
#    `which nvcc` -> /usr/local/cuda, 而这里是 **CUDA 12.4**。
#    而 compilation_context.py:74 对 major==12 要求 >= 12.9:
#        raise RuntimeError("SM 12.x requires CUDA >= 12.9")
#    这个异常被同文件 :107 的 `except Exception` 吞掉, TARGET_CUDA_ARCHS 留空,
#    最终 check_cuda_arch() 报出
#        "FlashInfer requires GPUs with sm75 or higher"
#    —— 和真实原因毫无关系 (5090 明明是 sm_120)。
#
# 2) tokenizer 必须走 HF fast 路径 (config.json/tokenizer.json 已修过), 否则
#    慢速 LlamaTokenizer 会吃掉空格并丢 BOS, 输出直接变乱码。
#
# 用法 (必须在 5090 节点的分配里跑):
#   bash bench/check_5090.sh                # eager + graph 都校验 (默认)
#   bash bench/check_5090.sh eager
#   bash bench/check_5090.sh graph
#   MODEL=/path/to/other bash bench/check_5090.sh
#
# 参考 token 来自 A100 上 BR-MoE 的原生 PyTorch 路径 (同一个 prompt、贪心解码)。

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/vllm5090/bin/python
CUDA=${BASE}/cuda-12.9                       # 必须 12.9, 见抬头注释
MODEL=${MODEL:-${BASE}/models/brmoe-3bit-vllm-int3dense}
RES=${REPO}/bench_results
LOGDIR=${RES}/logs

MODE=${1:-both}                              # eager | graph | both
# tag 前缀跟模型目录名走, 免得 int3dense 的结果覆盖 brmoe3bit 的
_m=$(basename "${MODEL}")
TAGPFX=${TAGPFX:-${_m#brmoe-3bit-vllm}}
TAGPFX=${TAGPFX#-}
if [ -z "${TAGPFX}" ]; then TAGPFX=brmoe3bit; fi

PROMPT=${PROMPT:-"The capital of France is"}
MAXTOK=${MAXTOK:-16}
GPU_MEM=${GPU_MEM:-0.85}
MAX_BATCHED=${MAX_BATCHED:-4096}

# ---- 环境 (顺序重要: CUDA_HOME 要在 python 起来之前设好) ----
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_HOME=${CUDA}
export PATH=${CUDA}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA}/lib64:${CUDA}/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${LOGDIR}"
STAMP=$(date +%m%d_%H%M%S)

# ---- 先自检 CUDA_HOME 真的生效 (否则直接给出可执行的修法) ----
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "!! CUDA_HOME=${CUDA_HOME} 下没有 nvcc。"
  exit 1
fi
NVCC_VER=$("${CUDA_HOME}/bin/nvcc" --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)
echo "=== check on $(hostname) / $(date '+%F %T') ==="
echo "MODEL      = ${MODEL}"
echo "CUDA_HOME  = ${CUDA_HOME}  (${NVCC_VER})"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置>}'"
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "!! CUDA_VISIBLE_DEVICES 为空 —— 这个分配没拿到 GPU, 请先:"
  echo "   srun -p 5090 -N1 -n1 --mem=48G --gres=gpu:1 -t 2:00:00 --pty bash"
  exit 1
fi

run_one () {
  local tag=$1 eager=$2
  local out=${RES}/chk_${tag}.json
  local log=${LOGDIR}/chk_${tag}_${STAMP}.log
  echo "################ ${tag}  (enforce_eager=${eager}) ################"
  echo "日志: ${log}"
  "${PY}" "${REPO}/bench/vllm_plugin_check.py" \
      --model "${MODEL}" \
      --enforce-eager "${eager}" \
      --prompt "${PROMPT}" --max-tokens "${MAXTOK}" \
      --max-model-len 512 \
      --max-num-batched-tokens "${MAX_BATCHED}" \
      --quantization brmoe_int3 \
      --tokenizer-mode hf \
      --gpu-mem-util "${GPU_MEM}" \
      --out "${out}" --tag "${tag}" 2>&1 | tee "${log}" | tail -25
  echo "-- ${tag} 结束 (exit=${PIPESTATUS[0]})"
  echo
  echo "---- 与 A100 参考对拍 ----"
  "${PY}" - "${out}" <<'PYEOF'
import json, sys
REF = [8913, 11, 588, 317, 6286, 331, 254, 8144, 95965, 13, 429, 3787, 317, 13429, 881, 207]
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"   !! 读不到结果文件: {e}")
    sys.exit(1)
got = d.get("token_ids")
print(f"   got : {got}")
print(f"   ref : {REF}")
if got == REF:
    print("   ✅ 16/16 完全一致")
else:
    same = sum(1 for a, b in zip(got or [], REF) if a == b)
    print(f"   ❌ 不一致, 命中 {same}/{len(REF)}")
    if got is None:
        print("      (token_ids 缺失 —— 可能是加载期就报错了)")
PYEOF
  echo "   text: $(grep -oE '\"text\": \"[^\"]*\"' "${out}" 2>/dev/null | head -1)"
  echo
}

rc=0
case "${MODE}" in
  eager) run_one "${TAGPFX}_eager" 1 ;;
  graph) run_one "${TAGPFX}_graph" 0 ;;
  both)
    run_one "${TAGPFX}_graph" 0 || rc=1
    run_one "${TAGPFX}_eager" 1 || rc=1
    ;;
  *) echo "!! 未知 MODE='${MODE}', 应为 eager|graph|both"; exit 2 ;;
esac

echo "=== 完成 ($(date '+%F %T')) ==="
echo "注意: graph 组只有在编译期无 guard 失败时才有意义; int3 的 attention 线性层"
echo "      已注册为不透明 custom op (torch.ops.brmoe_int3.linear), 应当能捕获成功。"
exit ${rc}
