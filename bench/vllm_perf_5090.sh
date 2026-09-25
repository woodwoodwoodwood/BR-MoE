#!/bin/bash
# 在 5090 节点上跑 vLLM 的 brmoe_int3 性能扫描 (TTFT / TPOT / 吞吐, batch 扫描)。
#
# 前提: 必须已经在 5090 节点的交互式分配里 (salloc 或 srun --pty bash),
#       否则这个脚本在登录节点上会因为看不到 GPU 而失败。
#   进节点:  bash bench/enter_5090.sh
#
# 用法:
#   bash bench/vllm_perf_5090.sh              # graph + eager 都跑 (默认)
#   bash bench/vllm_perf_5090.sh graph        # 只跑 CUDA Graph 版 (性能最好)
#   bash bench/vllm_perf_5090.sh eager        # 只跑 eager 版 (和 HF 口径对齐)
#   BATCHES=1,2,4 bash bench/vllm_perf_5090.sh
#   GPU_MEM=0.85 bash bench/vllm_perf_5090.sh
#   MODEL=${BASE}/models/brmoe-3bit-vllm-int3dense bash bench/vllm_perf_5090.sh graph
#     ^ 跑全 int3 版本 (attention 也是 int3)。结果自动写成
#       vllm_perf_int3dense_5090_graph.json, 不会覆盖 brmoe3bit 的。
#
# 为什么分 graph / eager 两组:
#   vLLM 默认开 CUDA Graph (capture_sizes 到 512), 而 BR-MoE 的 HF 路径没有,
#   所以严格对比 HF 要看 eager 那组, 看 vLLM 的实际上限要看 graph 那组。

set -uo pipefail

# ---------------- 路径 (全部绝对, 不依赖 cwd / 变量是否已设) ----------------
BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/vllm5090/bin/python
CUDA=${BASE}/cuda-12.9
MODEL_3BIT=${BASE}/models/brmoe-3bit-vllm
# 可用 MODEL=/path/to/other 覆盖 (例如 int3dense 全 int3 版本)。
# 注意: tag 前缀会自动跟着模型目录名变 —— 否则跑 int3dense 会把结果写进
#       vllm_perf_brmoe3bit_5090_*.json, 覆盖掉 brmoe3bit (attention fp16)
#       那份数据。两者是不同模型, 不能混。
MODEL=${MODEL:-${MODEL_3BIT}}
_m=$(basename "${MODEL}")
TAGPFX=${TAGPFX:-${_m#brmoe-3bit-vllm}}
TAGPFX=${TAGPFX#-}
if [ -z "${TAGPFX}" ]; then TAGPFX=brmoe3bit; fi
RES=${REPO}/bench_results
LOGDIR=${RES}/logs

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export BRMOE_DEBUG=1
export CUDA_HOME=${CUDA}
export PATH=${CUDA}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA}/lib64:${CUDA}/lib:${LD_LIBRARY_PATH:-}

# ---------------- 参数 ----------------
MODE=${1:-both}                 # graph | eager | both
BATCHES=${BATCHES:-1,2,4,8,16,32}
IN_LEN=${IN_LEN:-512}           # 单个输入长度 (默认 512)
# 多输入长度, 逗号分隔, 优先于 IN_LEN。复用同一个 LLM 实例, 不重复加载权重。
# 长上下文注意: 本模型 max_position_embeddings=4096, 超出即为 RoPE 外推,
# 只能看延迟。KV 每 token 224 KiB (MHA), 4k+512 只要 ~1 GiB, 不是瓶颈。
#   IN_LENS=1024,2048,4096 BATCHES=1 OUT_LEN=512 bash bench/vllm_perf_5090.sh graph
IN_LENS=${IN_LENS:-}
OUT_LEN=${OUT_LEN:-512}
GPU_MEM=${GPU_MEM:-0.90}
# ---- MAX_BATCHED 有两个约束 ----
# 1) 上限: BR-MoE 的 moe_align_block_size_triton 在 M*top_k > 32768 时退回
#    torch 实现, 而那条分支缺 'sv' 键会 KeyError。所以 M <= 5461。
# 2) 下限: 实测单次前向 M=4096 会触发 device-side assert (崩在
#    align_triton.py:192), 而 M<=2048 正常。输入比这个长时, chunked prefill
#    会按 max_num_batched_tokens 切块, 所以把它设成 2048 就能跑 4096 的输入:
#      MAX_BATCHED=2048 IN_LENS=4096 OUT_LEN=512 BATCHES=1 bash ... graph
MAX_BATCHED=${MAX_BATCHED:-2048}

mkdir -p "${LOGDIR}"
STAMP=$(date +%m%d_%H%M%S)

echo "=== vLLM perf on $(hostname) / $(date '+%F %T') ==="
echo "MODE=${MODE} BATCHES=${BATCHES} IN=${IN_LEN} IN_LENS=${IN_LENS:-<未用>} OUT=${OUT_LEN} GPU_MEM=${GPU_MEM}"
echo "MODEL =${MODEL}"
echo "TAGPFX=${TAGPFX}   (结果: ${RES}/vllm_perf_${TAGPFX}_5090_*.json)"
# KV 池约 6.5 万 token, 每 token 224 KiB (MHA: 28层 x 2 x 16头 x 128dim x 2B)。
# 并发上限 ≈ 64592/(in+out); batch 超过这个数 vLLM 会靠抢占(preemption)调度,
# 测出来的延迟语义就变了 —— 比大小没意义。
echo "提示: 并发上限约 $(( 64592 / (IN_LEN + OUT_LEN) )) 路 (按 IN_LEN 估), batch 超过会触发抢占"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置, 大概率看不到 GPU>}'"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "!! CUDA_VISIBLE_DEVICES 为空 —— 这个分配没拿到 GPU。"
  echo "   请退出后用: srun -p 5090 -N1 -n1 --mem=48G --gres=gpu:1 -t 2:00:00 --pty bash"
  exit 1
fi

run_one () {
  local tag=$1 eager=$2
  local log=${LOGDIR}/vllm_${tag}_${STAMP}.log
  echo "################ ${tag}  (enforce_eager=${eager}) ################"
  echo "日志: ${log}"
  "${PY}" "${REPO}/bench/vllm_perf.py" \
      --model "${MODEL}" \
      --tag "${tag}" --quantization brmoe_int3 \
      --batch-sizes "${BATCHES}" \
      --input-len "${IN_LEN}" --output-len "${OUT_LEN}" \
      ${IN_LENS:+--input-lens "${IN_LENS}"} \
      --max-num-batched-tokens "${MAX_BATCHED}" \
      --enforce-eager "${eager}" --gpu-mem-util "${GPU_MEM}" \
      --out "${RES}/vllm_perf_${tag}.json" 2>&1 | tee "${log}"
  echo "-- ${tag} 结束 (exit=${PIPESTATUS[0]})"
  echo
}

rc=0
case "${MODE}" in
  graph) run_one "${TAGPFX}_5090_graph" 0 || rc=1 ;;
  eager) run_one "${TAGPFX}_5090_eager" 1 || rc=1 ;;
  both)
    run_one "${TAGPFX}_5090_graph" 0 || rc=1
    run_one "${TAGPFX}_5090_eager" 1 || rc=1
    ;;
  *) echo "!! 未知 MODE='${MODE}', 应为 graph|eager|both"; exit 2 ;;
esac

echo "=== 汇总 ($(date '+%F %T')) ==="
for f in "${RES}"/vllm_perf_${TAGPFX}_5090_*.json; do
  [ -e "$f" ] || continue
  echo "-- $(basename "$f")"
  "${PY}" -c "
import json,sys
d=json.load(open('$f'))
print(f\"   {d['gpu']}  peak={d['peak_gib']:.2f} GiB\")
print(f\"   {'in':>7}{'batch':>7}{'TTFT_ms':>11}{'TPOT_ms':>11}{'E2E_ms':>12}{'tok/s':>10}\")
for r in d['rows']:
    if 'error' in r:
        print(f\"   {r.get('input_len','?'):>7}{r['batch']:>7}   {r['error']}\"); continue
    print(f\"   {r.get('input_len',0):>7}{r['batch']:>7}{r['ttft_ms']:>11.2f}{r['tpot_ms']:>11.2f}{r['e2e_ms']:>12.2f}{r['thr_tok_s']:>10.1f}\")
" || true
done
exit ${rc}
