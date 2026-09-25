#!/bin/bash
# 在 5090 上测 deepseek-moe-16b-gptq-4bit 的 vLLM 推理性能。
#
# 背景 / 为什么需要这个脚本:
#   该模型 desc_act=True (索引里 5380 个权重全带 g_idx), 而 vLLM 的
#   is_moe_wna16_compatible() 要求 `not desc_act` (moe_wna16.py:162)。
#   不过层级 get_quant_method() 对 RoutedExperts 是**无条件**返回
#   MoeWNA16Method (moe_wna16.py:195), 所以能不能跑、跑了对不对
#   只能实测 —— 因此本脚本分两阶段: 先冒烟确认能加载, 再跑性能。
#
# 用法 (必须在 ada-5090 节点的分配里):
#   bash bench/gptq4bit_5090.sh                 # 冒烟 + 全量
#   bash bench/gptq4bit_5090.sh smoke           # 只冒烟 (2 分钟)
#   bash bench/gptq4bit_5090.sh perf            # 只跑性能
#   QUANT=gptq_marlin bash bench/gptq4bit_5090.sh   # 强制 marlin 后端
#
# 可用环境变量:
#   MODEL  QUANT  BATCHES  IN_LENS  OUT_LEN  GPU_MEM  MAX_MODEL_LEN
#
# 提交成独立作业 (8 核, 避免 CPU 饿死):
#   sbatch --export=NONE -p 5090 --mem=48G --cpus-per-task=8 --gres=gpu:1 -t 01:00:00 \
#     -J gptq4bit --wrap "bash /mnt/709/data3/home/jianglei/ada/BR-MoE/bench/gptq4bit_5090.sh" \
#     -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/slurm_gptq4bit_%j.out

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/vllm5090/bin/python
CUDA=${BASE}/cuda-12.9
MODEL=${MODEL:-/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-gptq-4bit}
RES=${REPO}/bench_results
LOGDIR=${RES}/logs

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# 可覆盖: 在只有 1 核的分配里用 OMP_NUM_THREADS=1 更好, 8 个线程会互相抢占
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
export CUDA_HOME=${CUDA}
export PATH=${CUDA}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA}/lib64:${CUDA}/lib:${LD_LIBRARY_PATH:-}

STAGE=${1:-all}                  # all | smoke | perf
QUANT=${QUANT:-}                 # 空 = 让 vLLM 自动推断; 失败可试 gptq_marlin
MAX_MODEL_LEN=${MAX_MODEL_LEN:-0}
BATCHES=${BATCHES:-1,2,4,8,16}
IN_LENS=${IN_LENS:-1024,2048,4096}
OUT_LEN=${OUT_LEN:-512}
GPU_MEM=${GPU_MEM:-0.85}
# 2048 而非 4096: 一是与 brmoe-3bit 的实测口径对齐 (chunk 大小直接影响 TTFT),
# 二是 BR-MoE 的 align kernel 在 M=4096 会崩, 保持一致便于后续同口径比较。
MAX_BATCHED=${MAX_BATCHED:-2048}

mkdir -p "${LOGDIR}"
STAMP=$(date +%m%d_%H%M%S)
rc=0

echo "=== GPTQ-4bit perf on $(hostname) / $(date '+%F %T') ==="
echo "model=${MODEL}"
echo "stage=${STAGE} quant='${QUANT:-<auto>}' batches=${BATCHES} lens=${IN_LENS} out=${OUT_LEN}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置>}'"
echo

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "!! 这个分配没拿到 GPU。请用:"
  echo "   srun -p 5090 -N1 -n1 --mem=48G --cpus-per-task=8 --gres=gpu:1 -t 2:00:00 --pty bash"
  exit 1
fi
if [ ! -d "${MODEL}" ]; then
  echo "!! 模型目录不存在: ${MODEL}"; exit 1
fi

# ---- 先打印量化配置与 vLLM 准入结论 (纯读, 不占 GPU) ----
"${PY}" - "$MODEL" <<'PY' || true
import json, sys
p = sys.argv[1]
c = json.load(open(p + "/config.json"))
q = c.get("quantization_config") or {}
qm, da, nb = q.get("quant_method"), q.get("desc_act"), q.get("bits")
print("=== 量化配置 ===")
for k in ("quant_method", "bits", "group_size", "desc_act", "sym", "true_sequential"):
    if k in q:
        print(f"  {k:18} = {q[k]}")
print(f"  max_position_embeddings = {c.get('max_position_embeddings')}")
print(f"  n_routed_experts={c.get('n_routed_experts')} topk={c.get('num_experts_per_tok')}")
ok = (qm == "gptq" and not da and nb in (4, 8))
print()
print("=== vLLM MoE WNA16 准入 (moe_wna16.py:162) ===")
print(f"  quant_method={qm} desc_act={da} bits={nb} -> "
      f"{'通过' if ok else '不通过 (层级不做检查, 仍会尝试加载 -> 需实测)'}")
PY
echo

# ---- 阶段 1: 冒烟 (加载 + 一次生成) ----
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "smoke" ]; then
  echo "################ STAGE 1: 冒烟 (加载 + 生成) ################"
  smoke_log=${LOGDIR}/gptq4bit_smoke_${STAMP}.log
  smoke_args=(
    --model "${MODEL}" --tag gptq4bit_smoke
    --batch-sizes 1 --input-lens 1024 --output-len 32
    --max-num-batched-tokens "${MAX_BATCHED}"
    --enforce-eager 1 --gpu-mem-util "${GPU_MEM}"
    --out "${RES}/vllm_perf_gptq4bit_smoke.json"
  )
  [ -n "${QUANT}" ] && smoke_args+=(--quantization "${QUANT}")
  # 汇总表里读 input_len; MAX_MODEL_LEN=0 时脚本会自动推算
  [ "${MAX_MODEL_LEN}" != "0" ] && smoke_args+=(--max-model-len "${MAX_MODEL_LEN}")

  if "${PY}" "${REPO}/bench/vllm_perf.py" "${smoke_args[@]}" 2>&1 | tee "${smoke_log}"; then
    if grep -q 'Error\|Traceback\|error' "${smoke_log}"; then
      echo "!! 冒烟日志里出现错误关键字, 请看 ${smoke_log}"
      rc=1
    else
      echo "-- 冒烟通过"
    fi
  else
    echo "!! 冒烟失败 (exit=${PIPESTATUS[0]}), 见 ${smoke_log}"
    rc=1
  fi
  echo
  if [ "${rc}" -ne 0 ] && [ "${STAGE}" = "all" ]; then
    echo "冒烟未通过, 跳过性能阶段。"
    echo "排查建议:"
    echo "  1) 强制 marlin: QUANT=gptq_marlin bash bench/gptq4bit_5090.sh smoke"
    echo "  2) 若报 desc_act/g_idx 相关错误, 说明 vLLM 的 MoE 路径确实不支持 act-order,"
    echo "     需要先反量化再以 desc_act=False 重量化才能跑。"
    exit 1
  fi
fi

# ---- 阶段 2: 性能扫描 ----
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "perf" ]; then
  echo "################ STAGE 2: 性能扫描 ################"
  # KV 提醒: 每 token 224 KiB, 池约 6.5 万 token -> 并发上限 ~64592/(in+out)
  echo "注意: KV 每 token 224 KiB; 4096+512 时并发上限约 14, batch 超过会触发抢占"
  perf_log=${LOGDIR}/gptq4bit_perf_${STAMP}.log
  perf_args=(
    --model "${MODEL}" --tag gptq4bit_5090
    --batch-sizes "${BATCHES}" --input-lens "${IN_LENS}" --output-len "${OUT_LEN}"
    --max-num-batched-tokens "${MAX_BATCHED}"
    --enforce-eager 0 --gpu-mem-util "${GPU_MEM}"
    --out "${RES}/vllm_perf_gptq4bit_5090.json"
  )
  [ -n "${QUANT}" ] && perf_args+=(--quantization "${QUANT}")
  [ "${MAX_MODEL_LEN}" != "0" ] && perf_args+=(--max-model-len "${MAX_MODEL_LEN}")

  "${PY}" "${REPO}/bench/vllm_perf.py" "${perf_args[@]}" 2>&1 | tee "${perf_log}"
  [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1

  echo
  echo "=== 汇总 ==="
  "${PY}" - "${RES}/vllm_perf_gptq4bit_5090.json" <<'PY' || true
import json, os, sys
f = sys.argv[1]
if not os.path.exists(f):
    print("  结果文件不存在:", f); raise SystemExit
d = json.load(open(f))
print(f"  {d['gpu']}  peak={d['peak_gib']:.2f} GiB  quant={d['quantization']}")
print(f"  {'in':>7}{'batch':>7}{'TTFT_ms':>11}{'TPOT_ms':>11}{'E2E_ms':>12}{'tok/s':>10}")
for r in d["rows"]:
    if "error" in r:
        print(f"  {r.get('input_len','?'):>7}{r['batch']:>7}   {r['error'][:90]}")
        continue
    print(f"  {r.get('input_len',0):>7}{r['batch']:>7}{r['ttft_ms']:>11.2f}"
          f"{r['tpot_ms']:>11.2f}{r['e2e_ms']:>12.2f}{r['thr_tok_s']:>10.1f}")
PY
fi

echo
echo "=== $(date '+%F %T') DONE rc=${rc} ==="
exit ${rc}
