#!/bin/bash
# 在 5090 节点上跑 BR-MoE **HF 原生** 路径的性能, 用来和 vLLM 侧对照。
#
# 为什么用 vllm5090 环境: milo 环境的 torch 是 2.5.1+cu121, archs 只到 sm_86,
# 跑不了 5090 (sm_120)。vllm5090 的 torch 2.13+cu129 支持 sm_120,
# 且自带 transformers / triton / safetensors。
#
# 前提: 在 5090 节点的交互式分配里。进节点: bash bench/enter_5090.sh
#
# 用法:
#   bash bench/hf_perf_5090.sh                          # grouped + brmoe_auto
#   BACKENDS=grouped bash bench/hf_perf_5090.sh         # 只跑 grouped
#   BACKENDS="grouped brmoe_auto" BATCHES=1,2 bash bench/hf_perf_5090.sh
#
# 对照关系 (A100 上已有的 HF 数字, batch=1 / in=128):
#   grouped       TTFT  41.31 ms   TPOT  21.97 ms/tok   <- int3 fused kernel
#   brmoe_auto    TTFT 242.19 ms   TPOT  44.93 ms/tok   <- 逐专家循环
#   两者之差 = grouped kernel 的净收益
#   (注意那批是 A100 数, 这里跑的是 5090, 绝对值不可直接比)

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
PY=${BASE}/miniconda3/envs/vllm5090/bin/python
RES=${REPO}/bench_results
LOGDIR=${RES}/logs

# BR-MoE 原生 checkpoint (含 qmodel.pt + modeling_deepseek.py)
MODEL_HF=${MODEL_HF:-/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0}

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export RES          # 给下面汇总用的内嵌 python 读

BACKENDS=${BACKENDS:-"grouped brmoe_auto"}
BATCHES=${BATCHES:-1}
IN_LENS=${IN_LENS:-128}
OUT_LEN=${OUT_LEN:-128}
WARMUP=${WARMUP:-2}
REPEAT=${REPEAT:-3}

mkdir -p "${LOGDIR}"
STAMP=$(date +%m%d_%H%M%S)

echo "=== HF(BR-MoE) perf on $(hostname) / $(date '+%F %T') ==="
echo "model=${MODEL_HF}"
echo "backends=${BACKENDS} batches=${BATCHES} in=${IN_LENS} out=${OUT_LEN}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置>}'"
echo

if [ ! -d "${MODEL_HF}" ]; then
  echo "!! 模型目录不存在: ${MODEL_HF}"
  echo "   附近可用的 3bit 目录:"
  ls -d /mnt/4090/data/jianglei/models/MiLo/*3bit* 2>/dev/null | head -5
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "!! CUDA_VISIBLE_DEVICES 为空 —— 这个分配没拿到 GPU"
  exit 1
fi

rc=0
for bk in ${BACKENDS}; do
  for bs in ${BATCHES//,/ }; do
    tag="hf_${bk}_5090_bs${bs}"
    log=${LOGDIR}/${tag}_${STAMP}.log
    echo "################ ${tag} ################"
    echo "日志: ${log}"
    "${PY}" "${REPO}/bench/bench_ttft_tpot.py" \
        --impl brmoe \
        --model-path "${MODEL_HF}" \
        --brmoe-backend "${bk}" \
        --device cuda:0 \
        --batch-size "${bs}" \
        --input-lens "${IN_LENS}" \
        --output-len "${OUT_LEN}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" \
        --label "${tag}" \
        --out "${RES}/${tag}.json" 2>&1 | tee "${log}"
    rc_this=${PIPESTATUS[0]}
    echo "-- ${tag} 结束 (exit=${rc_this})"
    [ "${rc_this}" -ne 0 ] && rc=1
    echo
  done
done

echo "=== 汇总 ($(date '+%F %T')) ==="
"${PY}" - <<'PY' || true
import glob, json, os
for f in sorted(glob.glob(os.path.join(os.environ.get("RES",""), "hf_*_5090_bs*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    print(f"-- {os.path.basename(f)}  backend={d.get('brmoe_backend')} "
          f"load={d.get('load_seconds', 0):.1f}s")
    for c in d.get("cases", []):
        print(f"   in={c.get('input_len'):>5}  TTFT={c.get('ttft_ms', float('nan')):8.2f} ms  "
              f"TPOT={c.get('tpot_ms', float('nan')):8.2f} ms/tok")
PY
exit ${rc}
