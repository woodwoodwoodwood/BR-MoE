#!/bin/bash
#SBATCH -J vllm-perf
# !! 必须用 /mnt/709/... 全路径 !!
# 计算节点上 /home/jianglei 指向**另一个本地 home**, 里面没有 ada/ 目录:
#   登录节点: /home/jianglei -> /mnt/709/data3/home/jianglei   (能解析)
#   计算节点: /home/jianglei -> 本地 home, 没有 ada/            (解析失败)
# slurmd 建不出输出文件 -> 作业**秒失败**: JobState=FAILED / Reason=NonZeroExitCode
# / RunTime<=4s / StdOut 文件根本不存在。实测 38456 / 38460 / 38516 全是这个原因。
# (之前一直误判成"环境变量污染", 其实是路径。)
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_perf_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/vllm_perf_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# vLLM 上的 fp16 vs brmoe_int3 性能对比 (batch 扫描)。
#
# 显存约束 (决定了必须在哪个分区跑):
#   fp16 权重 30.5 GiB -> 5090(32GB) 放不下 (实测 OOM, PyTorch 已占 30.53 GiB)
#                        -> 必须 A100(80GB) 或 H200(141GB)
#   brmoe_int3 权重 8.1 GiB -> 都能跑
# 所以 "同卡对比" 只能在 A100/H200 上做。
#
# 用法:
#   sbatch -p a100 bench/vllm_perf_compare.slurm
#   CASES="fp16|$MODEL_FP16|none brmoe3bit|$MODEL_INT3|brmoe_int3" ...

set -uo pipefail

ENV=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
PY=${ENV}/bin/python
export PYTHONUNBUFFERED=1

# ---- CUDA_HOME 必须显式导出 (否则 FlashInfer 的 JIT 会炸) ----
# 计算节点上 `which nvcc` 落到 /usr/local/cuda = CUDA **12.4**, 而 FlashInfer 生成
# 的 ninja 构建脚本会传 `--compress-mode=size` (CUDA 12.8+ 才支持), 于是:
#     nvcc fatal   : Unknown option '--compress-mode=size'
#     ninja: build stopped: subcommand failed.
#     RuntimeError: Ninja build failed
# 实测 38908 三个 case 全死在这里 (fp16 / brmoe3bit 各一份, compress-mode 出现 36 次)。
# 手动跑时因为命令里带了 `env CUDA_HOME=...` 所以没事 —— sbatch 是干净环境, 会漏。
CUDA_HOME=${CUDA_HOME:-/mnt/709/data3/home/jianglei/cuda-12.9}
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# FlashInfer 解析 nvcc 的优先级 (jit/core.py:477):
#     FLASHINFER_NVCC  >  {cuda_home}/bin/nvcc  (cuda_home 来自 CUDA_HOME/CUDA_PATH)
#                      >  which nvcc 推导出的目录  >  /usr/local/cuda
# 38908 的日志里编译命令是 `/usr/bin/nvcc` —— 因为计算节点上 `which nvcc` 返回
# /usr/bin/nvcc, 于是 cuda_home 被推成 /usr/bin。显式钉死最稳, 免得再被 PATH 影响。
export FLASHINFER_NVCC="${CUDA_HOME}/bin/nvcc"
echo "[env] CUDA_HOME=${CUDA_HOME}"
echo "[env] FLASHINFER_NVCC=${FLASHINFER_NVCC}"
"${CUDA_HOME}/bin/nvcc" --version 2>&1 | tail -2 | sed 's/^/[env] /'

# JIT 缓存的 key 里**不含 CUDA 版本**, 所以用错 CUDA 编过一次后, 换回来也可能复用坏目标文件。
# 需要时用 FLUSH_FLASHINFER_CACHE=1 强制清空重建 (代价约 1-2 分钟)。
if [ "${FLUSH_FLASHINFER_CACHE:-0}" = "1" ]; then
  rm -rf "${HOME}/.cache/flashinfer"
  echo "[cache] 已清空 FlashInfer JIT 缓存: ${HOME}/.cache/flashinfer"
fi

MODEL_FP16=${MODEL_FP16:-/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-base}
MODEL_INT3=${MODEL_INT3:-/mnt/709/data3/home/jianglei/models/brmoe-3bit-vllm}

BATCHES=${BATCHES:-1,2,4,8,16,32}
IN_LEN=${IN_LEN:-128}
OUT_LEN=${OUT_LEN:-128}
EAGER=${EAGER:-1}
GPU_MEM=${GPU_MEM:-0.90}
# 上限 2048: 实测单次前向 M=4096 会触发 device-side assert
# (崩在 BR-MoE 的 align_triton.py:192), 而 M<=2048 正常。
# 默认 4096 时, bs=32 x in=128 = 4096 正好会撞上; chunked prefill 会按这个值切块。
MAX_BATCHED=${MAX_BATCHED:-2048}
# 每组重复几次取中位数。单次测量噪声可达 15%, 而 int3-MoE vs fp16-MoE 的
# 效应也就 12~16% —— 不重复的话噪声能盖过信号。
REPEAT=${REPEAT:-3}
# 可选: 独立 tokenizer 目录。
# 两组对比**必须**用同一个修好的 tokenizer —— fp16 基座的 tokenizer_config.json 里
# tokenizer_class=LlamaTokenizerFast 且没有 tokenizer.model, 会让 vLLM 退回慢速
# LlamaTokenizer (丢 BOS + 吃掉所有空格, 变成 'ThecapitalofFranceis')。
TOKENIZER=${TOKENIZER:-}

# --export=NONE 时 slurm 仍会设 SLURM_JOB_ID, 但保险起见给个兜底
# (脚本是 set -u, 未定义变量会直接退出)
JID=${SLURM_JOB_ID:-manual}

# 格式: "tag|model|quantization"  (quantization 为 none 表示 fp16 原生)
CASES=${CASES:-"fp16|${MODEL_FP16}|none
brmoe3bit|${MODEL_INT3}|brmoe_int3"}

ROOT=/mnt/709/data3/home/jianglei/ada/BR-MoE   # 同上: 不能用 /home/jianglei
RES=${ROOT}/bench_results
LOGDIR=${RES}/logs
mkdir -p "${LOGDIR}"
cd "${RES}" || exit 1

# 结果文件名带上模式 (graph/eager)。
# 否则 EAGER=0 那次会把 EAGER=1 的基线**覆盖掉** —— 实测 38901 就是这样:
# 同一个 JID 下 vllm_perf_fp16_38901.json 先存了 eager 数据, 再跑 graph 就被冲掉。
if [ "${EAGER}" = "0" ]; then MODE=graph; else MODE=eager; fi

# ---- 闸门: 等 A100 的 sm_80 编译完成 ----
# 现有 .so 是 TORCH_CUDA_ARCH_LIST=12.0 构建的 (70 个 sm_120 / 仅 4 个 sm_80),
# 在 A100 上跑会报 "no kernel image is available for execution on the device"。
# 编译是在登录节点用 tools/build_vllm_sm80.sh 后台跑的非 slurm 进程,
# 没法用 --dependency 关联, 所以这里主动轮询等它出结果。
# 5090 分区不需要等 (现有 .so 就支持), 用 SKIP_WAIT=1 跳过。
CUDA_BIN=/mnt/709/data3/home/jianglei/cuda-12.9/bin
SO=/mnt/709/data3/home/jianglei/framework/vllm/vllm/_C_stable_libtorch.abi3.so
REQUIRED_SM80=${REQUIRED_SM80:-20}
WAIT_MAX=${WAIT_MAX:-5400}

if [ "${SKIP_WAIT:-0}" != "1" ]; then
  t0=$(date +%s)
  while true; do
    n=$("${CUDA_BIN}/cuobjdump" --list-elf "${SO}" 2>/dev/null | grep -c sm_80)
    if [ "${n}" -ge "${REQUIRED_SM80}" ]; then
      echo "[gate] sm_80 kernel 数 = ${n} >= ${REQUIRED_SM80}, 编译已就绪"
      break
    fi
    el=$(( $(date +%s) - t0 ))
    if [ "${el}" -gt "${WAIT_MAX}" ]; then
      echo "[gate] !! 等待编译超时 (${el}s, sm_80=${n}), 放弃"
      exit 1
    fi
    echo "[gate] 等待 sm_80 编译... 当前 ${n}/${REQUIRED_SM80} (已等 ${el}s)"
    sleep 60
  done
fi

echo "=== node $(hostname) / $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "batches=${BATCHES} in=${IN_LEN} out=${OUT_LEN} eager=${EAGER}"
echo "max_num_batched_tokens=${MAX_BATCHED}  tokenizer=${TOKENIZER:-<随模型>}"
echo

rc=0
while IFS='|' read -r tag model quant; do
  [ -z "${tag}" ] && continue
  tag=$(echo "$tag" | xargs); model=$(echo "$model" | xargs); quant=$(echo "$quant" | xargs)
  echo ""
  echo "################ CASE ${tag}  (quant=${quant}) ################"

  extra=()
  [ "${quant}" != "none" ] && extra+=(--quantization "${quant}")
  tokarg=()
  [ -n "${TOKENIZER}" ] && tokarg=(--tokenizer "${TOKENIZER}")

  LOG=${LOGDIR}/cmp_${tag}_${MODE}_${JID}_$(date +%m%d_%H%M%S).log
  echo "日志: ${LOG}"
  # 原来是 `| tail -40` —— 它缓存全部输出、到进程退出才打印, 所以每个 case
  # 跑多久终端就静默多久, 看上去像卡死 (fp16 加载 30.5 GiB + 编译 + 图捕获
  # 要好几分钟)。改成 tee: 实时可见 + 落盘可复查。
  if "${PY}" "${ROOT}/bench/vllm_perf.py" \
        --model "${model}" --tag "${tag}" \
        --batch-sizes "${BATCHES}" \
        --input-len "${IN_LEN}" --output-len "${OUT_LEN}" \
        --enforce-eager "${EAGER}" --gpu-mem-util "${GPU_MEM}" \
        --max-num-batched-tokens "${MAX_BATCHED}" \
        --repeat "${REPEAT}" \
        --out "${RES}/vllm_perf_${tag}_a100_${MODE}_${JID}.json" \
        "${tokarg[@]}" "${extra[@]}" 2>&1 | tee -a "${LOG}"; then
    echo "-- ${tag} 完成"
  else
    echo "!! ${tag} 失败"
    rc=1
  fi
done <<< "${CASES}"

echo ""
echo "=== ${JID} DONE $(date '+%F %T') rc=${rc} ==="
exit ${rc}
