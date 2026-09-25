#!/bin/bash
#SBATCH --job-name=brmoe-trace
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/trace_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/trace_%j.err
#
# 采集 fp16 / brmoe_3bit 的 nsys trace, 并用 veloq 生成 parquet 缓存
#   sbatch -p a100 bench/trace_nsys.slurm
#   TRACE_CONFIGS="fp16" sbatch -p a100 bench/trace_nsys.slurm   # 只采一个

set -uo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
ENV_PREFIX=${BASE}/miniconda3/envs/milo
PY=${ENV_PREFIX}/bin/python
RESULT_DIR=${REPO}/bench_results
TRACE_DIR=${RESULT_DIR}/traces
VELOQ=${HOME}/.local/bin/veloq

MODEL_FP16=/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-base
MODEL_3BIT=/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0

mkdir -p "${TRACE_DIR}"
export BRMOE_GROUPED_CACHE_DIR="${RESULT_DIR}/grouped_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 包目录名含连字符, 用符号链接暴露成可导入的 BR_MoE
WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"

echo "=== node $(hostname) / $(date +%H:%M:%S) ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
nsys --version
"${VELOQ}" --version

run_trace () {   # tag impl model_path backend
  local tag=$1 impl=$2 path=$3 backend=$4
  if [[ ! -e "${path}" ]]; then
    echo "!! missing model: ${path}"; return 0
  fi
  echo ""
  echo "################ TRACE ${tag} (impl=${impl}, backend=${backend}) ################"
  # --sample=none / --cpuctxsw=none: 关掉 CPU 采样, 只留 CUDA+NVTX (也避开 perf_event_paranoid)
  nsys profile \
      --trace=cuda,nvtx \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      --stats=false \
      -o "${TRACE_DIR}/${tag}" \
      "${PY}" "${REPO}/bench/trace_run.py" \
          --impl "${impl}" \
          --model-path "${path}" \
          --brmoe-backend "${backend}" \
          --input-len "${IN_LEN:-128}" \
          --decode-steps "${DECODE_STEPS:-16}" \
          --repeats "${REPEATS:-2}" \
          --warmup "${WARMUP:-1}" \
    2>&1 | tail -6
  ls -lh "${TRACE_DIR}/${tag}.nsys-rep" 2>/dev/null
  # 预构建 veloq 的 parquet 缓存 (登录节点 nsys 版本过低, 必须在节点上做)
  if [[ -x "${VELOQ}" ]]; then
    echo "--- veloq prep ${tag} ---"
    "${VELOQ}" prep "${TRACE_DIR}/${tag}.nsys-rep" 2>&1 | tail -5
  fi
}

CONFIGS=${TRACE_CONFIGS:-"fp16 brmoe3bit_grouped brmoe3bit_pytorch"}
for cfg in ${CONFIGS}; do
  case "${cfg}" in
    fp16)      run_trace fp16 fp16 "${MODEL_FP16}" default ;;
    brmoe3bit_grouped) run_trace brmoe3bit_grouped brmoe "${MODEL_3BIT}" brmoe_grouped ;;
    brmoe3bit_pytorch) run_trace brmoe3bit_pytorch brmoe "${MODEL_3BIT}" default ;;
    brmoe3bit_cuda)    run_trace brmoe3bit_cuda brmoe "${MODEL_3BIT}" brmoe_auto ;;
    *) echo "unknown config ${cfg}" ;;
  esac
done

echo ""
echo "=== TRACES ==="
ls -lh "${TRACE_DIR}"/*.nsys-rep 2>/dev/null
echo "=== DONE $(date +%H:%M:%S) ==="
