#!/bin/bash
#SBATCH --job-name=qwen35-nvfp4-bench
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/framework/qwen35_nvfp4_bench_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/framework/qwen35_nvfp4_bench_%j.err

set -euo pipefail

MODEL=/mnt/4090/data/jianglei/models/Qwen/Qwen3.8-27B-QUASAR-NVFP4
SERVED_NAME=qwen3.8-27b-quasar-nvfp4
RESULT_DIR=/mnt/709/data3/home/jianglei/framework/qwen35_nvfp4_results
PORT=8321
BASE_URL=http://127.0.0.1:${PORT}

CUDA_HOME=/mnt/709/data3/home/jianglei/cuda-12.9
ENV_PREFIX=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
export PATH=${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=8

mkdir -p "${RESULT_DIR}"
SERVER_LOG="${RESULT_DIR}/server_${SLURM_JOB_ID}.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "=== STOPPING SERVER ==="
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "=== NODE / GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "MODEL=${MODEL}"
echo "RESULTS=${RESULT_DIR}"

echo "=== STARTING VLLM SERVER ==="
vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "SERVER_PID=${SERVER_PID}"
echo "SERVER_LOG=${SERVER_LOG}"

for i in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "=== SERVER READY AFTER ${i} CHECKS ==="
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "=== SERVER EXITED DURING STARTUP ==="
    tail -n 100 "${SERVER_LOG}"
    exit 1
  fi
  sleep 10
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "=== SERVER DID NOT BECOME READY ==="
  tail -n 150 "${SERVER_LOG}"
  exit 1
fi

echo "=== WARMUP ==="
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one short sentence about GPU throughput.\"}],\"max_tokens\":32,\"temperature\":0}" \
  > "${RESULT_DIR}/warmup_${SLURM_JOB_ID}.json"

VLLM_VERSION=$(${ENV_PREFIX}/bin/python -c 'import vllm; print(vllm.__version__)')

for CONCURRENCY in 1 8 32 64; do
  echo "=== BENCHMARK CONCURRENCY=${CONCURRENCY} INPUT=512 OUTPUT=128 PROMPTS=64 ==="
  vllm bench serve \
    --backend openai-chat \
    --base-url "${BASE_URL}" \
    --endpoint /v1/chat/completions \
    --model "${SERVED_NAME}" \
    --tokenizer "${MODEL}" \
    --dataset-name random \
    --seed 1234 \
    --num-prompts 64 \
    --random-input-len 512 \
    --random-output-len 128 \
    --request-rate inf \
    --max-concurrency "${CONCURRENCY}" \
    --ignore-eos \
    --temperature 0 \
    --top-p 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --disable-tqdm \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "concurrency_${CONCURRENCY}.json" \
    --label "qwen35_nvfp4_c${CONCURRENCY}" \
    --metadata "model=${MODEL}" "vllm_version=${VLLM_VERSION}" "concurrency=${CONCURRENCY}"
done

echo "=== GPU AFTER BENCHMARK ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "=== THROUGHPUT SUMMARY ==="
${ENV_PREFIX}/bin/python - "${RESULT_DIR}"/concurrency_*.json <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as f:
        r = json.load(f)
    print(
        f"concurrency={r.get('max_concurrency')} "
        f"requests={r.get('completed')} "
        f"request_throughput={r.get('request_throughput'):.3f} req/s "
        f"output_throughput={r.get('output_throughput'):.3f} tok/s "
        f"mean_e2el={r.get('mean_e2el_ms'):.1f} ms "
        f"p99_e2el={r.get('p99_e2el_ms'):.1f} ms"
    )
PY

echo "=== BENCHMARK COMPLETE ==="
