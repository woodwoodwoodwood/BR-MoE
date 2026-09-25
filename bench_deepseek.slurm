#!/bin/bash
#SBATCH --job-name=brmoe-bench
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# 注意: cpus-per-task 不能超过分区节点的核数(A100 只有 12 核, H200 96 核),
# 否则作业会以 Reason=PartitionConfig 挂住无法调度。
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/slurm_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/slurm_%j.err
#
# DeepSeek-MoE-16B: BR-MoE 3bit vs fp16 的 TTFT / TPOT 对比 (batch=1)
#
# 提交 (分区决定 CUDA 架构, 内核会按需重编译):
#   sbatch -p h200 bench_deepseek.slurm      # H200  sm_90
#   sbatch -p a100 bench_deepseek.slurm      # A100  sm_80
#   sbatch -p 4090 bench_deepseek.slurm      # 4090  sm_89  (注意: fp16 单卡放不下)
#
# 开关:
#   SMOKE=1       只跑 input=128 / output=8, 用于快速验证链路
#   RUN_CUDA=1    额外测 int3 CUDA 后端 (默认关闭)
#   BUILD_KERNEL=0  跳过内核编译
#   INPUT_LENS=256,512 OUTPUT_LEN=64 REPEAT=2  自定义参数

set -euo pipefail

BASE=/mnt/709/data3/home/jianglei
REPO=${BASE}/ada/BR-MoE
ENV_PREFIX=${BASE}/miniconda3/envs/milo
PY=${ENV_PREFIX}/bin/python
RESULT_DIR=${REPO}/bench_results

MODEL_FP16=/mnt/4090/data/jianglei/models/DeepSeek/deepseek-moe-16b-base
MODEL_3BIT=/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0

# ---- 按分区选 CUDA 目标架构 ----
case "${SLURM_JOB_PARTITION:-}" in
  h200) ARCH=sm_90;  TCA=9.0 ;;
  a100) ARCH=sm_80;  TCA=8.0 ;;
  4090) ARCH=sm_89;  TCA=8.9 ;;
  5090) ARCH=sm_120; TCA=12.0 ;;
  *)    ARCH=sm_80;  TCA=8.0 ;;
esac

BUILD_KERNEL=${BUILD_KERNEL:-1}
RUN_CUDA=${RUN_CUDA:-0}
RUN_GROUPED=${RUN_GROUPED:-1}
RUN_PYTORCH=${RUN_PYTORCH:-1}
RUN_FP16=${RUN_FP16:-1}
SMOKE=${SMOKE:-0}
# batch 扫描: 逗号分隔, 例如 BATCHES=1,2,4,8,16,32 (论文图 8 用的就是这组)
BATCHES=${BATCHES:-1}
if [[ "${SMOKE}" == "1" ]]; then
  INPUT_LENS=${INPUT_LENS:-128};     OUTPUT_LEN=${OUTPUT_LEN:-8}
  WARMUP=${WARMUP:-1};               REPEAT=${REPEAT:-1}
else
  INPUT_LENS=${INPUT_LENS:-128,512,1024}; OUTPUT_LEN=${OUTPUT_LEN:-128}
  WARMUP=${WARMUP:-2};               REPEAT=${REPEAT:-3}
fi

mkdir -p "${RESULT_DIR}"

# BR-MoE 包目录名含连字符, 用符号链接暴露成可导入的 BR_MoE
WORK=$(mktemp -d)
ln -s "${REPO}/BR-MoE" "${WORK}/BR_MoE"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# grouped 后端的权重打包缓存 (首次打包约 1 分钟, 之后秒级命中)
export BRMOE_GROUPED_CACHE_DIR="${RESULT_DIR}/grouped_cache"

# CUDA_HOME: torch 的 cpp_extension 编译扩展时需要它。
# 计算节点上 /usr/local/cuda 不一定存在, 优先用 conda 环境自带的 nvcc。
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x "${ENV_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${ENV_PREFIX}"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  echo "CUDA_HOME=${CUDA_HOME}  nvcc=$(command -v nvcc || echo none)"
fi

echo "=== NODE / GPU ==="
hostname
echo "partition=${SLURM_JOB_PARTITION:-?}  cuda_arch=${ARCH}"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo "cpus=$(nproc)  mem=$(free -g | awk '/^Mem:/{print $2}')G"
${PY} -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
echo "SMOKE=${SMOKE} RUN_CUDA=${RUN_CUDA} INPUT_LENS=${INPUT_LENS} OUTPUT_LEN=${OUTPUT_LEN} REPEAT=${REPEAT}"

# --- 按需编译 int3 CUDA kernel (不同分区架构不同, 用标记文件避免混用) ---
ARCH_FILE="${ENV_PREFIX}/lib/python3.10/site-packages/brmoe_cuda.arch"
if [[ "${BUILD_KERNEL}" == "1" ]]; then
  prev=$(cat "${ARCH_FILE}" 2>/dev/null || echo none)
  if [[ "${prev}" != "${ARCH}" ]]; then
    echo "=== BUILD brmoe_cuda (arch=${ARCH}, 当前已装=${prev}) ==="
    export BRMOE_CUDA_ARCH="${ARCH}"
    export TORCH_CUDA_ARCH_LIST="${TCA}"
    export MAX_JOBS=16
    if ( cd "${REPO}/BR-MoE/kernels" && ${PY} setup_brmoe.py install ) \
         > "${RESULT_DIR}/kernel_build_${SLURM_JOB_ID}.log" 2>&1; then
      echo "${ARCH}" > "${ARCH_FILE}"
      echo "build done"
    else
      echo "build FAILED, tail:"
      tail -n 15 "${RESULT_DIR}/kernel_build_${SLURM_JOB_ID}.log" || true
    fi
  else
    echo "=== brmoe_cuda 已是 ${ARCH}, 跳过编译 ==="
  fi
fi

# 必须真实执行一次 kernel 才能确认可用 (跨架构 .so 能 import 但 launch 失败)
KERNEL_OK=0
if timeout 300 ${PY} "${REPO}/bench/_check_kernel.py" \
     > "${RESULT_DIR}/kernel_check_${SLURM_JOB_ID}.log" 2>&1; then
  KERNEL_OK=1
  echo "brmoe_cuda 可用: $(tail -n 1 "${RESULT_DIR}/kernel_check_${SLURM_JOB_ID}.log")"
else
  echo "brmoe_cuda 不可用:"
  tail -n 6 "${RESULT_DIR}/kernel_check_${SLURM_JOB_ID}.log" || true
fi

run_case () {   # tag impl model_path backend
  local tag=$1 impl=$2 path=$3 backend=$4
  local b out
  for b in ${BATCHES//,/ }; do
    out="${RESULT_DIR}/${tag}_b${b}_${SLURM_JOB_ID}.json"
    echo ""
    echo "################ CASE ${tag} batch=${b}  (impl=${impl}, backend=${backend}) ################"
    if [[ ! -e "${path}" ]]; then
      echo "!! model path not found: ${path} -> skip"
      return 0
    fi
    # 每个 case 都用 timeout 保护: CUDA 后端出问题时不会拖垮整个作业
    timeout "${CASE_TIMEOUT:-3600}" ${PY} "${REPO}/bench/bench_ttft_tpot.py" \
        --impl "${impl}" \
        --model-path "${path}" \
        --brmoe-backend "${backend}" \
        --device cuda:0 \
        --batch-size "${b}" \
        --input-lens "${INPUT_LENS}" \
        --output-len "${OUTPUT_LEN}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --label "${tag}" \
        --out "${out}" \
      || echo "!! case ${tag} batch=${b} failed/timeout (continuing)"
  done
}

# 0) BR-MoE 3bit —— grouped int3 MoE (一个 launch 覆盖全部专家)
#    最关键的对比项: 逐专家循环 -> grouped GEMM
if [[ "${RUN_GROUPED:-1}" == "1" ]]; then
  CASE_TIMEOUT=${GROUPED_CASE_TIMEOUT:-1800} \
    run_case brmoe3bit_grouped brmoe "${MODEL_3BIT}" brmoe_grouped
else
  echo "(skip brmoe3bit_grouped: RUN_GROUPED=${RUN_GROUPED:-1})"
fi

# 1) BR-MoE 3bit —— int3 CUDA kernel
#    确认 workspace 修复之后 prefill 不再挂死。3bit 只加载 ~8GB, 比 fp16(32GB) 快得多。
if [[ "${RUN_CUDA}" == "1" && "${KERNEL_OK}" == "1" ]]; then
  CASE_TIMEOUT=${CUDA_CASE_TIMEOUT:-1200} \
    run_case brmoe3bit_cuda brmoe "${MODEL_3BIT}" brmoe_auto
else
  echo "(skip brmoe3bit_cuda: RUN_CUDA=${RUN_CUDA}, KERNEL_OK=${KERNEL_OK})"
fi

# 2) BR-MoE 3bit —— PyTorch 参考后端
if [[ "${RUN_PYTORCH}" == "1" ]]; then
  run_case brmoe3bit_pytorch brmoe "${MODEL_3BIT}" default
else
  echo "(skip brmoe3bit_pytorch)"
fi

# 3) fp16 基线 (32GB 权重, 加载最慢, 放最后)
if [[ "${RUN_FP16}" == "1" ]]; then
  run_case fp16 fp16 "${MODEL_FP16}" default
else
  echo "(skip fp16)"
fi

# --- 汇总对比 ---
echo ""
echo "#################### SUMMARY ####################"
${PY} - "${RESULT_DIR}" "${SLURM_JOB_ID}" <<'PY'
import glob
import json
import os
import sys

result_dir, job_id = sys.argv[1], sys.argv[2]
recs = []
for p in sorted(glob.glob(os.path.join(result_dir, f"*_{job_id}.json"))):
    with open(p) as f:
        recs.append(json.load(f))

if not recs:
    print("no results")
    raise SystemExit(0)

ORDER = ["fp16", "brmoe3bit_grouped", "brmoe3bit_cuda", "brmoe3bit_pytorch"]


def rank(lab):
    return ORDER.index(lab) if lab in ORDER else 99


# 按 label -> batch 归集 (batch 扫描后同一 label 会有多个文件, 不能互相覆盖)
curves = {}
for r in recs:
    curves.setdefault(r["label"], {})[r["batch_size"]] = r


def total_s(r, c):
    return (c["ttft_ms"] + c["tpot_ms"] * max(0, r["output_len"] - 1)) / 1000.0


print(f"{'case':<20}{'batch':>6}{'in_len':>8}{'TTFT(ms)':>11}{'TPOT(ms)':>11}"
      f"{'tot(s)':>10}{'tok/s':>9}{'mem(GiB)':>10}")
for lab in sorted(curves, key=rank):
    for b in sorted(curves[lab]):
        r = curves[lab][b]
        for c in r["cases"]:
            if c.get("oom"):
                print(f"{lab:<20}{b:>6}{c['input_len']:>8}{'OOM':>11}")
                continue
            print(f"{lab:<20}{b:>6}{c['input_len']:>8}{c['ttft_ms']:>11.2f}"
                  f"{c['tpot_ms']:>11.3f}{total_s(r, c):>10.2f}"
                  f"{c['decode_tok_per_s']:>9.2f}"
                  f"{r.get('peak_gpu_mem_gib', 0):>10.2f}")

if "fp16" in curves:
    print("\n--- 相对 fp16 的加速比 (>1 = 比 fp16 快) ---")
    print(f"{'case':<20}{'batch':>6}{'in_len':>8}{'TTFT x':>10}{'TPOT x':>10}"
          f"{'总时间 x':>11}")
    for lab in sorted(curves, key=rank):
        if lab == "fp16":
            continue
        for b in sorted(curves[lab]):
            if b not in curves["fp16"]:
                continue
            base = {c["input_len"]: c
                    for c in curves["fp16"][b]["cases"] if not c.get("oom")}
            for c in curves[lab][b]["cases"]:
                if c.get("oom") or c["input_len"] not in base:
                    continue
                bt = base[c["input_len"]]
                tf = total_s(curves["fp16"][b], bt)
                tl = total_s(curves[lab][b], c)
                print(f"{lab:<20}{b:>6}{c['input_len']:>8}"
                      f"{bt['ttft_ms'] / c['ttft_ms']:>10.2f}"
                      f"{bt['tpot_ms'] / c['tpot_ms']:>10.2f}"
                      f"{tf / tl:>11.2f}")
PY

echo ""
echo "=== RESULTS IN ${RESULT_DIR} ==="
echo "=== BENCHMARK COMPLETE ==="
