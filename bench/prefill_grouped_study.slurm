#!/bin/bash
#SBATCH -J prefill-grouped
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH -o /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/prefill_grouped_%j.out
#SBATCH -e /mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/prefill_grouped_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
SRC=${SRC:-$REPO}
export CUDA_HOME=$BASE/cuda-12.9 FLASHINFER_NVCC=$BASE/cuda-12.9/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$SRC/tools:$SRC/bench:${PYTHONPATH:-} BRMOE_PKG=$SRC/BR-MoE
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/tmp/brmoe_prefill_pycache_${SLURM_JOB_ID}
unset PYTHONDONTWRITEBYTECODE
OUT=$REPO/bench_results/prefill_grouped/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$REPO"
PY=$BASE/miniconda3/envs/vllm5090/bin/python
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.sm,clocks.mem --format=csv
if [[ ${PHASE:-sweep} == cuda_build ]]; then
  BUILD=$OUT/cuda
  mkdir -p "$BUILD"
  cp "$SRC/BR-MoE/kernels/marlin_int3_moe/"{int3_moe_ops.cpp,int3_moe_kernel.cu,setup_moe.py} "$BUILD/"
  cd "$BUILD"
  BRMOE_CUDA_ARCH=sm_80 BRMOE_ABI=1 MAX_JOBS=4 "$PY" setup_moe.py build_ext --inplace
  export BRMOE_STUDY_CUDA_SO=$BUILD/brmoe_moe_int3.cpython-312-x86_64-linux-gnu.so
  cd "$REPO"
  "$PY" "$SRC/bench/prefill_grouped_study.py" sweep --out "$OUT" ${ARGS:-}
elif [[ ${PHASE:-sweep} == validate ]]; then
  "$PY" "$SRC/bench/prefill_grouped_study.py" verify --out "$OUT/grouped" \
    --m-values 1,4,32,127,128,129,255,256,257,511,512,513,1024,2048 --configs fastreduce_128_128_64_3_4
  "$PY" "$SRC/bench/prefill_grouped_study.py" verify --out "$OUT/cuda" \
    --m-values 1,4,32,127,128,129,255,256,257,511,512,513,1024,2048 --configs cudafast_32_128_128_4_8
  "$PY" "$SRC/bench/int3_linear_study.py" verify --out "$OUT/linear" \
    --input "${INPUT:?set INPUT to a real-input capture directory}" --configs previous,production --prefill
elif [[ ${PHASE:-sweep} == e2e ]]; then
  for config in ${CASES:-baseline tc_64_64_128_3_4}; do
    "$PY" "$SRC/bench/prefill_grouped_study.py" e2e --config "$config" --out "$OUT/$config" ${ARGS:-}
  done
else
  "$PY" "$SRC/bench/prefill_grouped_study.py" "${PHASE:-sweep}" --out "$OUT" ${ARGS:-}
fi
