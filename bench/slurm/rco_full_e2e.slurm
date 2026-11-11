#!/bin/bash
#SBATCH --job-name=brmoe-rco-e2e
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/rco_e2e_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/artifacts/runs/rco_e2e_%j.err
set -euo pipefail
BASE=/mnt/709/data3/home/jianglei
REPO=$BASE/ada/BR-MoE
PY=$BASE/miniconda3/envs/vllm5090/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export BRMOE_RCO_DECODE_BACKEND=gpu BRMOE_RCO_GPU_CACHE_GB=48
export BRMOE_QUANT_THREADS=8
export BRMOE_RCO_KEEP_ORIGINAL_ON_GPU=1 BRMOE_RCO_KEEP_FACTORS_ON_GPU=1
export TRITON_CACHE_DIR=$REPO/artifacts/runs/triton_iq_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BRMOE_GGML_QUANT_LIB=$BASE/.cache/brmoe-ggml/ceefd16daa2093730bf3340315fa2b84b286ddbc/build/libbrmoe_ggml_quant.so
export RCO_RUN_DIR=$REPO/artifacts/runs/rco_e2e/run_${SLURM_JOB_ID}
mkdir -p "$RCO_RUN_DIR"
SRC=$RCO_RUN_DIR/source
mkdir -p "$SRC"
tar -xf "${RCO_SOURCE_ARCHIVE:?set a frozen source archive}" -C "$SRC"
cd "$SRC"
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" -c 'from examples.BR_compress_auto import load_hf_moe_model; from RCO_solver.full_model_search import run_cli; print("RCO_ENTRY_IMPORT_OK")'
"$PY" -m pytest -xq tests/test_iq_gpu.py tests/test_full_tensor_iq.py::test_real_hf_checkpoint_pipeline
"$PY" bench/rco_full_e2e.py \
 --mode e2e_kl \
 --model_path /mnt/4090/data/jianglei/models/Qwen/qwen1.5-moe \
 --dataset_path "$SRC/datasets/gsq_calib_no_marvis/n128/data.jsonl" \
 --granularity tensor \
 --quant_types Q2_K,Q3_K,Q4_K,IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S,IQ4_XS,IQ4_NL,F16 \
 --ranks 0,16,32,64,128 --max_rank 128 \
 --avg_bits 3.0 --num_calib_samples 128 --seq_len 256 \
 --steps 300 --lr 0.1 --seed 42 \
 --output_file "$RCO_RUN_DIR/best.json"
