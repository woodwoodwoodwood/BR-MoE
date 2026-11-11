# RCO GPU decoding and native encoding, 2026-10-22

This change preserves the pinned GGML encoders and format/rank search space.
It replaces repeated CPU decoding and full FP32 host-to-device copies with
Triton GPU decoding of the same packed bytes. Supported GPU types are Q2_K,
Q3_K, Q4_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL;
F16/F32 identity candidates skip redundant native encoding. Other registered
formats use CPU decoding in auto mode. GPU codebooks are exported directly
from the pinned native bridge, so rebuilding `tools/build_ggml_quant.py` is required.

## Repeated search work

- Original CPU weights move to CUDA only for a new residual SVD or identity
  candidate, rather than every candidate lookup.
- A shared 4 GiB LRU bounds retained packed CUDA bases and low-rank factors.
  Dense decoded weights are temporary. `BRMOE_RCO_GPU_CACHE_GB=0` disables retention.
- Candidate-score gradients stay on CUDA. Prefix corrections for all ranks are
  gathered together, removing per-format `.item()` and per-option scalar stores.
- Discrete choices transfer to the host once per step rather than synchronizing
  once per projection.
- CPU encoding can use disjoint row chunks through the upstream native API.
  Upstream initializes its read-only tables under a critical section; each
  worker writes a separate output row range. `BRMOE_QUANT_THREADS` defaults to 1.

No new IQ fused inference GEMM kernel is introduced. GPU decoding materializes
one dense candidate at a time for the existing PyTorch matmuls.

## Validation and measurements

Slurm job 54713 on ada-server, RTX 2080 Ti, passed 44 tests: native/GPU exact
FP16/FP32 decode parity across random and zero inputs, CPU/GPU RCO gradient
parity, scalar/vectorized rank-gradient parity, bounded-cache eviction, and
the existing IQ/RCO integration checks. BF16 output requires compute capability
8.0 or newer and was not tested on this sm75 card.

The 30 real-projection measurements use Qwen1.5-MoE attention, shared and routed
weights. Timings are medians of 10 synchronized iterations after warmup. Native
CPU encoding and GPU compilation are excluded. The CPU comparison decodes the
packed buffer to an FP32 host matrix and transfers it as CUDA FP16; the GPU
comparison transfers packed bytes on every iteration and then decodes them.
The resident-buffer measurement excludes packed-byte transfer.

Representative attention q_proj [2048, 2048] timings:

| Format | CPU decode + transfer (ms) | GPU decode + transfer (ms) | Ratio |
|---|---:|---:|---:|
| IQ2_XS | 23.52 | 0.434 | 54.1x |
| IQ3_S | 23.02 | 0.494 | 46.6x |
| IQ4_XS | 11.83 | 0.555 | 21.3x |

These ratios describe this stage under this job's resource allocation. They
are not an end-to-end RCO speedup or a prediction for a different GPU. The
larger shared projection had substantially higher CPU-path latency on this
node; retain the raw measurements when interpreting those ratios.

Raw data: `artifacts/runs/iq_gpu_54713.json`. Benchmark:
`bench/iq_decode_bench.py`.

Slurm job 54715 separately passed 11 cache/parallel-encoding checks. On the
real routed expert up projection [1408, 2048], eight encoder workers were
3.22–3.72x faster across ten IQ/K formats (three repetitions, warm codebooks),
with byte-identical output against one worker. For IQ2_XS the median changed
from 2.863 s to 0.781 s. Data: `artifacts/runs/iq_encode_54715.json`;
reproduction: `bench/slurm/iq_encode_verify.slurm`. Initialization of all model
candidates can still take hours; steady-state decode ratios do not remove it.

## Full search configuration

`bench/slurm/rco_full_e2e.slurm` runs the full Qwen1.5-MoE model with an average 3-bit
selected-payload budget, 128 calibration sequences of at most 256 tokens,
300 KL steps, and ranks 0/16/32/64/128. It requests one GPU in the A100/H200/5090 partitions, 8 CPU cores,
160 GiB host RAM and a 96-hour limit. It uses eight native encoder workers,
GPU decoding, and a 4 GiB GPU cache. Router, embedding and normalization weights
remain outside the allocation budget.

The source is frozen in a supplied tar archive. Progress includes reference
forward counts, initialized candidates, per-step wall time, rolling ETA and
GPU cache statistics. Every ten steps (also step 1 and the final step), it
saves alpha/optimizer/RNG snapshots and the last sampled feasible configuration.
These snapshots do not yet provide an automatic resume command. The final
allocation comes from the final no-noise DP selection, not the sampled file.

## H200 residency configuration (2026-10-24)

The full-search Slurm script now targets one H200 with 80 GiB host RAM.
`BRMOE_RCO_KEEP_ORIGINAL_ON_GPU=1` keeps the searchable original weights on
CUDA, and `BRMOE_RCO_KEEP_FACTORS_ON_GPU=1` keeps each format's maximum-rank
U/V factors on CUDA. Both flags default to 0 for smaller GPUs. Packed native
bases are still encoded and retained on CPU, with a 48 GiB CUDA packed-buffer
cache. Resident U/V tensors are not counted again in that LRU.

For the current Qwen1.5-MoE candidate set, header-derived estimates are:
25.5 GiB original searched weights, 27.2 GiB U/V factors and 38.5 GiB packed
bases, plus about 1.2 GiB fixed GPU weights. Thus final GPU payload residency
is about 92.4 GiB before activations, temporary decoded matrices and SVD work.
Host payload caches are about 47.8 GiB (packed bases + 9.3 GiB reference logits),
before the interpreter, encoder buffers and allocator overhead. These are
estimates rather than measured peaks. The probed H200 NVL reports 143771 MiB
of total memory. The runner logs peak host RSS and allocated/reserved CUDA
memory to check actual growth during reference forwards and candidate setup.
