# Attention / shared expert + native GGML IQ search

## Scope and formats

`e2e_kl` and `tensor_proxy` discover actual model weights. Routed experts retain
the default per-3D decision groups (`L0_gate_exps`, `L0_up_exps`,
`L0_down_exps`). Each attention/shared-expert Linear has its own 2D decision,
for example `L0.self_attn.q_proj` or `L0.mlp.shared_expert.up_proj`.
Hybrid `linear_attn` and shared ModuleLists are included. Routers, embeddings,
normalization, convolution and other non-Linear weights are excluded.

Options use concrete tensor types: `IQ4_XS_rank16` and `IQ4_NL_rank16` are distinct.
Historical `3bit_rank16` configs remain supported by compression and legacy search.

Native formats: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`,
`Q4_1`, `Q5_0`, `Q5_1`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
`IQ3_S`, `IQ4_XS`, `IQ4_NL`, `F16`, `F32`.
Model-wide presets such as `Q4_K_M` are not tensor types and are rejected.

| Type | Values/block | Bytes/block | Effective bpw |
|---|---:|---:|---:|
| IQ2_XS | 256 | 74 | 2.3125 |
| IQ3_S | 256 | 110 | 3.4375 |
| IQ4_XS | 256 | 136 | 4.25 |
| IQ4_NL | 32 | 18 | 4.5 |

Costs use these real block bytes plus FP16 U/V bytes, with rank clipped to the
matrix dimensions. Incompatible row widths mask a candidate; no hidden padding
or substitution occurs. Keep `F16_rank0` available for unaligned projections.
The budget is **selected weight + compensator payload MiB**, not total model
size, GPU memory, KV cache, or inference latency. Biases and fixed parameters
are outside it. `--avg_bits` converts this same selected element count to a budget.

## Build the native codecs

From the repository root, in the Python environment used for BR-MoE:

```bash
python tools/build_ggml_quant.py
```

Requires CMake, a C/C++ compiler and network access for initial source download.
The bridge uses `ggml_quantize_chunk` and the upstream decoders from llama.cpp
revision `ceefd16daa2093730bf3340315fa2b84b286ddbc`. Only GGML base sources/headers
are downloaded, with their LICENSE, into `~/.cache/brmoe-ggml/<revision>`.
Use `--no-proxy` if the environment's proxy is unavailable. A custom build can
be selected with `BRMOE_GGML_QUANT_LIB=/absolute/path/to/libbrmoe_ggml_quant.so`
(`.dylib` on macOS). Runtime never downloads or compiles implicitly.

Named formats use the same native encoder in search and compression. The older
integer-bit fast Torch path remains available for compatibility; it is not
used by the new named-format CLI search.

## Search

```bash
python -m RCO_solver.rco_search \
  --mode e2e_kl \
  --model_path /path/to/model \
  --dataset_path /path/to/calibration.jsonl \
  --granularity tensor \
  --quant_types Q2_K,Q3_K,IQ2_XS,IQ3_S,IQ4_XS,IQ4_NL,F16 \
  --ranks 0,16,32,64 \
  --avg_bits 4.0 \
  --num_calib_samples 16 --seq_len 256 --steps 100 \
  --output_file results/best.json
```

JSONL records need a `text` field. TXT (one sample per line) and Parquet (`text`
column; requires pandas/Parquet engine) are also accepted. Sequences are not
padded during statistics collection. A too-small budget reports the actual
minimum before optimization. `--skip_attention` and `--skip_shared_expert`
exclude those categories. `--granularity expert_tensor` allows different expert
choices, but exporting conflicting expert types into one GGUF 3D tensor fails
explicitly; the default `tensor` avoids this issue.

IQ2_XS/IQ2_XXS require input-channel importance. The CLI collects mean squared
input activations and saves `results/best.imatrix.npz`, keyed by full module
name. If an expert receives no calibration tokens, required-imatrix candidates
are masked for its entire 3D group. Zero-mass statistics count as unavailable.
No uniform statistics are fabricated. `--imatrix_file` reuses an existing NPZ;
this is a BR-MoE calibration file, **not** llama.cpp's binary imatrix format.
Direct API callers may explicitly provide uniform importance if desired.

Outputs:

- `best.json`: unit → concrete format/rank.
- `best_tensor_allocation.json`: module names, shapes, categories, effective
  ranks, actual payload bytes, calibration file reference and search history.
- `best.imatrix.npz`: statistics, when collected or provided.

Use `--mode tensor_proxy` with the same arguments for a cheaper model-derived
proxy: squared residual weight error, weighted by available input-channel
importance, with the same codec and prefix SVD candidates. This is a proxy,
not end-to-end KL. Existing `proxy`/`int3_codesign` modes retain their historical
expert-statistics-table scope and do not infer attention/shared errors from
those tables. Use the two new full-tensor paths for this task.

Native encoding runs on CPU and can use disjoint-row workers with
`BRMOE_QUANT_THREADS=8` (default: 1). CUDA offloaded search now transfers packed
blocks and decodes Q2_K/Q3_K/Q4_K and all seven supported IQ types with Triton.
The codebooks come from the same pinned GGML library. Rebuild the native bridge
when upgrading from the CPU-only implementation. Other registered types retain
the native CPU decode path in auto mode.

`BRMOE_RCO_DECODE_BACKEND=auto` selects GPU decode where supported; `cpu` retains
the reference path, and `gpu` fails if a selected format has no GPU decoder.
`BRMOE_RCO_GPU_CACHE_GB=4` controls the byte-bounded LRU for packed CUDA bases and
U/V factors (0 disables retention). Decoded dense matrices are temporary, rather
than retained for every rank/format. Original weights move to CUDA only when
needed for a new SVD or an identity candidate. BF16 Triton output requires sm80+;
FP16 and FP32 are also supported on sm75. Large models still require substantial
host RAM and first-use encoding/SVD time.

See [GPU decode measurements](rco_gpu_decode_20261022.md) for the tested scope.
Search uses randomized maximum-rank SVD prefixes; compression refits the chosen
rank, so the complete compensated result can differ numerically even when the
base quantized bytes match. Re-evaluate the final compressed model.

## Compress and export

```bash
python examples/BR_compress_auto.py \
  --model_path /path/to/model \
  --save_dir results/compressed \
  --config results/best_tensor_allocation.json \
  --device cuda
```

The manifest supplies per-module types/ranks and the importance file. Modules
outside its selected scope stay uncompressed. Passing `best.json` also detects
its matching sibling manifest automatically. Keep all output files together,
or pass `--imatrix_file` explicitly. Standalone compression accepts named types
in `--attn_bits`, `--shared_bits`, and `--default_bits` as well.

The GGUF exporter now preserves named IQ types in its quantization recipe and
exports 2D attention/shared compensators in addition to 3D routed compensators.
For example, for a Qwen3 MoE checkpoint:

```bash
python tools/export_brmoe_to_gguf.py \
  --brmoe_dir results/compressed --output_dir results/gguf \
  --arch qwen3moe --tensor_alloc_path results/best_tensor_allocation.json
```

Its low-level writer accepts exact native IQ payloads with shape/byte validation.
Standard attention/shared names are built in; architecture-specific hybrid
names use the optional official `gguf` package. Unknown mappings fail explicitly.

This exporter produces a LoRA adapter and a base-model quantization recipe;
it is not a complete HF-to-GGUF base-model converter. Re-quantizing a base model
externally can use different importance statistics and weights, so a recipe
alone does not guarantee byte-for-byte parity with the saved checkpoint.
IQ format support here does not add IQ kernels to the custom INT3 vLLM backend.
End-to-end llama.cpp loading/performance must be validated for the target
architecture and llama.cpp build.

## Verification

After building the bridge, install `pytest` and `gguf` in the test environment:

```bash
python -m pytest -q tests/test_full_tensor_iq.py RCO_solver/verify_and_benchmark.py \
  tests/test_mixed_vllm_e2e.py::test_a4_gguf_kquant_alignment_and_residual_svd
```

Tests compare native bytes against the independent official Python decoders,
exercise serialized types/ranks and restored compensation, cover calibration
masking, joint KL/proxy search, and read generated IQ/2D-adapter GGUF files.
Native integration tests skip with an explicit message if the bridge is absent.
