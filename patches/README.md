# llama.cpp patches

## Expert LoRA in the CUDA decode GEMV

`llama-cpp-expert-lora-mmvq.patch` folds an expert LoRA
(`mul_mat_id(W)`, `mul_mat_id(A)`, `mul_mat_id(B)`, `scale`, `add`)
into `ggml_cuda_mul_mat_vec_q` for single-token decode.
`h = Aᵀx` is accumulated while the activation is quantized, then each output row adds `scale * B_row · h`.
When gate and up have the same type, their LoRA epilogues stay inside the fused SwiGLU GEMV.

The patch applies to the llama.cpp tree used for the Qwen1.5-MoE-A2.7B GGUF runs
(the 2026-09-07 snapshot that already contains `GGML_TYPE_GSQ2`).
From that tree:

```bash
patch -p1 < /path/to/BR-MoE/patches/llama-cpp-expert-lora-mmvq.patch
```

`GGML_CUDA_DISABLE_LORA_FUSION=1` keeps the unfused graph.
Prefill (`n_tokens > 1`) is unchanged.

On one H20, a 2048-token prompt and 128 generated tokens:

| setup | decode |
|---|---|
| quantized experts, no compensator | 249.7 t/s |
| compensator, unfused | 166.3 t/s |
| compensator, this patch | 216.7 t/s |
