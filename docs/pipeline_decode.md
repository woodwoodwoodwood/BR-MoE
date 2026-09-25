# Decode 算子流程（小 batch, vLLM + brmoe_int3）

decode 每步每层的真实执行路径（`tools/brmoe_int3_vllm/` + `BR-MoE/kernels/triton_int3/int3_moe/`）。
路径分派阈值：A100 (sm_80) GEMV 用于 M≤4，5090 (sm_120) 用于 M≤8，超出走张量核心（TC）路径。

```mermaid
flowchart TD
    H["hidden states<br/>[M, 2048] fp16"] --> QKV

    subgraph ATTN["Attention（每层 4 个 int3 linear: q/k/v/o）"]
        QKV{"M ≤ 8 ?"}
        QKV -->|是| LGV["routed_int3_gemv<br/>单专家视角 + split-K 拉满<br/>K=2048 拆 16 段, fp32 atomic 归约<br/><i>5090: 4.3us, 比 cuBLAS fp16 快 0.74x</i>"]
        QKV -->|否| LTC["int3_moe_gemm (TC)<br/>block_m 自适应 16/32/64<br/>num_warps=4 num_stages=3"]
        LGV --> ATTO["FlashAttention + o_proj"]
        LTC --> ATTO
    end

    ATTO --> MOE

    subgraph MOE["MoE（64 专家, top-6, 3-bit 打包权重 K-major）"]
        RTR["router (fp16 gate)<br/>topk_weights / topk_ids"] --> SAN["_sanitize_routing<br/>-1 槽位 -> 专家0+权重清零<br/>(全程张量算子, CUDA Graph 安全)"]
        SAN --> DISP{"M ≤ 阈值 ?<br/>sm_80: 4 / sm_120: 8"}

        DISP -->|是: GEMV 路径| G1["routed_int3_gemv (w13)<br/>逐路由对 GEMV, 无 padding<br/>split-K: M=1 拆 8 段<br/>-> inter32 [fp32, atomic]"]
        G1 --> SILU1["silu_mul_routes"]
        SILU1 --> G2["routed_int3_gemv (w2)<br/>add=True, 乘路由权重<br/>-> out32 [fp32, atomic]"]

        DISP -->|否: TC 路径| AL["moe_align_block_size_triton<br/>按专家排序+对齐 (slot=16)<br/>零 host 同步"]
        AL --> T1["int3_moe_gemm (w13)<br/>grouped GEMM, tl.dot<br/>slot16/block_m16"]
        T1 --> SILU2["silu_mul"]
        SILU2 --> T2["int3_moe_gemm (w2)<br/>add=True -> out32"]

        G2 --> OUT["out fp16<br/>[M, 2048]"]
        T2 --> OUT
    end

    OUT --> NEXT["下一层 (x28) -> lm_head -> sample"]
```

## 关键设计

| 组件 | 文件 | 说明 |
|---|---|---|
| 权重布局 | `moe_method.py::process_weights_after_loading` | 加载后转置成 K-major `[E, Kpack, N]`，GEMV 的 w 载入沿 n 连续 → 合并访存（+5~19%）|
| GEMV split-K | `int3_moe/kernel.py::_routed_int3_gemv` | grid 第三维拆 K，部分和 fp32 atomic；小 M 下把 program 数从几百提到几千 |
| 自动分派 | `int3_moe/ops.py::fused_moe_int3` | 按 (架构, M) 选 GEMV/TC；阈值是 e2e 实测定的（随机路由 micro 会高估 GEMV 的适用范围）|
| 数值 | `bench/micro_moe.py --check` | 三条路径都与反量化金标准逐层比对（rel < 5e-4）|
