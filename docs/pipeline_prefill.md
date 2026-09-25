# Prefill 算子流程（大 batch / 长 prompt, vLLM + brmoe_int3）

prefill 时 M 大（chunked prefill, `max_num_batched_tokens=2048`），全部走张量核心（TC）
grouped GEMM 路径——GEMV 的逐路由对读取在这个区间没有任何优势。

```mermaid
flowchart TD
    P["prompt tokens<br/>in=128 x bs, chunk <= 2048"] --> EMB["embed + 每层循环 (x28)"]

    EMB --> ATTN

    subgraph ATTN["Attention（int3dense: 全 int3）"]
        AQ["q/k/v/o proj<br/>int3_moe_gemm 单专家视角<br/>block_m 按 M 自适应 (64..)<br/>prefill 时 padding 浪费 ≈ 0"] --> AF["FlashAttention (prefill)"]
    end

    ATTN --> MOE

    subgraph MOE["MoE（64 专家全激活, top-6）"]
        MR["router -> topk<br/>_sanitize_routing"] --> MA["moe_align_block_size_triton<br/>按专家排序, slot=64 对齐<br/>(M 大时 padding 占比小)"]
        MA --> M1["int3_moe_gemm (w13)<br/>grouped GEMM tl.dot<br/>block_m=64 block_n=64<br/>num_warps=2 stages=1"]
        M1 --> MS["silu_mul"]
        MS --> M2["int3_moe_gemm (w2)<br/>add=True"]
        M2 --> MO["out fp16"]
    end

    MO --> NX["下一层 -> ... -> lm_head -> 首 token"]
```

## 现状与已知边界（A100 实测，2026-09-26）

- prefill 是**算力受限**场景：权重被反复复用，3-bit 省字节的优势天然减弱
- int3 TC kernel 在 M=512/2048 只有 10.8% / 3.1% 峰值带宽；cuBLAS fp16 参照也只有
  51.5% / 23.5% —— 这个区间两家都远离带宽墙，瓶颈在反量化 ALU 与 MMA 的排布
- 大 M 下 `slot=64/block_m=64` 反而最优（padding 占比小，大 tile 喂得饱张量核心）
- 已知的下一步候选：int4d 布局（去掉 `tl.where` 解包链）在 TC 路径的 M>=16 区间
  尚未充分验证；`block_n=256` 在 M>=512 会出现灾难性回退（25ms, 勿用）

## 一次请求的完整生命周期

```mermaid
flowchart LR
    REQ["请求"] --> PF["prefill<br/>(本页: TC 路径)"]
    PF --> DC["decode 逐 token<br/>(见 pipeline_decode.md:<br/>GEMV + split-K)"]
    DC --> EOS["EOS / 长度上限"]
```
