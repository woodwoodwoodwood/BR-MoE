# int3 Grouped MoE CUDA Kernel 设计（tile 级融合，Marlin 式流水）

> 目标区间: decode bs≥16 与 prefill（M≥16），即 Triton TC 路径（`int3_moe_gemm`）
> 目前只有 11~34% 峰值带宽的区间。小 M 已由 GEMV+split-K 覆盖（5090 上 98% 峰值），
> 本 kernel 不碰那个区间。
>
> 讨论过的备选及否决原因:
> - 两段式（独立反量化 + cuBLAS fp16）: 放弃 tile 级融合, 多付 10 倍 HBM 流量
>   (读 int3 + 写 fp16 + 读 fp16 = 231MB vs 22.7MB), 天花板低 10 倍。否决。
> - 给 vLLM marlin_moe_wna16 加 3-bit 标量类型: 模板 87KB, 要进 vLLM 源码树构建,
>   标量类型系统假设 2 的幂 pack_factor, 3-bit 是异形。否（作为索引逻辑的参照保留）。

## 路线

扩展 `BR-MoE/kernels/brmoe/brmoe_cuda_with_zero_kernel.cu`（Marlin 改版的 int3 + fp16
零点，已是 tile 级融合 + `cp.async` 多级流水 + `mma.sync` + split-K 锁），加上
grouped MoE 调度。新 kernel 放在 `BR-MoE/kernels/marlin_int3_moe/`。

## 调度模型（对齐 vLLM marlin_moe 的约定）

```
grid = (num_blocks_m_max, N / BLOCK_N, [split-k])   # 超发启动
block (pid_m):
    if pid_m * BLOCK_M >= num_post: return          # META 早退, 零 host 同步
    expert = expert_ids[pid_m]                       # 一个 block 一个专家 (SLOT=BLOCK_M)
    B 基址 = Wq + expert * stride_e                  # 唯一新增的索引逻辑
    A 行    = sorted_token_ids[pid_m*BLOCK_M + j]    # gather (第一层) / 连续 (第二层)
    C 行    = sorted 行号 (第一层, 直写) / token 号 (第二层, atomic_add 乘路由权重)
```

元数据直接复用 Triton align（`align_triton.py`）的输出：`sorted_token_ids` /
`expert_ids` / `num_post` —— 约定与 marlin_moe 完全一致，无需新写 align。

## 权重布局（converter 侧新做一个 repacker）

brmoe kernel 的 B 是 Marlin 交错布局（ldmatrix 友好），与 Triton 路径的
`[E, N, Kpack]` 不同。repacker 按专家逐个做 Marlin 重排后 stack 成
`[E, marlin_layout]`，外加 per-expert scale/zero 重排。零点折叠: `z' = -z * s`
（CUDA 侧 `w = __hfma2(q, s, z')`）。

## 接口契约（与 Triton 路径逐项对应）

| 项 | Triton 路径 | 本 kernel |
|---|---|---|
| 打包 | 32 值 / 3 int32 交错 | Marlin 交错布局 (brmoe 同款) |
| 反量化 | `tl.where` 链 + `(q-z)*s` | lop3 + `__hfma2(q, s, z')` |
| 零点 | fp16 per-group, 减法 | fp16 per-group, 折叠为偏移 |
| align | Triton, 零 host 同步 | 复用同一份元数据 |
| split-K | grid 第三维 + fp32 atomic | brmoe 自带的 lock 屏障 |
| 校验 | `bench/micro_moe.py --check` 金标准 | 同一金标准, rel < 5e-4 |

## 里程碑

1. [ ] skeleton 独立构建通过（逻辑未改, 先证明构建链）
2. [ ] grouped 索引（expert_ids 寻址 + sorted gather/scatter）
3. [ ] repacker + 单层数值对齐（vs 反量化金标准）
4. [ ] 双投影 + silu + atomic reduce 接入 `fused_moe_int3` 的平行路径
5. [ ] micro 带宽达标（目标: M=16/32 时 ≥60% 峰值, 现 TC 为 33%）
6. [ ] vLLM 插件分派（M≥16 走 CUDA）+ e2e

## 风险

- brmoe kernel 的 B 加载循环假设单矩阵连续地址; per-block 专家基址改动会碰到
  `cp.async` 预取指针的生成逻辑 —— 需要保证 stage 流水在块内完整（Marlin MoE 同款处理:
  每块开头重建流水）
- sm_120 (5090) 与 sm_80 (A100) 的 `cp.async`/`mma` 都兼容 (brmoe kernel 已两卡跑过)
- 数值: lop3 解包与 Triton 的位序必须逐位一致, 用金标准卡死
