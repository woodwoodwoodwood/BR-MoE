# Qwen1.5-MoE：PIQA 56019 使用的压缩配置

来源：`best_tensor_allocation.json`，已验证其 SHA256 与 `compressed/COMPRESSION_COMPLETE.json` 完全一致。

层编号从 0 开始，共 24 层、240 个决策单元。格式 `IQ3_S/r16` 表示 IQ3_S 量化 + rank=16 的低秩残差补偿；r0 表示没有补偿。补偿矩阵保存为 FP16。

路由专家以层和投影类型为决策单元：同层的 60 个专家共用一种量化格式及 rank，各自的权重独立量化。Attention 和共享专家的每个二维投影矩阵独立选择配置。

Embedding、lm_head、router 和 LayerNorm 等未纳入搜索的参数保持 FP16。

## 分组统计

| 类别 | 决策数 | 格式分布 | rank 分布 | 所选载荷（GiB） |
|---|---:|---|---|---:|
| Attention | 96 | IQ2_S: 7, IQ2_XS: 7, IQ2_XXS: 1, IQ3_S: 13, IQ3_XXS: 19, IQ4_NL: 7, IQ4_XS: 15, Q2_K: 8, Q3_K: 10, Q4_K: 9 | r0: 47, r16: 22, r32: 20, r64: 7 | 0.172211 |
| 路由专家 | 72 | IQ2_S: 4, IQ2_XS: 3, IQ2_XXS: 37, IQ4_NL: 24, Q2_K: 4 | r0: 72 | 4.270020 |
| 共享专家 | 72 | IQ2_S: 6, IQ2_XS: 4, IQ2_XXS: 5, IQ3_S: 15, IQ3_XXS: 17, IQ4_NL: 1, IQ4_XS: 4, Q2_K: 7, Q3_K: 12, Q4_K: 1 | r0: 41, r16: 15, r32: 11, r64: 4, r128: 1 | 0.314995 |

## 路由专家：全部 24 层

| 层 | gate_proj | up_proj | down_proj |
|---|---|---|---|
| 0 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 1 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 2 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 3 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 4 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 5 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 6 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 7 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 8 | IQ2_XXS/r0 | IQ2_XS/r0 | IQ4_NL/r0 |
| 9 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 10 | IQ2_S/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 11 | IQ2_XXS/r0 | IQ2_S/r0 | IQ4_NL/r0 |
| 12 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 13 | IQ2_XS/r0 | Q2_K/r0 | IQ4_NL/r0 |
| 14 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 15 | IQ2_S/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 16 | IQ2_XXS/r0 | Q2_K/r0 | IQ4_NL/r0 |
| 17 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 18 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 19 | IQ2_XXS/r0 | IQ2_S/r0 | IQ4_NL/r0 |
| 20 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |
| 21 | IQ2_XXS/r0 | Q2_K/r0 | IQ4_NL/r0 |
| 22 | Q2_K/r0 | IQ2_XS/r0 | IQ4_NL/r0 |
| 23 | IQ2_XXS/r0 | IQ2_XXS/r0 | IQ4_NL/r0 |

## Attention：全部 24 层

| 层 | q_proj | k_proj | v_proj | o_proj |
|---|---|---|---|---|
| 0 | IQ2_XXS/r0 | Q2_K/r0 | Q2_K/r0 | IQ3_XXS/r16 |
| 1 | IQ2_S/r0 | IQ2_S/r0 | IQ3_XXS/r32 | IQ3_S/r64 |
| 2 | IQ2_S/r0 | IQ3_S/r0 | IQ4_XS/r16 | IQ4_NL/r16 |
| 3 | IQ2_S/r0 | IQ2_S/r32 | IQ4_XS/r32 | IQ4_XS/r32 |
| 4 | IQ3_XXS/r16 | Q2_K/r64 | Q4_K/r64 | Q3_K/r32 |
| 5 | Q3_K/r32 | IQ3_S/r32 | IQ3_S/r64 | IQ3_XXS/r64 |
| 6 | Q3_K/r0 | Q2_K/r0 | IQ4_XS/r16 | Q4_K/r0 |
| 7 | Q3_K/r0 | IQ3_S/r0 | IQ4_XS/r0 | Q3_K/r32 |
| 8 | IQ3_XXS/r16 | IQ3_XXS/r32 | IQ4_XS/r0 | IQ4_NL/r0 |
| 9 | IQ3_XXS/r16 | IQ3_XXS/r16 | IQ4_XS/r16 | IQ4_XS/r16 |
| 10 | IQ3_S/r0 | IQ2_XS/r0 | IQ3_S/r16 | Q4_K/r32 |
| 11 | IQ4_NL/r32 | IQ3_XXS/r0 | IQ4_XS/r0 | IQ4_XS/r16 |
| 12 | IQ3_XXS/r32 | IQ3_XXS/r16 | IQ4_XS/r0 | IQ3_S/r32 |
| 13 | IQ3_XXS/r0 | IQ4_XS/r0 | IQ4_NL/r32 | IQ4_XS/r0 |
| 14 | IQ4_XS/r16 | IQ4_NL/r16 | Q4_K/r0 | IQ3_S/r0 |
| 15 | Q2_K/r0 | IQ3_XXS/r0 | Q4_K/r32 | IQ4_NL/r0 |
| 16 | IQ3_XXS/r32 | IQ3_XXS/r0 | Q4_K/r16 | IQ3_XXS/r0 |
| 17 | Q4_K/r0 | Q3_K/r0 | IQ3_S/r64 | IQ3_S/r0 |
| 18 | IQ2_XS/r0 | IQ3_S/r0 | IQ2_XS/r32 | Q4_K/r0 |
| 19 | IQ2_XS/r16 | IQ2_XS/r0 | Q3_K/r0 | Q3_K/r0 |
| 20 | Q2_K/r16 | IQ2_S/r0 | IQ3_XXS/r32 | IQ2_XS/r32 |
| 21 | Q2_K/r0 | IQ2_S/r0 | IQ4_XS/r0 | IQ3_XXS/r16 |
| 22 | IQ2_XS/r16 | Q2_K/r16 | Q4_K/r0 | Q3_K/r32 |
| 23 | IQ3_XXS/r16 | Q3_K/r0 | IQ4_NL/r0 | IQ3_S/r64 |

## 共享专家：全部 24 层

| 层 | gate_proj | up_proj | down_proj |
|---|---|---|---|
| 0 | IQ2_S/r16 | IQ2_XXS/r32 | IQ2_XXS/r0 |
| 1 | IQ2_XXS/r0 | IQ2_S/r64 | IQ2_XS/r32 |
| 2 | IQ2_XS/r0 | IQ3_XXS/r32 | Q2_K/r64 |
| 3 | Q3_K/r0 | Q2_K/r16 | IQ2_S/r16 |
| 4 | IQ2_S/r0 | Q2_K/r64 | Q2_K/r32 |
| 5 | IQ2_XXS/r0 | IQ2_XS/r16 | IQ3_S/r32 |
| 6 | Q2_K/r32 | IQ3_XXS/r0 | IQ3_XXS/r0 |
| 7 | IQ3_S/r16 | Q2_K/r64 | IQ3_XXS/r128 |
| 8 | Q3_K/r32 | IQ3_S/r0 | IQ3_XXS/r32 |
| 9 | Q3_K/r0 | IQ3_XXS/r16 | IQ3_XXS/r16 |
| 10 | Q3_K/r0 | IQ3_S/r0 | IQ3_XXS/r0 |
| 11 | IQ3_XXS/r16 | IQ3_S/r0 | Q3_K/r0 |
| 12 | IQ3_XXS/r0 | Q3_K/r0 | Q4_K/r0 |
| 13 | Q3_K/r32 | Q3_K/r32 | IQ3_S/r0 |
| 14 | IQ3_S/r0 | IQ3_S/r16 | IQ3_S/r16 |
| 15 | Q3_K/r0 | IQ3_S/r0 | IQ4_NL/r0 |
| 16 | IQ3_S/r0 | IQ3_S/r0 | IQ3_S/r0 |
| 17 | Q2_K/r16 | IQ3_S/r16 | IQ2_S/r0 |
| 18 | IQ3_XXS/r16 | IQ3_XXS/r32 | IQ3_XXS/r0 |
| 19 | IQ2_XS/r16 | IQ2_S/r0 | IQ3_S/r0 |
| 20 | IQ3_XXS/r0 | IQ3_XXS/r0 | Q3_K/r0 |
| 21 | IQ2_XXS/r0 | Q3_K/r0 | Q3_K/r0 |
| 22 | IQ3_XXS/r0 | IQ4_XS/r0 | IQ3_XXS/r0 |
| 23 | IQ4_XS/r0 | IQ4_XS/r0 | IQ4_XS/r16 |
