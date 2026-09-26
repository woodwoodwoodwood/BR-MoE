# CUDA MoE 外围 kernel fusion 实测

## 实现与开关

本次保留原有 Marlin INT3 权重布局、CUDA 矩阵乘和默认 FP16 中间写出精度，
融合矩阵乘前后的操作。代码位于 `BR-MoE/kernels/marlin_int3_moe/fused_ops.py`。

```bash
export BRMOE_CUDA_FUSE=1
```

默认关闭；开启后：

- gather 在一个 Triton kernel 内完成索引、输入读取、padding 清零，只处理有效专家块。
- SiLU、乘 up、FP32/FP16 转换在一个 kernel 内完成，跳过无效缓冲范围。
- align 同时写入 `route_pos[original_route] = sorted_row`，无需单独启动反向索引 kernel。
- 最后按 token gather 各 top-k 路由结果，在一个 kernel 内完成 FP32 加权求和与输出类型转换。
  输出每个元素只有一个写者，省去原 scatter 的输出清零、原子累加及单独转换。
- sm_120 在开启融合时，M<=2 保留原 GEMV，4/8/16/32 等形状走融合 CUDA 路径；
  插件原有 M<=512 的 CUDA 上限继续生效，M>512 使用原 Triton grouped GEMM。
- 两个实验开关都开启且 CUDA 可用时，`BRMOE_CUDA_FUSE` 优先于 `BRMOE_GROUPED_GEMV`。
- `BRMOE_CUDA_FUSE=atomic` 保留第一版融合加权 scatter，用于对照。

所有有效长度与路由索引都留在 GPU 上，每次 CUDA Graph replay 重建，保持原工作区的单流使用约定。
split-K>1 仍保留 CUDA 矩阵乘内部的 FP32 原子归约；本次去掉的是最终 top-k scatter 的原子操作。

## 5090：完整 MoE 与端到端

全 INT3 checkpoint 为 `brmoe-3bit-vllm-int3dense`，本次只修改 routed MoE。
端到端同卡先后运行基线与融合版本：128 输入 / 128 输出，CUDA Graph，关闭 prefix cache，
每组重复 3 次，TPOT 为 `(E2E-TTFT)/127`。基线为两个实验开关均关闭的原 CUDA/Triton 分派。

| batch | 27 层完整 MoE 基线 ms | 融合 ms | MoE 降幅 | TPOT 基线 ms | 融合 TPOT ms | TPOT 降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.393 | 0.919 | 34.0% | 3.555 | 3.044 | 14.4% |
| 8 | 2.762 | 0.926 | 66.5% | 5.896 | 3.975 | 32.6% |
| 16 | 2.600 | 0.965 | 62.9% | 7.029 | 5.551 | 21.0% |
| 32 | 3.815 | 1.781 | 53.3% | 9.859 | 8.743 | 11.3% |

微观回放使用前次采集的 108 组真实输入（各 batch 的最后一个 decode step、每组 27 层），
依次执行所有层及其真实权重，包含 align、gather、两次矩阵乘、激活和归约。
端到端覆盖全部 128 个输出 token。重复相同 prompt 的测量不能代表任意混合请求流量。

对应结果：

- `bench_results/moe_fuse/run_39129/micro.json`：完整 MoE 的四种路径比较。
- `bench_results/moe_fuse/run_39130/e2e_baseline.json`、`e2e_fused4.json`：同卡端到端。
- `bench_results/moe_fuse/run_39132/e2e_fused_atomic4.json`：第一版融合 scatter 的端到端。
- `bench_results/moe_fuse/run_39143/`：实际插件开关复测，以及两开关组合下的分派计数。

实际插件开关复测确认 M=4/8/16/32 均调用融合路径；TPOT 约 3.05/3.98/5.54/8.70 ms，
与同卡强制分派实验一致。基线与这次复测的 23,040 个输出 token 同样全部一致，
详细比较在 `bench_results/moe_fuse/token_comparison.json`。

## Profile：原先的瓶颈已经改变

`run_39129/trace_*.json` 与 `profile_summary.json` 只累计 GPU kernel 时间：

- batch 16：每层 kernel 数 **23 → 9**。27 层 GPU kernel 总时间 **2.468 → 0.982 ms**。
- batch 16：两次 CUDA 矩阵乘累计约 **0.635 → 0.613 ms**，计算主体基本保持，
  收益来自外围操作减少。矩阵乘占比从约 **26% 上升到 62%**。
- batch 32：GPU kernel 总时间 **3.713 → 1.789 ms**，矩阵乘约占融合后总时间 **77%**。
- 原先每层需要处理静态上界的大缓冲；融合版由 device metadata 限制访问范围。

计时图与 profile 分别采集；profile 的 kernel 时间之和不等同于完整算子的事件计时。

继续优化的优先级：

1. batch 32 的主要成本已回到 CUDA 矩阵乘，适合对 tile/stages 或专家负载分布做对照。
2. 小 batch 的 align/count 仍有约 5–6 μs/层的计数成本；可以继续研究小路由的一次性分组。
3. 继续增加外围 fusion 的收益空间已比原先小，应同时检查端到端中 attention/dense 的占比。

这三项是后续方向，本次没有继续改动 CUDA 矩阵乘实现。

## 数值、边界与路由分散度

- 最终验证 `run_39139/verify.json`：M=1/2/4/8/16/32/64/512，80 组用例通过。
- 包含集中/随机/零权重路由、同一 Graph 更换路由、split-K 1/2 与两级不同 split 设置。
- 每次 replay 前将中间缓存填为 NaN、反向索引填为 -999，确认有效数据确实重建，padding 不污染输出。
- 覆盖非连续 x 和路由权重。展平的权重可能仍有 stride，已显式规整连续存储；gather 使用 x 的真实 stride。
- 对原 CUDA 路径的最大相对 L2 误差约 **6.15e-5**；与独立 Triton GEMV 对照不超过 **1.12e-3**。
  后者还包含原 CUDA FP16 中间写出与 GEMV FP32 中间累加之间的精度差异。
- `run_39130` 的基线与融合端到端：**23,040 个输出 token ID 全部相同**。

单层随机路由压力测试 `run_39141/micro.json` 使用实际权重/x，但替换路由为均匀随机 top-k，
不属于真实端到端路由采样：

| M | 活跃专家 | 原路径 μs | 融合 μs |
|---:|---:|---:|---:|
| 4 | 20 | 52.34 | 55.00 |
| 8 | 34 | 111.95 | 111.16 |
| 16 | 52 | 225.59 | 153.71 |
| 32 | 61 | 251.42 | 177.08 |

分散路由下 M=4 略慢，M=8 基本持平；因此保留为显式实验开关，不把本次阈值当作通用最优值。

## A100 与复现

A100 作业 **39144** 撰写报告时仍在排队：会进行数值验证、真实输入回放/profile、
完整全 INT3 端到端基线/融合对照。使用 `bench_results/moe_fuse/source_final` 的源码与扩展快照，
对应 SHA256 在 `manifest.json`。未得到结果前，不将 5090 收益外推至 A100。

```bash
# 5090，实际插件开关
PHASE=e2e ARGS='--config production' sbatch -p 5090 --export=ALL bench/moe_fuse_study.slurm

# 同一个 GPU 上先后测量基线与融合
PHASE=e2e_pair sbatch -p 5090 --export=ALL bench/moe_fuse_study.slurm

# 验证、27 层回放/profile、端到端完整流程
PHASE=all sbatch -p a100 --export=ALL bench/moe_fuse_study.slurm
```

实验脚本使用独立字节码缓存目录；早期作业出现过加载旧接口的问题。39128/39134/39137
为失败的调试轮次，39138 未确认融合优先分派，均不作为最终结果。

Grouped GEMV 交互讲解首次提交为 `afe5e51`。融合实现、实验脚本与性能表随本报告一起纳入仓库。
当前算子图见 [分派总览 / 融合流程 / grouped GEMV](brmoe-kernel-paths.html)，
独立 [SVG 总览](brmoe-kernel-paths.svg) 可直接在 GitHub README 预览。
