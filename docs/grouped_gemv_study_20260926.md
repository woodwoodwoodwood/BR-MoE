# Grouped GEMV 实测：2026-09-26

后续已实现 CUDA MoE 外围融合，完整 MoE 与端到端对照见
[CUDA MoE fusion 实测](moe_cuda_fusion_20260926.md)。本文保留 grouped GEMV 当时的实验结果。

## 结论与使用

在 RTX 5090 上，grouped GEMV 改善了本次真实路由下 batch 4/8/16 的完整
MoE 和全 INT3 端到端性能。batch 32 继续使用原 CUDA 路径。

插件已接入实验开关，默认关闭：

```bash
export BRMOE_GROUPED_GEMV=1
# 然后运行原有 vLLM 启动/性能测试命令
```

此开关仅在 sm_120、4 <= M <= 16 生效。A100 作业 **39094** 后续已完成，
实验脚本强制启用相同的 grouped GEMV 配置时，bs16 回退，未将生产开关的范围扩展到 A100。

### 后续补充：A100 的同卡对照（39094）

| batch | 原版 27 层 MoE ms | grouped 策略 ms | 原版 TPOT ms | grouped 策略 TPOT ms |
|---:|---:|---:|---:|---:|
| 4 | 3.479 | 2.511 | 8.233 | 7.263 |
| 8 | 4.495 | 4.016 | 12.187 | 11.685 |
| 16 | 5.471 | 7.045 | 11.840 | 13.419 |
| 32 | 6.590 | 6.589 | 16.196 | 16.246 |

同样是全 INT3 模型、128 输入 / 128 输出；72 组数值用例通过。bs32 两组均走原 CUDA。
这不是外围融合的测量；外围融合的 A100 作业 39144 仍在排队。

## 测量口径

- 全 INT3 模型：`/mnt/709/data3/home/jianglei/models/brmoe-3bit-vllm-int3dense`。
  该目录的 attention 和 MoE 都是 INT3；本次只修改 routed MoE。
- 基线为当前已启用且修复数值问题的 CUDA/Triton 混合分派。
- 端到端：128 输入、128 输出，CUDA Graph，关闭 prefix cache，每组 3 次取中位数。
  TPOT 使用原脚本的 `(E2E - TTFT) / 127`，包括主机侧开销。
- 与原性能脚本一致，同一 batch 内重复相同 prompt；不代表不同 prompt 的在线流量。
- 新采集 4/8/16/32 各 27 层最后一个 decode step 的真实 x、路由和权重。
  算子回放依次执行全部 27 层及其真实权重，包含分组、清零、两级 GEMV、激活、归约和转换。
- 采集/profile 与干净端到端计时分进程执行。微观回放只代表采样 step；端到端覆盖全部 128 token。

## 最终结果：5090

MoE 列为 **27 层合计毫秒**，不是单层，也不是整个模型。
端到端列来自实际插件开关复测（39091），不是仅使用实验 monkeypatch 的结果。

| batch | 完整 MoE 基线 ms | grouped/最终 ms | MoE 降幅 | 基线 TPOT ms | 最终 TPOT ms | TPOT 降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.407 | 0.922 | 34.4% | 3.566 | 3.043 | 14.7% |
| 8 | 2.733 | 1.482 | 45.8% | 5.916 | 4.627 | 21.8% |
| 16 | 2.596 | 2.322 | 10.6% | 7.049 | 6.121 | 13.2% |
| 32 | 3.900 | 3.883 | 测量波动 | 9.850 | 9.860 | 测量波动，原 CUDA 路径 |

batch 32 强行使用初版 grouped 配置时 TPOT 为 11.87 ms，明显退化，因此最终
策略上限为 M=16。首次直接广播原型和按行串行原型也有多组明显负收益，已保留原型
与扫描日志，未将它们作为最终配置。

## 实现

`BR-MoE/kernels/triton_int3/int3_moe/grouped_gemv.py`：

1. 单个 GPU kernel 对 `(expert, route)` 排序，按专家形成 4/8 行小组。
2. 同组 token 共用解包后的权重；中间结果保持原 route 顺序，方便第二级访问和 scatter。
3. 固定数量 CTA 循环处理 GPU 上的有效分组，避免按最坏情况启动大量空 CTA。
4. 两级 split-K 用 FP32 原子归约；每次调用清零，并在每次 Graph replay 重新分组。

实测参数如下；`groups=1` 表示每次 K 循环解包 32 个值，两个投影的 split-K 不同。

| M | 同组行数 | BLOCK_N | K 循环步长 | w13 split-K | w2 split-K | warps |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 128 | 32 | 16 | 8 | 4 |
| 8 | 8 | 128 | 32 | 8 | 4 | 4 |
| 16 | 8 | 256 | 32 | 8 | 4 | 4 |

这是 W3A16 的 SIMT GEMV，仍需解包和浮点乘加。权重流量下降并不意味着整模型
按位宽比例加速；共享专家较少时也可能产生额外无效计算，因此目前保留为显式开关。

## 数值与 Graph 验证

- 72 组小形状验证：M=1/2/4/8/16/32，3 个最终配置，各测随机路由、全部同一专家
  （含同一 token 重复专家）、全部无效路由、重新随机路由。
- 在同一个 CUDA Graph 上更换路由后重放；独立参考使用反量化权重及浮点矩阵乘。
  最大相对 L2 误差 `2.6463e-4`，无 NaN/Inf。
- 真实输入的最终配置与原 routed GEMV 相对 L2 误差不超过约 `1e-4`。
  原 CUDA 的 FP16 中间写出与 GEMV 的 FP32 中间累加本身存在精度差异。
- 端到端基线与最终插件开关：4 个 batch、各 3 次，共 **23,040 个输出 token ID 完全一致**。
  这是本次 prompt 的一致性检查，不等同于通用模型质量评测。

## Profile 揭示的下一处瓶颈

batch 16 的基线，27 层累计 GPU kernel 时间约 **2.493 ms**，其中两次 Marlin
矩阵乘合计仅 **0.653 ms（26.2%）**。每层共有 23 个 kernel，许多时间花在
gather、FP16/FP32 转换、激活和 index_add/scatter 上。

源码还能解释额外流量的来源：M=16、topk=6、E=64 时，CUDA wrapper 按
`96 + 64 * 15 = 1056` 行分配并执行外围 torch 操作。本次采样的每层只激活
6 个专家、每专家 16 个 token，实际只需 96 行。矩阵乘读取 device metadata
跳过无效块，但外围操作仍处理静态上界缓冲。

因此，继续优化 M=16/32 时，优先考虑：

1. 用 Triton 融合 gather/转换、SiLU×up、加权 scatter，并在设备端按有效行掩码执行。
2. 保留现有 CUDA 矩阵乘，先减少外围全缓冲读写和 kernel 数量。
3. 再与 grouped GEMV 做完整 MoE 和端到端选择；A100 单独校准阈值。

上述是基于 profile 和源码的优化方向，融合 CUDA wrapper 尚未在本次实现。

## 结果与复现

文件均位于 `bench_results/grouped_gemv/`：

- `run_39061/real_inputs.pt`：108 组当前版本真实采样。
- `run_39084/micro.json`、`trace_bs*.json`：27 层回放与原始 GPU profile。
- `profile_summary.json`：只累计 GPU kernel 事件，w13/w2 依执行顺序拆分。
- `run_39086/e2e_baseline.json`：第二轮干净端到端基线。
- `run_39091/e2e_production.json`：实际插件开关的最终端到端。
- `run_39087/verify.json`、`token_comparison.json`：数值及输出 token 对照。
- `prototypes/`：初版广播、串行共享、persistent 原型。
- `source_final/manifest.json`：最终源码及 CUDA 扩展 SHA256，A100 作业使用此冻结副本。

```bash
# 重跑实际插件开关（5090）
PHASE=e2e ARGS='--config production' sbatch -p 5090 --export=ALL bench/grouped_gemv_study.slurm

# 从数值验证、真实采集到完整 MoE/profile、端到端基线/候选的完整实验
PHASE=all sbatch -p a100 --export=ALL bench/grouped_gemv_study.slurm
```

A100 作业 39094 的输出位置：`bench_results/grouped_gemv/run_39094/`；
调度日志为 `bench_results/grouped_gemv_39094.out` 和 `.err`。
