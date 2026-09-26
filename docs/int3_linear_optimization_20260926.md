# 共享专家与 attention 的 INT3 线性层优化

日期：2026-09-26。目标是 attention、共享专家和 routed experts 全部使用 INT3 权重的模型，激活为 FP16（W3A16）。本轮在 RTX 5090 上完成；A100 不启用未经该卡实测的新配置。

## 改动

原实现把非路由线性层表示为 `E=1, top_k=1` 的 routed GEMM。虽然输入行共享同一份权重，`slot=16` 仍使 `BLOCK_M=32/64` 的 CTA 分别解码相同权重 2/4 次。源码中的 `_slot_dot` 按 slot 独立执行，这是本轮首先消除的重复工作。

1. **单 slot**：令 `slot=BLOCK_M`，同时按 slot 对齐 `num_post`，`num_valid` 保持真实 M。原输入无需补零，尾行由 kernel 的 mask 处理。
2. **专用单权重 TC kernel**：新增 `tools/brmoe_int3_vllm/linear_tc.py`。一次加载、解包 `BLOCK_K × BLOCK_N` 的权重块，供整个 M tile 使用；省去专家元数据，使用 `tl.dot(a, b, acc)` 沿 K 累加。
3. **按实际 M 分派**：中小 M 使用 `BM/BN/BK=32/64/128, stages=3, warps=4`；M≤16 时 BM 降至 16。大 M 保留原 GEMM 的 `BK=32`，采用单 slot。

这是 FP16 Tensor Core 矩阵乘配合 INT3 权重解包。权重打包、checkpoint、FP16 反量化中间舍入边界均保持，不建立完整 FP16 权重缓存。共享专家 SiLU、两分支相加以及 attention 的其他运算保持独立。

| 正式分派条件 | 路径 |
|---|---|
| M≤8 | 原 split-K GEMV |
| `auto`、sm_120、FP16、GS64，9≤M≤128 | 专用 INT3 TC，32/64/128，4 warps、3 stages |
| 同上 auto 条件，M>128 | 原 Triton GEMM，BM=slot=64、BN 自适应、BK32 |
| `legacy` 或其他架构 / GS / dtype | 原 GEMV / slot16 GEMM |

M 是一次 kernel 调用的输入行数，受 prefill 分块和 CUDA Graph padding 影响，不是固定的请求 batch。`BRMOE_LINEAR_BACKEND=auto` 为默认值；`legacy` 可回退。

[四张执行路径图](brmoe-kernel-paths.html)中的第 4 张解释专用 kernel；[独立 SVG](brmoe-int3-linear.svg)可直接下载。

## 真实输入与测量口径

- 模型：`brmoe-3bit-vllm-int3dense`，该目录名称对应的是全 INT3 模型。64 个 routed experts，top-k=6，27 个 MoE 层。
- 环境：RTX 5090 32GB；driver 610.57.04；Torch 2.13.0+cu129；Triton 3.7.1；CUDA 12.9。
- 本轮 baseline / candidate **均启用 `BRMOE_CUDA_FUSE=1`**；CUDA 配置 128/128/4，`BRMOE_MOE_SMEM=legacy`，`BRMOE_GROUPED_GEMV=0`。对照的是线性层改动的增量收益。
- job 39179 保存 112 份线性层权重：56 个 attention QKV/O、54 个共享专家 gate_up/down、2 个首层 MLP。采集 batch 8/16/32/64/128 的每层真实 decode 输入，以及每种形状的真实 prefill 输入，共 584 个线性输入样本、135 个 routed MoE 样本。样本是最后一次观测到的调用，不覆盖全部 decode step。
- 六种 `(K,N)`：`(2048,6144)`、`(2048,2048)`、`(2048,21888)`、`(10944,2048)`、`(2048,5632)`、`(2816,2048)`；GS=64。
- **线性层回放**：decode 每档使用全部 112 层真实权重与输入，CUDA Graph 计时；prefill M2048 只有 6 个代表层，未观测到 M512，不能把 6 层总和当作全模型时间。
- **完整 MoE 回放**：27 层的共享 gate_up → SiLU×up → shared down，加上 routed MoE 和两分支相加。断言真实 shared gate 输入等于 routed 输入。gate/top-k 使用已采集结果，不计入此回放。
- **端到端**：128 输入 / 128 输出，CUDA Graph，关闭 prefix cache，greedy，3 次取中位数；同一卡上先后运行 baseline / candidate。TPOT=`(E2E−TTFT)/127`。同 batch 内重复相同 prompt，结论范围限于该请求与路由分布。
- **分阶段 profile**：独立作业在实际模型路径插入 CUDA Events；采样和干净 e2e 分开。事件数值不能直接拼加为 e2e TPOT。

## 参数对照的结论

job 39179 / 39188 / 39190 / 39194 分别覆盖 slot、BK、BM/BN 和流水线阶段数。`single_BK_stages` 是原 GEMM 改为单 slot；`tc_BM_BN_BK_stages_warps` 是专用 kernel。

- 仅单 slot、BK32 即让 bs32/64/128 e2e TPOT 从 8.72/14.17/19.66 ms 降到 6.78/9.08/11.88 ms（39183）。
- BK64 + stages1 明显变慢；stages3 更合适。单 slot BK64 在部分 decode 优于 BK32，但 prefill 和 e2e 不稳定地获益，未作为统一配置。
- 专用 kernel 的 BK128 优于 BK64。继续增至 BK256，或把 stages 改为 2/4，并未形成更好的组合。
- 6 个代表层的小扫参一度偏向 BN32；**全部 112 层回放**显示 BN64 更好。因此最终采用 BM32/BN64/BK128。
- BK128 专用 kernel 在 M2048 不如单 slot BK32，最终策略按 M 组合两者。最终分派由实际 e2e 对照确认，未单凭微基准上线。

## 正式路径结果

**端到端：job 39201，同一卡、线性层正式入口，无线性调参 hook。**

| batch | TPOT baseline → auto（ms） | 降幅 | TTFT baseline → auto（ms） |
|---:|---:|---:|---:|
| 1 | 2.266 → **2.261** | 0.2% | 19.35 → **13.12** |
| 2 | 2.775 → **2.773** | 0.1% | 26.66 → **17.76** |
| 4 | 3.039 → **3.032** | 0.2% | 39.41 → **28.44** |
| 8 | 3.971 → **3.957** | 0.4% | 70.53 → **47.85** |
| 16 | 5.560 → **3.370** | 39.4% | 118.10 → **85.05** |
| 32 | 8.645 → **4.937** | 42.9% | 232.85 → **163.99** |
| 64 | 14.193 → **6.809** | 52.0% | 463.16 → **315.86** |
| 128 | 19.697 → **10.988** | 44.2% | 929.08 → **629.09** |

bs1–8 的 decode 基本持平。97,920 个输出 token ID 全部一致；本轮未重跑 FP16 基线，不用历史跨作业 FP16 数值计算加速比。

**真实输入 CUDA Graph 回放：job 39202，单位 ms。**

| M | 112 层线性 baseline → auto | 27 层完整 shared+routed MoE baseline → auto |
|---:|---:|---:|
| 8 | 2.150 → **2.161** | 1.981 → **1.982** |
| 16 | 4.035 → **1.674** | 3.060 → **1.851** |
| 32 | 6.596 → **1.918** | 5.046 → **2.801** |
| 64 | 12.373 → **2.296** | 8.276 → **3.658** |
| 128 | 12.618 → **3.005** | 9.530 → **5.147** |

M2048 的 6 个代表线性层合计 **5.531 → 2.306 ms**，这是 prefill 回放，不是全模型 TTFT。

[机器可读完整数据、各类别回放和源码指纹](perf/int3_linear_20260926.json)。

## 数值验证

专用 kernel 的候选验证 job 39194：960 项通过。正式自动分派 job 39202：448 项通过，对独立 INT3 解包、FP16 反量化后 FP32 GEMM 参考的最大相对 L2 为 **0.000281**，阈值 0.003。

正式验证覆盖六种真实权重形状、合成 K/N 尾部、GS128 回退，M=1/8/9/16/17/31/32/33/64/65/127/128/129/257；输入使用非连续 stride。每种情况捕获 CUDA Graph，两次修改输入后 replay，输出先填 NaN 检查是否完整写出。FP32 参考关闭 TF32。

正式 112 层线性与完整 MoE 的真实输入回放中，M≥16 的 FP16 输出与 baseline 逐位一致；M8 仍使用原 split-K atomic GEMV，重复调用存在约 1e-5 相对 L2 的原子归约顺序差异。

## 优化后的瓶颈

job 39206 在同卡顺序采集 baseline / production 的真实模型分阶段事件，单位 ms：

| batch | attention 线性 baseline → auto | 共享专家线性 baseline → auto | auto routed MoE 总计 | auto W13 + W2 |
|---:|---:|---:|---:|---:|
| 16 | 1.915 → **0.907** | 2.382 → **0.977** | 1.607 | 0.702 |
| 32 | 2.814 → **1.028** | 3.525 → **1.108** | 2.569 | 1.524 |
| 64 | 4.759 → **1.199** | 6.122 → **1.272** | 3.165 | 2.032 |
| 128 | 5.145 → **1.543** | 8.790 → **1.613** | 4.379 | 3.135 |

bs128 时，attention + 共享专家线性采样从 **13.935 ms 降到 3.156 ms**。优化后的 routed MoE 为 **4.379 ms**，其中 W13 + W2 为 **3.135 ms，约 72%**。下一优先级转向 routed grouped GEMM，尤其 W13；共享专家的 SiLU / 相加融合可以继续评估，但应先采集这两个小算子的占比。

独立 profile 的 bs32/64/128 分别有 2/7/9 层专家计数直方图发生少量变化；bs128 的直方图半 L1 合计为 25，padded 行合计 23488 → 23504。该 profile 用于识别优化后的耗时分布；routed MoE 的前后事件变化不能单独归因为线性 kernel 的收益。上面的完整 MoE 对照固定使用同一份真实输入、expert IDs 和路由权重。

后续可针对真实专家行数比较更大的 M tile、分组调度与 W13/W2 流水线；保留“固定真实路由完整 MoE + 干净 e2e”的双重验收。当前没有寄存器溢出、occupancy 或 HBM 带宽计数，不能仅凭耗时把剩余瓶颈归类为某一种硬件资源。

A100 本轮没有新性能结果；自动分派在 sm_80 保留旧路径。

## 使用与复现

```bash
# 正式插件默认自动选择新线性层配置；只在已验证的 sm_120 / FP16 / GS64 启用。
export BRMOE_LINEAR_BACKEND=auto
# 为复现本报告，MoE 外围融合也需启用。
export BRMOE_CUDA_FUSE=1
export BRMOE_GROUPED_GEMV=0
export BRMOE_MOE_SMEM=legacy

# 单独回退本轮线性层改动：
export BRMOE_LINEAR_BACKEND=legacy
```

以下在仓库根目录运行，Slurm 脚本使用现有 `vllm5090` 环境。参数含逗号时通过环境变量传入，避免被 `sbatch --export` 当作分隔符。

```bash
# 采集真实输入，然后做初始单 slot 扫参
PHASE=collect sbatch -p 5090 --export=ALL bench/int3_linear_study.slurm

# 修改 --input 为上一步输出目录。完整 112 层、完整 MoE、数值与 Graph 验证。
PHASE=compare ARGS='--input /absolute/path/to/run_capture --configs baseline,production --prefill' \
  sbatch -p 5090 --export=ALL bench/int3_linear_study.slurm

# 正式插件入口，baseline / production 同卡端到端
PHASE=e2e_pair CANDIDATE=production ARGS='--batch-sizes 1,2,4,8,16,32,64,128' \
  sbatch -p 5090 --export=ALL bench/int3_linear_study.slurm

# 同卡分阶段采样，与上面的干净 e2e 独立
PHASE=profile_pair CANDIDATE=production ARGS='--batch-sizes 16,32,64,128' \
  sbatch -p 5090 --export=ALL bench/int3_linear_study.slurm
```

`baseline` 显式设置 `BRMOE_LINEAR_BACKEND=legacy`，防止正式默认值更新后污染对照组；`production` 设置 auto，线性层无调参 monkeypatch。每个阶段记录源文件 SHA256、GPU 与软件版本。大张量保留在本机 `bench_results/int3_linear/`，汇总数字随仓库保存。
