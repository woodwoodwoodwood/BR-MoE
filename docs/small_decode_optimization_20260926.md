# A100 小 batch decode：兑现 INT3 权重流量收益

本轮从 `14766d2` 继续，目标为全 INT3 attention / shared / routed 模型的 batch 1/2，兼顾 batch 4/8。权重布局、checkpoint、activation 与 KV cache 精度均延续上一轮。

## 测量口径

- A100 80GB PCIe，TP1；128 输入 / 128 输出，CUDA Graph，prefix cache 关闭，3 次中位数。
- 同一 batch 内重复相同 prompt。真实路由不代表任意业务流量；不以随机路由替代性能验收。
- TTFT 使用独立单输出 token 请求；TPOT 为 `(E2E−TTFT)/127`。FP16 使用本机 vLLM 默认 Triton MoE 配置。
- 完整 MoE 回放包含 routed 分支、共享 gate_up / SiLU / down、最终相加，固定 27 层真实输入与路由；router/top-k 已采集。两分支串行回放，端到端使用 vLLM 的多 stream 调度。
- 独立 CUDA event profile 和 profiler trace 不用于端到端计时。采集 39294 的 `max_num_seqs=8`；干净 E2E 与整模型 trace 为 512，实际 decode M 均按目标 batch 验证。

## 发现的瓶颈

### 权重压缩比没有等比例转成带宽收益

当前 GS64 使用 FP16 scale 和 zero，平均位宽约 `3 + (16+16)/64 = 3.5 bit`，理想权重流量比为 `16/3.5 ≈ 4.57×`。这一比值要求比较相同权重访问，并忽略解包、临时缓冲与其他模型操作。

Nsight Compute 冷缓存对照（39309）：真实 `o_proj`，M1/K2048/N2048。原 GEMV HBM 吞吐约 **201 GB/s，峰值利用率 10.45%**，并未达到 HBM 带宽上限。第一版 half2 原型达到 **236 GB/s，12.28%**；核心时间 **9.312 → 7.872 µs**，执行指令数 **1,427,456 → 1,133,568**。寄存器从 69 增至 124，说明更少指令不自动意味着更高 occupancy，仍需联合调整分块与 split-K。

这些是独立投影的硬件计数，不能直接当作整个模型的带宽或端到端加速比。39305/39308 的 `cache-control=none` 对照受 profiler 多次 replay 后的缓存状态影响，仅保留为暖缓存诊断；报告冷缓存结论来自 `cache-control=all`。

Nsight 2025.2.1 与当前驱动不兼容（39296），改用 2024.1.0。系统临时目录的 profiler 锁不可写（39301），后续作业先检查没有其他 ncu 进程，再使用独立可写 TMPDIR；没有更改驱动或系统权限。


最终默认配置冷缓存对照（39324，同一投影）：

| 配置 | 核心 µs | 寄存器/线程 | HBM GB/s | 执行指令数 |
|---|---:|---:|---:|---:|
| baseline | 9.312 | 69 | 201.0 | 1,427,456 |
| production | 6.816 | 37 | 271.0 | 1,316,864 |

最终配置的活跃 warp 比例从 12.75% 提升到 24.26%，HBM 吞吐提高约 34.8%。与第一版 half2 原型相比，GROUPS1/split32 的总指令数略多，但寄存器更少、并发更高，核心反而更快；优化不能只看位宽或指令数。

该表为独立核心，归约另计；计数器采样清空缓存，不代表多 stream 全模型的实际缓存命中率。

### 小操作和分派也占时间

整模型 trace 39300 含一次 prefill + 15 步 decode：每步 decode 有 **112 个线性 GEMV + 54 个 routed GEMV = 166 次 GEMV**。原线性路径每次还需要清零和 FP16 转换，routed 路径也有清零与原子归约。路由规整由多个 PyTorch 操作组成。

最终正式路径 trace 39323 的 GPU kernel 总数为 **10,228**，基线为 **14,743**（相同的一次 prefill + 15 步 decode），减少 **4,515 次 / 30.6%**，即每步 decode 少 301 次。M1/M2 均确认 112 次线性归约、27 次路由规整、27 次融合激活归约与 27 次最终加权归约。

不能把各 stream 的 kernel 时间直接相加当作 TPOT；shared 和 routed 存在重叠。原版分阶段采样与完整调用次数见机器可读数据。

## 实现

1. **half2 反量化**：对 3-bit 布局内相隔 16 bit 的两个权重成对解码，使用 FP16x2 减零点与乘 scale。保持 FP16 的反量化舍入边界，不转存完整 FP16 权重。
2. **独立 split-K 部分和**：每个 split 覆盖自己的输出位置，包含空 K split；无需输出预清零或 FP32 原子累加。
3. **融合归约**：线性层的 split-K 归约与 FP16 转换合并；routed W13 归约与 SiLU 合并，W2 的 split-K / top-k 加权归约与最终转换合并。routed W13 继续保留原 GEMV 的 FP32 中间精度。
4. **batch 2 权重复用**：一个 CTA 为两行输入共享解包结果，分别累加输出。直接按行号计算，不假设两个 token 的内容相同。
5. **路由规整融合**：一个 kernel 完成 int64 转换、负 expert ID 替换与权重清零，保留调用者的原始张量。

### 最终自动分派

| 范围 | 配置 |
|---|---|
| A100 FP16/GS64 线性 M1 | BN64 / GROUPS1 / split32 / 2 warps，单行 half2 |
| 同上 M2 | 相同配置，两行共享解码 |
| A100 当前模型 routed M1/2 | BN64 / GROUPS4 / W13 split8、W2 split4 / 2 warps |
| 同上 routed M3…8 | 仅融合路由规整，保留现有 CUDA/TC 分派 |
| M>8 / 其他架构 | 延续前轮分派 |

线性层核心由清零/GEMV/cast 三次变为两次；routed 核心为四次，加上规整共五次，真实路由权重为 FP32。路由的自动策略限 E64/K2048/I1408/top-k6/K-major/GS64；`BRMOE_DEBUG` 保留原路径的诊断输出。

此处 GROUPS 表示每轮同时解码的 32-K 分组数；split32 会分配独立 FP32 部分和缓冲。减少 GROUPS 降低寄存器压力，增加 K-split 提供更多 CTA，代价是更多归约流量。完整 FP16 权重没有被缓存，checkpoint 和 CUDA 权重副本也无需重新打包。

## 完整 MoE 与端到端

正式端到端来自 **39318**，同一 A100 上顺序运行 FP16 / `14766d2` 等价基线 / 当前正式入口，单位 ms。结果采用固定源码，不使用实验 monkeypatch。

| batch | FP16 TPOT | INT3：之前 → 当前 | INT3 耗时降幅 | 相对 FP16 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 4.916 | 4.244 → **3.670** | 13.5% | 1.34× |
| 2 | 6.657 | 5.802 → **4.718** | 18.7% | 1.41× |
| 4 | 5.574 | 5.504 → **5.302** | 3.7% | 1.05× |
| 8 | 5.542 | 5.740 → **5.569** | 3.0% | 0.995× |
| 16 | 6.042 | 6.369 → **6.346** | 0.4% | 0.95× |
| 32 | 9.201 | 7.536 → **7.547** | -0.1% | 1.22× |
| 64 | 12.190 | 10.542 → **10.530** | 0.1% | 1.16× |
| 128 | 18.238 | 17.822 → **17.829** | ≈0% | 1.02× |

| batch | TTFT：FP16 | TTFT：INT3 之前 → 当前 | E2E：FP16 | E2E：INT3 之前 → 当前 |
|---:|---:|---:|---:|---:|
| 1 | 24.79 | 19.01 → 17.93 | 649.11 | 557.97 → 484.01 |
| 2 | 27.32 | 25.82 → 25.05 | 872.74 | 762.66 → 624.21 |
| 4 | 34.81 | 38.15 → 36.86 | 742.74 | 737.19 → 710.24 |
| 8 | 52.15 | 62.72 → 62.17 | 756.03 | 791.71 → 769.41 |
| 16 | 86.33 | 107.36 → 106.73 | 853.63 | 916.26 → 912.73 |
| 32 | 173.56 | 210.93 → 209.99 | 1342.13 | 1168.06 → 1168.51 |
| 64 | 343.80 | 418.78 → 418.58 | 1891.94 | 1757.68 → 1755.83 |
| 128 | 682.13 | 834.89 → 833.95 | 2998.42 | 3098.30 → 3098.29 |

M>8 的默认策略未改变，表中差异属于重复运行波动；prefill 沿用前一轮优化。TPOT 不能用 kernel 耗时简单相加估算。

完整算子来自 **39321 / 39322**，使用 **39294** 的真实输入，单位 ms。线性与 MoE 列有 shared 投影重叠，不能相加。

| M | 112 层线性：之前 → 当前 | 27 层 routed：之前 → 当前 | 27 层完整 MoE：之前 → 当前 |
|---:|---:|---:|---:|
| 1 | 1.646 → 1.230 | 1.763 → 1.350 | 2.624 → 2.052 |
| 2 | 2.246 → 1.447 | 2.607 → 2.104 | 3.790 → 2.939 |
| 4 | 2.712 → 2.712 | 1.969 → 1.754 | 3.397 → 3.177 |
| 8 | 2.715 → 2.715 | 2.090 → 1.877 | 3.524 → 3.309 |

分阶段 CUDA event 采样（39294 / 39325），27/28 层合计，单位 ms；包含插桩影响和 shared stream 重叠：

| batch | attention 投影 | shared 投影 | routed 总调用 | routed W13 | 激活/融合归约 | routed W2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.115 → 0.901 | 1.607 → 1.419 | 2.734 → 2.527 | 1.025 → 0.911 | 0.183 → 0.229 | 0.656 → 0.582 |
| 2 | 1.416 → 1.005 | 2.405 → 2.031 | 3.735 → 3.261 | 1.715 → 1.413 | 0.170 → 0.238 | 0.978 → 0.818 |
| 4 | 1.556 → 1.539 | 1.710 → 1.682 | 3.486 → 3.403 | 0.823 → 0.826 | 0.223 → 0.253 | 0.483 → 0.488 |
| 8 | 1.559 → 1.563 | 1.717 → 1.689 | 3.594 → 3.450 | 0.831 → 0.831 | 0.227 → 0.219 | 0.571 → 0.581 |

采样路由的 27 层在各 batch 均激活 6 个专家，每个活跃专家接收 M 条路由，反映本次重复 prompt 的高度相关性。逐层 64 专家直方图保存在 JSON；图回放正确性另用变化的随机与无效路由验证。


## 对照与取舍

- 39295：仅移除原子归约，6 个代表投影的 M1 为 132.30 → 124.11 µs，收益约 6%。
- 39306：112 个真实投影，第一版 half2 / 两行复用使 M1 **1.647 → 1.397 ms**、M2 **2.282 → 1.756 ms**。
- 39304：仅 half2 + 归约改造的端到端 bs1 **4.242 → 3.939 ms**，bs2 **5.801 → 5.407 ms**；5,760 个输出 token ID 一致。
- 39310：加上两行复用与路由融合，bs1 **4.243 → 3.741 ms**、bs2 **5.813 → 5.004 ms**；输出 token ID 一致。
- **未采用 batch 4 的两行 GEMV 默认分派**：完整 MoE 回放更快，但端到端为 5.429 ms，劣于只融合路由规整的 5.359 ms；保留原 TC 路径。batch 8 的逐路由 / 两行 GEMV 也不及现有 TC。
- grouped GEMV 的 routed 两行复用增加 GPU 分组操作，第一层的小幅收益不足以直接推广。完整 27 层与正式端到端决定默认配置。
- 冷缓存计数显示并发仍不足，随后对照 GROUPS=1 与更细 split-K；完整 112 层对照 39315 和端到端 39317 确认收益后，固定上述配置并完成 39318 正式复测。

## 正确性与回退

**1,004 项数值 / CUDA Graph 检查通过**：

- 线性 608 项：8 个实际/尾部形状，19 个 M，独立 FP32 参考，两次不同输入图回放。
- 直接 MoE 288 项：真实权重与 GS32/64/128 合成形状；普通/集中/负 ID/全无效/零权重路由；非连续输入与路由；污染中间缓冲后重复图回放。
- 正式分派 108 项：M=1/2/3/4/8/9/16/32/128，FP16/FP32 输出，各 6 种路由模式。比较同一输入数值，旧 GEMV 参考先转连续张量，新路径保留非连续输入；每次 replay 后先保存输出，避免旧 CUDA 工作缓冲别名掩盖差异；检查调用者路由未被修改。

最大相对 L2 为 **0.00023811**，阈值 0.003；没有放宽容差。正式 INT3 基线与新版本的 **97,920 个生成 token ID 完全一致**（8 个 batch × 3 次请求）。这验证本次执行等价性，不替代量化模型的任务质量评测。

39319 的验证命令漏传 `--input`，补充参数后由 39326 完成。39320 暴露旧 GEMV 对非连续输入的参考限制，校正参考张量布局后由 39328 完成；生产 kernel 未因这两次验证修订而改变。

缓存的临时缓冲沿用仓库已有 MoE 的有序 stream 使用约定；输出独立分配。端到端涵盖 vLLM 当前 shared/routed 双 stream 调度，未验证任意调用者并发复用同一工作缓冲。


```bash
export BRMOE_LINEAR_BACKEND=auto
export BRMOE_PREFILL_BACKEND=auto
export BRMOE_CUDA_FUSE=1
export BRMOE_SMALL_DECODE_BACKEND=auto  # 默认；legacy 回退本轮 small decode 策略
```

### 复现

本轮只增加 Triton kernel 和 Python 分派，无需重编 CUDA 扩展。已有的 prefill CUDA 32 行路径仍需上一轮扩展。所有 GPU 测量均由 SLURM 在 A100 上运行；作业启动前冻结源文件，性能 JSON 保存源码 SHA256。

```bash
# 在仓库根目录；脚本的 BASE/MODEL 路径按机器调整。
# 1. 采集目标 batch 的真实输入与路由
PHASE=collect ARGS='--batch-sizes 1,2,4,8 --linear-config production' \
  sbatch -p a100 --export=ALL bench/int3_linear_study.slurm
# 将下面 run_CAPTURE 改成采集作业 ID；完整 MoE 和 112 层线性回放
PHASE=moe ARGS='--input bench_results/int3_linear/run_CAPTURE --configs baseline,production' \
  sbatch --export=ALL bench/small_decode_study.slurm
PHASE=micro ARGS='--input bench_results/int3_linear/run_CAPTURE --configs baseline,production' \
  sbatch --export=ALL bench/small_decode_study.slurm
# 2. 正式入口端到端：baseline = 本轮之前，production = 本轮默认
PHASE=e2e CASES='fp16 baseline production' \
  ARGS='--m-values 1,2,4,8,16,32,64,128 --repeats 3' \
  sbatch --export=ALL bench/small_decode_study.slurm
# 3. 独立参考 / 动态图回放 / 正式路由分派检查
PHASE=verify ARGS='--input bench_results/int3_linear/run_CAPTURE --configs baseline,production' \
  sbatch --export=ALL bench/small_decode_study.slurm
PHASE=verify_dispatch ARGS='--m-values 1,2,3,4,8,9,16,32,128' \
  sbatch --export=ALL bench/small_decode_study.slurm
```

`small_decode_study` 的 baseline 保留 `LINEAR=auto/PREFILL=auto/FUSE=1`，仅关闭 SMALL；不是旧 `int3_linear_study` 的原始 slot16 baseline。`--quick` 只用于初筛；正式表格均覆盖全部 112 个投影、27 个完整 MoE。

冷缓存硬件计数使用 `PHASE=counters CASES='baseline production' NCU_CACHE=all`，通过 `NCU_BIN` 指定兼容当前驱动的 Nsight；`--input` 同上。整模型 trace 使用 `PHASE=model_profile ARGS='--config production --m-values 1,2'`。

## 后续仍可优化的部分

1. Routed GEMV 的更细 K 分块：39316 中 GROUPS1 / split16 在完整 MoE 回放优于当前 GROUPS4 / split8，但尚未通过对应端到端验收，未设为默认。
2. 继续控制寄存器和归约开销，评估权重加载与解包的流水重叠。单投影硬件计数显示距离带宽上限仍有差距；继续以真实完整 MoE 和 E2E 决定是否采用。
3. bs4/8 继续保留 TC；同专家复用需同时抵消 GPU 分组与并发调度成本。FP16 attention/KV、router、LM head 和框架开销也限制端到端加速比。

[性能 / 路由 / 源码指纹 JSON](perf/small_decode_20260926.json) · [README](../README.md) · [调用路径 HTML](brmoe-kernel-paths.html) · [上一轮 prefill 优化](prefill_grouped_optimization_20260926.md)。
