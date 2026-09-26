# A100：prefill 与 routed grouped GEMM 优化

本轮从 `cc3b5a1` 的全 INT3 版本继续优化。attention、共享专家和 routed experts 的权重均保持 INT3；激活与 KV cache 保持 FP16。所有结果在 A100 80GB PCIe 实测。

## 测量与瓶颈

配置延续上一轮：TP1、128 输入 / 128 输出、CUDA Graph、关闭 prefix cache、相同 prompt 重复组成 batch、greedy decode、每档 3 次中位数。`max_num_batched_tokens=2048`，引擎默认 `max_num_seqs=512`。TTFT 使用独立的单输出 token 请求，TPOT 为 `(E2E−TTFT)/127`。

job 39272 单独采集真实 prefill，request batch 为 4/8/16/32/128，实际算子 M 为 512/1024/2048。每档保存 27 层的最后一次 prefill 调用；bs32/128 分别有 2/8 个 M2048 chunk。不是用随机路由代替真实 prefill，也不是把最后一次 chunk 的事件耗时当作整个 TTFT。

优化前 M2048 的分阶段事件采样，27 层合计约：

- routed MoE 95 ms，W13/W2 合计 85 ms。
- attention 线性投影 21 ms，共享专家线性投影 21 ms。
- align 6 ms 左右；固定路由的独立回放为 4.29 ms。

真实 prefill 平均激活约 56/64 个专家。M2048 的 padding 比例：BM16 为 1.007×、BM64 为 1.123×、BM128 为 1.274×。大 tile 增加 padding，但通过减少重复解包获得了实际收益，需要以完整算子和端到端结果选择。

## 改动

### 1. 完整 BK tile 的 Triton grouped GEMM

旧 `_slot_dot` 即使设置更大的 `BLOCK_K`，内部仍静态展开成多个 32 宽 dot。新增 `int3_moe/grouped_tc.py`，一次解包完整 BK×BN 权重块，用 `tl.dot(a,b,acc)` 累加；一 CTA 处理一个专家的 BM 行。

最终配置 **BM/BN/BK=128/128/64、4 warps、3 stages**。对照覆盖 BK32/64/128、BM16/32/64/128/256、BN64/128/256、2/3/4 stages 与 4/8 warps 的若干组合。增加 warps 或 BK 并不必然变快；本次 8-warp 方案明显更慢。

W13 写出 FP16，SiLU×up 结果为 FP16；W2 写出排序空间中的 FP32 结果，随后按 `route_pos` 做确定性的 top-k 归约。保留原 Triton prefill 的数值边界，没有在 W2 后额外引入 FP16 舍入。每个最终输出元素只有一个写者，省去全局输出清零与 scatter 原子累加。

### 2. prefill 的路由排序

- CTA 内先统计 expert ID 直方图，再按专家聚合到全局计数，减少重复 ID 的全局原子操作。
- 长向量 scatter 使用 8 warps。
- M2048、27 层固定真实路由回放：**4.291 → 0.719 ms**；STI、expert IDs、metadata、排序权重和反向路由表逐项一致。
- profiler 中，scatter 本身从约 3.81 降到 0.40 ms；这些是 profiler 采样值，不能代替干净端到端计时。

### 3. 较小 prefill 的 CUDA 32 行 tile

CUDA 扩展增加 `thread_m`，align 粒度、gather 缓冲、激活和矩阵乘使用同一行块大小。保留原 16 行默认值；A100 / 当前 GS64 模型在 M256…512、开启 CUDA fusion 且新版扩展可用时选择 32 行。64 行方案保留为实验，没有提升为默认配置。

GS128 小形状在新增的独立参考检查中失败，原 16 行配置同样失败（39286/39289）。这是此次发现的已有 CUDA 限制；新增大 tile 在 C++ 入口明确拒绝非 GS64。当前模型为 GS64。Triton grouped kernel 的 GS32/64/128 则单独验证。

### 4. attention / shared expert 的大 M 投影

A100、FP16 / GS64、M≥512 使用 **BM/BN/BK=128/128/32、4 warps、3 stages** 的单权重 TC。6 个代表投影的 M2048 回放 **4.637 → 3.352 ms**。这个合计包括首层 MLP，不是所有 112 层的总时间。

M256 保留此前线性分派；M≤128 的 decode 分派保持原策略。5090 未启用本轮 A100 的 prefill 配置。

### BLOCK_K / 流水线对照

均为 M2048、27 层完整 shared+routed MoE 回放，单位 ms；以下调参阶段共享投影还使用上一轮线性配置。每组内部固定其余条件，不把不同 job 的小幅差异当成独立收益。

| job | BM / BN / BK | stages / warps | 对齐 / 输出 | 完整 MoE |
|---:|---|---|---|---:|
| 39276 | 64 / 128 / 32 | 3 / 4 | 原 align / atomic | 99.897 |
| 39276 | 64 / 128 / 64 | 3 / 4 | 原 align / atomic | 96.481 |
| 39276 | 64 / 128 / 128 | 3 / 4 | 原 align / atomic | 94.761 |
| 39276 | 64 / 128 / 64 | 1 / 4 | 原 align / atomic | 131.276 |
| 39276 | 64 / 128 / 64 | 2 / 4 | 原 align / atomic | 100.696 |
| 39278 | 128 / 128 / 32 | 3 / 4 | 快 align / atomic | 85.896 |
| 39278 | 128 / 128 / 64 | 3 / 4 | 快 align / atomic | 82.000 |
| 39288 | **128 / 128 / 64** | **3 / 4** | **快 align / top-k reduce** | **78.072** |
| 39288 | 128 / 128 / 64 | 2 / 4 | 快 align / top-k reduce | 81.810 |
| 39288 | 128 / 128 / 64 | 3 / 8 | 快 align / top-k reduce | 110.323 |
| 39288 | 128 / 128 / 32 | 4 / 4 | 快 align / top-k reduce | 82.379 |

BK128 在 BM64 的部分配置中较好；当 BM 增至 128 并优化外围操作后，BK64 / 3 stages / 4 warps 更合适。最终再加入共享投影优化，正式入口完整 MoE 降到 72.843 ms。

端到端也做了分步对照：39282 的 bs128 TTFT，原版 **1245.20 ms** → 仅新 grouped TC **931.09 ms** → 加上线性 prefill 配置 **839.32 ms**。39287 的中小 prefill CUDA 32 行组合达到 **827.68 ms**；最终正式代码复测值见下表。实验版线性阈值为 M>128，正式版收紧为 M≥512。上述各组优化前后 97,920 个输出 token ID 均一致。

## 最终自动分派

| 范围 | 路径 |
|---|---|
| A100 当前模型，M≤2 | 原 split-K GEMV |
| A100 当前模型，3≤M<256 | 原 CUDA 16 行 tile；缺少扩展时原 Triton 回退 |
| A100 当前模型，256≤M≤512 | fusion=1 且新版扩展可用：CUDA 32 行 tile；否则原路径 |
| A100 当前模型，M>512、路由总数≤32768 | 新 Triton grouped TC，128/128/64 + FP32 top-k 归约 |
| 其他模型形状、架构、GS / dtype | 原路径 |

新增 routed 自动分派限 E64 / K2048 / I1408 / top-k6 / GS64。直接调用新 Triton kernel 可测试其他形状。没有缓存完整 FP16 权重，没有更改 checkpoint。

## 正式入口的完整算子回放

job 39290 使用正式插件分派，固定真实输入与路由。完整 MoE 包含 routed 分支、共享 gate_up、SiLU×up、shared down 和两分支相加；router/top-k 已采集。回放串行计算两个分支，端到端可以重叠，因此不能直接相加推算 TPOT。

| 实际 M | routed 之前 | routed 当前 | 完整 MoE 之前 | 完整 MoE 当前 | 完整算子降幅 |
|---:|---:|---:|---:|---:|---:|
| 8 | 2.026 | 2.019 | 3.456 | 3.452 | 0.1% |
| 32 | 3.521 | 3.521 | 4.994 | 4.994 | 0.0% |
| 64 | 4.826 | 4.826 | 6.596 | 6.596 | 0.0% |
| 128 | 7.564 | 7.564 | 10.445 | 10.446 | 0.0% |
| 512 | 25.099 | 19.870 | 31.474 | 25.268 | 19.7% |
| 1024 | 49.258 | 34.532 | 61.382 | 43.618 | 28.9% |
| 2048 | 93.545 | 55.519 | 116.161 | 72.843 | 37.3% |

## 最终端到端与 FP16

| batch | FP16 TTFT | INT3 TTFT：之前 → 当前 | TPOT：FP16 → 当前 INT3 | FP16 E2E | INT3 E2E：之前 → 当前 |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.22 | 18.18 → 18.75 | 4.940 → 4.237 | 651.61 | 552.75 → 556.80 |
| 2 | 27.63 | 26.94 → 24.80 | 6.671 → 5.814 | 874.79 | 764.91 → 763.12 |
| 4 | 34.40 | 45.22 → 37.73 | 5.571 → 5.481 | 741.98 | 736.72 → 733.82 |
| 8 | 51.80 | 85.98 → 61.89 | 5.596 → 5.756 | 762.51 | 818.98 → 792.91 |
| 16 | 86.08 | 158.17 → 107.05 | 6.035 → 6.333 | 852.57 | 965.73 → 911.40 |
| 32 | 172.98 | 314.27 → 207.38 | 9.180 → 7.546 | 1338.87 | 1269.90 → 1165.75 |
| 64 | 341.31 | 624.76 → 417.08 | 12.189 → 10.498 | 1889.31 | 1958.27 → 1750.36 |
| 128 | 679.41 | 1249.69 → 834.20 | 18.220 → 17.741 | 2993.29 | 3514.52 → 3087.26 |

job **39292** 在同一张 A100 上顺序运行 FP16、previous 和 production，各自独立进程，全部为正式入口。表中单位 ms。previous 保留上一轮 decode 优化，仅关闭本轮 prefill 策略。

bs128 **TTFT 1249.69 → 834.20 ms（降低 33.2%）**，完整请求 **3514.52 → 3087.26 ms（降低 12.2%）**。decode 分派没有修改，TPOT 差别包含运行波动。

当前 bs128 完整请求相对本次 FP16 的 2993.29 ms 仍慢 3.1%；INT3 prefill 与 FP16 仍有差距。

FP16 使用同一 vLLM 部署自动选择的 Triton unquantized MoE。日志缺少对应 A100 E64/N1408 的调优文件，使用内置默认配置；这不是 FP16 的性能上限。

## 正确性与实际路径

job **39291** 使用冻结的最终源码与重编后的 sm_80 CUDA 扩展：

- Triton grouped TC：**224 项通过**，最大 relative L2 = 0.00017936；含真实 GS64 及 GS32/64/128 小形状。
- CUDA 32 行 tile：**112 项通过**，最大 relative L2 = 0.00115999；真实模型和小形状均为 GS64。
- 线性层 previous / production：**768 项通过**，最大 relative L2 = 0.000232177。
- routed 检查覆盖 M1…2048，包括 127/128/129、255/256/257、511/512/513、strided 输入、热点/随机/零权重路由、CUDA Graph 重放和预先填毒的 workspace；以独立反量化后 FP32 GEMM 为参考，关闭 TF32。
- 正式端到端：优化前后 **97,920 个输出 token ID 全部相同**（8 档 batch，每档 3 次完整生成）。

最终源码与二进制 91 个文件 SHA256 与冻结副本一致；CUDA 二进制 SHA256 为 `12aa730a5953a881a368fb04012f10b7cca44a3052c32fc4afb5f634d667c80d`。

job **39293** 采集最终实际 prefill 路径，与 39272 对照。以下为每层最后一次 prefill 调用的分阶段 CUDA event 合计，单位 ms；包含插桩影响，不作为干净端到端数据。

| batch | 实际 M | attention 投影 | shared 投影 | routed 总计 | align | W13 | W2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 512 | 6.32 → 5.02 | 6.29 → 5.56 | 25.62 → 20.17 | 0.69 → 0.56 | 13.61 → 10.05 | 7.63 → 5.72 |
| 8 | 1024 | 11.36 → 8.29 | 11.17 → 8.69 | 50.27 → 33.93 | 1.00 → 0.81 | 25.69 → 19.44 | 21.02 → 9.87 |
| 16 | 2048 | 20.79 → 15.64 | 20.75 → 16.33 | 94.49 → 53.27 | 6.12 → 1.09 | 44.10 → 30.82 | 40.48 → 15.80 |
| 32 | 2048 | 21.03 → 15.99 | 21.12 → 16.81 | 95.67 → 54.79 | 6.19 → 1.13 | 44.59 → 31.78 | 41.06 → 16.29 |
| 128 | 2048 | 20.98 → 15.83 | 21.06 → 16.70 | 95.43 → 54.42 | 6.20 → 1.13 | 44.50 → 31.53 | 40.90 → 16.17 |

bs32/128 分别有 2/8 次 M2048 调用，上表只保存最后一个 chunk。M256 的 shared 分支与 routed 分支重叠，event 时长包含资源竞争，也不能视作独立 kernel 延迟。

本轮请求使用重复相同 prompt，不代表全部业务流量或模型质量评估。INT3 优化前后比较生成 token；不要求量化模型与原始 FP16 模型的输出相同。

## 下一步优先级

最终 M2048 采样中，W13 约 30.8 ms、W2 约 15.8 ms，两次 GEMM 占 routed 总时间约 88%。attention 与共享投影另占约 32 ms。接下来优先减少 W13 的解包指令与寄存器压力，研究更有效的加载 / MMA 流水；8-warp 的简单增配本轮已证实回退。

SiLU 与 top-k reduce 合计约 4.6 ms，即使完全消除也只覆盖 routed 的约 9%，完整请求收益更小。W13 + SiLU epilogue 融合仍值得实验，但必须同时产出 gate/up 两半，并维持 FP16 舍入边界，需重新检查寄存器占用与完整算子收益。共享 gate_up + SiLU 是另一个范围较小的融合候选。

A100 这里执行的是 W3A16：INT3 解包后仍使用 FP16 Tensor Core MMA。大 M 时 FP16 GEMM 本身效率很高，因此 3-bit 不意味着 3 倍速度，权重带宽缩减也不会等比例反映到整模型。

## 复现与回退

```bash
export BRMOE_LINEAR_BACKEND=auto
export BRMOE_CUDA_FUSE=1
export BRMOE_PREFILL_BACKEND=auto  # 默认；legacy 回退本轮 prefill 优化

# 同卡独立进程：FP16、上一轮全 INT3、本轮正式版本。
CASES='fp16 previous production' sbatch --export=ALL bench/a100_full_int3_study.slurm

# 采集真实 prefill；输出目录见 job 日志（含 real_inputs.pt / linear_weights.pt）。
PHASE=collect ARGS='--prefill-only --linear-config previous --batch-sizes 4,8,16,32,128' \
  sbatch --partition=a100 --export=ALL bench/int3_linear_study.slurm

# 使用真实输入目录回放完整 MoE。
ARGS='--input /absolute/path/to/prefill_capture --m-values 512,1024,2048 --configs previous,production' \
  sbatch --export=ALL bench/prefill_grouped_study.slurm

# 正式 kernel、CUDA Graph 与线性层的独立数值检查。
PHASE=validate INPUT=/absolute/path/to/prefill_capture \
  sbatch --export=ALL bench/prefill_grouped_study.slurm
```

CUDA 32 行 tile 需要重新编译扩展；旧扩展会保留 16 行路径，Triton 大 prefill 优化仍可用：

```bash
cd BR-MoE/kernels/marlin_int3_moe
BRMOE_CUDA_ARCH=sm_80 BRMOE_ABI=1 python setup_moe.py build_ext --inplace
cp brmoe_moe_int3.cpython-312-x86_64-linux-gnu.so brmoe_moe_int3_sm80.cpython-312-x86_64-linux-gnu.so
```

ABI / Python 后缀需与实际环境一致。测试采用 CUDA 12.9、PyTorch 2.13.0+cu129、Triton 3.7.1、vLLM `27a94d1ce4e3fc100c4732439ccec10f8246a804`。CUDA 二进制指纹、冻结源码、路由统计、候选结果和最终数据见 [机器可读结果](perf/prefill_grouped_20260926.json)。

[调用逻辑 HTML](brmoe-kernel-paths.html) · [上一轮 A100 报告](a100_full_int3_20260926.md)。
