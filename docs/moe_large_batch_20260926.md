# 全 INT3 大 batch：真实路由、分阶段采样与 CUDA 流水线对照

> 后续更新：共享专家与 attention 线性层已优化，并完成 [A100 全 INT3 / FP16 对照](a100_full_int3_20260926.md)。下文保留当时的 5090 结果和资源限制实验；复现其旧线性路径需设置 `BRMOE_LINEAR_BACKEND=legacy`。

## 结论

5090 上，已验证的外围融合继续改善 bs64/128：TPOT 分别由 **15.87 → 14.17 ms**、
**20.85 → 19.68 ms**。大 batch 的主要成本同时分布在 routed MoE、共享专家和 attention
的 INT3 线性投影；只继续调整 routed MoE 的 CUDA tile，已不足以解决整个模型的延迟。

本轮另实现了两个默认关闭的 CUDA 实验选项：按实际需求申请 shared memory，
以及编译时限制寄存器以容纳两个 CTA。组合方案让 bs128 完整 routed MoE 回放减少约 **6.2%**，
但端到端 bs32/128 回退，**不提升为默认配置**。

这些测试始终使用 attention + MoE 全 INT3 的同一模型，checkpoint 目录名为
`brmoe-3bit-vllm-int3dense`。目录名不是本轮增加了另一条模型比较线。

## 1. 测量范围

- RTX 5090；PyTorch `2.13.0+cu129` / Triton `3.7.1`。
- 128 输入 / 128 输出，CUDA Graph，prefix cache 关闭，每个 batch 重复 3 次。
- TPOT = `(E2E − TTFT) / 127`。TTFT 通过另一次单输出 token 请求测量。
- batch 为 32 / 64 / 128；每个 batch 内重复相同 prompt。不能代表任意混合请求。
- 真实路由采集覆盖 27 个 routed MoE 层，保存每个 batch 最后一次实际 decode 调用的
  x、top-k IDs、权重，共 **81 组**。不是用随机路由替代真实路由。
- 完整 routed MoE 回放包含 align、gather、W13、SiLU×up、W2、加权归约与输出转换。
  回放按顺序执行 27 层各自真实权重及采样输入；不包含 gate/top-k、shared experts 或 attention。
- 分阶段事件采样覆盖上述 MoE 与 **112 个有唯一名称的 INT3 线性投影**。
  采样程序与干净端到端计时分开运行；MoE 总区间包含子阶段，不能再与它们相加。

可随仓库查阅的数字与二进制 SHA256：
[汇总 JSON](perf/moe_large_20260926.json)。

## 2. 当前融合版：完整算子与端到端

| batch | 27 层 routed MoE 原版 ms | 融合 ms | TPOT 原版 ms | 融合 ms | TPOT 降幅 |
|---:|---:|---:|---:|---:|---:|
| 32 | 3.906 | 1.792 | 9.832 | 8.721 | 11.3% |
| 64 | 4.553 | 2.408 | 15.868 | 14.173 | 10.7% |
| 128 | 6.086 | 3.600 | 20.850 | 19.682 | 5.6% |

原版为 grouped GEMV / CUDA fusion 两个实验开关均关闭。融合版开启 `BRMOE_CUDA_FUSE=1`，
保留原 CUDA 矩阵乘的 `(thread_n, thread_k, stages)=(128,128,4)`。

完整算子：`run_39165/micro.json`；同卡端到端：`run_39153/e2e_{baseline,fused4}.json`。
两次端到端的 **86,016 个输出 token ID 全部相同**。

## 3. 路由与真正占时间的部分

### 最后一次 decode 的路由

| batch | 每层平均活跃专家 | 活跃专家范围 | padding16 / 有效路由 |
|---:|---:|---:|---:|
| 32 | 16.41 | 14–18 | 1.444× |
| 64 | 23.15 | 19–28 | 1.228× |
| 128 | 31.74 | 25–41 | 1.134× |

batch 增大后 padding 相对浪费下降，活跃专家和实际计算量上升。
平均专家数与 padding 系数来自本轮最后一次 decode；不能代表全部 128 步的路由轨迹。

### 融合版的分阶段事件区间（ms）

| batch | routed MoE 总区间 | 其中 W13 / W2 | 共享专家 INT3 线性层 | attention INT3 线性层 | 第 0 层 MLP 线性层 |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.628 | 0.962 / 0.645 | 3.534 | 2.820 | 0.326 |
| 64 | 3.631 | 1.734 / 0.844 | 6.484 | 4.754 | 0.560 |
| 128 | 5.478 | 2.788 / 1.487 | 8.394 | 5.151 | 0.622 |

来源：`run_39165/stages.json`，最后一次实际 decode 的 device event。
这是加入事件采样的模型执行区间，包含区间内等待与采样扰动；不是 CUPTI kernel 时间之和，
也不是干净端到端 TPOT，不能直接拼加后解释为 TPOT。
attention 列仅为 qkv / o 投影，不包含 attention 核心、KV cache 或其他模型操作。

**后续优先检查共享专家和 attention 的 INT3 GEMM。** 当前
`tools/brmoe_int3_vllm/linear_method.py:pick_tiles` 在大 M 下仍返回 `slot=16`，
即使线性层只有一个专家，`block_m=32/64` 仍拆成多个专家子块计算。
下一轮应比较 `slot=block_m` 对权重加载、解包复用与寄存器占用的影响，随后再调
BLOCK_K / stages；这是代码检查得到的候选方向，本轮没有将它作为已验证收益。

## 4. CUDA 对照：stage 数、共享内存与寄存器

### 4.1 原始二进制，只改流水级数

所有数字为 27 层完整 routed MoE，单位 ms。N/K tile 均为 128。

| batch | stages=4 | stages=3 | stages=5 |
|---:|---:|---:|---:|
| 32 | 1.792 | 1.799 | 1.823 |
| 64 | 2.408 | 2.452 | 2.436 |
| 128 | 3.600 | 3.723 | 3.604 |

原配置 4 级仍更合适。既往 `(N=64,K=128)` 和 `(N=128,K=64)` 配置存在数值失败记录
（39043 / 39052），本轮没有将这些结果不正确的配置当作性能候选。

### 4.2 按实际布局申请共享内存

旧 launcher 对所有 tile/stages 固定申请 **96 KiB 动态共享内存**。
新运行时开关 `BRMOE_MOE_SMEM=rightsize` 根据 A、B、scale/zero 流水缓存的实际布局计算申请量，
并保留覆盖归约与 split-K 写出复用区间的保守下界。数据布局与计算流程保持相同。

本模型 gs=64、N/K tile=128 时：3 / 4 / 5 级分别申请 **39 / 52 / 65 KiB**；
编译资源另有 **1 KiB 静态共享内存**。

但原核每线程使用 **134–138 个寄存器**，仍受寄存器限制。
`run_39159` 的完整 MoE 对照中，单独减少共享内存基本无收益，不能声称它已经提高并发或性能。

### 4.3 组合限制寄存器

新增编译选项 `BRMOE_MOE_MIN_BLOCKS=2`，对 MoE kernel 使用 launch bounds。
编译后的 3 级流水使用 128 个寄存器/线程；有 32 字节/线程的栈空间。
配合 39 KiB 动态共享内存，profile 记录两个 CTA/SM 的资源上限。
这不是实测平均 occupancy；该工具对原 96 KiB 配置给出的 0 occupancy 估计无效，不用它作比较。

**同卡、两个独立进程回放（39169）：**

| batch | 当前融合版 ms | bounds2 + rightsize + stages3 ms | 变化 |
|---:|---:|---:|---:|
| 32 | 1.788 | 1.797 | 慢 0.5% |
| 64 | 2.402 | 2.337 | 快 2.7% |
| 128 | 3.600 | 3.376 | 快 6.2% |

**同卡、两个干净引擎的端到端（39166）：**

| batch | 当前融合 TPOT ms | 候选 TPOT ms | 变化 |
|---:|---:|---:|---:|
| 32 | 8.663 | 9.240 | 慢 6.7% |
| 64 | 14.184 | 13.922 | 快 1.8% |
| 128 | 19.980 | 20.324 | 慢 1.7% |

两组仍有 **86,016 个输出 token 全部一致**。端到端结果不支持全局开启该组合。
最后一个 decode 输入的独立回放无法覆盖所有步的路由、缓存状态和模型中其他算子的影响。
bs64 的小幅收益也不足以在这一轮就确定通用分派策略。

## 5. 验证、复现与已排除的轮次

- 39154：仅 rightsize 的最终 80 组数值/Graph 回放验证通过。
- 39161：bounds2 + rightsize + stages3 的 80 组验证通过。
- 覆盖 M=1/2/4/8/16/32/64/512，随机/集中/零权重路由、缓存填 NaN 后重放、
  非连续输入/权重、split-K=1/2 及两个投影使用不同 split 设置。
- CUDA 编译实验始终在独立源码/二进制目录中运行，没有替换主工作区默认 `.so`。
- 39150：采样计数器的 inference tensor 在普通模式下清零报错；已改用 inference mode。
- 39158：真实输入及 routed MoE 回放有效；attention 的空 prefix 造成事件键覆盖，
  其 linear 汇总作废。39165 改为加载期真实 `named_modules()` 路径并断言 112 个唯一线性层。
- 39164：同进程用同一 Python 扩展名加载两个 `.so` 时复用了原模块，
  不能作为新旧二进制对照。39169 改为两个独立进程，并记录实际 `.so` 路径和 SHA256。

本机完整产物位于 `bench_results/moe_large/`，实验脚本为
`bench/moe_large_batch_study.py` / `.slurm`。原始 tensor 和大 trace 没有纳入 Git。

```bash
# 采集 bs32/64/128 真实输入和分阶段事件，再做完整 routed MoE 流水线对照
PHASE=collect sbatch -p 5090 --export=ALL bench/moe_large_batch_study.slurm

# 原版 / 当前融合版：同卡全 INT3 端到端
PHASE=e2e_pair sbatch -p 5090 --export=ALL bench/moe_large_batch_study.slurm

# rightsize / bounds2 的 A/B 使用独立快照，避免覆盖默认扩展。
# 在 candidate 源码副本中构建（不是主工作区）：
# BRMOE_CUDA_ARCH=sm_120 BRMOE_MOE_MIN_BLOCKS=2 python setup_moe.py build_ext --inplace
SRC=/absolute/candidate/source REFERENCE_SRC=/absolute/reference/source \
  PHASE=e2e_ab sbatch -p 5090 --export=ALL bench/moe_large_batch_study.slurm
```

A100 外围融合复测作业 39144 截至本轮报告仍排队；上面的新实验结论只来自 5090。
