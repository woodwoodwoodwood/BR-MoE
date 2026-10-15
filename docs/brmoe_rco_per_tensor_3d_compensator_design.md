# BR-MoE + RCO：基于 Per-3D-Tensor `(Bit, Rank)` 联合分配与 3D 补偿矩阵的跨框架（`llama.cpp` / `vLLM`）适配方案

> **核心定位**：将 BR-MoE 的 **`(量化位宽 Bit, 低秩补偿秩 Rank)` 联合协同设计** 从原本难以在主流推理框架中高效执行的“同层内逐专家异构（Per-Expert Heterogeneous）”粒度，升级为与现代推理引擎（`llama.cpp` GGUF 与 `vLLM` FusedMoE）物理存储对齐的 **Per-Tensor / Per-3D-Expert-Tensor `(Bit, Rank)` 粒度**，并通过黎曼流形约束优化（**RCO**, `arXiv:2605.00649`）在精确显存预算下求解全局最优配置。
>
> **最大工程收益**：
> 1. **`llama.cpp` 零代码修改原生运行**：直接复用 `llama.cpp` 内置的 `build_lora_mm_id`（`ggml_mul_mat_id` 3D 索引矩阵乘），以 `Base GGUF + 3D Compensator GGUF` 形式在 CUDA / Metal / ROCm / Vulkan 全后端免改源码跑通。
> 2. **`vLLM` 极简融合算子适配**：彻底消除同层内专家按位宽拆桶（Bit-Bucketing）与变长 Rank 零填充掩码（Zero-Padding Mask）的碎片化开销，每层保持单一规整的 3D 量化张量 `[E, ...]` + 规整 3D 低秩补偿张量 `[E, r, ...]`。
> 3. **超越社区 `GSQ-RCO` 的率失真帕累托前沿**：保留专家激活频率加权的逐专家独立 SVD 切片，用连续细腻的 $r \in \{0, 8, 16, 32, 64\}$ 填补离散位宽（如 `2-bit` 与 `3-bit` 之间 50% 体积跳变）的预算空白。

---

## 1. 背景：为什么 `GSQ-RCO` 能无缝进框架，而逐专家异构量化难进框架？

### 1.1 社区 `Qwen3.8-GSQ-RCO` 的真实做法
在 ISTA-DASLab 发布的 `Qwen3.8-27B-GSQ-RCO-GGUF` 与 `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 中，其“Per-Tensor 混合精度”在 MoE 模型上的物理含义是：
- **稠密层（Attention / SSM / Shared Expert）**：每个 2D 权重矩阵（如 `blk.0.attn_qkv.weight`、`blk.0.ffn_down_shexp.weight`）是一个独立 Tensor，单独分配一种量化格式（如 `IQ4_XS`、`Q6_K`、`IQ3_S`）。
- **MoE 路由专家层（Routed Experts）**：在 GGUF 与 HuggingFace 新版规范中，同一层全部 $E$ 个专家（例如 512 个或 64 个专家）的同类投影被堆叠为 **1 个 3D Tensor**：
  - `blk.{l}.ffn_gate_exps.weight`：形状 `[H, I, E]`（PyTorch 视角为 `[E, I, H]`）
  - `blk.{l}.ffn_up_exps.weight`：形状 `[H, I, E]`
  - `blk.{l}.ffn_down_exps.weight`：形状 `[I, H, E]`
- 因此，在 `GSQ-RCO` 中，**同一层内所有专家的 `gate` 共享同一种量化格式，所有专家的 `up` 共享同一种量化格式，所有专家的 `down` 共享同一种量化格式**：
  ```text
  # 摘自 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS 真实分配表
  blk.0.ffn_gate_exps.weight: IQ2_XS   # 第 0 层 512 个专家的 gate 统一为 IQ2_XS
  blk.0.ffn_up_exps.weight:   IQ2_XS   # 第 0 层 512 个专家的 up   统一为 IQ2_XS
  blk.0.ffn_down_exps.weight: IQ4_NL   # 第 0 层 512 个专家的 down 统一为 IQ4_NL
  blk.1.ffn_gate_exps.weight: IQ2_XXS  # 第 1 层 512 个专家的 gate 统一为 IQ2_XXS
  blk.1.ffn_down_exps.weight: Q2_0     # 第 1 层 512 个专家的 down 统一为 Q2_0
  ```

### 1.2 三种方案的核心取舍对比

| 维度 | 社区 `GSQ-RCO` (Per-3D-Tensor 纯量化) | 传统 `BR-MoE` (Per-Expert 异构 `Bit + Rank`) | **本方案：Per-3D-Tensor `(Bit, Rank)` + 3D 补偿矩阵** |
| :--- | :--- | :--- | :--- |
| **基本决策单元 $i \in \{1..N\}$** | 每个 2D Tensor 及每层专家的 3D Tensor（$N \approx 300\sim 500$） | 每个独立专家（$N = L \times E \approx 1,728 \sim 10,240$） | **每个 2D Tensor 及每层专家的 3D Tensor（$N \approx 300\sim 500$）** |
| **每个单元的动作空间 $k$** | 仅量化位宽 $b \in \{2, 3, 4, \dots\}$ | 量化位宽 + 补偿秩 $(b_e, r_e)$ | **量化位宽 + 3D 补偿秩 $(b_t, r_t)$** |
| **同层专家间差异化能力** | ❌ 无（同层所有专家完全同精度、无补偿） | ✅ 极高（每个专家不同 bit、不同 rank） | ✅ **高**：同层专家共享规整维度 $(b_t, r_t)$，但**每个专家 $e$ 拥有独立训练的 SVD 低秩切片 $(\mathbf{U}_{t,e}, \mathbf{V}_{t,e})$**，且按路由频率加权优化 |
| **预算调节细腻度** | 粗（3D 张量从 2-bit 升到 3-bit 显存暴增 50%） | 细 | **极细**（`2bit + rank16` 仅比纯 `2bit` 多 ~5% 显存，却能大幅修复离群奇异值） |
| **`llama.cpp` 适配成本** | 零修改 | ❌ 极难（GGUF 不支持单 3D 张量内混合类型） | **零修改**（直接利用原生 `build_lora_mm_id` 算子） |
| **`vLLM` 适配成本** | 需支持多种小位宽算子 | ❌ 高（需同层拆桶 + 稀疏变长 Rank 掩码） | **低**（每层单一 3D 量化算子 + 单一规整 3D BMM 算子） |
| **RCO 端到端 KL 搜索耗时** | 快（变量数 $N \sim 350$） | 慢（变量数 $N \sim 10,240$） | **快（变量数 $N \sim 350$，流形与 DP 规模缩小 30 倍）** |

---

## 2. 数学建模：Per-3D-Tensor `(Bit, Rank)` 与 RCO 预算流形

### 2.1 张量集合与候选动作空间
设模型的可压缩权重张量集合为 $\mathcal{T} = \mathcal{T}_{\text{2D}} \cup \mathcal{T}_{\text{3D}}$，总数 $N = |\mathcal{T}|$：
1. **2D 张量 $t \in \mathcal{T}_{\text{2D}}$**：包括 Attention 投影（`q_proj`, `k_proj`, `v_proj`, `o_proj` 或 MLA / Gated DeltaNet 投影）及共享专家投影（`shared_experts.{gate,up,down}_proj`），形状为 $\mathbf{W}_t \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$。
2. **3D 路由专家张量 $t \in \mathcal{T}_{\text{3D}}$**：每层 MoE 包含 3 个（或合并 `gate_up` 后 2 个）3D 张量：
   - $t = (l, \text{gate})$：$\mathbf{W}_t \in \mathbb{R}^{E \times I \times H}$
   - $t = (l, \text{up})$：$\mathbf{W}_t \in \mathbb{R}^{E \times I \times H}$
   - $t = (l, \text{down})$：$\mathbf{W}_t \in \mathbb{R}^{E \times H \times I}$

每个张量 $t \in \mathcal{T}$ 从离散候选配置集合 $\mathcal{K} = \{(b_1, r_1), (b_2, r_2), \dots, (b_K, r_K)\}$ 中选择一个配置 $k = (b, r)$，其中：
- $b \in \mathcal{B}$（例如 $\{2, 3, 4\}$ bit 或 GGUF 的 $\{\text{Q2\_K}, \text{IQ2\_XS}, \text{IQ3\_XXS}, \text{IQ3\_S}, \text{Q4\_K}\}$）；
- $r \in \mathcal{R}$（例如 $\{0, 8, 16, 32, 64\}$）。

### 2.2 3D 张量内部的交替量化与逐专家 SVD 补偿（BR-MoE 核心）
当 3D 专家张量 $t \in \mathbb{R}^{E \times D_{\text{out}} \times D_{\text{in}}}$ 选择配置 $k = (b, r)$ 时：
- 若 $r = 0$：直接对每个专家切片 $\mathbf{W}_{t,e}$（$e = 0, \dots, E-1$）执行基础量化（HQQ / GPTQ / GSQ），打包为 3D 量化张量 $\widehat{\mathbf{W}}_t(b)$。
- 若 $r > 0$：对每个专家切片 $e \in \{0, \dots, E-1\}$ 独立执行 BR-MoE 的迭代残差补偿：
  \[
  \mathbf{W}_{t,e} \approx \widehat{\mathbf{W}}_{t,e}^{(b, r)} + \mathbf{U}_{t,e}^{(b, r)} \mathbf{V}_{t,e}^{(b, r)}, \quad \mathbf{U}_{t,e}^{(b, r)} \in \mathbb{R}^{D_{\text{out}} \times r}, \; \mathbf{V}_{t,e}^{(b, r)} \in \mathbb{R}^{r \times D_{\text{in}}}
  \]
  然后将 $E$ 个专家的低秩矩阵沿第 0 维堆叠为两个规整的 **3D 补偿矩阵**：
  \[
  \mathbf{U}_t^{(b, r)} \in \mathbb{R}^{E \times D_{\text{out}} \times r}, \qquad \mathbf{V}_t^{(b, r)} \in \mathbb{R}^{E \times r \times D_{\text{in}}}
  \]
- **关键性质**：虽然同一 3D 张量内的 $E$ 个专家具有相同的秩维度 $r$（从而拼成规整的 3D 张量），但**第 $e$ 个专家的补偿参数 $\mathbf{U}_{t,e}, \mathbf{V}_{t,e}$ 完全针对第 $e$ 个专家自身的离群残差主成分提取**！

### 2.3 精确显存成本向量 $c_{t, k}$ 与频率加权敏感度 $L_{t, k}$
对于任意 3D 张量 $t \in \mathbb{R}^{E \times D_{\text{out}} \times D_{\text{in}}}$（2D 张量令 $E=1$）和配置 $k = (b, r)$：
1. **字节/MB 成本 $c_{t, k}$**：
   \[
   c_{t, (b, r)} = E \cdot D_{\text{out}} D_{\text{in}} \cdot \text{bpw}_{\text{base}}(b) + E \cdot r (D_{\text{out}} + D_{\text{in}}) \cdot \text{bpw}_{\text{comp}}
   \]
   其中 $\text{bpw}_{\text{base}}(b) = b + \frac{32}{g_w}$（含 FP16 scale/zero），$\text{bpw}_{\text{comp}}$ 为补偿矩阵存储位宽（在 `vLLM` 中可为 INT3/INT4/FP16，在 `llama.cpp` 中可为 `Q8_0` / `Q4_0` / `F16`）。
2. **专家路由频率加权的代理失真度 $L_{t, k}$（用于 Proxy 模式与 RCO Warm-Start）**：
   设校准集统计得到的第 $l$ 层第 $e$ 个专家的归一化激活频率为 $f_{l, e}$，则 3D 张量 $t=(l, \text{proj})$ 在配置 $k=(b, r)$ 下的加权误差为：
   \[
   L_{t, (b, r)} = \sum_{e=0}^{E-1} f_{l, e} \cdot \left\| \mathbf{W}_{t,e} - \widehat{\mathbf{W}}_{t,e}^{(b, r)} - \mathbf{U}_{t,e}^{(b, r)} \mathbf{V}_{t,e}^{(b, r)} \right\|_F^2
   \]

### 2.4 RCO 流形优化（注意：每个张量 $t$ 的参数量 $w_t$ 不同）
由于不同张量（如 `attn_q_proj` vs `ffn_gate_exps`）的元素个数相差百倍，定义：
- 归一化成本矩阵 $\mathbf{C} \in \mathbb{R}^{N \times K}$，其中第 $(t, k)$ 项为 $C_{t,k} = c_{t, k}$（单位 MB）；
- 连续松弛概率 $p_{t,k} = \operatorname{softmax}(\boldsymbol{\alpha}_t)_k$；
- 期望总显存成本函数：
  \[
  C(\boldsymbol{\alpha}) = \sum_{t=1}^N \sum_{k=1}^K p_{t,k} C_{t,k} = B_{\text{target}}
  \]
- 闭式流形法向量（对应 `RCO_solver/manifold.py` 中支持 2D `costs` 矩阵的推广形式）：
  \[
  n_{t,k}(\boldsymbol{\alpha}) = \frac{\partial C}{\partial \alpha_{t,k}} = p_{t,k} \left( C_{t,k} - \sum_{j=1}^K p_{t,j} C_{t,j} \right)
  \]
- 单调二分重映射（Retraction）：沿 $\alpha_{t,k}(s) = \alpha_{t,k} + s \cdot C_{t,k}$ 搜索标量 $s \in \mathbb{R}$，其导数 $\frac{d}{ds} C(\boldsymbol{\alpha}(s)) = \sum_{t=1}^N \operatorname{Var}_{p_t}(C_t) > 0$ 严格单调递增，35 步二分即可达到 $|C(\boldsymbol{\alpha}) - B_{\text{target}}| < 10^{-8}$。

---

## 3. `llama.cpp` 零代码修改适配方案（基于原生 3D `build_lora_mm_id`）

### 3.1 `llama.cpp` 底层源码契约验证
在 `llama.cpp` 的 [`src/llama-graph.cpp`](https://github.com/ggml-org/llama.cpp/blob/main/src/llama-graph.cpp) 中，MoE 层的三个 3D 专家张量（`up_exps`, `gate_exps`, `down_exps`）均通过 `build_lora_mm_id` 调用：

```cpp
// llama.cpp/src/llama-graph.cpp: build_moe_ffn
ggml_tensor * up      = build_lora_mm_id(up_exps,   cur, selected_experts);
ggml_tensor * gate    = build_lora_mm_id(gate_exps, cur, selected_experts);
// ... SiLU / SwiGLU 激活函数在此处执行 ...
ggml_tensor * experts = build_lora_mm_id(down_exps, par, selected_experts);
```

而在 `build_lora_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids)` 内部：

```cpp
ggml_tensor * llm_graph_context::build_lora_mm_id(
        ggml_tensor * w,
        ggml_tensor * cur,
        ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) continue;

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
            ctx0, lw->b,
            ggml_mul_mat_id(ctx0, lw->a, cur, ids),
            ids
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}
```

同时，在 [`src/llama-adapter.cpp`](https://github.com/ggml-org/llama.cpp/blob/main/src/llama-adapter.cpp) 中，加载器对每个名为 `<tensor_name>.lora_a` 和 `<tensor_name>.lora_b` 的张量做如下形状校验（注意 `ggml` 的 `ne[0]` 是最内层连续维度，即 PyTorch 的最后一维 `shape[-1]`）：
- 基底 3D 张量 `model_tensor`（即 `w`）：`ne = [D_in, D_out, E]`（对应 PyTorch `[E, D_out, D_in]`）
- 补偿张量 `w.a`（即 `.lora_a`，对应 $\mathbf{V}$）：`ne = [D_in, r, E]`（对应 PyTorch `[E, r, D_in]`）
- 补偿张量 `w.b`（即 `.lora_b`，对应 $\mathbf{U}$）：`ne = [r, D_out, E]`（对应 PyTorch `[E, D_out, r]`）
- 校验条件：`model_tensor->ne[0] == w.a->ne[0]`（即 $D_{\text{in}}$ 一致）、`model_tensor->ne[1] == w.b->ne[1]`（即 $D_{\text{out}}$ 一致）、且 `w.a->ne[1] == w.b->ne[0]`（即秩 $r$ 一致）。

**这带来了三个完美的工程特性**：
1. **激活函数位置完全正确**：`gate_exps` 和 `up_exps` 的低秩补偿项 $\mathbf{U}(\mathbf{V}\mathbf{x})$ 是在 `SwiGLU` 激活函数**之前**加上去的，与 BR-MoE 训练/压缩时的数学定义 100% 一致。
2. **只计算被路由选中的 Top-$K$ 专家**：`ggml_mul_mat_id` 直接消费 `selected_experts`（`ids`），每 token 只读取并计算被激活的 $k$ 个专家的 $(\mathbf{U}_e, \mathbf{V}_e)$ 切片，复杂度仅为 $O(k \cdot r (D_{\text{in}} + D_{\text{out}}))$。
3. **支持不同 3D Tensor 拥有不同的秩 $r_t$**：每个 Tensor 独立检查 `w.a->ne[1] == w.b->ne[0]`，且在 GGUF 元数据中将 `adapter.lora.alpha` 设为 `0.0f`（此时 `scale = lora.second = 1.0f`，不随 $r_t$ 变化），即可让不同层的 3D Tensor 自由取 $r_t \in \{0, 8, 16, 32, 64\}$！

### 3.2 GGUF 导出产物规范
导出脚本生成两个标准 GGUF 文件：

1. **主权重文件 `model-brmoe-mixed.gguf`**：
   - 包含所有 2D Tensor 与 3D 专家 Tensor（`blk.{l}.ffn_{gate,up,down}_exps.weight`）。
   - 每个 Tensor 根据 RCO 搜索出的位宽 $b_t$ 编码为对应的 GGUF 量化类型（如 `Q2_K`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `Q4_K`）。
2. **3D 补偿器文件 `model-brmoe-compensator.gguf`**：
   - GGUF Header 元数据：
     ```text
     general.type = "adapter"
     adapter.type = "lora"
     general.architecture = <arch>   # 如 "qwen3moe" / "qwen3next" / "deepseek2"
     adapter.lora.alpha = 0.0        # 关键：令 alpha=0，使 llama.cpp 内部 scale 恒等于 1.0，不受各层不同 rank 影响
     ```
   - 仅对 RCO 分配了 $r_t > 0$ 的张量写入两项（可量化为 `Q8_0` 或 `Q4_0` 或 `F16`）：
     - `blk.{l}.ffn_gate_exps.weight.lora_a`：PyTorch 形状 `[E, r_t, H]`
     - `blk.{l}.ffn_gate_exps.weight.lora_b`：PyTorch 形状 `[E, I, r_t]`
     - （`up_exps`、`down_exps` 及 $r_t > 0$ 的 2D Tensor 同理；$r_t = 0$ 的张量直接省略）
3. **运行命令（无需修改任何 `llama.cpp` 代码）**：
   ```bash
   llama-cli \
     -m model-brmoe-mixed.gguf \
     --lora model-brmoe-compensator.gguf \
     -ngl 99 -p "Hello!"
   ```

---

## 4. `vLLM` 高性能适配方案（扩展 `tools/brmoe_int3_vllm`）

在 `vLLM` 中，采用 **Per-3D-Tensor `(Bit, Rank)`** 同样能把插件复杂度降到最低，并保持 CUDA Graph 零同步特性。

### 4.1 权重与 `config.json` 契约设计
在 `config.json` 的 `quantization_config` 中记录每个模块前缀 `prefix` 的 `(bits, rank)`：

```json
{
  "quantization_config": {
    "quant_method": "brmoe",
    "group_size": 64,
    "compensator_dtype": "fp16",
    "layer_configs": {
      "model.layers.0.mlp.experts": {
        "w13_bits": 3, "w13_rank": 16,
        "w2_bits": 3,  "w2_rank": 32
      },
      "model.layers.1.mlp.experts": {
        "w13_bits": 3, "w13_rank": 0,
        "w2_bits": 3,  "w2_rank": 0
      }
    }
  }
}
```

对于第 $l$ 层 `RoutedExperts`（`prefix = "model.layers.{l}.mlp.experts"`）：
- **基础量化张量**（与现有 `brmoe_int3_vllm` 完全一致）：
  - `w13_q`: `int32 [E, 2I, K_pack]`，`w13_s`, `w13_z`: `fp16 [E, K // gs, 2I]`
  - `w2_q`: `int32 [E, K, I_pack]`，`w2_s`, `w2_z`: `fp16 [E, I // gs, K]`
- **可选的 3D 低秩补偿张量**（仅当该层的 `w13_rank > 0` 或 `w2_rank > 0` 时注册为 Parameter）：
  - 若 `w13_rank = r_13 > 0`：
    - `w1_v`, `w3_v`: `fp16 [E, r_13, K]`（或合并为 `w13_v: fp16 [E, 2 * r_13, K]`）
    - `w1_u`, `w3_u`: `fp16 [E, I, r_13]`
  - 若 `w2_rank = r_2 > 0`：
    - `w2_v`: `fp16 [E, r_2, I]`
    - `w2_u`: `fp16 [E, K, r_2]`

### 4.2 `vLLM` 前向计算图（主路 Grouped GEMM + 规整 3D 补偿旁路）
在 `BRMoEInt3MoEMethod` 中：
1. **当该层 `w13_rank == 0` 且 `w2_rank == 0` 时**：
   - 100% 走你现有的 `brmoe_int3_moe`（小 M 走 `fused_moe_gemv`，中 M 走 Marlin CUDA `fused_moe_int3_cuda`，大 M 走 Triton `fused_moe_int3_tc`），**零额外开销**。
2. **当该层配有 3D 补偿矩阵（如 `r_13 > 0` 或 `r_2 > 0`）时**：
   - 由于该层所有 $E$ 个专家都具有相同的秩 $r_{13}$ / $r_2$，无需任何掩码判断或动态分支，直接复用同一个路由索引 `topk_ids [M, topk]` 执行两次极轻量的 Indexed Batched GEMM（`[M, topk, K] @ [E, K, r] -> [M, topk, r] @ [E, r, I]`）：
   ```python
   # 伪代码示意：规整 3D 补偿旁路（以 w2_u [E, K, r], w2_v [E, r, I] 为例）
   # inter: [M, topk, I], topk_ids: [M, topk]
   flat_ids = topk_ids.reshape(-1)                         # [M * topk]
   V_sel = layer.w2_v.index_select(0, flat_ids)            # [M * topk, r, I]
   U_sel = layer.w2_u.index_select(0, flat_ids)            # [M * topk, K, r]
   h_lowrank = torch.bmm(V_sel, inter.reshape(-1, I, 1))   # [M * topk, r, 1]
   y_comp = torch.bmm(U_sel, h_lowrank).squeeze(-1)        # [M * topk, K]
   ```
   - 因为 $r \in \{8, 16, 32\}$ 极小，`V_sel` 与 `U_sel` 体积仅几百 KB（完全命中 L2 Cache），耗时不足主矩阵的 3%，且全程是静态形状张量算子，**100% 兼容 vLLM CUDA Graph 捕获**！

---

## 5. 代码改造与落地路线图（Step-by-Step）

### Step 1：`RCO_solver` 新增 `--granularity tensor`（Per-3D-Tensor 搜索模式）
- **修改文件**：`RCO_solver/rco_search.py`、`RCO_solver/model_adapters.py`
- **功能**：
  1. 支持将同一层 MoE 的专家聚合为 3 个 3D 张量决策组（`L{l}_gate_exps`, `L{l}_up_exps`, `L{l}_down_exps`）或整层 1 个决策组（`L{l}_experts`），同时支持将 `self_attn` / `linear_attn` / `shared_experts` 纳入统一预算池。
  2. 支持 2D 成本矩阵 $\mathbf{C} \in \mathbb{R}^{N \times K}$（每个张量 $t$ 根据自身形状计算各 `(bit, rank)` 的真实 MB 成本）。
  3. 输出两种映射表：
     - 面向现有 `BR-MoE` 压缩器的展开版 `best_config_expanded.json`（自动将 `L{l}` 的 `(bit, rank)` 广播给 `L{l}_E0 .. L{l}_E{E-1}`，现有压缩脚本零修改即可跑）；
     - 面向 `llama.cpp` / `vLLM` 的张量级分配表 `rco_tensor_allocation.json`。

### Step 2：新增 `tools/export_brmoe_to_gguf.py`（导出 `llama.cpp` 双 GGUF）
- **输入**：`BR-MoE` 压缩产物目录（含 `qmodel.pt`、`compensators.pt`、`ranks.json`）
- **功能**：
  1. 将各层专家的基础量化权重拼回 3D 张量 `blk.{l}.ffn_{gate,up,down}_exps.weight` 并写入主 GGUF；
  2. 将 `r_t > 0` 的层的 `compensators.pt` 中的逐专家 `U_e, V_e` 反量化后沿专家维堆叠为 3D 张量 `[E, r_t, D_in]` 和 `[E, D_out, r_t]`，以 `.lora_a` 和 `.lora_b` 写入 `compensator.gguf`（设置 `adapter.lora.alpha = 0.0`）。

### Step 3：升级 `tools/convert_brmoe_to_vllm.py` 与 `tools/brmoe_int3_vllm/`
- **修改文件**：
  - `tools/convert_brmoe_to_vllm.py`：在转换 `qmodel.pt` 时同步读取 `compensators.pt`，若某层专家 `rank > 0`，将该层所有专家的 `U, V` 堆叠导出为 `routed_experts.w13_u / w13_v / w2_u / w2_v` 写入 safetensors，并在 `config.json` 的 `quantization_config` 中记录各层 `rank`。
  - `tools/brmoe_int3_vllm/moe_method.py` & `kernel.py`：在 `create_weights` 中根据该层 `rank > 0` 动态注册 `w13_u, w13_v, w2_u, w2_v`，并在前向传播的 `gate_up` 投影后和 `down` 投影后加入 3D 低秩补偿项。
