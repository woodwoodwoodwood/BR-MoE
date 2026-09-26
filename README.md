# BR-MoE: Bi-Level Tuning of Mixed-Precision Quantization and Low-Rank Compensators for Mixture-of-Experts Models

**Official Implementation of "BR-MoE: Bi-Level Tuning of Mixed-Precision Quantization and Low-Rank Compensators for Mixture-of-Experts Models"**


## 🎯 Overview

BR-MoE introduces a novel framework that jointly optimizes mixed-precision quantization and low-rank compensation for Mixture-of-Experts (MoE) models. Unlike existing approaches that treat bit-width selection and compensator rank allocation as independent problems, BR-MoE discovers and exploits the **hidden coupling** between these two dimensions through principled global optimization.

### 🔑 Key Innovations

- **Discovery of Bit-Rank Coupling**: First work to identify and formalize the complex interdependency between quantization bit-width and compensator rank in MoE models
- **Global Optimization Framework**: Transforms the intractable combinatorial problem into a solvable Integer Linear Programming (ILP) formulation
- **Efficient Proxy Metric**: Layer-wise quantization loss enables rapid evaluation of thousands of configurations without full model retraining
- **Superior Performance**: Achieves better accuracy-memory trade-offs than state-of-the-art methods across multiple MoE architectures

### 📊 Results Highlights

| Model | Method | Memory | WikiText2 PPL↓ | Average Score↑ |
|-------|---------|--------|---------------|----------------|
| Mixtral-8×7B | FP16 | 88.90GB | 3.700 | 80.48 |
| | GPTQ-3bit | 18.43GB | 4.730 | 73.80 |
| | HQQ-3bit | 20.55GB | 4.612 | 71.93 |
| | **BR-MoE** | **20.36GB** | **4.095** | **77.83** |
| DeepSeek-MoE | FP16 | 31.24GB | 5.832 | 68.82 |
| | GPTQ-3bit | 6.97GB | 6.843 | 62.14 |
| | **BR-MoE** | **8.18GB** | **6.180** | **67.70** |

### ⚡ 推理性能（vLLM 端到端, DeepSeek-MoE-16B 3-bit, 2026-09-26）

自研 Triton int3 kernel 栈（`BR-MoE/kernels/triton_int3/` + `tools/brmoe_int3_vllm/`），
decode 小 batch 走 **GEMV + K-major + split-K** 路径，大 batch / prefill 走张量核心
grouped GEMM。算子流程框图：[decode](docs/pipeline_decode.md) / [prefill](docs/pipeline_prefill.md)。

**RTX 5090（graph 模式, in=128 out=128, TPOT ms/tok ↓）**

| bs | int3dense 优化前 | brmoe3bit 优化前 | brmoe3bit 最终 | int3dense 最终 | +grouped GEMV¹ |
|---|---|---|---|---|---|
| 1 | 8.10 | 6.04 | **3.01** | **2.26** | — |
| 2 | 8.82 | 6.69 | **3.37** | **2.95** | — |
| 4 | 9.68 | 7.75 | **3.88** | **3.65** | **3.04** |
| 8 | 11.68 | 9.44 | **5.19** | **5.88** | **4.63** |
| 16 | 12.62 | 11.03 | **6.19** | **7.52** | **6.12** |

¹ 实验开关 `BRMOE_GROUPED_GEMV=1`（默认关，仅 sm_120 且 4≤M≤16），
详见 [docs/grouped_gemv_study_20260926.md](docs/grouped_gemv_study_20260926.md)。

"最终" = GEMV（小 M）+ Marlin CUDA grouped kernel（中段 M）+ Triton TC（大 M）
三级分派全部启用且数值校验通过（kernel 执行路径框图：
[docs/brmoe-kernel-paths.html](docs/brmoe-kernel-paths.html)）。

**Kernel 级：int3 分派路径 vs fp16（同一条 grouped 流水线，5090，zipf 路由）**

| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| int3 µs | 25.2 | 41.8 | 55.8 | 106.4 | 181.0 | 217.4 | 256.2 |
| fp16 µs | 116.7 | 150.7 | 218.3 | 331.0 | 423.7 | 569.6 | 654.8 |
| **加速比** | **4.64×** | **3.60×** | **3.91×** | **3.11×** | **2.34×** | **2.62×** | **2.56×** |

MoE GEMV 微基准在 5090 上 M=4 达 **1766 GB/s ≈ 98% 峰值带宽**；Marlin CUDA grouped
kernel 在 M=16~64 达 ~910 GB/s（50% 峰值）并 1.9× 于 Triton 张量核路径；
attention int3 linear 在 M=1 比 cuBLAS fp16 还快（4.3 µs vs 5.8 µs, **0.74×**）。

**A100 80GB PCIe（graph 模式, 同口径, 与 fp16 同卡对比）**

| bs | fp16 | brmoe3bit 前→后 | int3dense 最终 | brmoe3bit 最终 vs fp16 |
|---|---|---|---|---|
| 1 | 4.92 | 7.95 → **4.38** | **4.24** | **0.89× ✅ 反超** |
| 2 | 6.78 | 9.71 → **5.67** | **5.89** | **0.84× ✅ 反超** |
| 4 | 6.82 | 9.91 → **7.31** | 8.78 | 1.07× |
| 8 | 6.78 | → **8.42** | 12.66 | 1.24× |
| 16 | 8.35 | → **9.89** | 12.63 | 1.18× |
| 32 | 10.07 | → **11.30** | 16.50 | 1.12× |

A100 上 prefill 改善显著（TTFT bs1 31→22 ms，反超 fp16 的 26 ms）。
显存：权重 fp16 30.5 GiB → int3 **8.1 GiB**（-73%），KV cache 空间相应放大。

已知边界（诚实记录）：A100 bs≥4 的 decode 仍落后 fp16——Marlin CUDA kernel 在
sm_80 上只到 ~25% 峰值带宽（5090 为 50%），且 GEMV 逐路由重读权重在 M 增大后
流量反转；GEMV/CUDA 的交叉点按架构区分（sm_80: M≤2, sm_120: M≤8），由
`gemv_max_m=None` 自动选择。复现：`bench/vllm_perf_compare.slurm`、
`bench/verify_moe_cuda.py`、`bench/micro_moe.py`。

## 🚀 Quick Start

### Installation

1. **Set up conda environment**
```bash
cd BR-MoE
chmod +x conda_env_setup.sh
./conda_env_setup.sh
conda activate brmoe
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Compile CUDA kernels** (Optional for acceleration)
```bash
chmod +x kernel_setup.sh
./kernel_setup.sh
```

### Basic Usage

#### 1. Compress a MoE Model

```python
import torch
from transformers import AutoModelForCausalLM
from BR_MoE.models.hf.qwen import Qwen15MoEBRMoE as AutoBRMoEHFModel
from BR_MoE.core.quantize import BaseCompressConfig

# Load model
model_path = "path/to/your/qwen1.5-moe"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Configure compression settings
compress_config = BaseCompressConfig(
    nbits=3,                    # Base quantization bits
    group_size=128,             # Quantization group size
    sparse_rank=32,             # Compensator rank for experts
    dense_rank=512,             # Compensator rank for dense layers
    iter=20,                    # Optimization iterations
    compensator_dtype="int3",   # Compensator quantization
    quant_zero=False,           # Don't quantize zero points
    quant_scale=False,          # Don't quantize scales
    axis=1                      # Quantization axis
)

# Apply compression
device = "cuda"
AutoBRMoEHFModel.compress_model(model, compress_config=compress_config, device=device)

# Save compressed model
quant_model_dir = "path/to/save/compressed_model"
AutoBRMoEHFModel.save_compressed(model, quant_model_dir)
```

#### 2. Generate Optimal Configuration with ILP Solver

```bash
# Step 1: Collect quantization loss data for all expert configurations
python collect/collect_expert.py \
    --model_path "path/to/your/qwen1.5-moe" \
    --output_path "expert_impact_results.json" \
    --bit_options "2,3,4" \
    --rank_options "0,16,32,64,128"

# Step 2: Use ILP solver to find optimal allocation
python ILP_solver/solver.py \
    --l2_file "expert_impact_results.json" \
    --freq_file "expert_freq.json" \
    --memory_file "memory_usage.json" \
    --output_file "best_config.json" \
    --budget 8000.0 \
    --baseline "2bit_rank0" \
    --min_rank 16 \
    --max_rank 256
```

#### 3. Run with Optimal Configuration

```bash
# Edit BR_compress_Qwen15.py to set your model paths:
# model_path = "path/to/your/qwen1.5-moe"
# quant_model_dir = "path/to/save/compressed_model"

# Run compression with ILP-optimized configuration
python examples/BR_compress_Qwen15.py --config best_config.json
```

**Configuration file format (best_config.json):**
```json
{
  "L0_E0": "3bit_rank32",
  "L0_E1": "4bit_rank16", 
  "L1_E0": "2bit_rank64",
  "L1_E1": "3bit_rank0",
  ...
}
```

#### 3. Uniform Compression (Alternative)

```bash
# For uniform compression without ILP optimization
# Edit BR_compress_Mixtral_uniform.py to set model paths and run:
python examples/BR_compress_Mixtral_uniform.py
```

#### 4. Evaluation

```bash
# Edit BR_eval.py to set your model paths:
# quant_model_dir = "path/to/your/compressed_model"
# model_id = "mistralai/Mixtral-8x7B-v0.1"  # or your model ID

python examples/BR_eval.py
```

## 📁 Project Structure

```
BR-MoE-organized/
├── BR_MoE/                 # Core framework
│   ├── core/               # Quantization and compensation algorithms
│   ├── models/             # Model implementations (Mixtral, DeepSeek, Qwen)
│   ├── engine/             # Inference engine
│   ├── kernels/            # CUDA kernels for acceleration
│   ├── backends/           # Hardware backends
│   └── utils/              # Utility functions
├── ILP_solver/             # Integer Linear Programming solver
├── collect/                # Configuration collection scripts
├── evaluation/             # Evaluation framework
├── examples/               # Usage examples
├── model_statistics/       # Model analysis tools
├── imgs/                   # Documentation images
├── requirements.txt        # Python dependencies
├── conda_env_setup.sh      # Environment setup script
└── kernel_setup.sh         # CUDA kernel compilation
```

## 🧩 Additional Backends

### Triton int3 Grouped GEMM (`kernels/triton_int3/`)

A build-free **Triton** implementation of the 3.0-bit MoE grouped GEMM, as a portable counterpart
to the CUDA kernel in `kernels/brmoe`. One MoE layer = two grouped GEMMs + a fused
`silu(gate)*up`, with a fully device-side token alignment (zero host sync).

| M | int3 3.0 bpw | int4 4.0 bpw | fp16 16 bpw | per-expert int3 | per-expert fp16 |
|---|---|---|---|---|---|
| 1 | 0.724 | 0.769 | 0.617 | 3.173 | 2.638 |
| 8 | 1.594 | 1.724 | 1.466 | 5.190 | 8.081 |
| 512 | 6.585 | 7.387 | 6.152 | 9.533 | 17.028 |

ms per MoE layer, `E=8 K=2048 I=4096 topk=2 group=128`, Tesla T4 15 GB — **1.9–2.4×** faster than
this kernel's initial version and **1.7–4.4×** faster than the per-expert loop. `kernels/triton_int3/README.md`
documents the full methodology, the measured negative results (which tile shapes hurt and why) and
the bottleneck analysis.

> Run with `cd kernels/triton_int3 && python test_kernel.py --bench` (requires `triton>=2.1`).

## 🧠 Methodology

### The Co-Design Challenge

Traditional approaches treat quantization and compensation independently:
- **Mixed-precision quantization**: Allocates different bit-widths based on sensitivity
- **Low-rank compensation**: Adds compensator matrices to recover quantization errors

**BR-MoE's insight**: These dimensions are deeply coupled! The optimal compensator rank depends on the bit-width, and vice versa.

### Our Solution: ILP-Based Global Optimization

**BR-MoE Workflow:**

1. **Configuration Space Generation**: Create all possible (bit-width, rank) combinations
2. **Proxy Evaluation**: Measure layer-wise quantization loss for each expert-configuration pair
3. **ILP Formulation**: Cast as Multiple-Choice Knapsack Problem (MCKP)
4. **Global Optimization**: Use ILP solver to find optimal allocation under memory constraints

**Mathematical Formulation:**
```
Minimize: Σᵢ Σⱼ (Fᵢ · Lᵢⱼ) · xᵢⱼ
Subject to:
- Σⱼ xᵢⱼ = 1, ∀i (each expert gets exactly one config)
- Σᵢ Σⱼ Mⱼ · xᵢⱼ ≤ B (memory budget constraint)
```

Where:
- `Fᵢ`: Expert activation frequency (importance weight)
- `Lᵢⱼ`: Layer-wise quantization loss for expert i with config j
- `Mⱼ`: Memory cost of configuration j
- `B`: Total memory budget
- `xᵢⱼ`: Binary decision variable (1 if expert i uses config j)

The ILP solver systematically explores the configuration space and guarantees finding the globally optimal solution within the given memory budget.

## 🔬 Supported Models

| Model | Architecture | Experts | TopK | Status |
|-------|-------------|---------|------|--------|
| Mixtral-8×7B | Sparse MoE | 8 | 2 | ✅ Supported |
| DeepSeek-V2-Lite | Hybrid MoE | 64+2 | 6 | ✅ Supported |
| Qwen1.5-MoE | Dense+MoE | 60+4 | 4 | ✅ Supported |
| Switch Transformer | Custom | Variable | Variable | 🚧 Coming Soon |

## 📈 Performance Comparison

### Memory-Accuracy Trade-offs

![Performance Comparison](imgs/results.png)

BR-MoE consistently achieves superior accuracy at similar memory footprints compared to:
- **GPTQ**: Calibration-based quantization
- **HQQ**: Calibration-free quantization  
- **MiLo**: MoE-specific low-rank compensation
- **Mixed-Precision Heuristics**: Frequency-based allocation

### Scalability Analysis

Our ILP solver efficiently handles the combinatorial explosion:
- **Complexity**: O(|E| × |B| × |R|) for configuration evaluation + ILP solving
- **Practical Runtime**: < 10 seconds for typical MoE models
- **Memory Efficient**: Proxy evaluation avoids full model loading

## 📊 Evaluation Framework

### Supported Benchmarks

- **Language Modeling**: WikiText-2
- **Reading Comprehension**: HellaSwag, LAMBADA
- **Commonsense Reasoning**: PIQA, WinoGrande
- **Multi-task**: MMLU (57 tasks)

### Custom Evaluation

```python
from BR_MoE.evaluation import evaluate_model

results = evaluate_model(
    model_path="path/to/compressed_model",
    tasks=["wikitext2", "hellaswag", "piqa"],
    batch_size=8,
    device="cuda"
)
```

## 🔧 Configuration Options

### Quantization Settings

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `nbits` | Quantization bit-width | 3 | 2, 3, 4, 8 |
| `group_size` | Quantization group size | 128 | 32, 64, 128, 256 |
| `quant_zero` | Quantize zero points | False | True, False |
| `quant_scale` | Quantize scales | False | True, False |

### Compensation Settings

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `sparse_rank` | Expert compensator rank | 32 | 0, 16, 32, 64, 128, 256 |
| `dense_rank` | Dense layer compensator rank | 512 | 0, 256, 512, 1024 |
| `compensator_dtype` | Compensator quantization | "int3" | "fp16", "int4", "int3" |
| `rank_strategy` | Rank allocation strategy | "custom" | "uniform", "frequency", "custom" |

### Optimization Settings

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `iter` | Optimization iterations | 20 | 5-50 |
| `lr` | Learning rate | 0.01 | 0.001-0.1 |
| `solver_timeout` | ILP solver timeout (s) | 300 | 60-3600 |

### ILP Solver Settings

| Parameter | Description | Example | Notes |
|-----------|-------------|---------|-------|
| `--l2_file` | Expert quantization loss results | `expert_impact_results.json` | Generated by collect step |
| `--freq_file` | Expert activation frequencies | `expert_freq.json` | Importance weights |
| `--memory_file` | Memory usage for each config | `memory_usage.json` | Memory constraints |
| `--budget` | Memory budget (MB) | `8000.0` | Total allowed memory increase |
| `--baseline` | Baseline configuration | `"2bit_rank0"` | Reference config for memory calculation |
| `--min_rank` | Minimum compensator rank | `16` | Lower bound for rank allocation |
| `--max_rank` | Maximum compensator rank | `256` | Upper bound for rank allocation |

<!-- ## 📚 Citation

If you find BR-MoE useful in your research, please cite our paper:

```bibrex
@inproceedings{brmoe2025,
    title={BR-MoE: Beyond Heuristics in MoE Model Compression},
    author={Your Name and Co-authors},
    booktitle={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/}
}
``` -->


## 🙏 Acknowledgments

- Thanks to the open-source community for foundational tools
- Model providers: Meta (Mixtral), DeepSeek, Alibaba (Qwen)
- Evaluation framework: EleutherAI lm-evaluation-harness
