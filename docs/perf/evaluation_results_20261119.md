# RCO 与 baseline 实验结果

结果核对日期为 2026-11-19（北京时间）。大小为完整 checkpoint 的磁盘字节换算值。

| 模型与配置 | WikiText-2 PPL ↓ | HellaSwag acc_norm ↑ | PIQA acc ↑ | LAMBADA-openai acc ↑ | MMLU 5-shot acc ↑ | Checkpoint (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen1.5 FP16 | 7.2174 | 77.25% | 80.14% | 71.92% | 61.22% | 26.68 |
| Qwen1.5 RCO + 低秩补偿 | 8.0071 | 75.14% | 79.33% | 70.15% | 58.33% | 5.94 |
| Qwen1.5 纯混合精度（rank0） | 7.9436 | 75.15% | 79.27% | 72.07% | 57.42% | 5.94 |
| Qwen3.5-35B-A3B RCO | 7.1232 | 80.93% | 81.99% | 72.87% | 评测中断 | 15.96 |

WikiText-2 使用 2048 token 的非重叠完整窗口。HellaSwag、PIQA 和 LAMBADA-openai 为 0-shot，MMLU 使用 57 学科的 5-shot 题目加权准确率。Qwen1.5 与 Qwen3.5 的 tokenizer 不同，WikiText-2 分别有 146 和 145 个窗口。

Qwen1.5 的联合配置与 rank0 都以 4,896 MiB 为所选权重及补偿的预算，并各自重新搜索。联合配置的 MMLU 高 0.90 个百分点，LAMBADA-openai 低 1.92 个百分点。固定量化配置后单独移除补偿的收益仍待验证。

Qwen3.5 作业 61901 完成 300 步搜索、压缩和重载验证。430 个决策单元覆盖 31,030 个投影矩阵，150 个线性注意力单元中有 111 个选择非零 rank。完整 checkpoint 含视觉和 MTP 参数，磁盘体积比源目录减少 76.18%。MMLU 在批量评分与独立评分的一致性检查处中断，完成 0/14,042 题。同模型 FP16 对照仍待测量。

| Qwen1.5 baseline | 复现状态 |
|---|---|
| GPTQ W3A16，group128 | 作业 61905 完成压缩和重载，实际大小 6.23 GiB。作业 61927 在评测预检时遇到跨 GPU 设备错误。 |
| MiLo、MxMoE、GSQ | 顺序复现停在 GPTQ 评测，三项正式实验待启动。 |

逐项分数、样本数、作业号和源记录指纹见 [evaluation_results_20261119.json](evaluation_results_20261119.json)。
