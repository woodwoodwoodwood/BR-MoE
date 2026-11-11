# 量化配置

这里保存可复用的量化 bit/rank 分配配置，原根目录下的五份 `best_config*.json` 内容保持不变。

| 文件 | 用途 |
| --- | --- |
| `best_config.json` | Qwen1.5 示例压缩脚本的默认配置 |
| `best_config_deepseek_rco.json` | DeepSeek RCO 配置 |
| `best_config_mixtral_rco.json` | Mixtral RCO 配置 |
| `best_config_qwen3_6_35b_a3b.json` | Qwen3.6-35B-A3B 配置 |
| `best_config_rco_int3.json` | INT3 RCO 配置 |

新搜索结果默认写入 `artifacts/search/`。确认需要长期保留后，可将配置和它依赖的
`*_tensor_allocation.json`、`.imatrix.npz` 等配套文件一起放入此目录，保留相对路径。

从项目根目录运行，例如：

```bash
python examples/BR_compress_Qwen15.py --config configs/best_config.json
```
