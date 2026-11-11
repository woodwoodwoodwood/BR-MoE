# 基准与实验工具

本目录根部保留 24 个基准程序；辅助工具按用途归类。运行日志和中间结果统一写入
项目根目录下的 `artifacts/`。

| 位置 | 内容 |
| --- | --- |
| 根目录 `*.py` | 量化、线性层、MoE、端到端推理基准与实验程序 |
| `slurm/` | 38 个 Slurm 作业提交脚本 |
| `checks/` | 10 个数值校验、扩展加载和推理对拍程序 |
| `debug/` | 5 个内核排错和张量形状诊断程序 |
| `analysis/` | 结果汇总和绘图程序 |
| `scripts/` | 环境设置、交互运行和作业监控 Shell 脚本 |

从项目根目录运行：

```bash
mkdir -p artifacts/runs
sbatch bench/slurm/bench_deepseek.slurm
sbatch bench/slurm/iq_decode_verify.slurm
python bench/analysis/plot_batch_sweep.py --result-dir artifacts/runs --job 38416
source bench/scripts/env_5090.sh
```

`.slurm` 文件是作业提交脚本。Slurm 输出的 `.out/.err`、运行 `.log`、中间 JSON、
traces 和运行快照保存在 `artifacts/runs/`；ILP/RCO 搜索结果保存在 `artifacts/search/`。

2026-10-26 已清理已完成作业的日志、旧源码快照、追踪数据和多数历史张量，
保留实验汇总、脚本默认读取的两个输入和当前生成测试的依赖。
复现旧实验时请先运行对应的 `collect` 阶段，重新采集所需张量；
清理范围见 [`artifacts/README.md`](../artifacts/README.md)。
