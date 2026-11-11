# 运行产物

运行结果统一写入本目录。源码、可复用配置、数据集、模型统计和正式性能记录分别位于
项目各模块、`configs/`、`datasets/`、`model_statistics/` 和 `docs/perf/`。

| 路径 | 用途 |
| --- | --- |
| `runs/` | 实验汇总、图表、仍在使用的输入及当前作业产物 |
| `search/` | ILP/RCO 搜索输出，运行时创建 |
| `collect/` | 专家影响采集输出，运行时创建 |
| `cache/` | Python/pytest 等可重新生成的缓存，运行时创建 |
| `kernel_bench/triton_int3/` | 两份纳入版本管理的基准日志 |
| `organization/2026-10-26/` | 目录调整前的 Git 差异和状态 |
| `cleanup.json` | 2026-10-26 清理统计及保留项 |

2026-10-26 删除了 5,925 个旧产物文件和 768 个空目录，释放约 12.90 GiB：
包括可重新生成的权重缓存、历史采集张量、已完成作业日志、性能追踪数据、
旧源码快照、无引用的源码归档包、编译中间文件和迁移备份。

源码、配置、测试、数据集、模型统计、`docs/perf/` 的正式性能记录和实验汇总保留。
旧实验文档及历史 JSON 中的路径记录当时的运行位置，部分原始文件已删除；
复现这些实验需运行对应脚本的 `collect` 阶段重新采集输入，源码快照也需重新创建。

本次保留的主要运行依赖：

- `runs/full_int3_study/run_38936/real_inputs.pt`：grouped GEMV 基准的默认输入。
- `runs/grouped_gemv/run_39061/real_inputs.pt`：MoE fusion 基准的默认输入。
- `runs/rco_e2e/run_55343/`：生成测试使用的分配结果、搜索状态及检查点。
- 作业 55823 的输出目录、日志、源码归档，以及其使用的 `runs/triton_iq_cache/`。
- `runs/iq_smoke_source.tar`：现有 Slurm 脚本使用的源码归档。

旧入口 `bench_results` 已移除，维护中的脚本使用 `artifacts/runs`。
从项目根目录提交作业前先创建输出目录，Slurm 会在脚本执行前打开日志文件：

```bash
mkdir -p artifacts/runs
sbatch bench/slurm/bench_deepseek.slurm
```
