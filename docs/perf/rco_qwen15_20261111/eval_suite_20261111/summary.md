# Qwen1.5-MoE RCO per-tensor 压缩模型评测

Checkpoint：`artifacts/runs/rco_e2e/run_55343/compressed`。PIQA 复用相同 checkpoint 已完成的完整评测；其他任务为 2026-11-11 新增。

| 任务 | 协议 | 最终指标 | 状态 / 进度 |
|---|---|---|---|
| wikitext2 | raw test；2048 token 非重叠完整窗口；无 BOS；末尾不足一窗丢弃 | 待完成 | running 100/146，作业 61658 |
| hellaswag | validation，0-shot；主指标 acc_norm | 待完成 | running 125/10042，作业 61657 |
| piqa | validation，0-shot | acc **79.33%**；acc_norm 79.33% | 完成 1838/1838 |
| lambada_openai | test，0-shot；最后一个词的所有子 token 贪心匹配 | 待完成 | waiting_or_loading 0/?，作业 61658 |
| mmlu | 57 学科 test，5-shot；每学科前 5 道 dev 示例；按题数加权 | 待完成 | running 476/14042，作业 61656 |

各任务逐题或逐窗口记录在相应 `run_<jobid>/<task>/predictions.jsonl` 中，配置、数据哈希、参考 logits 检查与硬件记录在同目录的 `results.json` / `progress.json` 中。

进行中的成绩不作为最终结果。原模型同协议基线尚未测量，当前不能据此计算压缩前后的精度损失。
