#!/bin/bash
# 后台监控一个 slurm 作业, 每 60s 往日志追加一行进度; 作业结束后自动打印汇总并出图。
#
# 用法:
#   nohup bench/monitor_job.sh <jobid> [间隔秒] > /dev/null 2>&1 &
#   tail -f bench_results/monitor_<jobid>.log      # 看进度
#
# 说明: 完全只读, 不碰作业本身。

set -uo pipefail

JOB=${1:?用法: monitor_job.sh <jobid> [间隔秒]}
INTERVAL=${2:-60}

REPO=/mnt/709/data3/home/jianglei/ada/BR-MoE
RESULT_DIR="${REPO}/bench_results"
LOG="${RESULT_DIR}/monitor_${JOB}.log"
PY=/mnt/709/data3/home/jianglei/miniconda3/envs/milo/bin/python

# $0 可能是相对路径, 用绝对路径引用同目录脚本
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] 开始监控 job ${JOB} (间隔 ${INTERVAL}s)" >> "${LOG}"

start_ts=$(date +%s)
prev_count=-1

while true; do
  state=$(squeue -h -j "${JOB}" -o '%T' 2>/dev/null | head -1)
  if [[ -z "${state}" ]]; then
    state=$(sacct -n -j "${JOB}" -o State 2>/dev/null | head -1 | tr -d ' ')
    [[ -z "${state}" ]] && state="UNKNOWN(已离开队列)"
    echo "[$(ts)] 作业结束, 状态=${state}" >> "${LOG}"
    break
  fi

  n=$(ls -1 "${RESULT_DIR}"/*_b*_"${JOB}".json 2>/dev/null | wc -l)
  el=$(( $(date +%s) - start_ts ))
  if [[ "${n}" != "${prev_count}" ]]; then
    echo "[$(ts)] 运行中 ${state} 已跑 ${el}s | 已完成 ${n} 个 case" >> "${LOG}"
    prev_count=${n}
  fi
  sleep "${INTERVAL}"
done

# ---- 结束后: 汇总 + 出图 ----
{
  echo ""
  echo "=============== 最终结果 ==============="
  grep -E '^#### CASE|=> TTFT|^  in=|load\] done' "${RESULT_DIR}/slurm_${JOB}.out" 2>/dev/null | tail -60
  echo ""
  echo "=============== 汇总表 + 加速比 ==============="
  sed -n '/#### SUMMARY/,$p' "${RESULT_DIR}/slurm_${JOB}.out" 2>/dev/null
} >> "${LOG}" 2>&1

echo "" >> "${LOG}"
echo "[$(ts)] 生成图表..." >> "${LOG}"
( cd "${REPO}" && "${PY}" "${SELF_DIR}/plot_batch_sweep.py" \
    --result-dir bench_results --job "${JOB}" ) >> "${LOG}" 2>&1

echo "[$(ts)] 监控结束" >> "${LOG}"
