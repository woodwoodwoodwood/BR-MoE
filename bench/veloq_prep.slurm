#!/bin/bash
#SBATCH --job-name=veloq-prep
#SBATCH --partition=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/veloq_prep_%j.out
#SBATCH --error=/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/veloq_prep_%j.err
#
# 为 traces/*.nsys-rep 建立 veloq 的 parquet 缓存, 供后续 stats/slices/gaps 查询。
#
# 必须在计算节点跑: 登录节点 nsys 是 2023.4, 读不了计算节点 nsys 2025.3 产出的
# report (报 "Please update your Nsight Systems to the latest version")。
#
# 提交 (等 trace 作业结束后自动开始):
#   sbatch --dependency=afterany:<trace_jobid> bench/veloq_prep.slurm
#   sbatch bench/veloq_prep.slurm          # 或直接跑, 处理已有的 report

set -uo pipefail

# 注意: 不要用 ${HOME} —— 计算节点的 $HOME 解析路径与登录节点不同,
# 会得到 "No such file or directory"。这里用 /mnt/709 下的实体绝对路径。
REPO=/mnt/709/data3/home/jianglei/ada/BR-MoE
TRACE_DIR=${REPO}/bench_results/traces
VELOQ=${VELOQ:-/mnt/709/data3/home/jianglei/.local/bin/veloq}

# veloq 的 current-user daemon 需要一个可写的 runtime 目录。
# 批处理环境里 $XDG_RUNTIME_DIR 通常缺失 (登录节点是 /run/user/<uid>),
# 不设置会直接报 `daemon.lifecycle-failed: cannot inspect ... runtime directory`。
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]]; then
  export XDG_RUNTIME_DIR="$(mktemp -d)"
fi

echo "=== node $(hostname) / $(date +%H:%M:%S) ==="
echo "veloq=${VELOQ}"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
nsys --version
if [[ ! -x "${VELOQ}" ]]; then
  echo "!! veloq 不可执行: ${VELOQ}"
  ls -l "${VELOQ}" 2>&1 || true
  exit 1
fi

shopt -s nullglob
reps=("${TRACE_DIR}"/*.nsys-rep)
if [[ ${#reps[@]} -eq 0 ]]; then
  echo "!! ${TRACE_DIR} 下没有 .nsys-rep"
  exit 0
fi

rc=0
for rep in "${reps[@]}"; do
  echo ""
  echo "################ PREP $(basename "${rep}") ################"
  ls -lh "${rep}"
  # 注意: 不能用 `cmd | tail` 的退出码判断成败 (管道退出码来自 tail, 恒为 0)
  if out=$("${VELOQ}" prep "${rep}" 2>&1); then
    echo "${out}" | tail -4
    echo "prep ok"
  else
    echo "${out}" | tail -20
    echo "!! prep failed: ${rep}"
    rc=1
  fi
done

echo ""
echo "=== parquetdir 就绪情况 ==="
for rep in "${reps[@]}"; do
  pq="${rep}.veloq/parquetdir"
  if [[ -d "${pq}" ]]; then
    echo "OK   $(basename "${rep}")  ($(du -sh "${pq}" 2>/dev/null | cut -f1))"
  else
    echo "MISS $(basename "${rep}")"
  fi
done

echo ""
echo "=== veloq 缓存目录 ==="
ls -ld "${TRACE_DIR}"/*.veloq 2>/dev/null || echo "(无)"
du -sh "${TRACE_DIR}"/*.veloq 2>/dev/null || true
echo "=== DONE $(date +%H:%M:%S) rc=${rc} ==="
exit ${rc}
