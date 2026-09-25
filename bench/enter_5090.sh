#!/bin/bash
# 进入 / 观察 5090 节点的辅助脚本。
#
# 关键坑: Slurm 在不显式给 --mem 时 **默认申请整节点内存** (250G),
# 而节点常驻别人的作业只余 ~110G, 于是永远停在 PD(Resources) ——
# 看起来像"卡被占满", 其实不是。所以本脚本一律显式带 --mem。
#
# 用法 (在登录节点 ada-709 上跑):
#   bash bench/enter_5090.sh status     # 只看队列和节点占用, 不申请
#   bash bench/enter_5090.sh            # = srun, 直接进节点拿一个 shell (推荐)
#   bash bench/enter_5090.sh salloc     # 用 salloc, 之后可开多个终端附加
#   bash bench/enter_5090.sh monitor    # 另起一个分配专跑 nvitop 看全部 8 张卡
#
# 参数可用环境变量覆盖: HOURS=3 MEM=64G GPU=1 PART=5090
#
# 注意: srun 创建的分配 **不能** 被 `srun --jobid=` 附加 step
#       (会报 Unable to create step)。要多终端请用 salloc 模式。

set -uo pipefail

PART=${PART:-5090}
HOURS=${HOURS:-2}
MEM=${MEM:-48G}
CPUS=${CPUS:-8}
GPU=${GPU:-1}
MODE=${1:-srun}

NVITOP=/mnt/709/data3/home/jianglei/miniconda3/bin/nvitop
SBATCH_OPTS=(-p "${PART}" -N1 -n1 --mem="${MEM}" --cpus-per-task="${CPUS}"
             --gres=gpu:"${GPU}" -t "${HOURS}:00:00")

show_status () {
  echo "=== 我的作业 ==="
  squeue -u "$(whoami)" -o "%.8i %.2t %.10P %.12M %.12L %.20R %.18j" 2>/dev/null | head -8
  echo
  echo "=== ${PART} 分区上的作业 (看别人占了几张卡) ==="
  squeue -p "${PART}" -o "%.8i %.9u %.2t %.12M %.12b %.20j" 2>/dev/null | head -8
  echo
  echo "=== 节点资源 ==="
  scontrol show node ada-5090 2>/dev/null | grep -oE 'State=[^ ]*|Gres=[^ ]*|CPUTot=[^ ]*|CPUAlloc=[^ ]*|RealMemory=[^ ]*|FreeMem=[^ ]*'
  echo
  echo "提示: STATE=PD 且 REASON=Resources 时, 先确认自己的提交带了 --mem"
}

case "${MODE}" in
  status)
    show_status
    ;;

  monitor)
    # nvitop 走 NVML, 能看到整机 8 张卡; 但它需要 device 可见,
    # 所以这个分配也得带 --gres, 否则会 "No devices were found"。
    echo "申请一个专用于监控的分配 (会占 1 张卡, 节点上通常有空余)..."
    exec srun "${SBATCH_OPTS[@]}" --pty \
      bash -lc "echo '监控中 (Ctrl-C 退出):'; ${NVITOP}"
    ;;

  salloc)
    echo "申请分配 (salloc)。就绪后你会进到一个 shell, 然后:"
    echo "   终端 A:  srun --pty bash          # 跑测试"
    echo "   终端 B:  srun --pty bash  →  ${NVITOP}   # 监控"
    echo "   多个终端都可以 srun --pty bash (同一分配, 同一张卡)"
    echo
    exec salloc "${SBATCH_OPTS[@]}"
    ;;

  srun|*)
    echo "进入 ${PART} 节点: ${SBATCH_OPTS[*]}"
    echo "进去后跑:  bash /mnt/709/data3/home/jianglei/ada/BR-MoE/bench/vllm_perf_5090.sh"
    echo
    exec srun "${SBATCH_OPTS[@]}" --pty bash
    ;;
esac
