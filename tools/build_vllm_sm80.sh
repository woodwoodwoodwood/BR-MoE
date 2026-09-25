#!/bin/bash
# 在 vllm5090 环境里重编 vLLM，**追加 sm_80 (A100)** 支持，同时保留 sm_120 (5090)。
#
# 背景: 现有构建是 TORCH_CUDA_ARCH_LIST="12.0" 生成的，
#   _C_stable_libtorch.abi3.so   有 70 个 sm_120 / 仅 4 个 sm_80
#   _moe_C_stable_libtorch.abi3.so 同理
# 且没有 PTX 兜底，所以 A100 上跑 forward 会报
#   CUDA error: no kernel image is available for execution on the device
#
# 用法:
#   bash tools/build_vllm_sm80.sh            # 前台
#   nohup bash tools/build_vllm_sm80.sh > /path/build.log 2>&1 &
#
# 完成后用 tools/check_vllm_archs.sh 复核架构。

set -uo pipefail

SRC=/mnt/709/data3/home/jianglei/framework/vllm
ENV_PREFIX=/mnt/709/data3/home/jianglei/miniconda3/envs/vllm5090
CUDA_HOME_DIR=/mnt/709/data3/home/jianglei/cuda-12.9
BACKUP=/mnt/709/data3/home/jianglei/framework/vllm_so_backup_sm120

export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# ---- 代理 ----
# CMake 的 FetchContent 在配置阶段要 `git clone https://github.com/nvidia/cutlass.git`，
# 而节点没有外网直连，会卡 135s 后失败:
#   fatal: unable to access 'https://github.com/nvidia/cutlass.git/'
#   CMake Error at cutlass-subbuild/.../cutlass-populate-gitclone.cmake:39
#   ninja: build stopped: subcommand failed.
# 注意: 代理只监听 127.0.0.1，所以本脚本必须在代理所在主机 (ada-709 登录节点) 上跑;
#       提交到计算节点会连不上。
PROXY="${PROXY:-http://127.0.0.1:17890}"
if ! curl -s -o /dev/null --max-time 20 -x "${PROXY}" https://github.com; then
  echo "!! 代理不可用: ${PROXY}"
  echo "   (代理只监听 127.0.0.1, 请在 ada-709 上直接跑本脚本, 不要 sbatch)"
  exit 1
fi
export http_proxy="${PROXY}"   https_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"   HTTPS_PROXY="${PROXY}"
export all_proxy="${PROXY}"    ALL_PROXY="${PROXY}"
# 本机/节点内通信不走代理
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
echo "PROXY      = ${PROXY} (已验证可达 github)"

# 关键: 同时编 sm_80 与 sm_120，两个平台都能用。
# 想要更快的单架构构建可以改成 "8.0"，但会让 5090 失效。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 12.0}"
export VLLM_TARGET_DEVICE=cuda
# 登录节点 16 核 / 62 GB；给别的用户留些余量，同时避免 nvcc OOM
export MAX_JOBS="${MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
# 关掉 ninja 的进度刷屏，日志更好读
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"

echo "=== $(date '+%F %T') 开始构建 vLLM ==="
echo "SRC        = ${SRC}"
echo "CUDA_HOME  = ${CUDA_HOME}"
echo "ARCH_LIST  = ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS   = ${MAX_JOBS}"
"${CUDA_HOME}/bin/nvcc" --version | tail -1
"${ENV_PREFIX}/bin/python" -c "import torch;print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# ---- 备份现有（sm_120 专用）产物，构建失败可立刻回滚 ----
mkdir -p "${BACKUP}"
echo ""
echo "=== 备份现有 .so 到 ${BACKUP} ==="
for f in "${SRC}"/vllm/*.so; do
  [ -e "$f" ] || continue
  b="${BACKUP}/$(basename "$f")"
  if [[ ! -e "$b" ]]; then
    cp -p "$f" "$b" && echo "  backup $(basename "$f")"
  else
    echo "  已存在备份 $(basename "$f")"
  fi
done

# ---- 构建 ----
echo ""
echo "=== pip install -e . (这会现场编译 CUDA 扩展) ==="
cd "${SRC}" || exit 1
"${ENV_PREFIX}/bin/pip" install --no-cache-dir --no-build-isolation -e . 2>&1
rc=$?

echo ""
echo "=== $(date '+%F %T') 构建结束 rc=${rc} ==="
if [[ ${rc} -eq 0 ]]; then
  echo "重新检查架构:"
  for so in "${SRC}"/vllm/_C_stable_libtorch.abi3.so "${SRC}"/vllm/_moe_C_stable_libtorch.abi3.so; do
    [[ -e "$so" ]] || continue
    echo "  $(basename "$so"):"
    "${CUDA_HOME}/bin/cuobjdump" --list-elf "$so" 2>/dev/null \
      | grep -oE 'sm_[0-9]+' | sort | uniq -c | sed 's/^/    /'
  done
else
  echo "构建失败。回滚: cp ${BACKUP}/*.so ${SRC}/vllm/"
fi
exit ${rc}
