"""构建 int3 grouped MoE CUDA kernel (tile 级融合, Marlin 式流水)。

用法:
    cd BR-MoE/kernels/marlin_int3_moe
    BRMOE_CUDA_ARCH=sm_80  BRMOE_ABI=1 python setup_moe.py build_ext --inplace   # A100
    BRMOE_CUDA_ARCH=sm_120 BRMOE_ABI=1 python setup_moe.py build_ext --inplace   # 5090

BRMOE_ABI 必须与当前 torch 一致:
    python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"  # True -> 1

BRMOE_MOE_MIN_BLOCKS=2 是寄存器占用实验，默认不设置；需与运行时
BRMOE_MOE_SMEM=rightsize 配合对照，不能仅凭微基准启用。
5090 bs32/128 端到端回退记录见 docs/moe_large_batch_20260926.md。
"""
import os

from setuptools import setup
from torch.utils import cpp_extension

_CUDA_ARCH = os.environ.get("BRMOE_CUDA_ARCH", "sm_80")
_ABI = os.environ.get("BRMOE_ABI", "1")
_MIN_BLOCKS = os.environ.get("BRMOE_MOE_MIN_BLOCKS")
if _MIN_BLOCKS not in (None, "1", "2"):
    raise ValueError("BRMOE_MOE_MIN_BLOCKS must be 1 or 2 when set")

setup(
    name="brmoe_moe_int3",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "brmoe_moe_int3",
            [
                "int3_moe_ops.cpp",
                "int3_moe_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-DNDEBUG", f"-D_GLIBCXX_USE_CXX11_ABI={_ABI}"],
                "nvcc": [
                    "-O3",
                    "-DNDEBUG",
                    f"-arch={_CUDA_ARCH}",
                    "-Xcompiler", f"-D_GLIBCXX_USE_CXX11_ABI={_ABI}",
                    "--compiler-options", "-fPIC", "-lineinfo",
                ] + ([f"-DBRMOE_MOE_MIN_BLOCKS={_MIN_BLOCKS}"] if _MIN_BLOCKS else []),
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
