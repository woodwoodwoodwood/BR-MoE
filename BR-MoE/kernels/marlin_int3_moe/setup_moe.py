"""构建 int3 grouped MoE CUDA kernel (tile 级融合, Marlin 式流水)。

用法:
    cd BR-MoE/kernels/marlin_int3_moe
    BRMOE_CUDA_ARCH=sm_80  BRMOE_ABI=1 python setup_moe.py build_ext --inplace   # A100
    BRMOE_CUDA_ARCH=sm_120 BRMOE_ABI=1 python setup_moe.py build_ext --inplace   # 5090

BRMOE_ABI 必须与当前 torch 一致:
    python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"  # True -> 1
"""
import os

from setuptools import setup
from torch.utils import cpp_extension

_CUDA_ARCH = os.environ.get("BRMOE_CUDA_ARCH", "sm_80")
_ABI = os.environ.get("BRMOE_ABI", "1")

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
                ],
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
