import os

from setuptools import setup
from torch.utils import cpp_extension

# 目标架构: 默认 sm_80 (A100)。
# 可用环境变量覆盖: H200 -> sm_90, 4090 -> sm_89, 5090 -> sm_120
_CUDA_ARCH = os.environ.get("BRMOE_CUDA_ARCH", "sm_80")

# libstdc++ ABI: 必须与目标 torch 一致, 否则链接期报
#   undefined symbol: std::__cxx11::basic_string...
# 默认 0 = 与 BR-MoE 原构建一致 (milo 环境 torch 2.5.1 的 ABI=False)。
# vllm5090 环境的 torch 2.13 是 ABI=True -> 传 BRMOE_ABI=1。
# 检查方法: python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"
_ABI = os.environ.get("BRMOE_ABI", "0")

setup(
    name='brmoe',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'brmoe_cuda', 
            [
                'brmoe/brmoe_cuda.cpp',
                'brmoe/brmoe_cuda_kernel.cu',
                'brmoe/brmoe_cuda_with_zero_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3', '-DNDEBUG', f'-D_GLIBCXX_USE_CXX11_ABI={_ABI}'],
                'nvcc': [
                    '-O3',
                    '-DNDEBUG',
                    f'-arch={_CUDA_ARCH}',
                    '-Xcompiler', f'-D_GLIBCXX_USE_CXX11_ABI={_ABI}',
                    '--compiler-options', '-fPIC'
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['brmoe'],
    install_requires=['torch']
)
