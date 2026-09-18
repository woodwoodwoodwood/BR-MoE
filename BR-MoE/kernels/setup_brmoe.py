from setuptools import setup
from torch.utils import cpp_extension

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
                'cxx': ['-O3', '-DNDEBUG', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '-O3',
                    '-DNDEBUG',
                    '-arch=sm_80',
                    '-Xcompiler', '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '--compiler-options', '-fPIC'
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['brmoe'],
    install_requires=['torch']
)
