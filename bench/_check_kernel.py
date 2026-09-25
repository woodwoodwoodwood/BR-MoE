"""校验 brmoe_cuda 扩展在当前 GPU 上真的可执行。

注意: 只 import 是不够的 —— 用别的架构编译出来的 .so 能 import 成功,
但第一次 kernel launch 会报 "no kernel image is available for execution on the device"。
所以这里真实跑一次 int3 GEMM 才算通过。

外层请加 timeout 保护 (kernel 有问题时可能挂住):
    timeout 300 python _check_kernel.py
"""

import sys

import torch


def main():
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_properties(0).major}"
          f"{torch.cuda.get_device_properties(0).minor})", flush=True)

    import brmoe_cuda  # noqa: F401  pybind 绑定

    from BR_MoE.kernels.brmoe import Layer3bit

    k, n, gs = 2048, 2048, 128
    lin = torch.nn.Linear(k, n, bias=False).to(device, dtype=torch.half)
    with torch.no_grad():
        lin.weight.data.normal_(0, 0.02)
    scales = torch.full((n, k // gs), 0.01, dtype=torch.half, device=device)

    lay = Layer3bit(k, n, gs).to(device)
    lay.pack(lin, scales)

    x = torch.randn(8, k, dtype=torch.half, device=device)
    y = lay(x)
    torch.cuda.synchronize()
    print(f"KERNEL_OK shape={tuple(y.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
