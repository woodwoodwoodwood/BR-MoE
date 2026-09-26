"""逐段隔离 mul_3bit_moe 的 illegal instruction。"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "marlin_int3_moe"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "triton_int3"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))
import brmoe_moe_int3 as ext  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)

# 最小形状: E=1, K=256, N=256, 一个 m-tile, 全有效
E, K, N, gs = 1, 256, 256, 64
rows = 16  # 一个 m-tile
A = torch.randn(rows, K, dtype=torch.float16, device=dev)
B1 = torch.zeros(E, K // 16, N, dtype=torch.int32, device=dev)
B2 = torch.zeros(E, K // 16, N // 2, dtype=torch.int32, device=dev)
s = torch.ones(E, K // gs, N, dtype=torch.float16, device=dev)
z = torch.zeros(E, K // gs, N, dtype=torch.float16, device=dev)
C = torch.zeros(rows, N, dtype=torch.float16, device=dev)
eid = torch.zeros(1, dtype=torch.int32, device=dev)
meta = torch.tensor([rows, 1], dtype=torch.int32, device=dev)

print("step 0: 对照 mul_3bit_with_zeros (原线性 kernel, MOE=false)...", flush=True)
ws = torch.zeros(N // 128 * 16 + 64, dtype=torch.int32, device=dev)
A2 = torch.randn(16, K, dtype=torch.float16, device=dev)
C2 = torch.zeros(16, N, dtype=torch.float16, device=dev)
ext.mul_3bit_with_zeros(A2, B1[0], B2[0], C2, s[0], z[0], ws, -1, -1, -1, 16)
torch.cuda.synchronize()
print("step 0 OK (config 1,8,8,4). C2 均值:",
      C2.float().abs().mean().item(), flush=True)

# (已确认: config (4,16,4,4) 即 64 行大 tile 在 sm_120 上 illegal instruction,
#  与 MOE 改动无关 —— 见 38980。所以 MoE 路径只用 thread_m_blocks=1 的 16 行 tile。)

print("step 1: 调用 mul_3bit_moe (E=1, 单块)...", flush=True)
ext.mul_3bit_moe(A, B1, B2, C, s, z, eid, meta[0:1], 1)
torch.cuda.synchronize()
print("step 1 OK. C 均值:", C.float().abs().mean().item(), flush=True)

print("step 2: 超发块 (m_blocks_max=3, num_post=64 -> 2 个块应早退)...", flush=True)
eid3 = torch.zeros(3, dtype=torch.int32, device=dev)
ext.mul_3bit_moe(A, B1, B2, C, s, z, eid3, meta[0:1], 3)
torch.cuda.synchronize()
print("step 2 OK", flush=True)

# ---- step 3: 数值对比 —— MOE kernel vs 原 kernel, 同一权重同一输入 ----
print("step 3: MOE vs 原 kernel 数值对比 (随机权重)...", flush=True)
torch.manual_seed(0)
B1r = torch.randint(-2**31, 2**31 - 1, (E, K // 16, N), dtype=torch.int32, device=dev)
B2r = torch.randint(-2**31, 2**31 - 1, (E, K // 16, N // 2), dtype=torch.int32, device=dev)
sr = (torch.rand(E, K // gs, N, device=dev) * 0.02 + 0.005).half()
zr = (torch.rand(E, K // gs, N, device=dev) * 0.02 + 0.005).half()
C_ref = torch.zeros(rows, N, dtype=torch.float16, device=dev)
ext.mul_3bit_with_zeros(A, B1r[0], B2r[0], C_ref, sr[0], zr[0], ws,
                        128, 128, -1, 16)   # 显式同 MOE 的配置 (1,8,8,4)
C_moe = torch.zeros(rows, N, dtype=torch.float16, device=dev)
ext.mul_3bit_moe(A, B1r, B2r, C_moe, sr, zr, eid, meta[0:1], 1)
torch.cuda.synchronize()
d = (C_moe.float() - C_ref.float()).abs().max().item()
print(f"step 3: max|diff| = {d:.6e}  {'OK 一致' if d == 0 else '!! 不一致'}",
      flush=True)
print("全部完成")
