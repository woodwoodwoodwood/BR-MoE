"""with_zeros kernel 的结构性探针: 用 q=单位矩阵把 kernel 的实际读取布局逼出来。

三组探针 (K=N=128, gs=64, M=16, 线性 binding):
  A) q=I, s=1, z=0  -> y 应 = A; 若错, 用列匹配推出 kernel 的 n->k 映射表
  B) q=I, s=随机, z=0 -> y[i,n] = A[i,n] * s[n//64, n]   (分离 scale 路径)
  C) q=I, s=1, z=随机 -> y[i,n] = A[i,n] - z[n//64,n] * sum_k A[i,k] (分离 zero 路径)

用法: python bench/debug_kernel_probe.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))

import marlin_int3_moe  # noqa: E402


def _load_ext():
    d = os.path.dirname(marlin_int3_moe.__file__)
    cap = torch.cuda.get_device_capability(0)
    tag = f"_sm{cap[0]}{cap[1]}"
    cands = [f for f in os.listdir(d)
             if f.startswith("brmoe_moe_int3") and f.endswith(".so")]
    pref = [f for f in cands if tag in f] or [f for f in cands
                                              if "_sm" not in f]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brmoe_moe_int3", os.path.join(d, sorted(pref)[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = _load_ext()
from marlin_int3_moe.repack import repack_one  # noqa: E402


def run(q, s, z, A, N, K, gs, dev):
    B1, B2, sp, zp = repack_one(q, s, z, gs)
    C = torch.zeros(A.shape[0], N, dtype=torch.float16, device=dev)
    ws = torch.zeros((N // 64 + 1) * 16, dtype=torch.int32, device=dev)
    ext.mul_3bit_with_zeros(A, B1, B2, C, sp, zp, ws, 128, 128, -1, 8)
    torch.cuda.synchronize()
    return C.float()


def main():
    dev = "cuda"
    N = K = 128
    gs = 64
    g = torch.Generator(device="cpu").manual_seed(0)
    A = (torch.randn(16, K, generator=g) * 0.5).half().to(dev)
    qI = torch.eye(N, K, dtype=torch.int32).to(dev)      # W = I (s=1,z=0 时)
    ones = torch.ones(K // gs, N, dtype=torch.float16, device=dev)
    zeros = torch.zeros(K // gs, N, dtype=torch.float16, device=dev)

    # ---- A) 布局探针 ----
    C = run(qI, ones, zeros, A, N, K, gs, dev)
    d = (C - A.float()).abs().max().item()
    print(f"[A] q=I,s=1,z=0: max|C-A| = {d:.4e}  {'OK' if d < 1e-2 else '!! 错'}")
    if d > 1e-2:
        af, cf = A.float(), C
        match = []
        for n in range(N):
            dd = (af - cf[:, n:n + 1]).abs().sum(dim=0)
            k = int(dd.argmin())
            match.append(k if dd[k] < 1e-3 else -1)
        print("    C 列 n -> A 列 k 的映射 (前 64):")
        for r in range(0, 64, 16):
            print("    ", match[r:r + 16])
        hit = sum(1 for m in match if m >= 0)
        print(f"    匹配上的列: {hit}/{N}")
        n0 = next((i for i, m in enumerate(match) if m >= 0), None)
        if n0 is not None:
            k0 = match[n0]
            ratio = (cf[:, n0] / af[:, k0]).median().item()
            print(f"    例: C[:,{n0}] ≈ A[:,{k0}] * {ratio:.4f}")

    # ---- B') scale 编码探针: s[g,n] = g*1000 + n (fp16 可精确表示 <=2048 的整数) ----
    # 读出 kernel 对 (g, n) 位置实际用的 scale 值 -> 直接反解它读的是哪个 (g', n')
    s_enc = torch.arange(0, (K // gs) * 1000, 1000, dtype=torch.float16
                         ).view(K // gs, 1).expand(K // gs, N).contiguous()
    s_enc = (s_enc + torch.arange(N, dtype=torch.float16)[None, :]).to(dev)
    C = run(qI, s_enc, zeros, A, N, K, gs, dev)
    ratio = (C / A.float()).nan_to_num(-1.0)
    r0 = ratio[0]                      # 行 0 足矣 (列间互不干扰)
    print("[B'] scale 编码探针: kernel 读到的 (g',n') 映射:")
    ok = True
    maps = []
    for n in range(N):
        v = r0[n].item()
        gp, np_ = int(round(v)) // 1000, int(round(v)) % 1000
        maps.append((gp, np_))
        if (gp, np_) != (n // gs, n):
            ok = False
    print("     全部正确" if ok else "     存在错位的 scale!")
    if not ok:
        for gg in range(K // gs):
            row = maps[gg * gs:(gg + 1) * gs]
            print(f"     n={gg*gs}..{gg*gs+gs-1} (group {gg}): "
                  f"{[f'{a},{b}' for a, b in row[:16]]}")
            print(f"        ... {[f'{a},{b}' for a, b in row[16:32]]}")
            print(f"        ... {[f'{a},{b}' for a, b in row[32:48]]}")
            print(f"        ... {[f'{a},{b}' for a, b in row[48:64]]}")

    # ---- B) scale 探针 ----
    s = (torch.rand(K // gs, N, generator=g) * 0.5 + 0.5).half().to(dev)
    C = run(qI, s, zeros, A, N, K, gs, dev)
    sg = s.repeat_interleave(gs, dim=0).t().float()      # [N, K], diag 上 s[n//gs,n]
    yg = A.float() * torch.diag(sg)[None, :]
    d = (C - yg).abs().max().item() / yg.abs().max().item()
    print(f"[B] q=I,s=rand,z=0: max rel = {d:.4e}  {'OK' if d < 1e-2 else '!! scale 路径错'}")
    if d > 1e-2:
        ratio = (C / A.float()).nan_to_num(0.0)
        print("    ratio[0,:8] =", [round(v, 3) for v in ratio[0, :8].tolist()])
        print("    s diag[:8]  =", [round(v, 3) for v in torch.diag(sg)[:8].tolist()])

    # ---- C) zero 探针 ----
    # 注意金标准: zero 作用于整个 W (不止 q 的非零位):
    #   W[n,k] = q[n,k] - z[k//gs,n]  ->  y = A @ (q - zrep)^T
    z = (torch.rand(K // gs, N, generator=g) * 2 + 0.5).half().to(dev)
    C = run(qI, ones, z, A, N, K, gs, dev)
    zrep = z.repeat_interleave(gs, dim=0).t().float()        # [N, K]
    yg = A.float() @ (qI.float() - zrep).t()
    d = (C - yg).abs().max().item() / max(yg.abs().max().item(), 1e-9)
    print(f"[C] q=I,s=1,z=rand: max rel = {d:.4e}  {'OK' if d < 1e-2 else '!! zero 路径错'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
