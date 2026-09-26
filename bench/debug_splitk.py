"""MOE split-K 的最小隔离测试: 直接调 mul_3bit_moe, 不经过 fused 流程。

用法 (5090):
    python bench/debug_splitk.py            # E=1 N=128 K=2048 单块单 tile
    python bench/debug_splitk.py --n 256    # 2 个 n-tile
    python bench/debug_splitk.py --e 2      # 2 专家 2 m-block
"""
import argparse
import os
import sys

import torch  # 必须先于扩展加载 (libc10)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))

import marlin_int3_moe  # noqa: E402


# 按 GPU 架构选 .so (普通 import 只会撞到无后缀的 sm_120 版)
def _load_ext():
    d = os.path.dirname(marlin_int3_moe.__file__)
    cap = torch.cuda.get_device_capability(0)
    tag = f"_sm{cap[0]}{cap[1]}"
    cands = [f for f in os.listdir(d)
             if f.startswith("brmoe_moe_int3") and f.endswith(".so")]
    pref = [f for f in cands if tag in f] or [f for f in cands
                                              if "_sm" not in f]
    import importlib.util
    path = os.path.join(d, sorted(pref)[0])
    print(f"    加载扩展: {os.path.basename(path)}")
    spec = importlib.util.spec_from_file_location("brmoe_moe_int3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = _load_ext()  # noqa: E402
from marlin_int3_moe.repack import repack_one  # noqa: E402


def run_case(E, N, K, gs, dev):
    """E 个专家, 每专家一个满 16 行 m-block (num_post = E*16)。"""
    g = torch.Generator(device="cpu").manual_seed(42)
    q = torch.randint(0, 8, (E, N, K), generator=g).to(torch.int32).to(dev)
    s = (torch.rand(E, K // gs, N, generator=g) * 0.02 + 0.005).half().to(dev)
    z = (torch.rand(E, K // gs, N, generator=g) * 5 + 0.5).half().to(dev)

    B1s, B2s, ss, zs = [], [], [], []
    for e in range(E):
        B1, B2, sp, zp = repack_one(q[e], s[e], z[e], gs)
        B1s.append(B1); B2s.append(B2); ss.append(sp); zs.append(zp)
    B1 = torch.stack(B1s).contiguous()
    B2 = torch.stack(B2s).contiguous()
    sp = torch.stack(ss).contiguous()
    zp = torch.stack(zs).contiguous()

    rows = E * 16
    A = (torch.randn(rows, K, generator=g) * 0.1).half().to(dev)
    eid = torch.arange(E, dtype=torch.int32, device=dev)   # block e -> expert e
    meta = torch.tensor([rows], dtype=torch.int32, device=dev)

    # ---- 金标准: y = A @ W^T, W[n,k] = (q - z[k//gs, n]) * s[k//gs, n] ----
    sb = s.repeat_interleave(gs, dim=1).transpose(1, 2).float()   # [E, N, K]
    zb = z.repeat_interleave(gs, dim=1).transpose(1, 2).float()
    yg = torch.stack([A[e * 16:(e + 1) * 16].float()
                      @ ((q[e].float() - zb[e]) * sb[e]).t()
                      for e in range(E)]).reshape(E * 16, N)

    # ---- split-K 分区诊断: 权重只留一半 K, 看各 split 是否真算了不同区间 ----
    # s=1/z=0 (W=q), ks=2: split0 应算 [0,K/2), split1 应算 [K/2,K)
    def _half(which):
        qh = q.clone()
        if which == 0:
            qh[:, :, K // 2:] = 0
        else:
            qh[:, :, :K // 2] = 0
        B1s, B2s, ss, zs = [], [], [], []
        one = torch.ones(K // gs, N, dtype=torch.float16, device=dev)
        zer = torch.zeros(K // gs, N, dtype=torch.float16, device=dev)
        for e in range(E):
            b1, b2, s2, z2 = repack_one(qh[e], one, zer, gs)
            B1s.append(b1); B2s.append(b2); ss.append(s2); zs.append(z2)
        C32 = torch.zeros(rows, N, dtype=torch.float32, device=dev)
        Cd = torch.zeros(rows, N, dtype=torch.float16, device=dev)
        ext.mul_3bit_moe(A, torch.stack(B1s).contiguous(),
                         torch.stack(B2s).contiguous(), Cd,
                         torch.stack(ss).contiguous(),
                         torch.stack(zs).contiguous(), eid, meta, E,
                         C32=C32, k_splits=2)
        torch.cuda.synchronize()
        yh = torch.stack([A[e * 16:(e + 1) * 16].float()
                          @ qh[e].float().t() for e in range(E)]
                         ).reshape(E * 16, N)
        d = (C32 - yh).abs().max().item() / max(yh.abs().max().item(), 1e-9)
        return d

    d0, d1 = _half(0), _half(1)
    print(f"  [分区诊断 ks=2] 前半K权重: rel={d0:.3e}  后半K权重: rel={d1:.3e}"
          f"  (两个都小 = 分区正确)")
    # ks=1 + C32: 全 K + 原子写出 (分区逻辑不激活) -> 纯 writer 映射测试
    Cd = torch.zeros(rows, N, dtype=torch.float16, device=dev)
    C32_1 = torch.zeros(rows, N, dtype=torch.float32, device=dev)
    ext.mul_3bit_moe(A, B1, B2, Cd, sp, zp, eid, meta, E,
                     C32=C32_1, k_splits=1)
    torch.cuda.synchronize()
    d = (C32_1 - yg).abs().max().item() / max(yg.abs().max().item(), 1e-9)
    print(f"  [writer 隔离 ks=1+C32] vs 金标准: {d:.3e} "
          f"{'OK' if d < 5e-3 else '!! 原子写出映射错'}")
    if d >= 5e-3:
        match = []
        for n in range(min(N, 32)):
            dd = (yg - C32_1[:, n:n + 1]).abs().sum(dim=0)
            k = int(dd.argmin())
            match.append(k if dd[k] < 1e-3 else -1)
        print("    C32 列 n -> gold 列 k (前 32):", match)
        print("    C32[0,:8] =", [round(v, 3) for v in C32_1[0, :8].tolist()])
        print("    gold[0,:8]=", [round(v, 3) for v in yg[0, :8].tolist()])

    # ks=2 与 ks=4 输出是否逐位相同
    _c = []
    for ks in (2, 4):
        C32 = torch.zeros(rows, N, dtype=torch.float32, device=dev)
        ext.mul_3bit_moe(A, B1, B2, Cd, sp, zp, eid, meta, E,
                         C32=C32, k_splits=ks)
        torch.cuda.synchronize()
        _c.append(C32)
    print(f"  [诊断] ks=2 与 ks=4 输出逐位相同: {torch.equal(_c[0], _c[1])}")

    C1 = torch.zeros(rows, N, dtype=torch.float16, device=dev)
    ext.mul_3bit_moe(A, B1, B2, C1, sp, zp, eid, meta, E)
    torch.cuda.synchronize()

    print(f"  [ks=1] vs 金标准: max rel = "
          f"{(C1.float() - yg).abs().max().item() / yg.abs().max().item():.3e}")

    for ks in (2, 4):
        C32 = torch.zeros(rows, N, dtype=torch.float32, device=dev)
        ext.mul_3bit_moe(A, B1, B2, C1, sp, zp, eid, meta, E,
                         C32=C32, k_splits=ks)
        torch.cuda.synchronize()
        d = (C32 - yg).abs().max().item() / yg.abs().max().item()
        d1 = (C32 - C1.float()).abs().max().item() / yg.abs().max().item()
        print(f"  [ks={ks}] vs 金标准: {d:.3e}   vs ks=1: {d1:.3e}")
        if d > 1e-2:
            # 模式诊断: 是不是"所有段都算了同一段 K"?
            kh = K // ks
            yp = torch.stack([A[e * 16:(e + 1) * 16, :kh].float()
                              @ ((q[e][:, :kh].float() - zb[e][:, :kh])
                                 * sb[e][:, :kh]).t() for e in range(E)]
                             ).reshape(E * 16, N)
            r_same = (C32 - ks * yp).abs().max().item() / yg.abs().max().item()
            r_full = (C32 - ks * yg).abs().max().item() / yg.abs().max().item()
            print(f"        假设: C32≈ks*前段部分和 {r_same:.3e}   "
                  f"C32≈ks*全K {r_full:.3e}")
            print("        C32[0,:4] =", [round(v, 4) for v in C32[0, :4].tolist()])
            print("        gold[0,:4]=", [round(v, 4) for v in yg[0, :4].tolist()])
            print("        C32 非零占比:",
                  f"{(C32 != 0).float().mean().item():.3f}")


def run_linear(N, K, gs, dev):
    """同一组权重走**线性版** binding (MOE=false 模板实例), 隔离 MOE 分支。"""
    g = torch.Generator(device="cpu").manual_seed(42)
    q = torch.randint(0, 8, (N, K), generator=g).to(torch.int32).to(dev)
    s = (torch.rand(K // gs, N, generator=g) * 0.02 + 0.005).half().to(dev)
    z = (torch.rand(K // gs, N, generator=g) * 5 + 0.5).half().to(dev)
    B1, B2, sp, zp = repack_one(q, s, z, gs)

    A = (torch.randn(16, K, generator=g) * 0.1).half().to(dev)
    C = torch.zeros(16, N, dtype=torch.float16, device=dev)
    ws = torch.zeros((N // 64 + 1) * 16, dtype=torch.int32, device=dev)
    ext.mul_3bit_with_zeros(A, B1, B2, C, sp, zp, ws, 128, 128, -1, 8)
    torch.cuda.synchronize()

    sb = s.repeat_interleave(gs, dim=0).t().float()      # [N, K]
    zb = z.repeat_interleave(gs, dim=0).t().float()
    yg = A.float() @ ((q.float() - zb) * sb).t()
    d = (C.float() - yg).abs().max().item() / yg.abs().max().item()
    print(f"  [linear MOE=false] vs 金标准: max rel = {d:.3e} "
          f"{'OK' if d < 5e-3 else '!! 错'}")
    if d > 5e-3:
        print("        C[0,:4]  =", [round(v, 4) for v in C[0, :4].tolist()])
        print("        gold[0,:4]=", [round(v, 4) for v in yg[0, :4].tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=1)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--gs", type=int, default=64)
    args = ap.parse_args()
    assert args.n % 128 == 0 and args.k % 128 == 0
    print(f"=== debug_splitk: E={args.e} rows={args.e * 16} N={args.n} "
          f"K={args.k} gs={args.gs} ===")
    print("  -- 线性版 binding (MOE=false) --")
    run_linear(args.n, args.k, args.gs, "cuda")
    print("  -- MOE 版 binding (MOE=true) --")
    run_case(args.e, args.n, args.k, args.gs, "cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
