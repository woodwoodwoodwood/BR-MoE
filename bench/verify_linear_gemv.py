"""验证 + 测速: linear_method 的 GEMV+split-K 分支 (M<=8)。

为什么需要
----------
attention 的 112 个 int3 linear (K=N=2048) 是 int3dense 剩下的最大瓶颈:
e2e 里它比 brmoe3bit 慢 ~2.65ms (bs=1), 折合每层 94.6us。TC 路径在 M=1 只有
32 个 CTA (每 SM 0.3 个) 且 M 补齐到 16 行。GEMV+split-K 在 MoE 路径上已验证
(88->49us), 这里把它复用到 linear (单专家, top_k=1)。

本脚本:
  1. 数值: 对称 int3 真权重 -> GEMV 分支 vs 反量化金标准 (cuBLAS), 逐 M 比对;
     TC 路径 (gemm_only) 也一并比对。
  2. 测速: GEMV (含 ksplit 扫描) vs TC vs cuBLAS fp16, CUDA Graph 口径。

用法 (需 GPU):
    python bench/verify_linear_gemv.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "triton_int3"))

from brmoe_int3_vllm.linear_method import (  # noqa: E402
    brmoe_int3_linear, _kernel, _get_gemv_ws,
)
from micro_linear import bench_graph, make_bufs, setup_only, gemm_only  # noqa: E402
from int3_moe.packing import quantize_int3_symmetric, pack_int3  # noqa: E402


def make_linear_weights(K, N, gs, dev):
    """对称 int3 (zero=4) 真权重 + 反量化金标准。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    W = (torch.randn(N, K, generator=g) * 0.02).half().to(dev)
    amax = W.reshape(N, K // gs, gs).abs().amax(dim=2).clamp_min(1e-8)
    s = (amax / 3.0).half()                        # [N, K//gs]
    q = quantize_int3_symmetric(W, s, gs)          # 整数值 in [-4, 3]
    qw = pack_int3(q, transposed=True)             # [Kpack, N] (K-major)
    scales = s.t().contiguous()                    # [K//gs, N]
    zeros = torch.full((K // gs, N), 4.0, dtype=torch.float16, device=dev)
    Wdeq = ((q - 4).to(torch.float32)
            * s.to(torch.float32).repeat_interleave(gs, dim=1))   # [N, K] fp32
    return qw, scales, zeros, Wdeq


def gemv_call(x, qw, scales, zeros, gs, ksplit):
    """复刻 impl 的 GEMV 分支, ksplit 可调 (生产是自动档位)。"""
    M, K = x.shape
    N = scales.shape[1]
    Wp = qw.unsqueeze(0)
    S = scales.view(1, K // gs, N)
    Z = zeros.view(1, K // gs, N)
    ids, out32 = _get_gemv_ws(M, N, x.device)
    out32.zero_()
    _kernel.routed_int3_gemv(x, Wp, S, Z, ids, None, out32, 1, gs,
                             w_transposed=True, block_n=64, num_warps=2,
                             ksplit=ksplit)
    return out32.to(x.dtype)


def main():
    if not torch.cuda.is_available():
        print("!! 需要 GPU")
        return 1
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"=== linear GEMV 验证 on {prop.name} (sm_{prop.major}{prop.minor}) ===")

    gs = 64
    rc = 0
    for K, N in ((2048, 2048), (2048, 2816), (2816, 2048)):
        qw, scales, zeros, Wdeq = make_linear_weights(K, N, gs, dev)
        W16 = Wdeq.t().contiguous().half()         # cuBLAS 参照用 [K, N]
        bfp16 = K * N * 2
        bint3 = (K // 32 * 3) * N * 4 + 2 * (K // gs) * N * 2
        print(f"\n########## K={K} N={N}  (int3 {bint3/2**20:.2f} MiB vs "
              f"fp16 {bfp16/2**20:.2f} MiB) ##########")

        for M in (1, 2, 4, 8, 16):
            x = (torch.randn(M, K, device=dev) * 0.1).half()
            gold = (x.float() @ Wdeq.t())          # [M, N] fp32
            scl = max(gold.abs().max().item(), 1e-9)

            # ---- 数值 ----
            y_g = gemv_call(x, qw, scales, zeros, gs,
                            ksplit=max(1, min(16, (K + 127)//128)))
            d_g = (y_g.float() - gold).abs().max().item() / scl
            ok_tc = True
            d_tc = float("nan")
            try:
                b = make_bufs(x, qw, scales, zeros, gs)
                xp, eid, sti, out = setup_only(b, x)
                y_t = gemm_only(b, xp, eid, sti, out)[:M]
                d_tc = (y_t.float() - gold).abs().max().item() / scl
                ok_tc = d_tc < 5e-3
            except Exception as exc:               # noqa: BLE001
                ok_tc = False
                d_tc = str(exc)[:50]
            ok_g = d_g < 5e-3
            if not (ok_g and ok_tc):
                rc = 1

            # ---- 测速 (graph 口径; 捕获失败返回 None, 打印要兜底) ----
            t_fp = bench_graph(lambda: F.linear(x, W16))
            t_tc = bench_graph(lambda: gemm_only(b, xp, eid, sti, out))
            t_gv = bench_graph(lambda: gemv_call(
                x, qw, scales, zeros, gs,
                ksplit=max(1, min(16, (K + 127)//128))))
            def _f(v): return f"{v:7.2f}" if v is not None else "    n/a"
            def _r(v): return f"{v/t_fp:4.2f}x" if (v and t_fp) else " n/a"
            line = (f"  M={M:>3}  fp16 {_f(t_fp)}us | TC {_f(t_tc)}us | "
                    f"GEMV {_f(t_gv)}us | GEMV/fp16 {_r(t_gv)} | "
                    f"TC/fp16 {_r(t_tc)}")
            line += f" | 数值 GEMV {'OK' if ok_g else f'!! {d_g:.2e}'}"
            line += (f" TC {'OK' if ok_tc else f'!! {d_tc}'}" if ok_tc
                     else f" TC !! {d_tc}")
            print(line)

        # ---- ksplit 扫描 (M=1, 4, 8) ----
        for M in (1, 4, 8):
            x = (torch.randn(M, K, device=dev) * 0.1).half()
            nblk = (N + 63) // 64
            cells = []
            for ks in (2, 4, 8, 16):
                if ks > (K + 127) // 128:
                    continue
                t = bench_graph(lambda: gemv_call(x, qw, scales, zeros, gs, ks))
                cells.append(f"ks={ks}: {t:6.2f}us" if t else f"ks={ks}: n/a")
            print(f"    ksplit 扫描 M={M} (program=M*{nblk}*ks): "
                  + "  ".join(cells))

    print()
    print("=== rc=%d ===" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
