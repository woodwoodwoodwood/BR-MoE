"""int3 grouped MoE CUDA kernel 的数值验证 + 性能对比。

验证: 非对称量化 (随机 fp16 零点) 真权重 -> CUDA 路径输出 vs 反量化金标准
      (与 micro_moe.py --check 同一口径, rel < 5e-3)。
测速: CUDA (mul_3bit_moe) vs Triton TC (自动) vs cuBLAS bmm 参照,
      M = 16/32/64/512/2048 —— 目标区间是 TC 目前只有 33% 峰值的 M>=16。

用法 (需 GPU; 先在 marlin_int3_moe/ 下 build_ext --inplace):
    python bench/verify_moe_cuda.py                # 真实形状 E=64 K=2048 I=1408
    python bench/verify_moe_cuda.py --small        # 小形状快速冒烟
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))     # brmoe_int3_vllm (micro_moe 依赖)
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "triton_int3"))

import marlin_int3_moe  # noqa: E402  (包目录, 触发不了什么)

# 按 GPU 架构选 .so: brmoe_moe_int3_sm80/sm120.cpython-*.so, 无后缀 = 默认(5090 版)
import glob as _glob  # noqa: E402
import importlib.util as _ilu  # noqa: E402


def _load_ext():
    d = os.path.dirname(marlin_int3_moe.__file__)
    cap = torch.cuda.get_device_capability(0)
    tag = f"_sm{cap[0]}{cap[1]}"
    cands = [f for f in os.listdir(d)
             if f.startswith("brmoe_moe_int3") and f.endswith(".so")]
    pref = [f for f in cands if tag in f] or [f for f in cands if "_sm" not in f]
    if not pref:
        raise ImportError(f"没有匹配 sm{cap[0]}{cap[1]} 的 brmoe_moe_int3 .so: {cands}")
    path = os.path.join(d, sorted(pref)[0])
    print(f"    加载扩展: {os.path.basename(path)}")
    spec = _ilu.spec_from_file_location("brmoe_moe_int3", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = _load_ext()  # noqa: E402

from marlin_int3_moe.repack import repack_moe  # noqa: E402
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda  # noqa: E402
from micro_moe import bench_graph, make_routing, make_weight_pair, per_expert_bytes  # noqa: E402


def make_routing_zipf(M, E, topk, dev, alpha=1.2, seed=0):
    """相关路由 (贴近真实 decode): 专家热度 p_e ∝ (e+1)^-alpha,
    每个 token 用 Gumbel-topk 不放回按 p 采样 topk 个专家。
    与 make_routing 的均匀随机对照 —— 均匀随机在小 M 就把 64 个专家全激活,
    严重高估 MoE 成本 (padding 到 64*slot 行), 微观结论与 e2e 对不上。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = (torch.arange(E, dtype=torch.float64) + 1) ** (-alpha)
    logp = torch.log(p).expand(M, E).clone()
    gum = -torch.log(-torch.log(torch.rand(M, E, generator=g).double()
                                    .clamp_min(1e-12)))
    ids = torch.argsort(logp + gum, dim=1, descending=True)[:, :topk]
    ids = ids.to(device=dev, dtype=torch.int64).contiguous()
    w = torch.rand(M, topk, generator=g)
    w = (w / w.sum(1, keepdim=True)).half().to(dev).contiguous()
    return w, ids


def quantize_asym(W, gs, dev, seed=0):
    """非对称 int3: q = clamp(round(w/s + z), 0, 7), z 随机。返回 (q, s, z) 与金标准 Wdeq。
    有效权重 w = (q - z) * s (fp16 零点减法, 与生产格式一致)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    E, N, K = W.shape
    s = (torch.rand(E, K // gs, N, generator=g) * 0.02 + 0.005).half().to(dev)
    z = (torch.rand(E, K // gs, N, generator=g) * 5 + 0.5).half().to(dev)
    s_b = s.repeat_interleave(gs, dim=1).transpose(1, 2).float()  # [E, N, K]
    z_b = z.repeat_interleave(gs, dim=1).transpose(1, 2).float()
    q = torch.clamp(torch.round(W.float() / s_b + z_b), 0, 7).to(torch.int32)
    Wdeq = ((q.float() - z_b) * s_b).half()
    return q, s, z, Wdeq


def pack_triton(q, K):
    """[E, N, K] int (0..7) -> [E, N, Kpack] int32 (Triton N-major)。"""
    from int3_moe.packing import pack_int3
    E, N, _ = q.shape
    return pack_int3(q.reshape(E * N, K), transposed=False).reshape(E, N, K // 32 * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--perf-only", action="store_true",
                    help="跳过数值校验, 只测速 (数值调试期间用)")
    ap.add_argument("--ms", default="16,32,64,512,2048")
    ap.add_argument("--ksplits", default="1",
                    help="CUDA 路径的 split-K 段数扫描, 逗号分隔 (>1 时走 fp32 atomic 归约)")
    ap.add_argument("--gemv-max-m", type=int, default=None,
                    help="M<=此值时 CUDA 列委托 GEMV; None = 按架构自选 (sm_80:1, 其他:8); 0 = 纯 CUDA kernel")
    ap.add_argument("--peak-gbs", type=float, default=1792.0)
    ap.add_argument("--routing", default="random", choices=["random", "zipf"],
                    help="random=均匀随机(旧口径); zipf=相关路由(贴近 e2e)")
    ap.add_argument("--fp16-grouped", action="store_true",
                    help="加一列 fp16 权重走同一 grouped 流水线 (LAYOUT16) —— "
                         "这才是 vLLM 生产里 fp16 MoE 的形态 (cuBLAS-ref 是逐专家 bmm)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! 需要 GPU")
        return 1
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"=== MoE CUDA kernel 验证 on {prop.name} (sm_{prop.major}{prop.minor}) ===")

    if args.small:
        E, K, I, gs, topk = 8, 256, 128, 64, 2
    else:
        E, K, I, gs, topk = 64, 2048, 1408, 64, 6
    print(f"    E={E} K={K} I={I} gs={gs} top_k={topk}")

    # ---- 权重: 随机 fp16 -> 非对称量化 -> 两种布局 ----
    W13, W2 = make_weight_pair(E, K, I, dev)
    q13, s13, z13, W13d = quantize_asym(W13, gs, dev, seed=0)
    q2, s2, z2, W2d = quantize_asym(W2, gs, dev, seed=1)
    packed_t = {  # Triton 布局 (repack 的输入契约)
        "w13_q": pack_triton(q13, K), "s13": s13, "z13": z13,
        "w2_q": pack_triton(q2, I), "s2": s2, "z2": z2,
        "group_size": gs,
    }
    pk = repack_moe(packed_t)   # CUDA 布局
    print(f"    repack 完成: B13_1 {tuple(pk['B13_1'].shape)}  "
          f"B2_1 {tuple(pk['B2_1'].shape)}")

    # ---- 数值 ----
    if args.perf_only:
        pk = pk  # 跳过数值, 直接测速
        print("\n  -- 数值跳过 (--perf-only) --")
    else:
        # 三个口径互相对拍: CUDA / Triton(已验证) / 金标准(反量化+bmm)
        print("\n  -- 数值校验 (CUDA vs Triton vs 金标准) --")
    from brmoe_int3_vllm.kernel import get_fused_moe_int3
    fused_t = get_fused_moe_int3()
    from micro_moe import make_ref_fn
    rc = 0
    if not args.perf_only:
        for M in [1, 8, 16, 32, 64]:
            x = (torch.randn(M, K, device=dev) * 0.1).half()
            tw, tid = make_routing(M, E, topk, dev)
            gold_fn, _, _, _ = make_ref_fn(x, tw, tid, W13d, W2d, I, K)
            gold = gold_fn().float()
            scl = max(gold.abs().max().item(), 1e-9)
            y_c = fused_moe_int3_cuda(x, tw, tid, pk, ext).float()
            y_t = fused_t(x, tw, tid, packed_t, fast=True).float()
            d_c = (y_c - gold).abs().max().item() / scl
            d_t = (y_t - gold).abs().max().item() / scl
            d_ct = (y_c - y_t).abs().max().item() / scl
            ok = d_c < 5e-3
            rc |= (not ok)
            print(f"    M={M:>3}  CUDA vs 金标准={d_c:.2e}  Triton vs 金标准={d_t:.2e}  "
                  f"CUDA vs Triton={d_ct:.2e}  {'OK' if ok else '!! 偏差过大'}")
    if rc:
        print("\n数值未过, 不进入测速")
        return rc

    # ---- 性能: CUDA vs Triton TC(自动) vs cuBLAS 参照 ----
    print(f"\n  -- 性能 (CUDA Graph 口径, routing={args.routing}) --")
    from brmoe_int3_vllm.kernel import get_fused_moe_int3  # 需要 tools/ 在 path
    fused_t = get_fused_moe_int3()
    bpe = per_expert_bytes(E, K, I, gs, "int3")
    packed_f = None
    if args.fp16_grouped:
        from int3_moe.ops import pack_moe_weights
        packed_f = pack_moe_weights(W13, W2, gs, layout="fp16")

    _rt = make_routing_zipf if args.routing == "zipf" else make_routing
    for M in [int(v) for v in args.ms.split(",")]:
        x = (torch.randn(M, K, device=dev) * 0.1).half()
        tw, tid = _rt(M, E, topk, dev)
        uniq = int(torch.unique(tid).numel())

        kss = [int(v) for v in args.ksplits.split(",")]

        # split-K 一致性: 各档输出应与 ks=1 吻合 (仅 fp32 归约顺序差异)。
        # (绝对数值另由 --small 校验; 此处防 split-K 实现本身出错)
        if len(kss) > 1:
            y1 = fused_moe_int3_cuda(x, tw, tid, pk, ext, ksplit=1).float()
            s1 = max(y1.abs().max().item(), 1e-9)
            for ks in kss:
                if ks == 1:
                    continue
                yk = fused_moe_int3_cuda(x, tw, tid, pk, ext, ksplit=ks).float()
                d = (yk - y1).abs().max().item() / s1
                print(f"    M={M:>5}  ks={ks} vs ks=1 一致性: max rel = {d:.2e}"
                      f"  {'OK' if d < 1e-2 else '!! 不一致'}")

        # packed=packed_t + gemv_max_m=8: 与 vLLM 插件生产分派一致
        # (M<=8 时 fused_moe_int3_cuda 内部委托 GEMV, 此时 "CUDA" 列实为 GEMV)
        parts = []
        best_ks, best_t = None, None
        for ks in kss:
            t = bench_graph(lambda ks=ks: fused_moe_int3_cuda(
                x, tw, tid, pk, ext, ksplit=ks, packed=packed_t,
                gemv_max_m=getattr(args, "gemv_max_m")))
            parts.append(f"ks{ks}={t:7.2f}us" if t is not None else f"ks{ks}=  FAIL ")
            if t is not None and (best_t is None or t < best_t):
                best_ks, best_t = ks, t
        t_tri = bench_graph(lambda: fused_t(x, tw, tid, packed_t, fast=True))
        t_fg = (bench_graph(lambda: fused_t(x, tw, tid, packed_f, fast=True))
                if packed_f is not None else None)
        from micro_moe import make_ref_fn
        ref_fn, _, _, ref_bytes = make_ref_fn(x, tw, tid, W13, W2, I, K)
        t_ref = bench_graph(ref_fn)

        if best_t is None:
            print(f"    M={M:>5}  CUDA {' '.join(parts)}  "
                  f"Triton {t_tri:8.2f}us  cuBLAS-ref {t_ref:8.2f}us")
            continue
        gbs = uniq * bpe / (best_t * 1e-6) / 1e9
        line = (f"    M={M:>5}  CUDA {' '.join(parts)}  best=ks{best_ks} "
                f"({gbs:6.1f} GB/s, {gbs/args.peak_gbs*100:4.1f}%)  "
                f"Triton {t_tri:8.2f}us  cuBLAS-ref {t_ref:8.2f}us")
        if best_t and t_tri:
            line += f"  CUDA/TC = {t_tri/best_t:.2f}x"
        if t_fg is not None:
            line += f"  fp16-grp {t_fg:8.2f}us (CUDA/fp16 = {t_fg/best_t:.2f}x)"
        print(line)

    print(f"\n=== rc={rc} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
