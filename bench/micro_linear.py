"""单独微基准: int3 线性层 vs fp16 cuBLAS。

目的
----
端到端 benchmark 只能告诉我们 "int3dense 比 brmoe3bit 慢 1.14~1.34x",
但**说不清慢在哪**。这里把三层拆开单独计时:

    A. full   : brmoe_int3_linear(...)  —— 真实的调用, 含全部 setup
    B. gemm   : 只用预建好的缓冲调 int3_moe_gemm —— 纯 kernel
    C. setup  : xp=zeros / xp[:M2]=x2 / eid=zeros / sti=arange —— 我们自己加的辅助 kernel
    参照 fp16 : F.linear(x, W16)  —— cuBLAS

    A - B - C ≈ 0  (若不为 0 说明有我没拆到的开销)
    A - fp16     = int3 相对 cuBLAS 的真实惩罚

以及一个反量化的理论对照: int3 读的字节数只有 fp16 的 1/4.6,
所以如果 kernel 是**带宽受限**, 它应该快 4.6x。实测慢 1.3x => 它完全不是带宽受限。

被量化的 196 个线性层 (int3dense 里全是 int3, brmoe3bit 里全是 fp16):
    2048 x 2048    112 个   attention q/k/v/o   (28 层 x 4)      <- 占 57%
    2048 x 2816     54 个   shared expert gate/up (27 层 x 2)
    2816 x 2048     27 个   shared expert down
    2048 x 10944     2 个   dense MLP gate/up    (仅第 0 层)
    10944 x 2048     1 个   dense MLP down
    ---------------------
    合计           196 个
所以默认只测 2048x2048 (占 57%) 就能回答大半问题; 要完整跑用:
    --shapes 2048x2048,2048x2816,2816x2048,2048x10944,10944x2048

用法 (必须在 5090 节点的分配里跑):
    python bench/micro_linear.py
    python bench/micro_linear.py --shapes 2048x2048,2048x2816,2816x2048
    python bench/micro_linear.py --ms 1,8,64 --graph 0     # 快速看
"""
import argparse
import sys

import torch

# 直接复用插件里的实现与 kernel, 保证测的就是真实路径
from brmoe_int3_vllm.linear_method import (
    brmoe_int3_linear,
    int3_moe_gemm,
    pick_tiles,
)

# ---------------------------------------------------------------------------
# 计时 helper
# ---------------------------------------------------------------------------


def bench(fn, warmup=10, rep=50):
    """返回单次调用耗时 (微秒)。eager: 含 CPU 侧发射开销。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1e3


def bench_graph(fn, iters=20, warmup=3, rep=20):
    """用 CUDA Graph 捕获后测 replay 时间 (微秒/次)。

    这才是和真实端到端 benchmark 可比的口径 —— graph replay 时没有 CPU 发射开销,
    那 4 个 setup kernel 只剩 GPU 执行时间。捕获失败返回 None (不致命)。
    """
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()

        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(rep):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / rep / iters * 1e3
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 把 _brmoe_int3_linear_impl 里的步骤逐条复刻, 以便单独计时
# ---------------------------------------------------------------------------


def make_bufs(x, qweight, scales, zeros, gs):
    """复刻 impl 里 shape 推导 + 缓冲创建, 但**不启动 kernel**。"""
    M, K = x.shape
    N = scales.shape[1]
    x2 = x.reshape(-1, K).contiguous()
    M2 = x2.shape[0]

    # w_transposed=True 要求 [E, Kpack, N]
    Wp = qweight.unsqueeze(0).contiguous()
    S = scales.view(1, K // gs, N).contiguous()
    Z = zeros.view(1, K // gs, N).contiguous()

    block_m, block_n, block_k, slot = pick_tiles(K, N, gs, M2)

    num_post = (M2 + slot - 1) // slot * slot
    grid_m = (num_post + block_m - 1) // block_m
    n_rows = grid_m * block_m
    nseg = block_m // slot

    return dict(K=K, N=N, gs=gs, x2=x2, M2=M2, Wp=Wp, S=S, Z=Z,
                block_m=block_m, block_n=block_n, block_k=block_k, slot=slot,
                num_post=num_post, grid_m=grid_m, n_rows=n_rows, nseg=nseg)


def setup_only(b, x):
    """只跑辅助 kernel (zeros / copy / arange / empty), 返回调用 gemm 所需的一切。"""
    M2, K, N, gs = b["M2"], b["K"], b["N"], b["gs"]
    x2 = b["x2"]
    n_rows, nseg, grid_m = b["n_rows"], b["nseg"], b["grid_m"]

    if n_rows != M2:
        xp = torch.zeros(n_rows, K, dtype=x2.dtype, device=x2.device)
        xp[:M2] = x2
    else:
        xp = x2
    eid = torch.zeros(grid_m * nseg, dtype=torch.int32, device=x.device)
    sti = torch.arange(n_rows, dtype=torch.int32, device=x.device)
    out = torch.empty((n_rows, N), dtype=x.dtype, device=x.device)
    return xp, eid, sti, out


def gemm_only(b, xp, eid, sti, out, num_warps=4, num_stages=3, block_n=None):
    """纯 kernel, 不含任何创建/初始化。

    num_warps / num_stages 是关键: int3_moe_gemm 的默认值是 **2 / 1**,
    那是给 MoE 路径调的。线性层的 M 很小 (grid = (grid_m, N//block_n) 只有
    32 个 CTA, 而 5090 有 170 个 SM), 默认值下完全被内存延迟卡死。
    block_n 同理: 调小它能增加 CTA 数 (N//block_n)。
    """
    int3_moe_gemm(
        xp, b["Wp"], b["S"],
        sti, eid,
        b["num_post"],
        b["M2"],
        group_size=b["gs"],
        a_gather=False,
        add=False,
        out=out,
        meta=None,                 # HAS_META=False -> 用 num_post_arg
        grid_m=b["grid_m"],
        block_m=b["block_m"], block_n=block_n or b["block_n"],
        block_k=b["block_k"], slot=b["slot"],
        layout4=False, layout16=False,
        zeros=b["Z"],
        w_transposed=True,
        direct4=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def sweep(b, x, label=""):
    """扫 num_warps x num_stages x block_n, 找小 M 下的最优启动参数。"""
    K, N, M2 = b["K"], b["N"], b["M2"]
    xp, eid, sti, out = setup_only(b, x)
    base = gemm_only(b, xp, eid, sti, out)

    print(f"\n  --- 参数扫描 M={M2} ({label}) grid_m={b['grid_m']} "
          f"block_m={b['block_m']} block_k={b['block_k']} ---")
    print(f"    {'num_warps':>10}" + "".join(f"{'st='+str(s):>10}" for s in (1, 2, 3, 4)))
    best = (None, 1e9)
    for nw in (1, 2, 4, 8):
        cells = []
        for ns in (1, 2, 3, 4):
            try:
                t = bench(lambda: gemm_only(b, xp, eid, sti, out, nw, ns), 5, 30)
                # 顺带确认改参数不影响结果
                r = gemm_only(b, xp, eid, sti, out, nw, ns)
                ok = torch.equal(r, base)
                cells.append(f"{t:>9.2f}{'' if ok else '!'}")
                if t < best[1]:
                    best = ((nw, ns, b["block_n"]), t)
            except Exception as e:
                cells.append(f"{'ERR':>10}")
        print(f"    {nw:>10}" + "".join(f"{c:>10}" for c in cells))

    # block_n 影响 CTA 数: N//block_n; 越大越小 -> 小 M 时该调小
    bn_opts = [v for v in (16, 32, 64, 128) if N % v == 0]
    if len(bn_opts) > 1:
        print(f"    block_n (num_warps/stages 用默认 2/1 与最优 {best[0][:2]} 各试):")
        for bn in bn_opts:
            row = []
            for nw, ns in {  # 去重
                (2, 1), (best[0][0], best[0][1]),
            }:
                t = bench(lambda: gemm_only(b, xp, eid, sti, out, nw, ns, bn), 5, 30)
                row.append(f"w{nw}s{ns}={t:7.2f}")
                if t < best[1]:
                    best = ((nw, ns, bn), t)
            print(f"      block_n={bn:>4} (CTA={b['grid_m']*N//bn:>4})  " + "  ".join(row))

    print(f"    >>> 最优: num_warps={best[0][0]} num_stages={best[0][1]} "
          f"block_n={best[0][2]}  ->  {best[1]:.2f} µs")
    return best


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="2048x2048",
                    help="逗号分隔的 KxN 列表, 如 2048x2048,2048x10944")
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--rep", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--graph", type=int, default=1,
                    help="1 = 额外测 CUDA Graph 模式 (与端到端口径一致)")
    ap.add_argument("--sweep", type=int, default=0,
                    help="1 = 额外扫描 num_warps / num_stages / block_n")
    ap.add_argument("--sweep-ms", default="1,8,64",
                    help="扫描用哪些 M")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! 没有 GPU, 请进节点: "
              "srun -p 5090 -N1 -n1 --mem=48G --gres=gpu:1 -t 1:00:00 --pty bash")
        sys.exit(1)

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  ({props.total_memory/2**30:.1f} GiB)")
    # 粗略估算: clock(kHz) -> Hz, DDR 按 x2, 乘 bus_width/8 字节 (GDDR7 是 QDR,
    # 这个公式会**低估**, 所以结果只当数量级参考; 5090 规格值约 1792 GB/s)
    bw = (props.memory_clock_rate * 1e3 * 2 * (props.memory_bus_width / 8)) / 1e9
    print(f"理论显存带宽 ≈ {bw:.0f} GB/s (公式估算; 5090 规格约 1792 GB/s)")
    print(f"group_size={args.gs}  rep={args.rep}\n")

    print("口径说明: 本脚本是 **eager** 调用 (每次都在 CPU 侧发射 kernel),")
    print("          所以 'setup' 那一列把 CPU 发射开销也算进去了。")
    print("          真实的 graph 模式下这 4 个小 kernel 只剩 GPU 执行时间,")
    print("          故 setup 列是**上界**; 而 gemm 列是可信的纯 kernel 时间。\n")

    shapes = []
    for s in args.shapes.split(","):
        k, n = s.lower().split("x")
        shapes.append((int(k), int(n)))
    ms = [int(v) for v in args.ms.split(",")]

    for (K, N) in shapes:
        gs = args.gs
        assert K % 32 == 0 and K % gs == 0, (K, gs)

        # ---- 造权重 (值不重要, 只看时间) ----
        qweight = torch.randint(-2**31, 2**31 - 1, (K // 32 * 3, N),
                                dtype=torch.int32, device=dev)
        scales = torch.rand(K // gs, N, dtype=torch.float16, device=dev) + 0.01
        zeros = torch.rand(K // gs, N, dtype=torch.float16, device=dev)

        # fp16 等效权重, 只为给 cuBLAS 当参照 (字节数按真实 fp16 权重算)
        W16 = torch.randn(K, N, dtype=torch.float16, device=dev)

        bytes_fp16_w = K * N * 2
        bytes_int3_w = (K // 32 * 3) * N * 4 + 2 * (K // gs) * N * 2

        print("=" * 108)
        print(f"K={K} N={N}  权重字节: fp16 {bytes_fp16_w/2**20:.2f} MiB | "
              f"int3 {bytes_int3_w/2**20:.2f} MiB | "
              f"压缩比 {bytes_fp16_w/bytes_int3_w:.2f}x")
        print("=" * 108)
        hdr = (f"{'M':>6}{'fp16 µs':>10}{'int3 full':>11}{'int3 gemm':>11}"
               f"{'setup':>9}{'比值 full':>11}{'Δµs':>9}"
               f"{'fp16 GB/s':>11}{'int3 GB/s':>11}")
        print(hdr)
        print("-" * len(hdr))

        grows = []
        for M in ms:
            x = torch.randn(M, K, dtype=torch.float16, device=dev)

            # 参照: cuBLAS fp16  (y = x @ W^T)
            t_fp16 = bench(lambda: torch.nn.functional.linear(x, W16),
                           args.warmup, args.rep)

            # A. full
            t_full = bench(lambda: brmoe_int3_linear(x, qweight, scales, zeros, gs),
                           args.warmup, args.rep)

            # B/C 需要预建缓冲; 但缓冲内容依赖 M, 所以每次重建 (重建不计时)
            b = make_bufs(x, qweight, scales, zeros, gs)
            xp, eid, sti, out = setup_only(b, x)

            t_gemm = bench(lambda: gemm_only(b, xp, eid, sti, out),
                           args.warmup, args.rep)
            t_setup = bench(lambda: setup_only(b, x), args.warmup, args.rep)

            # 拆分自检: full 应当 ≈ setup + gemm
            ref = brmoe_int3_linear(x, qweight, scales, zeros, gs)
            got = gemm_only(b, xp, eid, sti, out)
            # 两条路都写同一个 kernel、同样的输入; add=False 无 atomic ->
            # 必须逐位相同。不同就说明这里的缓冲复刻与真实 impl 有出入。
            same = torch.equal(ref, got[:M])

            bw_fp16 = (M * K * 2 + bytes_fp16_w) / (t_fp16 * 1e-6) / 1e9
            bw_int3 = (M * K * 2 + bytes_int3_w) / (t_full * 1e-6) / 1e9

            flag = "" if same else "  <!! 拆分与 full 结果不一致>"
            print(f"{M:>6}{t_fp16:>10.2f}{t_full:>11.2f}{t_gemm:>11.2f}"
                  f"{t_setup:>9.2f}{t_full/t_fp16:>10.2f}x{t_full-t_fp16:>9.2f}"
                  f"{bw_fp16:>11.0f}{bw_int3:>11.0f}{flag}")

            if args.graph:
                grows.append((
                    M,
                    bench_graph(lambda: torch.nn.functional.linear(x, W16)),
                    bench_graph(
                        lambda: brmoe_int3_linear(x, qweight, scales, zeros, gs)),
                ))

        if args.graph and grows:
            print()
            print("  CUDA Graph 模式 (无 CPU 发射开销, 与端到端结果可比)")
            gh = f"  {'M':>6}{'fp16 µs':>12}{'int3 µs':>12}{'int3/fp16':>12}{'Δµs':>10}"
            print(gh)
            for M, gf, gi in grows:
                if gf is None or gi is None:
                    print(f"  {M:>6}     n/a (捕获失败)")
                else:
                    print(f"  {M:>6}{gf:>12.2f}{gi:>12.2f}"
                          f"{gi/gf:>11.2f}x{gi-gf:>10.2f}")

        if args.sweep:
            print()
            print("=" * 108)
            print("参数扫描: int3_moe_gemm 的默认启动参数是 num_warps=2, num_stages=1")
            print("          grid = (grid_m, N//block_n) —— 小 M 下只有 32 个 CTA,")
            print("          而 5090 有 170 个 SM, 且 st=1 没有软件流水 -> 全被内存延迟卡住。")
            print("          带 ! 的格子表示结果与默认配置不一致 (不该出现, 出现了要查)")
            print("=" * 108)
            for M in [int(v) for v in args.sweep_ms.split(",")]:
                xs = torch.randn(M, K, dtype=torch.float16, device=dev)
                bb = make_bufs(xs, qweight, scales, zeros, gs)
                sweep(bb, xs, label=f"K={K} N={N}")

        print()
        print("读法:")
        print("  int3 full / fp16 的比值 < 1 才算 int3 快; 现在 > 1 说明慢")
        print("  int3 GB/s 若远低于理论带宽 -> kernel 不是带宽受限, 省的字节没用上")
        print("  graph 那一栏才是和端到端结果可比的数 (eager 含 CPU 发射开销)")
        print("  扫描若显示某个 (num_warps, num_stages, block_n) 明显更快,")
        print("    那就是注释里说的启动参数问题 —— 修 linear_method.py 即可")
        print()

    print("=" * 108)
    print("注意: 本脚本故意不校验数值正确性 (端到端已用 check_5090.sh 验过 16/16)。")
    print("      随机 qweight 只是为了让 kernel 有活干, 结果值无意义。")
    print("      唯一的一致性检查是 'int3 gemm' 与 'int3 full' 必须逐位相同 ——")
    print("      若不同, 说明这里的缓冲复刻与真实 impl 有出入, 计时不可信。")


if __name__ == "__main__":
    main()
