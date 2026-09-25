"""实测 BR-MoE 原生 CUDA int3 kernel (Marlin 改版) vs cuBLAS fp16 —— 选项②的 go/no-go。

为什么这个实验现在就能做
------------------------
现有 `brmoe_cuda.cpython-310-x86_64-linux-gnu.so` 是 **py3.10 + sm_80** 编的,
而 `milo` 环境正好是 py3.10.16 / torch 2.5.1+cu121 (archs 含 sm_80)。
=> **A100 上不需要重编就能直接测**。这是评估选项②最便宜的一步。

要回答的问题
------------
我们的 Triton int3 kernel 在 M=1 时是 35.4 µs, 而 cuBLAS fp16 是 13.7 µs (慢 2.6x)。
BR-MoE 自己的 CUDA kernel (Marlin 系, thread tile 256x64 / 128x128 + split-K)
能不能打赢 cuBLAS? 如果能, 选项②就值得做; 如果不能, 就别折腾了。

四个测量
--------
1. 数值自检  —— 打包朝向对不对 (错了输出会明显偏离 fp16 参考)
2. eager 扫描 —— M = 1..512, 对比 F.linear
3. thread_k/thread_n 变体 —— (256,64) / (128,128) / (64,256)
4. **CUDA Graph 捕获** —— vLLM 必用; 这个 kernel 有全局 barrier + workspace,
   是选项②最大的未知风险。**可能死锁**, 所以放在最后且可用 --no-graph 跳过。

用法 (必须在 A100 节点上跑):
    srun -p a100 -N1 -n1 --mem=32G --gres=gpu:1 -t 00:30:00 --pty bash
    /home/jianglei/miniconda3/envs/milo/bin/python bench/micro_brmoe_cuda.py
    ... --no-graph         # 跳过有死锁风险的 CUDA Graph 测试
"""
import argparse
import os
import sys

# 默认指向 BR-MoE 原有的 py3.10/sm_80 产物 (milo 环境用)。
# 指向新编的 py3.12/sm_120 版本: BRMOE_SO_DIR=/path/to/build/lib python ...
BR_SO = os.environ.get(
    "BRMOE_SO_DIR",
    "/home/jianglei/ada/BR-MoE/BR-MoE/kernels/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, BR_SO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    import brmoe
except Exception as e:  # pragma: no cover
    print(f"!! import brmoe 失败: {e}")
    print(f"   期望 .so 在 {BR_SO}")
    print("   注意: 这个 .so 是 cpython-310, 必须用 milo 环境的 python")
    sys.exit(1)


def bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(rep):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / rep * 1e3      # µs


def build_layer(K, N, gs=64, device="cuda"):
    """按 BRMoE_Asymmetric_Linear 的朝向构造 layer, 并做数值自检。

    pack(linear, scales, zeros) 的朝向 (从 kernels/brmoe/__init__.py 反推):
        linear.weight  : (N, K)        即标准 nn.Linear(K, N)
        scales / zeros : (N, K//gs)    <- 注意是 (out, groups), 进 pack 后转置
    """
    layer = brmoe.Layer3bitWithZeros(K, N, gs).to(device)
    lin = torch.nn.Linear(K, N, bias=False)

    # 造一份"量化友好"的权重: 均匀分布 + 对称的 scale/zero, 保证 q 落在 [0,7]
    W = (torch.rand(K, N, device=device) * 2 - 1) * 0.1        # (K, N)
    lin.weight.data = W.t().half().contiguous()                # (N, K)

    g = K // gs
    s_val, z_val = 0.2 / 7, -0.1
    scales = torch.full((N, g), s_val, dtype=torch.float16, device=device)
    zeros = torch.full((N, g), z_val, dtype=torch.float16, device=device)

    layer.pack(lin, scales, zeros)
    return layer, W.half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--rep", type=int, default=50)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! 没有 GPU")
        sys.exit(1)

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  ({p.total_memory/2**30:.1f} GiB)   cc={p.major}.{p.minor}")
    print(f"kernel: BR-MoE 原生 CUDA (Marlin 系 int3),  K={args.K} N={args.N} gs={args.gs}")
    print("对照: 我们的 Triton int3 在 5090 上 M=1 是 35.4 µs (A100 上会不同, 只看相对值)")
    print()

    layer, W16 = build_layer(args.K, args.N, args.gs)

    # ---------- 1. 数值自检 ----------
    print("=" * 92)
    print("1. 数值自检 (打包朝向)")
    print("=" * 92)
    x0 = (torch.rand(16, args.K, device="cuda") * 2 - 1).half()
    try:
        y = layer(x0)
        yref = F.linear(x0, W16)
        rel = ((y.float() - yref.float()).norm() / yref.float().norm()).item()
        print(f"   输出 shape={tuple(y.shape)}  dtype={y.dtype}")
        print(f"   相对误差 {rel:.4f}  (量化误差应约 5e-3 ~ 5e-2; 明显更大 = 朝向错了)")
        print(f"   -> {'OK' if rel < 0.2 else '!! 朝向可能不对, 后面只看时间不看数值'}")
    except Exception as e:
        print(f"   !! 前向失败: {type(e).__name__}: {e}")
        return

    # ---------- 2. eager 扫描 ----------
    print()
    print("=" * 92)
    print("2. eager 扫描: BR-MoE CUDA kernel vs cuBLAS fp16")
    print("=" * 92)
    ms = [int(v) for v in args.ms.split(",")]
    bytes_w16 = args.K * args.N * 2
    # B1 + B2 + s + z
    bytes_int3 = (args.K // 16) * (args.N * 2) * 4 \
        + (args.K // 16) * (args.N // 2) * 4 \
        + 2 * (args.K // args.gs) * args.N * 2

    print(f"   权重字节: fp16 {bytes_w16/2**20:.2f} MiB | int3 {bytes_int3/2**20:.2f} MiB"
          f" | 压缩比 {bytes_w16/bytes_int3:.2f}x")
    hdr = (f"   {'M':>6}{'fp16 µs':>10}{'brmoe µs':>11}{'比值':>8}{'Δµs':>9}"
           f"{'fp16 GB/s':>11}{'brmoe GB/s':>12}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for M in ms:
        x = (torch.rand(M, args.K, device="cuda") * 2 - 1).half()
        t16 = bench(lambda: F.linear(x, W16), 10, args.rep)
        tb = bench(lambda: layer(x), 10, args.rep)
        bw16 = (M * args.K * 2 + bytes_w16) / (t16 * 1e-6) / 1e9
        bwb = (M * args.K * 2 + bytes_int3) / (tb * 1e-6) / 1e9
        print(f"   {M:>6}{t16:>10.2f}{tb:>11.2f}{tb/t16:>7.2f}x{tb-t16:>9.2f}"
              f"{bw16:>11.0f}{bwb:>12.0f}")

    # ---------- 3. thread tile 变体 ----------
    print()
    print("=" * 92)
    print("3. thread_k/thread_n 变体 (backends/brmoe.py 的优先序)")
    print("=" * 92)
    print("   gs=64 时优先 (256,64), 其次 (128,128), 再次 (64,256)")
    for M in (1, 16, 64):
        x = (torch.rand(M, args.K, device="cuda") * 2 - 1).half()
        out = torch.empty(M, args.N, dtype=torch.float16, device="cuda")
        line = f"   M={M:>3}  "
        for tk, tn in ((256, 64), (128, 128), (64, 256)):
            if args.K % tk or args.N % tn:
                line += f"({tk},{tn})=N/A  "
                continue
            try:
                f = lambda: brmoe.mul_3bit_with_zeros(              # noqa: E731
                    x, layer.B1, layer.B2, out, layer.s, layer.z,
                    layer.workspace, thread_k=tk, thread_n=tn)
                f()
                torch.cuda.synchronize()
                line += f"({tk},{tn})={bench(f, 5, 30):7.2f}  "
            except Exception as e:
                line += f"({tk},{tn})={type(e).__name__}  "
        print(line)

    # ---------- 4. CUDA Graph ----------
    if args.no_graph:
        print("\n(--no-graph: 跳过 CUDA Graph 测试)")
        return
    print()
    print("=" * 92)
    print("4. CUDA Graph 捕获 (vLLM 必用; 这是选项②最大的未知风险)")
    print("=" * 92)
    print("   !! 这个 kernel 用 workspace + 全局 barrier 做 split-K。")
    print("      若捕获失败或 replay 挂住, 就是选项②的否决项。")
    print("      (挂住的话 Ctrl-C 退出, 用 --no-graph 重跑)")
    sys.stdout.flush()
    for M in (1, 16, 64):
        x = (torch.rand(M, args.K, device="cuda") * 2 - 1).half()
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    layer(x)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    y_g = layer(x)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            t = bench(lambda: g.replay(), 3, 20) / 10
            # replay 结果 vs eager 结果
            y_e = layer(x)
            torch.cuda.synchronize()
            same = torch.equal(y_g, y_e)
            print(f"   M={M:>3}  捕获成功  replay={t:7.2f} µs/次  "
                  f"结果与 eager {'一致' if same else '不一致!!'}")
        except Exception as e:
            print(f"   M={M:>3}  !! 捕获失败: {type(e).__name__}: {str(e)[:90]}")

    print()
    print("=" * 92)
    print("怎么判")
    print("=" * 92)
    print("  · 表格里 brmoe/fp16 < 1        -> BR-MoE 的 CUDA kernel 打赢 cuBLAS")
    print("    我们 Triton 版是 2.6x 慢; 只要它 < 1, 选项②就有明确价值")
    print("  · brmoe 的 GB/s 接近 fp16 的   -> 说明它真的用上了带宽 (我们的只有 1/12)")
    print("  · CUDA Graph 捕获成功且 replay 不死不慢 -> 可以在 vLLM 里用")
    print("  · 若 CUDA Graph 挂住/失败      -> 选项② 直接否决 (vLLM 离不开 graph)")


if __name__ == "__main__":
    main()
