"""隔离微基准: brmoe_int3_linear (Triton int3) vs fp16 F.linear。

目的 —— 回答"int3 的 attention 投影为什么比 fp16 慢"：
  int3 该读的权重只有 fp16 的 3/8, 理论上该快 2.7x。若实测更慢,
  原因只可能是两类, 本脚本把它们分开:

    (A) kernel 配置问题 —— 占用率不足。
        `int3_moe_gemm` 的默认 tile (block_m=64, block_n=64, num_warps=2)
        是 BR-MoE 在 **T4** 上调的 (见 kernel.py 注释)。attention 的
        q_proj 是 M=1, N=2048, K=2048 -> 只起 32 个 block, 而 5090 有 170 个 SM,
        带宽利用率可能只有个位数百分比。graph 救不了这一类。
        本脚本扫 block_n / num_warps 找占用率拐点。

    (B) Python 级开销 —— 每 token 196 次调用, 每次 3 个分配 + 视图 + pick_tiles。
        eager 下是纯开销, CUDA Graph 能吃掉大部分。

用法 (必须带 CUDA 环境, 见 bench/env_5090.sh):
    python bench/bench_int3_linear.py --model <int3dense 目录>
"""
import argparse
import importlib.util
import os
import sys
import time

import torch


def load_mod(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BRMOE = os.environ.get("BRMOE_PKG",
                       "/mnt/709/data3/home/jianglei/ada/BR-MoE/BR-MoE")
_k = load_mod("be_kernel", os.path.join(
    BRMOE, "kernels", "triton_int3", "int3_moe", "kernel.py"))
int3_moe_gemm = _k.int3_moe_gemm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_method as lm  # noqa: E402


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3      # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="1",
                    help="用哪一层的 self_attn.q_proj 做测试")
    ap.add_argument("--ms", default="1,8,32,128,512")
    ap.add_argument("--sweep", type=int, default=1,
                    help="1 = 扫 block_n / num_warps (找占用率拐点)")
    args = ap.parse_args()

    from safetensors import safe_open
    import json

    idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    def get(name):
        with safe_open(os.path.join(args.model, idx[name]), framework="pt") as f:
            return f.get_tensor(name)

    p = f"model.layers.{args.layer}.self_attn.q_proj"
    qw = get(f"{p}.qweight").cuda()      # [Kpack, N]
    sc = get(f"{p}.scales").cuda()       # [K//gs, N]
    zz = get(f"{p}.zeros").cuda()
    Kpack, N = qw.shape
    K = Kpack // 3 * 32
    gs = K // sc.shape[0]
    print(f"层 {p}:  qweight{qw.shape}  K={K} N={N} gs={gs}")

    # 参考 fp16 权重 (反量化出来, 只用于对比 fp16 路径的速度)
    q = None
    W = None                                 # fp16 权重延迟构造

    dev = "cuda"
    rows = []
    print()
    print(f"{'M':>6} | {'fp16 ms':>9} {'int3 ms':>9} {'倍数':>6} | "
          f"{'fp16 GB/s':>10} {'int3 GB/s':>10}")
    print("-" * 66)
    for M in [int(x) for x in args.ms.split(",")]:
        x = torch.randn(M, K, dtype=torch.float16, device=dev)

        # fp16 基线: 读 fp16 权重 (K*N*2 字节)
        if W is None:
            q = lm._load_mod("be_pack", os.path.join(
                BRMOE, "kernels", "triton_int3", "int3_moe", "packing.py"))
            # 直接用随机 fp16 权重即可, 只为测速度
            W = torch.randn(N, K, dtype=torch.float16, device=dev)
        t_fp16 = timeit(lambda: torch.nn.functional.linear(x, W))

        # int3: 复用插件里的实现
        t_int3 = timeit(lambda: lm.brmoe_int3_linear(x, qw, sc, zz, gs))

        bytes_fp16 = K * N * 2
        bytes_int3 = Kpack * N * 4
        print(f"{M:>6} | {t_fp16:>9.3f} {t_int3:>9.3f} {t_int3/t_fp16:>5.2f}x | "
              f"{bytes_fp16/(t_fp16*1e-3)/1e9:>10.1f} "
              f"{bytes_int3/(t_int3*1e-3)/1e9:>10.1f}")
        rows.append((M, t_fp16, t_int3))

    if args.sweep:
        print()
        print("=== 扫 tile 配置 (M=1, 看占用率能否救回来) ===")
        M = rows[0][0]
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        x2 = x.reshape(-1, K).contiguous()
        Wp = qw.unsqueeze(0).contiguous()
        S = sc.view(1, -1, N).contiguous()
        Z = zz.view(1, -1, N).contiguous()
        print(f"{'block_n':>8}{'warps':>7}{'stages':>8}{'ms':>10}{'GB/s':>10}")
        for bn in (16, 32, 64, 128):
            if N % bn:
                continue
            for nw in (2, 4, 8):
                for ns in (1, 2):
                    block_m, slot = 64, 16
                    num_post = (M + slot - 1) // slot * slot
                    grid_m = (num_post + block_m - 1) // block_m
                    n_rows = grid_m * block_m
                    xp = torch.zeros(n_rows, K, dtype=x.dtype, device=dev)
                    xp[:M] = x2
                    eid = torch.zeros(grid_m * (block_m // slot),
                                      dtype=torch.int32, device=dev)
                    sti = torch.arange(n_rows, dtype=torch.int32, device=dev)
                    out = torch.empty(n_rows, N, dtype=x.dtype, device=dev)

                    def run(bn=bn, nw=nw, ns=ns):
                        int3_moe_gemm(
                            xp, Wp, S, sti, eid, num_post, M,
                            group_size=gs, a_gather=False, add=False, out=out,
                            meta=None, grid_m=grid_m,
                            block_m=block_m, block_n=bn, block_k=32, slot=slot,
                            layout4=False, layout16=False, zeros=Z,
                            w_transposed=True, direct4=False,
                            num_warps=nw, num_stages=ns)
                    try:
                        t = timeit(run, warmup=5, iters=30)
                    except Exception as e:
                        print(f"{bn:>8}{nw:>7}{ns:>8}   {type(e).__name__}: {str(e)[:40]}")
                        continue
                    nblk = grid_m * (N // bn)
                    gbs = (Kpack * N * 4) / (t * 1e-3) / 1e9
                    print(f"{bn:>8}{nw:>7}{ns:>8}{t:>10.3f}{gbs:>10.1f}   blocks={nblk}")


if __name__ == "__main__":
    main()
