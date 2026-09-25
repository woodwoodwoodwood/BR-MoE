"""Compare the direct INT3 route GEMV with grouped GEMM on one GPU.

Usage: python bench/bench_gemv_compare.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BR-MoE", "kernels", "triton_int3"))
from int3_moe.ops import fused_moe_int3, pack_moe_weights  # noqa: E402


def graph_us(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", default="1,8,32", help="comma-separated token counts")
    args = parser.parse_args()
    e, k, i, top_k, gs = 64, 2048, 1408, 6, 64
    torch.manual_seed(11)
    device = "cuda"
    w13 = torch.randn(e, 2 * i, k, device=device, dtype=torch.float16) * .02
    w2 = torch.randn(e, k, i, device=device, dtype=torch.float16) * .02
    packed = pack_moe_weights(w13, w2, gs)
    packed["z13"] = torch.full_like(packed["s13"], 4)
    packed["z2"] = torch.full_like(packed["s2"], 4)
    fp16 = pack_moe_weights(w13, w2, gs, layout="fp16")
    print(torch.cuda.get_device_name(), flush=True)
    print("M   legacy64  tuned16  gemv  auto  fp16  max_abs_diff (us)", flush=True)
    for m in (int(v) for v in args.ms.split(",")):
        x = torch.randn(m, k, device=device, dtype=torch.float16) * .1
        ids = torch.argsort(torch.rand(m, e, device=device), dim=1)[:, :top_k].contiguous()
        weights = torch.full((m, top_k), 1 / top_k, device=device, dtype=torch.float16)

        def run(p, **options):
            return fused_moe_int3(x, weights, ids, p, **options)

        legacy = lambda: run(packed, gemv=False, slot=64, block_m=64)
        tuned = lambda: run(packed, gemv=False, slot=16, block_m=16,
                            num_stages=3 if m <= 16 else 1)
        direct = lambda: run(packed, gemv=True)
        auto = lambda: run(packed)
        reference = legacy().clone()
        diff = (reference - direct()).abs().max().item()
        times = [graph_us(fn) for fn in
                 (legacy, tuned, direct, auto, lambda: run(fp16, gemv=False))]
        print(f"{m:<3} " + " ".join(f"{t:8.1f}" for t in times) + f"  {diff:.4g}",
              flush=True)


if __name__ == "__main__":
    main()
