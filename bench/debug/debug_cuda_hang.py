"""定位 brmoe int3 CUDA backend 挂死 (走真实推理路径 backends.brmoe.BRMoE_Symmetric_Layer)。

背景:
  prepare_for_inference(backend=brmoe_auto) 的逐层 probe 用 M=1, 5463 层全部通过;
  但 prefill (M=128) 会挂死, 3 分钟无输出。

假设:
  workspace 按 n // 128 * 16 分配, 当 n 不是 128 的整数倍时不足:
      n=10944 -> n//128*16 = 85*16 = 1360
      而 kernel 需要 n_tiles(10944/64=171) * max_par(8) = 1368  -> 少 8
  越界写破坏 kernel 内部的全局 barrier 计数 -> 死锁。
  已验证: n=10944 + thread_k=256,thread_n=64 + M=128 必挂; n=2048/n=512 正常。

外层务必加 timeout:  timeout 300 python debug_cuda_hang.py
"""

import os
import sys
import time

import torch

import BR_MoE.kernels.brmoe as brmoe_mod
from BR_MoE.backends.brmoe import BRMoE_Symmetric_Layer

SHAPES = [
    (2048, 2048, 128, "attn q_proj/o_proj"),
    (2048, 512, 128, "attn k_proj/v_proj"),
    (2048, 10944, 128, "layer0 dense gate/up  <-- 原死锁形状"),
    (10944, 2048, 128, "layer0 dense down"),
    (2048, 1408, 128, "expert gate/up"),
    (1408, 2048, 128, "expert down"),
]

TILES = [(128, 128), (256, 64), (64, 256)]


def build_layer(k, n, gs, device="cuda"):
    W = torch.randn(k, n, dtype=torch.half, device=device) * 0.02
    scales = torch.full((n, k // gs), 0.01, dtype=torch.half, device=device)
    lay = BRMoE_Symmetric_Layer(W, scales, groupsize=gs)
    return lay


def run(lay, M, tk, tn):
    x = torch.randn(M, lay.in_features, dtype=torch.half, device="cuda")
    C = torch.empty(M, lay.out_features, dtype=torch.half, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    brmoe_mod.mul_3bit(
        x, lay.Wq_packed1, lay.Wq_packed2, C, lay.scales, lay.workspace_fp,
        thread_k=tk, thread_n=tn,
    )
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    print(f"[env] CUDA_LAUNCH_BLOCKING={os.getenv('CUDA_LAUNCH_BLOCKING')}", flush=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    for k, n, gs, tag in SHAPES:
        print(f"\n##### k={k} n={n} gs={gs}  ({tag}) #####", flush=True)
        try:
            lay = build_layer(k, n, gs)
        except Exception as e:
            print(f"   [build] {type(e).__name__}: {str(e)[:140]}"
                  f"  -> 形状不支持, patch 时会回退 PyTorch 后端", flush=True)
            continue
        print(f"[ws] numel={lay.workspace_fp.numel()}  "
              f"(旧公式 n//128*16={n // 128 * 16}, 新公式 (n//64+1)*16={(n // 64 + 1) * 16}, "
              f"kernel 上界 n//64*8={n // 64 * 8})", flush=True)
        for M in [1, 2, 8, 64, 128, 256]:
            for tk, tn in TILES:
                if (k % tk != 0) or (n % tn != 0):
                    continue
                print(f"   M={M:>4} tk={tk:>3} tn={tn:>3} ...", end="", flush=True)
                try:
                    dt = run(lay, M, tk, tn)
                    print(f" OK {dt * 1000:.2f} ms", flush=True)
                except Exception as e:
                    print(f" EXC {type(e).__name__}: {str(e)[:110]}", flush=True)
    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
