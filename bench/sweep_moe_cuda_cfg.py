"""MOE CUDA kernel 的 tile/stages 配置扫描: 数值冒烟 + zipf 路由性能。

每个配置应在独立进程里跑 (sm_120 上未验证配置可能 illegal instruction,
进程隔离防止连坐) —— slurm 里按配置循环调用本脚本。

用法:
    python bench/sweep_moe_cuda_cfg.py --cfg 128,128,4 [--ms 4,8,16,32]
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "triton_int3"))

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
from marlin_int3_moe.repack import repack_moe  # noqa: E402
from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda  # noqa: E402
from micro_moe import bench_graph, make_weight_pair  # noqa: E402
from verify_moe_cuda import make_routing_zipf, quantize_asym, pack_triton  # noqa: E402
from brmoe_int3_vllm.kernel import get_fused_moe_int3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="thread_n,thread_k,stages")
    ap.add_argument("--ms", default="4,8,16,32")
    args = ap.parse_args()
    cfg = tuple(int(v) for v in args.cfg.split(","))
    assert len(cfg) == 3

    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    E, K, I, gs, topk = 64, 2048, 1408, 64, 6

    W13, W2 = make_weight_pair(E, K, I, dev)
    q13, s13, z13, _ = quantize_asym(W13, gs, dev, seed=0)
    q2, s2, z2, _ = quantize_asym(W2, gs, dev, seed=1)
    packed_t = {
        "w13_q": pack_triton(q13, K), "s13": s13, "z13": z13,
        "w2_q": pack_triton(q2, I), "s2": s2, "z2": z2,
        "group_size": gs,
    }
    pk = repack_moe(packed_t)
    fused_t = get_fused_moe_int3()

    # ---- 数值冒烟 (Triton 是已验证参照) ----
    x = (torch.randn(8, K, device=dev) * 0.1).half()
    tw, tid = make_routing_zipf(8, E, topk, dev)
    y_t = fused_t(x, tw, tid, packed_t, fast=True).float()
    y_c = fused_moe_int3_cuda(x, tw, tid, pk, ext, cfg=cfg).float()
    rel = (y_c - y_t).abs().max().item() / max(y_t.abs().max().item(), 1e-9)
    if rel > 5e-3:
        print(f"cfg={cfg}  数值 FAIL rel={rel:.2e} (sm_{prop.major}{prop.minor})")
        return 1
    line = f"cfg={cfg}  rel={rel:.1e}  "
    # ---- 性能 (zipf, graph) ----
    for M in [int(v) for v in args.ms.split(",")]:
        x = (torch.randn(M, K, device=dev) * 0.1).half()
        tw, tid = make_routing_zipf(M, E, topk, dev)
        t = bench_graph(lambda: fused_moe_int3_cuda(x, tw, tid, pk, ext,
                                                    cfg=cfg))
        t_t = bench_graph(lambda: fused_t(x, tw, tid, packed_t, fast=True))
        line += f"M{M}={t:7.2f}us(TC {t_t:7.2f})  "
    print(line + f"(sm_{prop.major}{prop.minor})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
