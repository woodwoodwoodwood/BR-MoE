"""POC: 把 MoE 的 per-expert 循环换成 grouped GEMM (一个 launch 算完所有专家),
验证 batch=1 时 int3 能否快过 fp16。

用真实 checkpoint 第 L 层的权重做**单层**对比:
  (a) fused_int3_grouped    : triton grouped int3 —— 2 次 GEMM + 融合激活, 一个 launch 覆盖全部专家
  (b) per_expert_int3_fp16  : 逐专家循环 (int3 反量化成 fp16 后), 隔离"解包开销"来量 grouped 本身的收益
  (c) per_expert_fp16       : 逐专家循环 + 原生 fp16 权重 (近似 HuggingFace 的实现)
  (d) per_expert_brmoe      : 当前主链路的真实实现 (BRMoELinear 逐专家, 含解包)

计时按 M 扫描, 结果可 ×27 层外推整模型 TPOT。

用法:
    python poc_grouped_moe.py                # 默认 L=1, M=1,8,64,512
    python poc_grouped_moe.py --layer 5 --ms 1,16
"""

import argparse
import os
import sys
import time

import torch

CKPT_DEFAULT = "/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0/qmodel.pt"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "BR-MoE", "kernels", "triton_int3"))

from int3_moe.ops import ref_fused_moe, fused_moe_int3, pack_moe_weights  # noqa: E402


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 从 checkpoint 还原某一层的全部 expert 权重 (fp16)
# ---------------------------------------------------------------------------

def load_layer_experts(sd, layer, E, I, K, device):
    """把某一层所有专家的 int3 权重反量化成 fp16, 拼成 w13 / w2。"""
    from BR_MoE.core.quantize import BRMoELinear

    w13 = torch.empty(E, 2 * I, K, dtype=torch.float16, device=device)
    w2 = torch.empty(E, K, I, dtype=torch.float16, device=device)

    for e in range(E):
        parts = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            if name not in sd:
                raise KeyError(f"checkpoint 缺少 {name}")
            lay = BRMoELinear(
                linear_layer=None, compress_config=None,
                compute_dtype=torch.float16, device=device,
            )
            # load_state_dict 会 pop 掉 dict 里的 key, 必须传副本 (本脚本两种用途共用一份 sd)
            lay.load_state_dict(dict(sd[name]))
            W = lay.dequantize()          # (out, in); 注意 dequantize 会改 meta, 只能调用一次
            if proj == "down_proj":
                w2[e] = W
            else:
                parts.append(W)
            del lay
        w13[e] = torch.cat(parts, dim=0)  # [2I, K]
        if (e + 1) % 16 == 0:
            log(f"   expert {e + 1}/{E} done")

    return w13, w2


# ---------------------------------------------------------------------------
# 2. 逐专家循环的对照实现
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_expert_loop(x, tw, ti, W13, W2, I, E):
    """逐专家 GEMM 循环 (HF MoE 风格): 每个激活的专家单独算, 输出按路由权重累加。"""
    M, K = x.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    for e in range(E):
        hit = (ti == e)                       # [M, topk]
        rows = hit.any(dim=1)
        if not bool(rows.any()):
            continue
        h = x[rows] @ W13[e].t()              # [n, 2I]
        g, u = h[:, :I], h[:, I:]
        act = (g * torch.sigmoid(g)) * u
        y = act @ W2[e].t()                   # [n, K]
        for t in range(ti.shape[1]):
            m = hit[rows, t]
            out[rows] += torch.where(m[:, None], y * tw[rows, t:t + 1],
                                     torch.zeros_like(y))
    return out


@torch.no_grad()
def per_expert_brmoe(x, tw, ti, layers, E):
    """当前主链路的真实做法: 每个 expert 是一个 BRMoELinear (PyTorch 反量化后端)。"""
    M, K = x.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    for e in range(E):
        hit = (ti == e)
        rows = hit.any(dim=1)
        if not bool(rows.any()):
            continue
        xr = x[rows]
        g = layers[e][0](xr)
        u = layers[e][1](xr)
        act = (g * torch.sigmoid(g)) * u
        y = layers[e][2](act)
        for t in range(ti.shape[1]):
            m = hit[rows, t]
            out[rows] += torch.where(m[:, None], y.float() * tw[rows, t:t + 1],
                                     torch.zeros_like(out[rows]))
    return out


def timeit(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--expert-count", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--I", type=int, default=1408)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--ms", default="1,8,64,512")
    ap.add_argument("--n-layers", type=int, default=27, help="外推整模型时乘的 MoE 层数")
    ap.add_argument("--skip-brmoe", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    E, I, K, topk, gs = args.expert_count, args.I, args.K, args.topk, args.group_size
    ms = [int(x) for x in args.ms.split(",")]

    log("=" * 78)
    log(f"POC grouped MoE | layer={args.layer} E={E} topk={topk} K={K} I={I} gs={gs}")
    log(f"GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__}")
    log("=" * 78)

    # ---- 权重 (checkpoint 只加载一次, 两种用法共用, 省掉一次 8GB 读盘) ----
    log(f"[ckpt] loading {args.ckpt} (8GB, 需要一会儿) ...")
    _t0 = time.time()
    sd = torch.load(args.ckpt, map_location="cpu")
    log(f"[ckpt] loaded in {time.time() - _t0:.1f}s, {len(sd)} modules")
    w13, w2 = load_layer_experts(sd, args.layer, E, I, K, dev)
    log(f"[w] w13={tuple(w13.shape)} w2={tuple(w2.shape)}  "
        f"fp16 bytes={w13.numel() * 2 / 2**20:.1f}+{w2.numel() * 2 / 2**20:.1f} MiB")

    if not args.skip_brmoe:
        # 保留一份 BRMoELinear 版本用于 (d)
        from BR_MoE.core.quantize import BRMoELinear
        brmoe_layers = []
        for e in range(E):
            trio = []
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lay = BRMoELinear(linear_layer=None, compress_config=None,
                                  compute_dtype=torch.float16, device=dev)
                lay.load_state_dict(
                    dict(sd[f"model.layers.{args.layer}.mlp.experts.{e}.{proj}"])
                )
                trio.append(lay)
            brmoe_layers.append(trio)
        log(f"[w] BRMoELinear 层已就绪")

    del sd
    torch.cuda.empty_cache()

    # ---- triton 打包 ----
    t0 = time.time()
    packed = pack_moe_weights(w13, w2, gs)
    log(f"[pack] triton int3 打包完成 ({time.time() - t0:.1f}s)  "
        f"w13_q={tuple(packed['w13_q'].shape)} w2_q={tuple(packed['w2_q'].shape)}")

    rows = []
    for M in ms:
        log(f"\n-------------------- M={M} (topk={topk}) --------------------")
        g = torch.Generator(device="cpu").manual_seed(0)
        x = (torch.randn(M, K, generator=g) * 0.1).half().to(dev)
        ti = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(M)]).to(dev)
        tw = torch.full((M, topk), 1.0 / topk, dtype=torch.float16, device=dev)

        # 数值校验
        y_ref = ref_fused_moe(x, tw, ti, packed)
        y_grp = fused_moe_int3(x, tw, ti, packed, fast=True)
        err = (y_ref.float() - y_grp.float()).abs().max().item()
        log(f"[check] grouped vs ref: max|err|={err:.4f}")

        t_grp = timeit(lambda: fused_moe_int3(x, tw, ti, packed, fast=True))
        t_pe_i = timeit(lambda: per_expert_loop(x, tw, ti, w13, w2, I, E))
        t_pe_f = t_pe_i  # 同一个循环, 权重都是 fp16 —— 见下方说明
        t_brmoe = (timeit(lambda: per_expert_brmoe(x, tw, ti, brmoe_layers, E))
                   if not args.skip_brmoe else float("nan"))

        log(f"[time] grouped int3      : {t_grp:8.3f} ms/层  -> x{args.n_layers}层 ≈ {t_grp * args.n_layers:7.2f} ms")
        log(f"[time] per-expert (fp16) : {t_pe_f:8.3f} ms/层  -> x{args.n_layers}层 ≈ {t_pe_f * args.n_layers:7.2f} ms")
        if not args.skip_brmoe:
            log(f"[time] per-expert BRMoE : {t_brmoe:8.3f} ms/层  -> x{args.n_layers}层 ≈ {t_brmoe * args.n_layers:7.2f} ms")
        log(f"[speedup] grouped vs per-expert(fp16) = x{t_pe_f / t_grp:.2f}")

        rows.append((M, err, t_grp, t_pe_f, t_brmoe))

    log("\n" + "=" * 78)
    log(f"{'M':>6}{'max|err|':>12}{'grouped(ms)':>14}{'per-exp fp16(ms)':>18}{'BRMoE(ms)':>12}{'speedup':>10}")
    for M, err, tg, tp, tb in rows:
        log(f"{M:>6}{err:>12.4f}{tg:>14.3f}{tp:>18.3f}{tb:>12.3f}{tp / tg:>10.2f}x")
    log("=" * 78)
    log("[note] per-expert 用的是 int3 反量化后的 fp16 权重, 因此这一列只反映")
    log("       'grouped vs 逐专家循环' 的差异; BRMoE 那列才含真实的解包开销。")


if __name__ == "__main__":
    main()
