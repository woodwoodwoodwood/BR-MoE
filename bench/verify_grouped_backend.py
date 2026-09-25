"""验证 brmoe_grouped 后端: 端到端数值一致性 + 时延对比 + 权重级诊断。

流程:
  1. from_compressed 加载 3bit 模型 (PyTorch 后端)
  2. warmup 后多次前向取中位数 -> baseline
  3. patch_moe_to_grouped (可限制只接管前 N 层)
  4. 再测一次, 对比 logits 误差与耗时
  5. 诊断: 对比 BRMoE 原始 fp16 权重 与 triton 打包后解包出来的权重, 定位转换损耗

用法:
  python verify_grouped_backend.py --limit 2
  python verify_grouped_backend.py --limit 0     # 全部 27 层
"""

import argparse
import os
import sys
import time

import torch

MODEL_DEFAULT = "/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0"
CACHE_DEFAULT = "/mnt/709/data3/home/jianglei/ada/BR-MoE/bench_results/grouped_cache"


def log(*a):
    print(*a, flush=True)


@torch.inference_mode()
def _fwd(model, in_len, device, seed=0):
    vocab = getattr(model.config, "vocab_size", 102400)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1, vocab, (1, in_len), generator=g).to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    return out.logits.float(), time.perf_counter() - t0


def bench_fwd(model, in_len, device, warmup=2, repeat=3, seed=0):
    logits = None
    for _ in range(warmup):
        logits, _ = _fwd(model, in_len, device, seed)
    ts = []
    for r in range(repeat):
        logits, dt = _fwd(model, in_len, device, seed)
        ts.append(dt * 1000)
    ts.sort()
    return logits, ts[len(ts) // 2], ts


def diag_weights(experts, packed, info, device):
    """对比 BRMoE 的 fp16 权重 vs triton 打包->解包出来的权重。"""
    from int3_moe.ops import dequant_int3

    I, K, gs = info["I"], info["K"], info["group_size"]
    W_gate = experts[0].gate_proj.dequantize()          # (I, K) fp16
    W_up = experts[0].up_proj.dequantize()              # (I, K)
    W13_ref = torch.cat([W_gate, W_up], dim=0)          # (2I, K)

    W13_tri = dequant_int3(packed["w13_q"], packed["s13"], K, gs, "int3",
                           packed.get("w_transposed", False),
                           zeros=packed.get("z13"))[0]  # (2I, K)

    d = (W13_ref.float() - W13_tri.float()).abs()
    ref_max = W13_ref.abs().max().item()
    log(f"[diag] w13 (expert0): max|diff|={d.max().item():.6f}  "
        f"mean|diff|={d.mean().item():.6f}  ref_max={ref_max:.6f}  "
        f"rel={d.max().item() / (ref_max + 1e-8):.3e}")

    # 量化网格是否对齐: 看 (W/scale) 与 3bit 网格的距离
    s = experts[0].gate_proj.meta.get("scale", None)
    z = experts[0].gate_proj.meta.get("zero", None)
    if s is not None:
        log(f"[diag] brmoe scale shape={tuple(s.shape)}  zero="
            f"{None if z is None else float(z.flatten()[0])}")
    log(f"[diag] brmoe meta: nbits={experts[0].gate_proj.meta.get('nbits')} "
        f"gs={gs} packing={experts[0].gate_proj.meta.get('packing')} "
        f"quant_zero={experts[0].gate_proj.meta.get('quant_zero')} "
        f"quant_scale={experts[0].gate_proj.meta.get('quant_scale')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_DEFAULT)
    ap.add_argument("--cache-dir", default=CACHE_DEFAULT)
    ap.add_argument("--limit", type=int, default=2, help="只接管前 N 个 MoE 层; 0=全部")
    ap.add_argument("--in-len", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-diag", action="store_true")
    ap.add_argument("--free-original", action="store_true",
                    help="释放原 int3 权重(省显存); 默认保留以便做权重级诊断")
    args = ap.parse_args()

    from BR_MoE.models.hf.deepseek import DeepSeekMoEBRMoE
    import BR_MoE.backends.brmoe_grouped as bg

    log("=" * 78)
    log(f"verify brmoe_grouped | limit={args.limit} in_len={args.in_len} device={args.device}")
    log(f"GPU={torch.cuda.get_device_name(0)}")
    log("=" * 78)

    t0 = time.time()
    model = DeepSeekMoEBRMoE.from_compressed(args.model_path, device=args.device)
    model.eval()
    log(f"[load] {time.time() - t0:.1f}s")

    log("\n--- baseline: PyTorch 后端 (warmup 后取中位数) ---")
    base_logits, base_t, base_all = bench_fwd(model, args.in_len, args.device)
    log(f"  forward = {base_t:.1f} ms   (all: {[round(x, 1) for x in base_all]})")

    log("\n--- patch: brmoe_grouped ---")
    os.makedirs(args.cache_dir, exist_ok=True)
    t0 = time.time()
    patched = bg.patch_moe_to_grouped(
        model, save_dir=args.cache_dir, limit=(args.limit or None), verbose=True,
        free_original=args.free_original,
    )
    log(f"[patch] 接管 {patched} 层, 用时 {time.time() - t0:.1f}s")

    if not args.skip_diag and not args.free_original and patched > 0:
        log("\n--- 权重级诊断 (转换损耗) ---")
        moe = [m for n, m in model.named_modules() if type(m).__name__ == "DeepseekMoE"][0]
        diag_weights(moe.experts, moe._brmoe_grouped_packed, moe._brmoe_grouped_info,
                     args.device)

    log("\n--- after: grouped 后端 ---")
    new_logits, new_t, new_all = bench_fwd(model, args.in_len, args.device)
    log(f"  forward = {new_t:.1f} ms   (all: {[round(x, 1) for x in new_all]})")
    log(f"  speedup = x{base_t / new_t:.2f}")

    err = (base_logits - new_logits).abs().max().item()
    ref = base_logits.abs().max().item()
    log(f"\n[check] max|dlogits| = {err:.4f}   ref_max = {ref:.2f}   "
        f"relative = {err / (ref + 1e-6):.3e}")
    log(f"[check] argmax 前5列 base={base_logits[0, :5].argmax(-1).tolist()} "
        f"new={new_logits[0, :5].argmax(-1).tolist()}")
    log("=" * 78)


if __name__ == "__main__":
    main()
