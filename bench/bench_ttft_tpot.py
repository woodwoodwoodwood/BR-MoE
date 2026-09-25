"""BR-MoE(3bit) vs fp16 的推理性能基准 —— TTFT / TPOT。

两种实现共用同一套计时逻辑, 保证可比性:
    --impl fp16   : transformers 原生加载 fp16 权重
    --impl brmoe  : BR_MoE 的 3-bit 量化权重 (qmodel.pt + compensators.pt)

指标定义 (与 vLLM 一致):
    TTFT (Time To First Token) = prefill 一次前向的耗时 (含首个 token 的 argmax)
    TPOT (Time Per Output Token) = decode 阶段平均每 token 耗时
                                 = decode 总时间 / (output_len - 1)
    另记录逐 token 延迟的 p50 / p90, 以及 decode 吞吐 (tok/s)。

用法:
    python bench_ttft_tpot.py --impl fp16  --model-path <fp16_dir>  --out fp16.json
    python bench_ttft_tpot.py --impl brmoe --model-path <3bit_dir>  --out brmoe.json \
        --brmoe-backend brmoe_symmetric
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_fp16(model_path: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device.startswith("cuda") and "," in device:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": device}
    log(f"[load] transformers fp16 from {model_path} (device_map={kwargs.get('device_map')})")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model


def load_brmoe(model_path: str, device: str, backend: str):
    from BR_MoE.models.hf.deepseek import DeepSeekMoEBRMoE

    log(f"[load] BR-MoE 3bit from {model_path}")
    model = DeepSeekMoEBRMoE.from_compressed(
        model_path,
        compute_dtype=torch.float16,
        device="cuda" if device == "auto" else device,
    )
    if backend and backend != "default":
        from BR_MoE.utils.patching import prepare_for_inference

        log(f"[patch] prepare_for_inference(backend={backend})")
        prepare_for_inference(model, backend=backend)
    return model


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------

@torch.inference_mode()
def measure_once(model, vocab_size, batch_size, in_len, out_len, device, seed):
    """跑一次 prefill + decode, 返回 (ttft_ms, tpot_ms, per_token_ms)。"""
    if device == "auto":
        dev = next(model.parameters()).device
    else:
        dev = torch.device(device if device.startswith("cuda") else device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (batch_size, in_len), generator=gen).to(dev)
    attn = torch.ones_like(input_ids)

    # ---- prefill ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
    past = out.past_key_values
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ttft_ms = (t1 - t0) * 1000.0

    # ---- decode ----
    per_token = []
    for _ in range(max(0, out_len - 1)):
        attn = torch.cat(
            [attn, torch.ones((batch_size, 1), dtype=attn.dtype, device=dev)], dim=1
        )
        ts = time.perf_counter()
        out = model(
            input_ids=next_ids,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
        )
        next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
        past = out.past_key_values
        torch.cuda.synchronize()
        per_token.append((time.perf_counter() - ts) * 1000.0)

    decode_ms = sum(per_token)
    tpot_ms = decode_ms / len(per_token) if per_token else float("nan")

    del out, past, next_ids, input_ids, attn
    return ttft_ms, tpot_ms, per_token


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))
    return s[idx]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["fp16", "brmoe"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / auto(多卡)")
    ap.add_argument("--brmoe-backend", default="default",
                    help="default / brmoe_symmetric / brmoe_asymmetric / brmoe_auto / marlin")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-lens", default="128,512,1024")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label = args.label or f"{args.impl}" + (
        f"_{args.brmoe_backend}" if args.impl == "brmoe" and args.brmoe_backend != "default" else ""
    )
    input_lens = [int(x) for x in args.input_lens.split(",") if x.strip()]

    log("=" * 78)
    log(f"impl={args.impl} label={label} device={args.device} batch={args.batch_size}")
    log(f"input_lens={input_lens} output_len={args.output_len} "
        f"warmup={args.warmup} repeat={args.repeat}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log(f"  GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GiB "
                f"sm_{p.major}{p.minor}")
    log("=" * 78)

    t_load0 = time.perf_counter()
    if args.impl == "fp16":
        model = load_fp16(args.model_path, args.device)
    else:
        model = load_brmoe(args.model_path, args.device, args.brmoe_backend)
    model.eval()
    load_s = time.perf_counter() - t_load0
    log(f"[load] done in {load_s:.1f}s")

    vocab_size = getattr(model.config, "vocab_size", 102400)
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    cases = []
    for in_len in input_lens:
        log(f"--- input_len={in_len} ---")
        try:
            for w in range(args.warmup):
                measure_once(model, vocab_size, args.batch_size, in_len,
                             args.output_len, args.device, args.seed + w)
            reps = []
            for r in range(args.repeat):
                ttft, tpot, per_tok = measure_once(
                    model, vocab_size, args.batch_size, in_len,
                    args.output_len, args.device, args.seed + r,
                )
                reps.append((ttft, tpot, per_tok))
                log(f"    rep{r}: TTFT={ttft:.1f} ms  TPOT={tpot:.2f} ms/tok")
        except torch.cuda.OutOfMemoryError as e:
            log(f"    OOM at input_len={in_len}: {e}")
            torch.cuda.empty_cache()
            cases.append(dict(input_len=in_len, oom=True))
            continue

        ttfts = [x[0] for x in reps]
        tpots = [x[1] for x in reps]
        all_tok = [v for x in reps for v in x[2]]
        case = dict(
            input_len=in_len,
            output_len=args.output_len,
            batch_size=args.batch_size,
            ttft_ms=round(statistics.median(ttfts), 2),
            ttft_ms_reps=[round(v, 2) for v in ttfts],
            tpot_ms=round(statistics.median(tpots), 3),
            tpot_ms_reps=[round(v, 3) for v in tpots],
            tpot_p50_ms=round(pct(all_tok, 50), 3),
            tpot_p90_ms=round(pct(all_tok, 90), 3),
            decode_tok_per_s=round(1000.0 / statistics.median(tpots), 2)
            if statistics.median(tpots) > 0 else None,
        )
        cases.append(case)
        log(f"    => TTFT={case['ttft_ms']} ms  TPOT={case['tpot_ms']} ms/tok  "
            f"(p50={case['tpot_p50_ms']}, p90={case['tpot_p90_ms']})  "
            f"decode={case['decode_tok_per_s']} tok/s")

    result = dict(
        label=label,
        impl=args.impl,
        brmoe_backend=args.brmoe_backend if args.impl == "brmoe" else None,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        output_len=args.output_len,
        warmup=args.warmup,
        repeat=args.repeat,
        load_seconds=round(load_s, 2),
        peak_gpu_mem_gib=round(peak_mem, 2),
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        torch_version=torch.__version__,
        host=platform.node(),
        cases=cases,
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log(f"[out] wrote {args.out}")

    log("=== SUMMARY ===")
    for c in cases:
        if c.get("oom"):
            log(f"  in={c['input_len']:>5}  OOM")
        else:
            log(f"  in={c['input_len']:>5}  TTFT={c['ttft_ms']:>9.2f} ms  "
                f"TPOT={c['tpot_ms']:>7.3f} ms/tok  decode={c['decode_tok_per_s']:>8.2f} tok/s")
    log(f"  load={load_s:.1f}s  peak_mem={peak_mem:.2f} GiB")


if __name__ == "__main__":
    sys.exit(main())
