"""在 NVTX 标注下跑一小段 prefill + decode, 供 nsys 采集、再由 veloq 分析。

NVTX 层级 (便于 veloq --group-by nvtx-path 聚合):
    load
    warmup{w}
    iter{r} / prefill
    iter{r} / decode / step{i}

用法 (由 nsys 包裹, 不要直接跑):
    nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none -o trace -f true \
        python trace_run.py --impl fp16 --model-path <dir>
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ttft_tpot import load_brmoe, load_fp16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["fp16", "brmoe"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--brmoe-backend", default="default")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    nvtx = torch.cuda.nvtx

    print(f"[trace] impl={args.impl} backend={args.brmoe_backend} "
          f"in_len={args.input_len} decode_steps={args.decode_steps} "
          f"repeats={args.repeats} warmup={args.warmup}", flush=True)

    nvtx.range_push("load")
    t0 = time.time()
    if args.impl == "fp16":
        model = load_fp16(args.model_path, args.device)
    else:
        model = load_brmoe(args.model_path, args.device, args.brmoe_backend)
    model.eval()
    nvtx.range_pop()
    print(f"[trace] load done in {time.time() - t0:.1f}s", flush=True)

    vocab = getattr(model.config, "vocab_size", 102400)
    dev = torch.device(args.device)

    def one_iter():
        g = torch.Generator(device="cpu").manual_seed(1234)
        ids = torch.randint(1, vocab, (1, args.input_len), generator=g).to(dev)
        attn = torch.ones_like(ids)

        nvtx.range_push("prefill")
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=attn, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            past = out.past_key_values
        torch.cuda.synchronize()
        nvtx.range_pop()

        nvtx.range_push("decode")
        with torch.inference_mode():
            for i in range(args.decode_steps):
                nvtx.range_push(f"step{i}")
                attn = torch.cat(
                    [attn, torch.ones((1, 1), dtype=attn.dtype, device=dev)], dim=1
                )
                out = model(
                    input_ids=nxt, attention_mask=attn,
                    past_key_values=past, use_cache=True,
                )
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                past = out.past_key_values
                torch.cuda.synchronize()
                nvtx.range_pop()
        nvtx.range_pop()

    for w in range(args.warmup):
        nvtx.range_push(f"warmup{w}")
        one_iter()
        nvtx.range_pop()
    torch.cuda.synchronize()

    for r in range(args.repeats):
        nvtx.range_push(f"iter{r}")
        one_iter()
        nvtx.range_pop()
    torch.cuda.synchronize()

    print("[trace] done", flush=True)


if __name__ == "__main__":
    main()
