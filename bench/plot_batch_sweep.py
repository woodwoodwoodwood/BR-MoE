#!/usr/bin/env python
"""把 batch 扫描结果画成论文 Figure 8 那样的分组柱状图 (Total Time vs batch size)。

数据来源: bench_results/ 下的 `<label>_b<batch>_<jobid>.json`,
每个文件的 fields: label / batch_size / output_len / cases[].{input_len,ttft_ms,tpot_ms,oom}

Total time = ttft_ms + tpot_ms * (output_len - 1)   [秒]

用法:
    python bench/plot_batch_sweep.py --result-dir bench_results --job 38416
    python bench/plot_batch_sweep.py --result-dir bench_results      # 每个组合取最新一份
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# label -> (显示名, 颜色)
STYLE = {
    "fp16":               ("FP16",                    "#4C72B0"),
    "brmoe3bit_grouped":  ("BR-MoE 3bit (grouped)",   "#55A868"),
    "brmoe3bit_cuda":     ("BR-MoE 3bit (per-expert)", "#DD8452"),
    "brmoe3bit_pytorch":  ("BR-MoE 3bit (PyTorch)",   "#C44E52"),
}
ORDER = list(STYLE)


def load(result_dir, job=None, input_len=None):
    pat = f"*_b*_{job}.json" if job else "*_b*.json"
    best = {}
    for p in glob.glob(os.path.join(result_dir, pat)):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        lab, b = r.get("label"), r.get("batch_size")
        if lab is None or b is None:
            continue
        cases = [c for c in r["cases"] if not c.get("oom")]
        if input_len is not None:
            cases = [c for c in cases if c["input_len"] == input_len]
        if not cases:
            continue
        # 同一 (label, batch) 取最新的文件
        key = (lab, b)
        mt = os.path.getmtime(p)
        if key not in best or mt > best[key][0]:
            best[key] = (mt, r, cases[0])
    return {k: v[2] for k, v in best.items()}, {k: v[1] for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="bench_results")
    ap.add_argument("--job", default=None, help="只读某个 job 的结果")
    ap.add_argument("--input-len", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases, recs = load(args.result_dir, args.job, args.input_len)
    if not cases:
        raise SystemExit(f"在 {args.result_dir} 下没找到结果 (job={args.job})")

    batches = sorted({b for _, b in cases})
    labels = [l for l in ORDER if any(k[0] == l for k in cases)]
    labels += [l for l, _ in cases if l not in labels]

    def total(lab, b):
        c = cases.get((lab, b))
        if c is None:
            return None
        out_len = recs[(lab, b)].get("output_len", 64)
        return (c["ttft_ms"] + c["tpot_ms"] * max(0, out_len - 1)) / 1000.0

    # ---- 打印表 ----
    in_len = next(iter(cases.values()))["input_len"]
    out_len = next(iter(recs.values())).get("output_len", 64)
    print(f"# {in_len} input + {out_len} output tokens, A100, batch sweep")
    hdr = f"{'batch':>6}" + "".join(f"{STYLE[l][0]:>28}" if l in STYLE else f"{l:>28}"
                                     for l in labels)
    print(hdr)
    for b in batches:
        row = f"{b:>6}"
        for l in labels:
            t = total(l, b)
            row += f"{'--':>28}" if t is None else f"{t:>27.3f}s"
        print(row)

    if "fp16" in labels:
        print(f"\n# speedup vs FP16 (total time ratio, >1 = faster than FP16)")
        print(f"{'batch':>6}" + "".join(f"{(STYLE[l][0] if l in STYLE else l):>28}"
                                        for l in labels if l != "fp16"))
        for b in batches:
            base = total("fp16", b)
            row = f"{b:>6}"
            for l in labels:
                if l == "fp16":
                    continue
                t = total(l, b)
                row += f"{'--':>28}" if (t is None or not base) else f"{base / t:>27.2f}x"
            print(row)

    # ---- 画图 ----
    n = len(labels)
    x = np.arange(len(batches))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, l in enumerate(labels):
        vals = [total(l, b) for b in batches]
        xs = [xx + (i - (n - 1) / 2) * w for xx, v in zip(x, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        name, color = STYLE.get(l, (l, None))
        bars = ax.bar(xs, vs, width=w * 0.92, label=name, color=color,
                      edgecolor="white", linewidth=0.6)
        for rect, v in zip(bars, vs):
            ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, rotation=90,
                        xytext=(0, 1.5), textcoords="offset points")

    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Total time (s)", fontsize=11)
    ax.set_title(f"End-to-end generation — DeepSeek-MoE-16B, A100 "
                 f"({in_len} input + {out_len} output tokens)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = args.out or os.path.join(
        args.result_dir, "figures",
        f"batch_sweep{'_' + args.job if args.job else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
