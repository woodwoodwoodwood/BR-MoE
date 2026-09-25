"""真实模型的 CUDA profile: 对比 fp16-attn+int3-MoE 与 全 int3 的差别在哪。

为什么必须做这个
----------------
微基准 (bench/micro_linear.py) 是**隔离测量**: 单层跑时 grid 只有 32 个 CTA,
5090 的 170 个 SM 空着 138 个, 纯延迟受限 -> 每层 ~28-36 µs。
按这个外推 196 层 = 5.5~7 ms/步, 但端到端实测差值只有 2.06 ms —— 差 3 倍。

隔离测量必然高估 (真实图里几十个 kernel 共存, 空闲 SM 会被邻居填上)。
所以需要一个**真实场景**的数, 这个数才能写进论文。

两个模型唯一的差别
------------------
已逐键核实: 196 个非专家线性层 (112 attention + 81 shared_expert + 3 dense_mlp)
    brmoe3bit : <proj>.weight                (fp16, 走 cuBLAS)
    int3dense : <proj>.qweight/scales/zeros  (int3, 走我们的 Triton kernel)
权重体积差 1.46 GiB, 与实测 8.75 vs 7.29 GiB 完全吻合。
=> 两个 profile 的**差**就是这 196 层的代价, 干净的一阶差分。

口径
----
* 统计的是**内核 CUDA 时间之和** (不含 CPU 发射开销), 所以用 eager 也能拿到
  可信的 GPU 工作量; 而 eager 下每个 kernel 都独立可见, 便于归因。
* 默认 --enforce-eager 1 正是为此 (graph 下内核会被折叠进 replay, 归因变差)。
* 必须在 import vllm **之前** 设 VLLM_ENABLE_V1_MULTIPROCESSING=0,
  否则 EngineCore 在子进程跑, 父进程的 profiler 什么都看不到。

用法 (必须在 5090 节点的分配里跑):
    python bench/profile_models.py
    python bench/profile_models.py --tokens 128 --prompt-len 256
    python bench/profile_models.py --cases "brmoe3bit|/path|brmoe_int3"
"""
import argparse
import json
import os
import subprocess
import sys

# ---- 必须在 import vllm 之前 ----
# 直接赋值而非 setdefault: 引擎在子进程跑的话 profiler 看不到任何内核
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# FlashInfer 采样器的 JIT 与归因无关, 关掉既省时间又让 profile 更干净
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# 内核归因
# ---------------------------------------------------------------------------
# 顺序重要: int3 必须在 "gemm" 之前判, 否则 moe_gemm 会被算进 cuBLAS 桶
BUCKET_RULES = [
    ("① int3 kernel (我们的)", ("int3", "moe_gemm")),
    ("② cuBLAS/fp16 GEMM",     ("xmma", "cutlass", "cublas", "gemm", "wgrad", "hgmm")),
    ("③ attention (flash)",    ("flash", "fmha", "attention", "mha_fwd", "mha_bwd")),
    ("④ elementwise/mem",      ("elementwise", "vectorized", "reduce", "memset",
                                "fill", "copy", "arange", "unrolled", "cast",
                                "index", "gather", "scatter", "silu", "rope")),
]


def bucket_of(name: str) -> str:
    n = name.lower()
    for label, kws in BUCKET_RULES:
        if any(k in n for k in kws):
            return label
    return "⑤ 其它"


def _cuda_us(ev) -> float:
    """torch 不同版本字段名不同。"""
    for attr in ("cuda_time_total", "device_time_total"):
        v = getattr(ev, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def summarize(prof):
    """-> (rows, per_bucket) ; rows = [(name, us, count), ...] 按时间降序"""
    rows = []
    tot = {}
    for ev in prof.key_averages():
        us = _cuda_us(ev)
        if us <= 0:
            continue
        rows.append((ev.key, us, int(ev.count)))
        b = bucket_of(ev.key)
        u, c = tot.get(b, (0.0, 0))
        tot[b] = (u + us, c + int(ev.count))
    rows.sort(key=lambda r: -r[1])
    return rows, tot


# ---------------------------------------------------------------------------


def make_prompt(tok, n_tok):
    base = ("The history of the Roman Empire spans more than a thousand years, "
            "from its legendary founding to the fall of Constantinople. ")
    t = base
    while len(tok.encode(t)) < n_tok:
        t += base
    return tok.decode(tok.encode(t)[:n_tok])


def profile_one(tag, model, quant, args, tok):
    from vllm import LLM, SamplingParams

    kw = dict(
        model=model, trust_remote_code=True, dtype="float16",
        max_model_len=args.prompt_len + args.tokens + 512,
        max_num_batched_tokens=args.max_batched,
        gpu_memory_utilization=args.gpu_mem,
        tokenizer_mode="hf",
        enforce_eager=bool(args.enforce_eager),
        disable_log_stats=True,
    )
    if quant:
        kw["quantization"] = quant
    if args.tokenizer:
        kw["tokenizer"] = args.tokenizer

    print(f"\n{'#'*100}\n## {tag}   model={model}  quant={quant}\n{'#'*100}", flush=True)
    llm = LLM(**kw)

    prompt = make_prompt(tok, args.prompt_len)
    prompts = [prompt] * args.batch
    sp = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)

    # 预热: 触发 Triton JIT / graph 捕获 / autotune
    llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    prof = profile(activities=[ProfilerActivity.CUDA], with_stack=False)
    with prof:
        llm.generate(prompts, sp)
        torch.cuda.synchronize()

    rows, buckets = summarize(prof)
    total = sum(r[1] for r in rows)
    n_decode = max(1, args.tokens)
    steps = max(1, args.batch)

    print(f"\n总内核 CUDA 时间 = {total/1e3:.2f} ms  "
          f"(batch={args.batch}, prompt={args.prompt_len} tok, 生成 {args.tokens} tok)")
    print(f"  ≈ {(total/steps)/1e3:.3f} ms / 条序列")
    print()

    print(f"  {'分桶':<26}{'CUDA ms':>12}{'占比':>9}{'调用次数':>12}")
    print("  " + "-" * 60)
    for b, (u, c) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print(f"  {b:<26}{u/1e3:>12.3f}{u/total*100:>8.1f}%{c:>12}")

    print(f"\n  {'Top 20 内核':<58}{'CUDA ms':>10}{'次数':>10}")
    print("  " + "-" * 78)
    for name, us, cnt in rows[:20]:
        print(f"  {name[:57]:<58}{us/1e3:>10.3f}{cnt:>10}")

    rec = dict(tag=tag, model=model, quantization=quant,
               enforce_eager=bool(args.enforce_eager),
               batch=args.batch, prompt_len=args.prompt_len, tokens=args.tokens,
               total_cuda_ms=total / 1e3,
               buckets={k: [v[0] / 1e3, v[1]] for k, v in buckets.items()},
               kernels=[[n, u / 1e3, c] for n, u, c in rows[:60]])

    del llm
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------


def print_diff(a, b):
    """一阶差分: 两个模型唯一的差别就是那 196 个非专家线性层。"""
    print(f"\n{'='*100}")
    print(f"一阶差分: {b['tag']} - {a['tag']}")
    print("  两个模型唯一的差别就是 196 个非专家线性层 (fp16 vs int3),")
    print("  所以下面的差就是这 196 层的真实代价 —— 这是能写进论文的数。")
    print(f"{'='*100}")
    dt = b["total_cuda_ms"] - a["total_cuda_ms"]
    print(f"\n  总 CUDA 时间: {a['total_cuda_ms']:.3f} -> {b['total_cuda_ms']:.3f} ms"
          f"   Δ = {dt:+.3f} ms  ({dt/a['total_cuda_ms']*100:+.1f}%)")
    print(f"  折合每层: {dt/196*1e3:+.2f} µs  (196 层)")
    print()
    print(f"  {'分桶':<26}{a['tag']:>14}{b['tag']:>14}{'Δ ms':>12}")
    print("  " + "-" * 68)
    for k in sorted(set(a["buckets"]) | set(b["buckets"])):
        ua = a["buckets"].get(k, [0, 0])
        ub = b["buckets"].get(k, [0, 0])
        print(f"  {k:<26}{ua[0]:>14.3f}{ub[0]:>14.3f}{ub[0]-ua[0]:>+12.3f}"
              f"   (次数 {ua[1]} -> {ub[1]})")
    print()
    print("  怎么读:")
    print("    · ① 桶 Δ 就是我们的 int3 线性层净增的 GPU 时间")
    print("    · ② 桶 Δ 应该是负的 (fetch 少了 196 个 fp16 cuBLAS)")
    print("    · 若 Δ 总时间 ≈ 端到端测到的 2.06 ms -> 微基准的 28-36 µs/层是")
    print("      **隔离测量的高估**, 真实图里空闲 SM 被邻居填上了")
    print("    · 若 Δ 总时间 ≈ 5-7 ms -> 微基准是对的, 端到端另有解释")


def main():
    BASE = "/mnt/709/data3/home/jianglei"
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=(
        f"brmoe3bit|{BASE}/models/brmoe-3bit-vllm|brmoe_int3;"
        f"int3dense|{BASE}/models/brmoe-3bit-vllm-int3dense|brmoe_int3"),
        help="分号分隔的 tag|model|quant")
    ap.add_argument("--tokenizer", default=f"{BASE}/models/brmoe-tokfix",
                    help="强制两组用同一个修好的 tokenizer (fp16 基座的会退化成慢速版)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=64, help="生成的 token 数 (decode 步数)")
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--max-batched", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", default=None,
                    help="只做差分: 两个 profile JSON 的路径 (逗号分隔), 不加载模型")
    ap.add_argument("--single", action="store_true",
                    help="内部用: 只处理这一个 case, 不再派生子进程")
    args = ap.parse_args()

    # ---- 差分模式: 不需要 GPU, 也不需要加载模型 ----
    # 用途: 同一个进程里连续创建两个 LLM 有失败风险 (分布式全局状态),
    #       所以可以分两次跑、各自 --out, 然后用这里对比。
    if args.diff:
        pa, pb = [p.strip() for p in args.diff.split(",")]
        ra = json.load(open(pa))
        rb = json.load(open(pb))
        print_diff(ra[-1] if isinstance(ra, list) else ra,
                   rb[-1] if isinstance(rb, list) else rb)
        return

    # ---- 多个 case: 必须用子进程 ----
    # 同一个进程里加载第二个 LLM 会 OOM —— vLLM 的 worker 在 `del llm` 之后
    # 仍持有权重和 KV cache (实测第一个跑完后只剩 2.46/31.4 GiB 空闲),
    # 而 torch.cuda.empty_cache() 释放不了活引用。
    # 所以每个 case 派生一个干净子进程, 最后读回 JSON 做差分。
    cases = [c.strip() for c in args.cases.split(";") if c.strip()]
    if len(cases) > 1 and not args.single:
        outs = []
        for c in cases:
            tag = c.split("|")[0].strip()
            out = (args.out.replace(".json", f"_{tag}.json")
                   if args.out else f"/tmp/profile_{tag}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--single",
                   "--cases", c, "--out", out,
                   "--tokenizer", args.tokenizer,
                   "--batch", str(args.batch),
                   "--prompt-len", str(args.prompt_len),
                   "--tokens", str(args.tokens),
                   "--enforce-eager", str(args.enforce_eager),
                   "--max-batched", str(args.max_batched),
                   "--gpu-mem", str(args.gpu_mem)]
            print(f"\n{'='*100}\n>>> 子进程 [{tag}]\n>>> {' '.join(cmd)}\n{'='*100}",
                  flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"!! 子进程 [{tag}] 退出码 {rc}, 终止")
                sys.exit(rc)
            outs.append(out)
        ra = json.load(open(outs[0]))
        rb = json.load(open(outs[-1]))
        print_diff(ra[-1] if isinstance(ra, list) else ra,
                   rb[-1] if isinstance(rb, list) else rb)
        return

    if not torch.cuda.is_available():
        print("!! 没有 GPU, 请先进节点")
        sys.exit(1)
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print("!! VLLM_ENABLE_V1_MULTIPROCESSING 不是 0, profiler 会看不到内核")
        sys.exit(1)

    # 两组必须用同一个 (修好的) tokenizer, 否则 fp16 基座会退回慢速版,
    # 输入都不一样就谈不上对比
    tok_dir = args.tokenizer or args.cases.split(";")[0].split("|")[1].strip()
    if args.prompt_len + args.tokens + 512 > 4096:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    print(f"tokenizer = {type(tok).__name__}  (来自 {tok_dir})")

    recs = []
    for case in args.cases.split(";"):
        case = case.strip()
        if not case:
            continue
        parts = [p.strip() for p in case.split("|")]
        tag, model = parts[0], parts[1]
        quant = parts[2] if len(parts) > 2 else None
        if quant in ("none", ""):
            quant = None
        recs.append(profile_one(tag, model, quant, args, tok))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"\n已写 {args.out}")

    # ---------------- 一阶差分 ----------------
    if len(recs) >= 2:
        print_diff(recs[0], recs[-1])
    else:
        print("\n(只跑了一个模型, 直接对比请用 --diff <a.json>,<b.json>，"
              "或一次传两个 --cases)")


if __name__ == "__main__":
    main()
