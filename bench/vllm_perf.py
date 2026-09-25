"""vLLM 性能扫描: 对给定模型跑 TTFT / TPOT / 吞吐 随 batch 的变化。

为什么需要它: fp16 (30.5 GiB 权重) 在 32 GB 的 5090 上放不下 (实测 OOM),
所以 "fp16 vs brmoe_int3 同卡对比" 只能在 A100(80GB) / H200(141GB) 上做。

计时口径 (与 bench/bench_ttft_tpot.py 一致, 便于和 HF 的数据对照):
  TTFT  = 单独跑一次 max_tokens=1 的墙钟
  E2E   = max_tokens=output_len 的墙钟
  TPOT  = (E2E - TTFT) / (output_len - 1)
  thr   = batch * output_len / E2E

用法:
  python vllm_perf.py --model <dir> --tag fp16 --batch-sizes 1,2,4,8,16,32
"""
import argparse
import json
import os
import statistics
import time

import torch


def make_prompt(tok, n_tok: int) -> str:
    """构造编码后恰好有 n_tok 个 token 的 prompt。"""
    base = ("The history of the Roman Empire spans more than a thousand years, "
            "from its legendary founding to the fall of Constantinople. ")
    text = base
    while len(tok.encode(text)) < n_tok + 8:
        text += base
    ids = tok.encode(text)
    # decode/encode is not an identity around BOS and subword boundaries.
    for size in range(max(1, n_tok - 8), n_tok + 9):
        prompt = tok.decode(ids[:size])
        if len(tok.encode(prompt)) == n_tok:
            return prompt
    raise ValueError(f"cannot construct a prompt of exactly {n_tok} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--input-len", type=int, default=128,
                    help="单个输入长度; 要测多个长度用 --input-lens")
    ap.add_argument("--input-lens", default=None,
                    help="逗号分隔的多个输入长度 (如 1024,2048,4096)。"
                         "复用同一个 LLM 实例, 避免每个长度都重载权重。"
                         "注意本模型 max_position_embeddings=4096。")
    ap.add_argument("--output-len", type=int, default=128)
    # 每组重复多少次取中位数。单次测量噪声可达 15%, 和我们要分辨的效应同量级
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--enforce-eager", type=int, default=0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    # 0 = 自动 = max(input_lens) + output_len + 512
    ap.add_argument("--max-model-len", type=int, default=0)
    # 必须 <= 5461: BR-MoE 的 moe_align_block_size_triton 在 total > 32768 时
    # 退回它的 torch 实现, 而那条分支返回的 buf 没有 'sv' 键, 会直接
    # KeyError: 'sv' 崩掉 (ops.py:216)。vLLM 默认 8192 -> 8192*6=49152 必崩,
    # 而且 profile_run 正好用 max_num_batched_tokens 个 dummy token 触发它。
    # 4096 时 4096*6=24576 < 32768, 且 chunked prefill 保证后续每步也不超。
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    # tokenizer 必须走 HF fast 路径, 否则慢速 LlamaTokenizer 会吃掉空格并丢 BOS
    ap.add_argument("--tokenizer-mode", default="hf")
    # 可选: 单独指定 tokenizer 目录。
    # 场合: fp16 基座模型的 tokenizer_config.json 里 tokenizer_class=LlamaTokenizerFast
    # 而目录下没有 tokenizer.model, 会让 vLLM 退回**慢速** LlamaTokenizer:
    #     HF   : [100000, 549, 6077, 280, 7239, 317]  -> 'The capital of France is'
    #     vLLM : [549, 42394, 994, 36715, 262]        -> 'ThecapitalofFranceis'
    # (丢 BOS + 吃掉所有空格)。对比实验里两组必须用**同一个**修好的 tokenizer,
    # 否则测的根本不是同一个输入。
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batches = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    if args.input_lens:
        lens = [int(v) for v in args.input_lens.split(",") if v.strip()]
    else:
        lens = [args.input_len]
    quant = None if args.quantization in (None, "", "none") else args.quantization

    # max_model_len: 未指定则按最长输入自动推算 (留 512 余量)。
    # 旧版硬编码 1024, 导致 1024 输入 + 128 输出被 vLLM 直接拒绝:
    #   "This model's maximum context length is 1024 tokens ... prompt 1025"
    max_len = args.max_model_len or (max(lens) + args.output_len + 512)

    # 本模型 max_position_embeddings=4096, 超了会被 vLLM 拒绝, 需显式放行
    # (必须在 import vllm 之前设置)。
    if max_len > 4096:
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        print(f"[perf] 注意: 模型训练上下文仅 4096, max_model_len={max_len} "
              f"属 RoPE 外推 —— 只看延迟, 别评判输出质量", flush=True)

    print(f"[perf] tag={args.tag} model={args.model} quant={quant} "
          f"eager={bool(args.enforce_eager)} batches={batches} "
          f"lens={lens} out={args.output_len} max_model_len={max_len}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                        trust_remote_code=True)
    if args.tokenizer:
        print(f"[perf] tokenizer 单独指定: {args.tokenizer}", flush=True)

    prompts_by_len = {}
    for n in lens:
        p = make_prompt(tok, n)
        prompts_by_len[n] = p
        print(f"[perf] 目标 in={n} -> 实际 {len(tok.encode(p))} tokens", flush=True)

    from vllm import LLM, SamplingParams
    kw = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=max_len,   # 必须用算好的值; args.max_model_len=0 只是"自动"的哨兵,
                                 # 直接传会被 vLLM 拒绝: "max_model_len must be a positive integer, got 0"
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        tokenizer_mode=args.tokenizer_mode,
        enforce_eager=bool(args.enforce_eager),
        enable_prefix_caching=False,  # repeated prompts must execute the 128-token prefill
        disable_log_stats=True,
    )
    if quant:
        kw["quantization"] = quant
    if args.tokenizer:
        kw["tokenizer"] = args.tokenizer
    llm = LLM(**kw)

    rows = []
    # 扁平化遍历: 输入长度 x batch。单层循环, 循环体缩进不变。
    for n_in, bs in [(n, b) for n in lens for b in batches]:
        prompts = [prompts_by_len[n_in]] * bs
        try:
            # 预热: 触发 graph 捕获与 kernel autotune
            llm.generate(prompts, SamplingParams(
                max_tokens=1, temperature=0.0, ignore_eos=True))

            # --- 重复测量取中位数 ---
            # 单次测量噪声太大: 实测 fp16 在 bs=4/8 之间能跳 15% (19.41 -> 22.33),
            # 而我们要分辨的效应 (int3 MoE 比 fp16 慢) 也就 12~16% -- 同量级。
            # 所以每组重复 N 次, 取中位数, 并把 min/max 一起记下来看离散度。
            ttfts, e2es, outs = [], [], None
            for _ in range(max(1, args.repeat)):
                # TTFT: 只生成 1 个 token
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                llm.generate(prompts, SamplingParams(
                    max_tokens=1, temperature=0.0, ignore_eos=True))
                torch.cuda.synchronize()
                ttfts.append((time.perf_counter() - t0) * 1e3)

                # E2E: 生成 output_len 个 token
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outs = llm.generate(prompts, SamplingParams(
                    max_tokens=args.output_len, temperature=0.0, ignore_eos=True))
                torch.cuda.synchronize()
                e2es.append((time.perf_counter() - t0) * 1e3)

            ttft_ms = statistics.median(ttfts)
            e2e_ms = statistics.median(e2es)
            ttft_lo, ttft_hi = min(ttfts), max(ttfts)
            e2e_lo, e2e_hi = min(e2es), max(e2es)

            n_out = args.output_len
            tpot_ms = (e2e_ms - ttft_ms) / max(1, n_out - 1)
            thr = bs * n_out / (e2e_ms / 1e3)

            # 顺带读一下 vLLM 自己的 metrics (若可用), 用于交叉验证
            m_ttft = None
            try:
                o = outs[0]
                ft = o.metrics.first_token_time
                ar = getattr(o.metrics, "arrival_time", None)
                if ft is not None and ar is not None:
                    m_ttft = (ft - ar) * 1e3
            except Exception:
                pass

            # TPOT 的上下界: 由 ttft/e2e 的极端组合给出 (保守区间)
            tpot_lo = (e2e_lo - ttft_hi) / max(1, n_out - 1)
            tpot_hi = (e2e_hi - ttft_lo) / max(1, n_out - 1)

            row = dict(input_len=n_in, batch=bs, ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                       e2e_ms=e2e_ms, thr_tok_s=thr,
                       ttft_lo=ttft_lo, ttft_hi=ttft_hi,
                       e2e_lo=e2e_lo, e2e_hi=e2e_hi,
                       tpot_lo=tpot_lo, tpot_hi=tpot_hi,
                       repeat=max(1, args.repeat),
                       vllm_ttft_ms=m_ttft,
                       sample=outs[0].outputs[0].text[:60])
            rows.append(row)
            print(f"[perf] in={n_in:>5} bs={bs:>3}  TTFT={ttft_ms:8.2f} ms  "
                  f"TPOT={tpot_ms:7.2f} ms/tok (±{(tpot_hi-tpot_lo)/2:4.2f})  "
                  f"E2E={e2e_ms:9.2f} ms  thr={thr:8.1f} tok/s"
                  + (f"  (vLLM TTFT={m_ttft:.2f})" if m_ttft else ""), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[perf] in={n_in} bs={bs} OOM: "
                  f"{str(e).splitlines()[0][:120]}", flush=True)
            torch.cuda.empty_cache()
            rows.append(dict(input_len=n_in, batch=bs, error="OOM"))
        except Exception as e:
            print(f"[perf] in={n_in} bs={bs} 失败 {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
            rows.append(dict(input_len=n_in, batch=bs,
                             error=f"{type(e).__name__}: {str(e)[:160]}"))

    # vLLM 把模型放在 EngineCore 子进程里, 父进程的 max_memory_allocated()
    # 恒为 0 (之前 peak 显示 0.00 GiB 就是这个原因)。改读设备级已用显存。
    peak = 0.0
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        peak = (total_b - free_b) / 2**30
    except Exception:
        pass
    summary = dict(tag=args.tag, model=args.model, quantization=quant,
                   enforce_eager=bool(args.enforce_eager),
                   input_lens=lens, output_len=args.output_len,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?",
                   peak_gib=peak, rows=rows)

    out = args.out or f"vllm_perf_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[perf] 峰值显存 {peak:.2f} GiB, 结果已写 {out}", flush=True)


if __name__ == "__main__":
    main()
