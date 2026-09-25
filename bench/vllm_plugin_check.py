"""用 vLLM + brmoe_int3 插件跑贪心解码, 输出 token ids 供与 BR-MoE 参考实现对拍。

用法:
    python bench/vllm_plugin_check.py \
        --model /mnt/709/data3/home/jianglei/models/brmoe-3bit-vllm \
        --out /path/out.json --enforce-eager 1
"""

import argparse
import json
import os
import sys

# 让 vLLM 发现插件 (已 pip install -e 的话不需要, 这里兜底)
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
if os.path.isdir(os.path.join(_repo, "tools", "brmoe_int3_vllm")):
    sys.path.insert(0, os.path.join(_repo, "tools"))
    try:
        import brmoe_int3_vllm  # noqa: F401
        brmoe_int3_vllm.register()
        print("[check] 已手动注册 brmoe_int3 插件", flush=True)
    except Exception as e:
        print(f"[check] 手动注册失败({e}), 依赖 entry point", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enforce-eager", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--prompt", default="The capital of France is")
    # 'none' 表示不量化 (fp16 基线)
    ap.add_argument("--quantization", default="brmoe_int3")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    # BR-MoE 的 Triton align 有 BLOCK_TOTAL_MAX=32768 的上限 (超出会退回 torch 版,
    # 那个版本含 .item() host 同步, 会破坏 CUDA Graph)。total = tokens * top_k,
    # top_k=6, 所以批 ≤ 5461 才安全。vLLM 默认 8192 会越界, 这里降下来。
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    # 必须用 HF 的 fast tokenizer。vLLM 默认的 "auto" 对
    # tokenizer_class=LlamaTokenizerFast + 无 tokenizer.model 的目录会退回
    # **慢速** LlamaTokenizer(sentencepiece 路径), 后果是:
    #   HF   : [100000, 549, 6077, 280, 7239, 317]  -> 'The capital of France is'
    #   vLLM : [549, 42394, 994, 36715, 262]        -> 'ThecapitalofFranceis'
    # 即丢掉 BOS 且吃掉所有空格, 模型吃到的是错句子, 输出必崩。
    ap.add_argument("--tokenizer-mode", default="hf")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    quant = None if args.quantization.lower() in ("none", "", "fp16") else args.quantization
    print(f"[check] enforce_eager={bool(args.enforce_eager)} "
          f"quantization={quant} gpu_mem_util={args.gpu_mem_util} "
          f"max_num_batched_tokens={args.max_num_batched_tokens} "
          f"tokenizer_mode={args.tokenizer_mode}", flush=True)

    # 先核对 tokenization 与 HF 一致 (这是之前输出乱码的真正原因)
    try:
        from vllm.tokenizers.registry import get_tokenizer as _gt
        _t = _gt(args.model, trust_remote_code=True, tokenizer_mode=args.tokenizer_mode)
        _ids = _t.encode(args.prompt) if hasattr(_t, "encode") else _t(args.prompt)["input_ids"]
        print(f"[check] tokenizer={type(_t).__name__} ids={list(_ids)} "
              f"decode={_t.decode(list(_ids))!r}", flush=True)
    except Exception as e:
        print(f"[check] tokenizer 自检失败: {type(e).__name__}: {e}", flush=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        quantization=quant,
        enforce_eager=bool(args.enforce_eager),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        tokenizer_mode=args.tokenizer_mode,
        dtype="float16",
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    out = llm.generate([args.prompt], sp)[0].outputs[0]

    # 顺带记一下显存占用，方便判断 fp16 在 32 GB 卡上是否放得下
    try:
        import torch as _t
        free_b, total_b = _t.cuda.mem_get_info()
        mem = {"total_gib": total_b / 2**30, "free_gib": free_b / 2**30,
               "used_gib": (total_b - free_b) / 2**30,
               "torch_peak_gib": _t.cuda.max_memory_allocated() / 2**30}
    except Exception:
        mem = {}

    res = {
        "impl": args.tag or ("vllm+brmoe_int3" if quant else "vllm+fp16"),
        "quantization": quant,
        "enforce_eager": bool(args.enforce_eager),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "text": out.text,
        "token_ids": list(out.token_ids),
        "gpu": __import__("torch").cuda.get_device_name(0),
        "memory": mem,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("[check] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
