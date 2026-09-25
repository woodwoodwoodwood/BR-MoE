"""BR-MoE 原生实现 (PyTorch 后端) 的贪心解码, 作为对拍参考。

和 vllm_plugin_check.py 用同一 prompt / 同一 max_tokens / temperature=0,
两边 token ids 逐位相同即说明: 转换器 + 插件链路数值正确。

用法 (需要 BR_MoE 可导入, 见 bench_deepseek.slurm 里的符号链接技巧):
    python bench/brmoe_ref_check.py --model-path <3bit 目录> --out <json>
"""

import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from BR_MoE.models.hf.deepseek import DeepSeekMoEBRMoE
    from BR_MoE.utils.patching import prepare_for_inference

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepSeekMoEBRMoE.from_compressed(
        args.model_path, compute_dtype=torch.float16, device="cuda:0")
    # 用 PyTorch 参考后端 (与 vLLM 的 kernel 无关, 只验证量化权重本身)
    prepare_for_inference(model, backend="default")
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.inference_mode():
        out = model.generate(
            ids, max_new_tokens=args.max_tokens, do_sample=False,
            num_beams=1, use_cache=True, pad_token_id=tok.pad_token_id or 0,
        )
    gen = out[0, ids.shape[1]:].tolist()
    text = tok.decode(gen, skip_special_tokens=True)

    res = {"impl": "brmoe-pytorch", "prompt": args.prompt,
           "text": text, "token_ids": gen}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("[ref] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
