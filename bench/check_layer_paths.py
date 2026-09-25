"""确认那 196 个 int3 非专家层, 每一类**实际**走哪条路径。

为什么需要这个
--------------
profile 显示: brmoe3bit 与 int3dense 的差异**只**在 112 个 attention 投影
(int3_moe_gemm 调用 3456 -> 10624, 差 7168 = 112 x 64)。shared_expert(81)
和 dense_mlp(3) 的内核次数、耗时**完全相同**。

但逐键核实过 checkpoint: 这 84 层在两个模型里确实是 fp16 vs int3。
=> 要么它们走了别的路径, 要么 int3 权重压根没被用上。
后者会影响"忠实 3bit 模型"这个描述, 所以要查清。

两部分
------
A. 静态: 遍历模块树, 打印每类层的 quant_method 类型 + 参数字段
   (有 qweight 说明 create_weights 建了 int3 权重)
B. 动态: **monkey-patch** BRMoEInt3LinearMethod.apply 计调用次数
   —— 这才是"实际走哪条路"的决定性证据。配置对不等于真的被调用。

用法 (必须在 5090 节点的分配里跑; 不需要 GPU 计算, 但要能建模型):
    python bench/check_layer_paths.py
    python bench/check_layer_paths.py --model /mnt/.../brmoe-3bit-vllm
    python bench/check_layer_paths.py --no-run     # 只做静态检查, 不跑推理
"""
import argparse
import collections
import os
import re
import sys

# ---- 必须在 import vllm 之前 ----
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def kind(prefix: str) -> str:
    """把 model.layers.7.self_attn.q_proj -> model.layers.N.self_attn.q_proj"""
    parts = [("N" if p.isdigit() else p) for p in str(prefix).split(".")]
    return ".".join(parts)


def short_kind(prefix: str) -> str:
    """只保留有区分度的尾部, 如 self_attn.q_proj / shared_experts.gate_up_proj"""
    parts = [("N" if p.isdigit() else p) for p in str(prefix).split(".")]
    idx = [i for i, p in enumerate(parts) if p in ("self_attn", "mlp", "shared_experts")]
    if idx:
        return ".".join(parts[idx[0]:])
    return ".".join(parts[-2:])


# ---------------------------------------------------------------------------
# A. 静态: 模块树
# ---------------------------------------------------------------------------


def find_model(llm):
    """vLLM v1 + InprocClient 的模型路径有几种可能, 逐个试。"""
    candidates = [
        ("llm_engine", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "engine_core", "model_executor", "driver_worker", "model_runner"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner"),
    ]
    for path in candidates:
        o = llm
        for a in path:
            o = getattr(o, a, None)
            if o is None:
                break
        if isinstance(o, nn.Module):
            return o, ".".join(path)
    # 兜底: 从 engine 的 __dict__ 里找体量最大的 nn.Module
    best, bestn = None, 1
    eng = getattr(llm, "llm_engine", None)
    if eng is not None:
        for name in dir(eng):
            if name.startswith("_"):
                continue
            try:
                o = getattr(eng, name)
            except Exception:
                continue
            if isinstance(o, nn.Module):
                n = sum(1 for _ in o.modules())
                if n > bestn:
                    best, bestn = o, n
    return (best, f"<fallback: {bestn} modules>") if best is not None else (None, None)


def part_a(model):
    print("=" * 104)
    print("A. 静态检查: 每类层的 quant_method 与参数字段")
    print("=" * 104)

    rows = []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        params = [p for p in ("weight", "qweight", "scales", "zeros")
                  if isinstance(getattr(mod, p, None), nn.Parameter)]
        rows.append((name, type(mod).__name__, type(qm).__name__, tuple(params)))

    if not rows:
        print("  !! 没找到任何带 quant_method 的模块, find_model 的路径可能不对")
        return []

    agg = collections.Counter((cls, qmc, p) for _, cls, qmc, p in rows)
    print(f"\n  共 {len(rows)} 个带 quant_method 的模块, 归为 {len(agg)} 类:\n")
    print(f"  {'层类型':<34}{'quant_method':<32}{'参数字段':<28}{'个数':>6}")
    print("  " + "-" * 100)
    for (cls, qmc, ps), n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {cls[:33]:<34}{qmc[:31]:<32}{str(list(ps))[:27]:<28}{n:>6}")

    by_kind = collections.Counter(short_kind(n) for n, _, _, _ in rows)
    print(f"\n  按名分组的 {len(by_kind)} 种:")
    for k, n in sorted(by_kind.items()):
        qmc = next(q for nn_, _, q, _ in rows if short_kind(nn_) == k)
        ps = next(p for nn_, _, _, p in rows if short_kind(nn_) == k)
        print(f"    {k:<44}{n:>5} 个   {qmc:<30}{list(ps)}")
    return rows


def part_a_moe(model):
    """MoE 层里的 shared expert 是否被融合掉了。"""
    print("\n" + "=" * 104)
    print("A2. MoE 层的 shared expert 结构")
    print("=" * 104)
    seen = 0
    for name, mod in model.named_modules():
        if not (name.endswith("experts") or name.endswith("mlp.experts")):
            continue
        print(f"\n  [{name}]  {type(mod).__name__}")
        try:
            n_exp = getattr(mod, "num_experts", None) or getattr(mod, "n_routed_experts", None)
            print(f"    num_experts / n_routed_experts = {n_exp}")
        except Exception as e:
            print(f"    (读 experts 数失败: {e})")

        # 直接把所有含 shared / fuse 的属性打出来, 不猜名字
        hits = []
        for a in dir(mod):
            if a.startswith("_"):
                continue
            la = a.lower()
            if "shared" in la or "fuse" in la:
                try:
                    v = getattr(mod, a)
                except Exception:
                    continue
                if callable(v):
                    continue
                hits.append((a, type(v).__name__ if v is not None and not isinstance(v, bool)
                             else v))
        if hits:
            print("    含 shared/fuse 的属性:")
            for a, v in hits:
                print(f"      {a} = {v!r}"[:110])
        else:
            print("    (没有含 shared/fuse 的属性 —— 可能属性名不同, 需另找)")

        # moe_config 里也可能有
        mc = getattr(mod, "moe_config", None)
        if mc is not None:
            sub = [a for a in dir(mc) if not a.startswith("_")
                   and ("shared" in a.lower() or "fuse" in a.lower())]
            if sub:
                print(f"    moe_config 里的相关字段: "
                      f"{ {a: getattr(mc, a, None) for a in sub} }"[:160])

        seen += 1
        if seen >= 2:
            break
    if seen == 0:
        print("  !! 没找到 experts 模块 (名字规则可能不同)")


# ---------------------------------------------------------------------------
# B. 动态: 计数实际调用
# ---------------------------------------------------------------------------


def part_b(llm, model, prompt):
    print("\n" + "=" * 104)
    print("B. 动态检查: 给 BRMoEInt3LinearMethod.apply 打桩, 数实际调用")
    print("=" * 104)

    # vLLM 的 LinearBase 不一定保留 prefix 属性, 所以用 id(module) -> 名字 的映射
    name_by_id = {}
    if model is not None:
        for n, m in model.named_modules():
            name_by_id[id(m)] = n

    import brmoe_int3_vllm.linear_method as LM
    try:
        import brmoe_int3_vllm.moe_method as MM
    except Exception:
        MM = None

    calls = collections.Counter()
    moe_calls = collections.Counter()

    orig_apply = LM.BRMoEInt3LinearMethod.apply

    def patched_apply(self, layer, x, bias=None):
        nm = name_by_id.get(id(layer)) or getattr(layer, "prefix", None) \
            or type(layer).__name__
        calls[kind(nm)] += 1
        return orig_apply(self, layer, x, bias)

    LM.BRMoEInt3LinearMethod.apply = patched_apply

    orig_moe = None
    if MM is not None and hasattr(MM, "brmoe_int3_moe"):
        orig_moe = MM.brmoe_int3_moe

        def patched_moe(*a, **kw):
            moe_calls["brmoe_int3_moe"] += 1
            return orig_moe(*a, **kw)

        MM.brmoe_int3_moe = patched_moe

    try:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True)
        llm.generate([prompt], sp)
        torch.cuda.synchronize()
    finally:
        LM.BRMoEInt3LinearMethod.apply = orig_apply
        if MM is not None and orig_moe is not None:
            MM.brmoe_int3_moe = orig_moe

    total = sum(calls.values())
    print(f"\n  BRMoEInt3LinearMethod.apply 总调用 {total} 次 "
          f"(max_tokens=3 -> 1 prefill + 2 decode = 3 次 forward)")
    if total == 0:
        print("  !! 一次都没被调用 —— 说明所有 196 层都没走我们的线性路径!")
        return calls, moe_calls

    print(f"\n  {'层(按类归并)':<50}{'调用次数':>10}{'次/forward':>12}")
    print("  " + "-" * 74)
    for k, n in sorted(calls.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:49]:<50}{n:>10}{n/3:>12.1f}")

    print(f"\n  MoE 路径 (brmoe_int3_moe) 调用 {sum(moe_calls.values())} 次")

    # 结论
    print("\n  " + "-" * 74)
    has_se = any("shared_experts" in k for k in calls)
    has_mlp = any((".mlp." in k) and ("shared_experts" not in k) for k in calls)
    has_attn = any("self_attn" in k for k in calls)
    print(f"  attention 走了我们的 kernel 吗?    {'是' if has_attn else '否'}")
    print(f"  shared_experts 走了吗?             {'是' if has_se else '否'}")
    print(f"  dense mlp (非 shared) 走了吗?      {'是' if has_mlp else '否'}")
    print()
    if has_attn and not has_se:
        print("  => 只有 attention 走我们的线性 kernel。shared_experts / dense mlp")
        print("     走的是别的路径 —— 结合 A2 看是不是被融进了 FusedMoE。")
        print("     若 A2 显示 shared_experts=None 且 fuse_shared_experts=True,")
        print("     那就是融合: 它们作为额外专家共用 MoE 的那 2 次 GEMM,")
        print("     因此不产生额外的 linear 调用 (与 profile 完全一致)。")
    return calls, moe_calls


# ---------------------------------------------------------------------------


def main():
    BASE = "/mnt/709/data3/home/jianglei"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{BASE}/models/brmoe-3bit-vllm-int3dense")
    ap.add_argument("--tokenizer", default=f"{BASE}/models/brmoe-tokfix")
    ap.add_argument("--quantization", default="brmoe_int3")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--no-run", action="store_true",
                    help="只做静态检查 A/A2, 不跑推理 (B)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! 没有 GPU, 请先进节点")
        sys.exit(1)

    from vllm import LLM
    print(f"加载 {args.model} ...", flush=True)
    kw = dict(
        model=args.model, trust_remote_code=True, dtype="float16",
        tokenizer=args.tokenizer, tokenizer_mode="hf",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True, disable_log_stats=True,
    )
    if args.quantization:
        kw["quantization"] = args.quantization
    llm = LLM(**kw)

    model, path = find_model(llm)
    print(f"拿到模型对象: {path}")
    if model is None:
        print("!! 找不到模型对象, 只能做动态检查")
    else:
        part_a(model)
        part_a_moe(model)

    if args.no_run:
        print("\n(--no-run: 跳过动态检查)")
        return

    part_b(llm, model, args.prompt)


if __name__ == "__main__":
    main()
