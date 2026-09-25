#!/usr/bin/env python
"""把 BR-MoE 的压缩 checkpoint 转成 vLLM 可加载的 HF safetensors。

背景
----
BR-MoE 的产物 (`qmodel.pt`) 是一份 **嵌套 dict** 的私有格式:
    {模块路径: {字段名: tensor/标量}}          # 5466 个模块
    e.g. "model.layers.1.mlp.experts.0.gate_proj" -> {
             "W_q":   int32 [ceil(n*k/gs/10), gs]   # 3-bit 码, 每 int32 装 10 个
             "scale": fp16  [n*k/gs, 1]             # 每 (输出通道, K 组) 一个
             "zero":  fp16  [n*k/gs, 1]
             "nbits": 3, "group_size": 64, "packing": "3bit_32",
             "shape": (n, k), "axis": 1, ... }

    反量化语义:  W = (unpack(W_q)[:n*k].view(n, k) - zero.view(n, gs_groups)) * scale...

vLLM 只认 safetensors + `quantization_config`, 而且**内置量化方法里没有 3-bit**
(`moe_wna16` 硬断言 weight_bits in [4,8]), 所以必须配一个自定义量化方法插件。
本脚本只负责 **数据格式转换**; 插件在 tools/brmoe_int3_vllm/ 下单独实现。

输出
----
    <dst>/config.json                     原 config + quantization_config
    <dst>/model.safetensors[.index.json]  分片权重
    <dst>/{configuration_deepseek.py, modeling_deepseek.py, tokenizer*.json, ...}

权重命名 (与 brmoe_int3 插件契约一致)
-------------------------------------
专家层 (layer 1..27) —— 直接采用 BR-MoE grouped kernel 吃的那套布局。
注意中间多一段 `routed_experts`（vLLM 0.29 的 MoERunner 子模块名，
是参数名的**精确匹配**要求，见 DEFAULT_EXPERT_SUBMODULE）:

    model.layers.N.mlp.experts.routed_experts.w13_q   int32 [E, 2I, K//32*3]
    model.layers.N.mlp.experts.routed_experts.w13_s   fp16  [E, K//gs, 2I]
    model.layers.N.mlp.experts.routed_experts.w13_z   fp16  [E, K//gs, 2I]
    model.layers.N.mlp.experts.routed_experts.w2_q    int32 [E, K, I//32*3]
    model.layers.N.mlp.experts.routed_experts.w2_s    fp16  [E, I//gs, K]
    model.layers.N.mlp.experts.routed_experts.w2_z    fp16  [E, I//gs, K]

非专家线性层 (attention / dense MLP / shared_experts)
    --dense-mode fp16 (默认)   ...weight            fp16  [n, k]        (反量化)
    --dense-mode int3          ...qweight  int32 [K//32*3, n]  (K-major, 便于合并访存)
                               ...scales   fp16  [k//gs, n]
                               ...zeros    fp16  [k//gs, n]

普通层 (embed_tokens / layernorm / mlp.gate / lm_head) 原样透传

用法
----
    python tools/convert_brmoe_to_vllm.py \
        --src /mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0 \
        --dst /mnt/709/data3/home/jianglei/models/brmoe-3bit-vllm

    # 只看规模不写盘
    python tools/convert_brmoe_to_vllm.py --src ... --dst ... --dry-run
"""

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import torch

# ---------------------------------------------------------------------------
# 按文件路径加载 BR-MoE 的两个打包模块 (包目录名含连字符 "BR-MoE", 不能直接 import)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRMOE_PKG = os.path.join(os.path.dirname(_HERE), "BR-MoE")


def _load_module(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bitpack = _load_module("brmoe_bitpack", os.path.join(_BRMOE_PKG, "core", "bitpack.py"))
_packing = _load_module(
    "brmoe_int3_packing",
    os.path.join(_BRMOE_PKG, "kernels", "triton_int3", "int3_moe", "packing.py"),
)
BitPack = _bitpack.BitPack
pack_int3 = _packing.pack_int3

# BR-MoE 的 packing 名 -> 解包函数
UNPACK = {"3bit_32": BitPack.unpack_3bit_32}

# 需要透传的普通层（不是量化层）
PASSTHROUGH_SUFFIX = ("embed_tokens", "layernorm", "mlp.gate", "lm_head")

# vLLM 侧专家参数所在的子模块名。
#
# vLLM 0.29 的 MoE 被拆成 MoERunner + RoutedExperts 两层，MoERunner 用
# `self.routed_experts = routed_experts` 挂载持有权重的 RoutedExperts
# (moe_runner.py)，而专家参数的 mapping 生成器默认
# `routed_experts_prefix="routed_experts"` (fused_moe_make_expert_params_mapping)。
# 因此参数全名是:
#     model.layers.N.mlp.experts.routed_experts.<param>
# 而 DeepseekV2Model.load_weights 里是 `param = params_dict[name]` 的**名字精确匹配**，
# 少一段就直接 KeyError，所以 checkpoint 的键必须带上这一段。
DEFAULT_EXPERT_SUBMODULE = "routed_experts"


def is_moe_expert(path: str) -> bool:
    return ".mlp.experts." in path


def is_quantized(d) -> bool:
    return isinstance(d, dict) and "W_q" in d


def layer_index(path: str):
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# 解包: BR-MoE 私有格式 -> (q[n,k] int, s[n,groups] fp16, z[n,groups] fp16)
# ---------------------------------------------------------------------------

def unpack_quantized(d: dict):
    """对应 BR-MoE `backends/brmoe_grouped.py::_extract_quantized` 的逻辑。"""
    packing = d["packing"]
    if packing not in UNPACK:
        raise NotImplementedError(f"暂不支持 packing={packing}")
    n, k = int(d["shape"][0]), int(d["shape"][1])
    gs = int(d["group_size"])
    if k % gs:
        raise ValueError(f"K={k} 不是 group_size={gs} 的整数倍: {d.get('shape')}")
    groups = k // gs

    # W_q [ceil(n*k/gs/10), gs] -> 解包成 [10*step, gs] -> 展平后取前 n*k 个 -> [n, k]
    q = UNPACK[packing](d["W_q"], dtype=torch.int32)
    q = q.reshape(-1)[: n * k].reshape(n, k).to(torch.int32)
    s = d["scale"].reshape(n, groups).to(torch.float16)
    z = d["zero"].reshape(n, groups).to(torch.float16)
    return q, s, z


def dequantize(d: dict) -> torch.Tensor:
    """W = (q - z) * s, 按 group 广播回 [n, k]。只用于 --dense-mode fp16。"""
    q, s, z = unpack_quantized(d)
    n, k = q.shape
    gs = int(d["group_size"])
    w = (q.to(torch.float32) - z.repeat_interleave(gs, dim=1).to(torch.float32))
    w = w * s.repeat_interleave(gs, dim=1).to(torch.float32)
    return w.to(torch.float16)


# ---------------------------------------------------------------------------
# 打包: -> vLLM 侧布局
# ---------------------------------------------------------------------------

def pack_one_expert_layer(get, expert_ids, gs, device="cpu"):
    """把一层专家的全部门打包成 BR-MoE grouped kernel 的布局。

    get(expert, proj) -> {'W_q','scale','zero','group_size','shape',...}
    对应 `backends/brmoe_grouped.py::_pack_one_layer_lossless`。
    """
    E = len(expert_ids)
    g0 = get(expert_ids[0], "gate_proj")
    I, K = int(g0["shape"][0]), int(g0["shape"][1])
    gs = int(g0["group_size"])

    q13 = torch.empty(E, 2 * I, K, dtype=torch.int32, device=device)
    q2 = torch.empty(E, K, I, dtype=torch.int32, device=device)
    s13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    s2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)
    z13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    z2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)

    for e, eid in enumerate(expert_ids):
        pq, ps, pz = [], [], []
        for proj in ("gate_proj", "up_proj"):
            q, s, z = unpack_quantized(get(eid, proj))
            pq.append(q)
            ps.append(s)
            pz.append(z)
        q13[e] = torch.cat(pq, dim=0)              # [2I, K]
        s13[e] = torch.cat(ps, dim=0).t()          # [K//gs, 2I]
        z13[e] = torch.cat(pz, dim=0).t()

        qd, sd, zd = unpack_quantized(get(eid, "down_proj"))
        q2[e] = qd                                  # [K, I]
        s2[e] = sd.t()                              # [I//gs, K]
        z2[e] = zd.t()
        del pq, ps, pz

    return {
        "w13_q": pack_int3(q13.reshape(E * 2 * I, K), transposed=False)
                 .reshape(E, 2 * I, K // 32 * 3).to(torch.int32),
        "w13_s": s13, "w13_z": z13,
        "w2_q": pack_int3(q2.reshape(E * K, I), transposed=False)
                .reshape(E, K, I // 32 * 3).to(torch.int32),
        "w2_s": s2, "w2_z": z2,
    }


def pack_one_linear(d: dict):
    """单个量化线性层 -> int3 布局。

    `pack_int3(..., transposed=True)` 给出 [K//32*3, N] (K-major)。
    packing.py 的文档说明过为什么必须 K-major: 若按 [N, Kpack] 存, 相邻 n 相隔
    一整个 Kpack (本模型 192 B), 每个 4 B 元素各拉一条 128 B cache line,
    过取 32 倍, 有效带宽掉到 ~9 GB/s。
    """
    q, s, z = unpack_quantized(d)          # q[n,k] s/z[n,groups]
    qw = pack_int3(q, transposed=True)     # [k//32*3, n]
    return {
        "qweight": qw.to(torch.int32),
        "scales": s.t().contiguous().to(torch.float16),    # [groups, n]
        "zeros": z.t().contiguous().to(torch.float16),
    }


# ---------------------------------------------------------------------------
# safetensors 分片写出
# ---------------------------------------------------------------------------

def save_sharded(tensors: dict, dst: str, max_shard_bytes: int, prefix="model"):
    from safetensors.torch import save_file

    def nbytes(t):
        return t.numel() * t.element_size()

    shards, cur, cur_bytes = [], {}, 0
    for name, t in tensors.items():
        b = nbytes(t)
        if cur and cur_bytes + b > max_shard_bytes:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = t.contiguous()
        cur_bytes += b
    if cur:
        shards.append(cur)

    total = len(shards)
    weight_map = {}
    for i, sh in enumerate(shards, 1):
        fn = f"{prefix}-{i:05d}-of-{total:05d}.safetensors" if total > 1 \
            else f"{prefix}.safetensors"
        save_file(sh, os.path.join(dst, fn), metadata={"format": "pt"})
        for k in sh:
            weight_map[k] = fn
        print(f"  [shard {i}/{total}] {fn}  {len(sh)} tensors  "
              f"{sum(nbytes(t) for t in sh.values()) / 2**30:.2f} GiB")
        sh.clear()

    if total > 1:
        with open(os.path.join(dst, f"{prefix}.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": sum(
                nbytes(t) for t in tensors.values())}, "weight_map": weight_map}, f,
                indent=2)
    return total


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BR-MoE 压缩模型目录")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--dense-mode", choices=["fp16", "int3"], default="fp16",
                    help="非专家线性层: fp16=反量化(只需 MoE 插件, 推荐先跑通) / "
                         "int3=保留 3bit(还需要 LinearMethod 插件)")
    ap.add_argument("--max-shard-gib", type=float, default=4.0)
    ap.add_argument("--expert-submodule", default=DEFAULT_EXPERT_SUBMODULE,
                    help="vLLM 侧专家参数所在的子模块名 (默认 routed_experts)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
    print(f"[1/5] 读取 {src}/qmodel.pt ...")
    sd = torch.load(os.path.join(src, "qmodel.pt"), map_location="cpu",
                    weights_only=False)
    print(f"      模块数 {len(sd)}")

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg.get("first_k_dense_replace", 1)
    n_experts = cfg["n_routed_experts"]
    expert_layers = [i for i in range(n_layers) if i >= first_dense]
    print(f"      层数 {n_layers}, 专家层 {expert_layers[0]}..{expert_layers[-1]} "
          f"({len(expert_layers)} 层 × {n_experts} 专家)")

    # ---- 分类 ----
    quant_mods, plain_mods = [], []
    for path, d in sd.items():
        (quant_mods if is_quantized(d) else plain_mods).append(path)
    print(f"[2/5] 量化模块 {len(quant_mods)} / 普通模块 {len(plain_mods)}")

    # 每个专家投影的 group_size 必须一致
    gs_all = Counter()
    for path in quant_mods:
        gs_all[sd[path]["group_size"]] += 1
    print(f"      group_size 分布: {dict(gs_all)}")

    out = {}

    # ---- 专家层 ----
    print("[3/5] 打包专家层 ...")
    for li in expert_layers:
        src_prefix = f"model.layers.{li}.mlp.experts"
        # 输出键名要带 vLLM 的子模块段，见 DEFAULT_EXPERT_SUBMODULE 的说明
        out_prefix = f"model.layers.{li}.mlp.experts.{args.expert_submodule}"

        def get(eid, proj, _li=li):
            return sd[f"model.layers.{_li}.mlp.experts.{eid}.{proj}"]

        eids = list(range(n_experts))
        packed = pack_one_expert_layer(
            get, eids, sd[f"{src_prefix}.0.gate_proj"]["group_size"])
        for k, v in packed.items():
            out[f"{out_prefix}.{k}"] = v
        print(f"      layer {li}: w13_q {tuple(packed['w13_q'].shape)} "
              f"w2_q {tuple(packed['w2_q'].shape)}")

        # 逐层释放源权重, 否则 8.2 GiB 的 qmodel 会一直驻留
        for eid in eids:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                sd.pop(f"{src_prefix}.{eid}.{proj}", None)
        gc.collect()

    # ---- 非专家量化层 ----
    print(f"[4/5] 处理非专家量化层 (dense-mode={args.dense_mode}) ...")
    stat = Counter()
    for path in quant_mods:
        if is_moe_expert(path):
            continue
        if args.dense_mode == "fp16":
            out[f"{path}.weight"] = dequantize(sd[path])
            stat["fp16"] += 1
        else:
            for k, v in pack_one_linear(sd[path]).items():
                out[f"{path}.{k}"] = v
            stat["int3"] += 1
        del sd[path]
    print(f"      {dict(stat)}")

    # ---- 普通层透传 ----
    for path in list(sd):
        d = sd[path]
        if isinstance(d, dict):
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[f"{path}.{k}"] = v.contiguous()
        elif torch.is_tensor(d):
            out[path] = d.contiguous()
    print(f"      普通层透传 {len(plain_mods)} 个")

    total_gib = sum(t.numel() * t.element_size() for t in out.values()) / 2**30
    print(f"[5/5] 输出 {len(out)} 个张量, 合计 {total_gib:.2f} GiB")

    if args.dry_run:
        print("(dry-run, 不写盘)")
        return

    os.makedirs(dst, exist_ok=True)

    # ---- config.json + quantization_config ----
    qcfg = {
        "quant_method": "brmoe_int3",
        "bits": 3,
        "group_size": int(list(gs_all)[0]) if len(gs_all) == 1 else None,
        "expert_layout": {
            "w13_q": "int32 [E, 2I, K//32*3]",
            "w13_s": "fp16 [E, K//gs, 2I]",
            "w13_z": "fp16 [E, K//gs, 2I]",
            "w2_q": "int32 [E, K, I//32*3]",
            "w2_s": "fp16 [E, I//gs, K]",
            "w2_z": "fp16 [E, I//gs, K]",
        },
        "dense_mode": args.dense_mode,
        "dequant": "W = (unpack_int3(W_q) - zero) * scale",
    }
    cfg_out = dict(cfg)
    cfg_out["quantization_config"] = qcfg
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"      config.json (+ quantization_config)")

    # ---- 权重 ----
    save_sharded(out, dst, int(args.max_shard_gib * 2**30))

    # ---- 附属文件 ----
    for fn in ("configuration_deepseek.py", "modeling_deepseek.py",
               "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst, fn))
    print(f"      附属文件已复制 -> {dst}")
    print("完成。运行时记得加 --trust-remote-code (配置里 model_type='deepseek'，"
          "vLLM 靠它选标准 MHA 分支)。")


if __name__ == "__main__":
    main()
