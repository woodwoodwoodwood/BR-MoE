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
_brmoe_utils = _load_module("brmoe_core_utils", os.path.join(_BRMOE_PKG, "core", "utils.py"))
_packing = _load_module(
    "brmoe_int3_packing",
    os.path.join(_BRMOE_PKG, "kernels", "triton_int3", "int3_moe", "packing.py"),
)
BitPack = _bitpack.BitPack
dequantize_UV = _brmoe_utils.dequantize_UV
pack_int3 = _packing.pack_int3
pack_uniform_bit = _packing.pack_uniform_bit
pack_uint1_msb = _packing.pack_uint1_msb
pack_varbit_experts = _packing.pack_varbit_experts

# BR-MoE 的 packing 名 -> 解包函数 (支持 2/3/4/8-bit 混合精度)
UNPACK = {
    "8bit_u8": BitPack.unpack_8bit_u8,
    "4bit_u8": BitPack.unpack_4bit_u8,
    "4bit_32": BitPack.unpack_4bit_32,
    "3bit_32": BitPack.unpack_3bit_32,
    "2bit_u8": BitPack.unpack_2bit_u8,
}

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
    return (".mlp.experts." in path) or (".block_sparse_moe.experts." in path) or (".feed_forward.experts." in path)


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


def dequantize(d: dict, uv=None) -> torch.Tensor:
    """W = (q - z) * s (+ U @ V), 按 group 广播回 [n, k]。用于 --dense-mode fp16。"""
    q, s, z = unpack_quantized(d)
    n, k = q.shape
    gs = int(d["group_size"])
    w = (q.to(torch.float32) - z.repeat_interleave(gs, dim=1).to(torch.float32))
    w = w * s.repeat_interleave(gs, dim=1).to(torch.float32)
    if uv is not None:
        u_dq, v_dq = uv
        if u_dq is not None and v_dq is not None and v_dq.shape[0] > 0:
            w = w + (u_dq.to(torch.float32) @ v_dq.to(torch.float32))
    return w.to(torch.float16)


def extract_uv(path: str, d: dict, compensators: dict | None):
    """从 compensators.pt 或模块字典中还原低秩补偿器 (U_dq [n, r], V_dq [r, k]) fp16。"""
    if isinstance(d, dict) and d.get("U") is not None and d.get("V") is not None:
        return d["U"].to(torch.float16), d["V"].to(torch.float16)
    if not compensators or path not in compensators:
        return None
    uv_q = compensators[path]
    if uv_q is None:
        return None
    n, k = int(d["shape"][0]), int(d["shape"][1])
    comp_dtype = d.get("compensator_dtype", "int3")
    comp_gs = int(d.get("compensator_quant_gs", 64))
    u_scale = uv_q[1]
    rank = int(u_scale.numel() * comp_gs // n)
    if rank <= 0:
        return None
    u_dq, v_dq = dequantize_UV(
        uv_q,
        n=n,
        k=k,
        rank=rank,
        compensator_dtype=comp_dtype,
        compensator_quant_gs=comp_gs,
    )
    return u_dq.to(torch.float16), v_dq.to(torch.float16)


# ---------------------------------------------------------------------------
# 打包: -> vLLM 侧布局
# ---------------------------------------------------------------------------

def pack_one_expert_layer(
    get,
    get_uv,
    expert_ids,
    gs,
    proj_triple=("gate_proj", "up_proj", "down_proj"),
    device="cpu",
):
    """把一层专家的全部门打包成 BR-MoE grouped kernel 的布局 (支持 2/3/4-bit 混合位宽 + CSR 变秩补偿)。

    get(expert, proj) -> {'W_q','scale','zero','group_size','shape','nbits',...}
    get_uv(expert, proj) -> (U_dq [out, r], V_dq [r, in]) or None
    返回: (packed_tensors_dict, layer_spec_dict)
    """
    gate_name, up_name, down_name = proj_triple
    E = len(expert_ids)
    g0 = get(expert_ids[0], gate_name)
    I, K = int(g0["shape"][0]), int(g0["shape"][1])
    gs = int(g0["group_size"])

    q13_list = []
    q2_list = []
    s13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    s2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)
    z13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    z2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)

    expert_bits = []
    expert_ranks = []
    w13_bits_list = []
    w2_bits_list = []
    w13_ranks_list = []
    w2_ranks_list = []

    comp_offsets = [0]
    vg_list, ug_list = [], []
    vu_list, uu_list = [], []
    vd_list, ud_list = [], []
    w1_v_3d, w1_u_3d, w3_v_3d, w3_u_3d = [], [], [], []
    w2_v_3d, w2_u_3d = [], []

    for e, eid in enumerate(expert_ids):
        pq, ps, pz = [], [], []
        eb = 0
        b13_e = 0
        for proj in (gate_name, up_name):
            entry = get(eid, proj)
            b_cur = int(entry.get("nbits", 3))
            eb = max(eb, b_cur)
            b13_e = max(b13_e, b_cur)
            q, s, z = unpack_quantized(entry)
            pq.append(q)
            ps.append(s)
            pz.append(z)
        q13_e = torch.cat(pq, dim=0)               # [2I, K]
        q13_list.append(q13_e)
        s13[e] = torch.cat(ps, dim=0).t()          # [K//gs, 2I]
        z13[e] = torch.cat(pz, dim=0).t()

        d_entry = get(eid, down_name)
        b2_e = int(d_entry.get("nbits", 3))
        eb = max(eb, b2_e)
        qd, sd, zd = unpack_quantized(d_entry)
        q2_list.append(qd)                          # [K, I]
        s2[e] = sd.t()                              # [I//gs, K]
        z2[e] = zd.t()
        del pq, ps, pz

        if eb not in (2, 3, 4):
            raise ValueError(f"MoE 专家不支持位宽 nbits={eb} (仅支持 2/3/4-bit)")
        expert_bits.append(eb)
        w13_bits_list.append(b13_e)
        w2_bits_list.append(b2_e)

        uv_g = get_uv(eid, gate_name)
        uv_u = get_uv(eid, up_name)
        uv_d = get_uv(eid, down_name)
        rg = int(uv_g[1].shape[0]) if uv_g is not None else 0
        ru = int(uv_u[1].shape[0]) if uv_u is not None else 0
        rd = int(uv_d[1].shape[0]) if uv_d is not None else 0
        r13_e = max(rg, ru)
        r2_e = rd
        re = max(r13_e, r2_e)
        expert_ranks.append(re)
        w13_ranks_list.append(r13_e)
        w2_ranks_list.append(r2_e)
        comp_offsets.append(comp_offsets[-1] + re)

        def _pad_uv(uv_pair, out_dim, in_dim, target_r):
            v_pad = torch.zeros((target_r, in_dim), dtype=torch.float16, device=device)
            ut_pad = torch.zeros((target_r, out_dim), dtype=torch.float16, device=device)
            if uv_pair is not None and target_r > 0:
                u_t, v_t = uv_pair
                r_cur = min(target_r, int(v_t.shape[0]))
                v_pad[:r_cur] = v_t[:r_cur].to(device=device, dtype=torch.float16)
                ut_pad[:r_cur] = u_t[:, :r_cur].to(device=device, dtype=torch.float16).t()
            return v_pad, ut_pad

        if re > 0:
            vg, ug = _pad_uv(uv_g, I, K, re)
            vu, uu = _pad_uv(uv_u, I, K, re)
            vd, ud = _pad_uv(uv_d, K, I, re)
            vg_list.append(vg)
            ug_list.append(ug)
            vu_list.append(vu)
            uu_list.append(uu)
            vd_list.append(vd)
            ud_list.append(ud)
        if r13_e > 0:
            vg13, ug13 = _pad_uv(uv_g, I, K, r13_e)
            vu13, uu13 = _pad_uv(uv_u, I, K, r13_e)
            w1_v_3d.append(vg13)
            w1_u_3d.append(ug13.t().contiguous())
            w3_v_3d.append(vu13)
            w3_u_3d.append(uu13.t().contiguous())
        if r2_e > 0:
            vd2, ud2 = _pad_uv(uv_d, K, I, r2_e)
            w2_v_3d.append(vd2)
            w2_u_3d.append(ud2.t().contiguous())

    is_uniform_tensor_bits = (len(set(w13_bits_list)) == 1 and len(set(w2_bits_list)) == 1)
    is_uniform_tensor_ranks = (len(set(w13_ranks_list)) == 1 and len(set(w2_ranks_list)) == 1)
    w13_bits = int(w13_bits_list[0]) if is_uniform_tensor_bits else None
    w2_bits = int(w2_bits_list[0]) if is_uniform_tensor_bits else None
    w13_rank = int(w13_ranks_list[0]) if is_uniform_tensor_ranks else None
    w2_rank = int(w2_ranks_list[0]) if is_uniform_tensor_ranks else None

    total_bits = sum(expert_bits)
    num_2bit = sum(1 for b in expert_bits if b == 2)
    num_3bit = sum(1 for b in expert_bits if b == 3)
    num_4bit = sum(1 for b in expert_bits if b == 4)
    total_rank = comp_offsets[-1]
    max_rank = max(expert_ranks) if expert_ranks else 0

    if is_uniform_tensor_bits:
        q13 = torch.stack(q13_list, dim=0)
        q2 = torch.stack(q2_list, dim=0)
        packed = {
            "w13_q": pack_uniform_bit(q13.reshape(E * 2 * I, K), w13_bits, transposed=False)
                     .reshape(E, 2 * I, (K // 32) * w13_bits).to(torch.int32),
            "w13_s": s13, "w13_z": z13,
            "w2_q": pack_uniform_bit(q2.reshape(E * K, I), w2_bits, transposed=False)
                    .reshape(E, K, (I // 32) * w2_bits).to(torch.int32),
            "w2_s": s2, "w2_z": z2,
        }
    else:
        w13_q, bit_offsets = pack_varbit_experts(q13_list, w13_bits_list, transposed=False)
        w2_q, _ = pack_varbit_experts(q2_list, w2_bits_list, transposed=False)
        packed = {
            "w13_q": w13_q.to(torch.int32),
            "w13_s": s13, "w13_z": z13,
            "w2_q": w2_q.to(torch.int32),
            "w2_s": s2, "w2_z": z2,
            "bit_offsets": bit_offsets.to(torch.int32),
        }

    if total_rank > 0:
        if is_uniform_tensor_ranks:
            if w13_rank > 0 and len(w1_v_3d) == E:
                packed["w1_v"] = torch.stack(w1_v_3d, dim=0).contiguous()  # [E, r_13, K]
                packed["w1_u"] = torch.stack(w1_u_3d, dim=0).contiguous()  # [E, I, r_13]
                packed["w3_v"] = torch.stack(w3_v_3d, dim=0).contiguous()  # [E, r_13, K]
                packed["w3_u"] = torch.stack(w3_u_3d, dim=0).contiguous()  # [E, I, r_13]
            if w2_rank > 0 and len(w2_v_3d) == E:
                packed["w2_v"] = torch.stack(w2_v_3d, dim=0).contiguous()  # [E, r_2, I]
                packed["w2_u"] = torch.stack(w2_u_3d, dim=0).contiguous()  # [E, K, r_2]
        else:
            packed["comp_offsets"] = torch.tensor(comp_offsets, dtype=torch.int32, device=device)
            packed["w13_v"] = torch.stack([torch.cat(vg_list, dim=0), torch.cat(vu_list, dim=0)], dim=0).contiguous()
            packed["w13_u"] = torch.stack([torch.cat(ug_list, dim=0), torch.cat(uu_list, dim=0)], dim=0).contiguous()
            packed["w2_v"] = torch.cat(vd_list, dim=0).contiguous()
            packed["w2_u"] = torch.cat(ud_list, dim=0).contiguous()

    layer_spec = {
        "group_size": gs,
        "num_experts": E,
        "layout": "int3" if is_uniform_tensor_bits else "varbit",
        "w13_bits": w13_bits,
        "w2_bits": w2_bits,
        "w13_rank": w13_rank if is_uniform_tensor_ranks else None,
        "w2_rank": w2_rank if is_uniform_tensor_ranks else None,
        "total_expert_bits": 0 if is_uniform_tensor_bits else int(total_bits),
        "num_2bit_experts": num_2bit,
        "num_3bit_experts": num_3bit,
        "num_4bit_experts": num_4bit,
        "total_comp_rank": 0 if is_uniform_tensor_ranks else total_rank,
        "max_comp_rank": max_rank,
        "bits_hist": {str(k): int(v) for k, v in sorted(Counter(expert_bits).items())},
        "ranks_hist": {str(k): int(v) for k, v in sorted(Counter(expert_ranks).items())},
    }
    return packed, layer_spec


def _quantize_asymmetric_int3(w: torch.Tensor, gs: int):
    """将 fp16 权重 [n, k] 非对称量化为 3-bit (q [n, k], s [n, k//gs], z [n, k//gs])。"""
    n, k = w.shape
    wf = w.to(torch.float32).reshape(n, k // gs, gs)
    w_min = wf.amin(dim=-1, keepdim=True)
    w_max = wf.amax(dim=-1, keepdim=True)
    s = ((w_max - w_min) / 7.0).clamp_min(1e-8)
    z = torch.clamp(torch.round(-w_min / s), 0.0, 7.0)
    q = torch.clamp(torch.round(wf / s + z), 0.0, 7.0).to(torch.int32).reshape(n, k)
    return q, s.squeeze(-1).to(torch.float16), z.squeeze(-1).to(torch.float16)


def pack_one_linear(d: dict, uv=None):
    """单个量化线性层 -> int3 布局。

    若原层是 3-bit 且无低秩补偿器，直接无损重排为 K-major；
    若原层为 2/4/8-bit 或带有低秩补偿器 (U, V)，先合成精确权重 W + U@V 再重打包为 3-bit。
    """
    nbits = int(d.get("nbits", 3))
    gs = int(d["group_size"])
    if nbits == 3 and uv is None:
        q, s, z = unpack_quantized(d)          # q[n,k] s/z[n,groups]
    else:
        w_total = dequantize(d, uv=uv)
        q, s, z = _quantize_asymmetric_int3(w_total, gs)
    qw = pack_int3(q, transposed=True)         # [k//32*3, n]
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

    comp_path = os.path.join(src, "compensators.pt")
    compensators = None
    if os.path.exists(comp_path):
        print(f"      读取低秩补偿器 {comp_path} ...")
        compensators = torch.load(comp_path, map_location="cpu", weights_only=False)
        non_empty_comp = sum(1 for v in compensators.values() if v is not None)
        print(f"      补偿器条目数 {len(compensators)} (非空 rank>0: {non_empty_comp})")

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config") or cfg.get("language_config") or cfg
    n_layers = int(text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers"))
    first_dense = int(text_cfg.get("first_k_dense_replace", 0))
    mlp_only_set = set(text_cfg.get("mlp_only_layers") or [])
    n_experts = int(
        text_cfg.get("n_routed_experts")
        or text_cfg.get("num_experts")
        or text_cfg.get("num_local_experts")
        or text_cfg.get("moe_num_experts")
    )
    expert_layers = [
        i for i in range(n_layers) if (i >= first_dense and i not in mlp_only_set)
    ]
    print(f"      层数 {n_layers}, 专家层 {expert_layers[0]}..{expert_layers[-1]} "
          f"({len(expert_layers)} 层 × {n_experts} 专家)")

    # ---- 分类 ----
    quant_mods, plain_mods = [], []
    for path, d in sd.items():
        (quant_mods if is_quantized(d) else plain_mods).append(path)
    print(f"[2/5] 量化模块 {len(quant_mods)} / 普通模块 {len(plain_mods)}")

    gs_all = Counter()
    bits_all = Counter()
    for path in quant_mods:
        gs_all[int(sd[path]["group_size"])] += 1
        bits_all[int(sd[path].get("nbits", 3))] += 1
    print(f"      group_size 分布: {dict(gs_all)} | nbits 分布: {dict(bits_all)}")

    out = {}
    moe_layer_specs = {}

    # ---- 专家层 ----
    print("[3/5] 打包专家层 (支持 2/3/4-bit 混合精度 + CSR 变秩低秩补偿) ...")
    for li in expert_layers:
        # 自动探测该层的专家前缀 (支持 model.layers, model.language_model.layers, mlp.experts, block_sparse_moe.experts)
        cand_prefixes = [
            f"model.layers.{li}.mlp.experts",
            f"model.language_model.layers.{li}.mlp.experts",
            f"language_model.model.layers.{li}.mlp.experts",
            f"model.layers.{li}.block_sparse_moe.experts",
        ]
        src_prefix = None
        proj_triple = ("gate_proj", "up_proj", "down_proj")
        for cp in cand_prefixes:
            if f"{cp}.0.gate_proj" in sd:
                src_prefix = cp
                proj_triple = ("gate_proj", "up_proj", "down_proj")
                break
            if f"{cp}.0.w1" in sd:
                src_prefix = cp
                proj_triple = ("w1", "w3", "w2")
                break
        if src_prefix is None:
            print(f"      layer {li}: 未找到专家投影，作为稠密层跳过")
            continue

        out_prefix = f"{src_prefix}.{args.expert_submodule}"

        def get(eid, proj, _sp=src_prefix):
            return sd[f"{_sp}.{eid}.{proj}"]

        def get_uv(eid, proj, _sp=src_prefix):
            mpath = f"{_sp}.{eid}.{proj}"
            return extract_uv(mpath, sd[mpath], compensators)

        eids = list(range(n_experts))
        first_proj = proj_triple[0]
        packed, lspec = pack_one_expert_layer(
            get,
            get_uv,
            eids,
            sd[f"{src_prefix}.0.{first_proj}"]["group_size"],
            proj_triple=proj_triple,
        )
        moe_layer_specs[str(li)] = lspec
        for k, v in packed.items():
            out[f"{out_prefix}.{k}"] = v
        print(
            f"      layer {li}: w13_q {tuple(packed['w13_q'].shape)} "
            f"w2_q {tuple(packed['w2_q'].shape)} | "
            f"bits={lspec['bits_hist']} ranks={lspec['ranks_hist']} "
            f"(4bit_exps={lspec['num_4bit_experts']}, total_rank={lspec['total_comp_rank']})"
        )

        # 逐层释放源权重, 否则 qmodel/compensators 会一直驻留
        for eid in eids:
            for proj in proj_triple:
                mpath = f"{src_prefix}.{eid}.{proj}"
                sd.pop(mpath, None)
                if compensators is not None:
                    compensators.pop(mpath, None)
        gc.collect()

    # ---- 非专家量化层 ----
    print(f"[4/5] 处理非专家量化层 (dense-mode={args.dense_mode}) ...")
    stat = Counter()
    for path in quant_mods:
        if is_moe_expert(path):
            continue
        entry = sd[path]
        uv = extract_uv(path, entry, compensators)
        if isinstance(entry, dict) and torch.is_tensor(entry.get("bias")):
            out[f"{path}.bias"] = entry["bias"].to(torch.float16).contiguous()
        if args.dense_mode == "fp16":
            out[f"{path}.weight"] = dequantize(entry, uv=uv)
            stat["fp16"] += 1
        else:
            for k, v in pack_one_linear(entry, uv=uv).items():
                out[f"{path}.{k}"] = v
            stat["int3"] += 1
        del sd[path]
        if compensators is not None:
            compensators.pop(path, None)
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

    layer_configs = {}
    for li_str, lspec in moe_layer_specs.items():
        if lspec.get("w13_bits") is not None and lspec.get("w2_bits") is not None:
            layer_configs[f"model.layers.{li_str}.mlp.experts"] = {
                "w13_bits": int(lspec["w13_bits"]),
                "w13_rank": int(lspec.get("w13_rank") or 0),
                "w2_bits": int(lspec["w2_bits"]),
                "w2_rank": int(lspec.get("w2_rank") or 0),
            }

    # ---- config.json + quantization_config ----
    bits_val = int(list(bits_all)[0]) if len(bits_all) == 1 else "mixed"
    qcfg = {
        "quant_method": "brmoe_int3",
        "bits": bits_val,
        "supported_bits": sorted(int(b) for b in bits_all.keys()),
        "group_size": int(list(gs_all)[0]) if len(gs_all) == 1 else 64,
        "compensator_dtype": "fp16",
        "layer_configs": layer_configs,
        "expert_layout": {
            "w13_q": "int32 [E, 2I, (K//32)*w13_bits]",
            "w13_s": "fp16 [E, K//gs, 2I]",
            "w13_z": "fp16 [E, K//gs, 2I]",
            "w2_q": "int32 [E, K, (I//32)*w2_bits]",
            "w2_s": "fp16 [E, I//gs, K]",
            "w2_z": "fp16 [E, I//gs, K]",
            "w1_v / w3_v": "optional fp16 [E, r_13, K] (Per-3D-Tensor compensator)",
            "w1_u / w3_u": "optional fp16 [E, I, r_13] (Per-3D-Tensor compensator)",
            "w2_v": "optional fp16 [E, r_2, I] (Per-3D-Tensor compensator) or [R_tot, I] (CSR)",
            "w2_u": "optional fp16 [E, K, r_2] (Per-3D-Tensor compensator) or [R_tot, K] (CSR)",
        },
        "moe_layer_specs": moe_layer_specs,
        "dense_mode": args.dense_mode,
        "dequant": "W = (unpack_uniform_bit(W_q) - zero) * scale + U @ V",
    }
    cfg_out = dict(cfg)
    cfg_out["quantization_config"] = qcfg
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"      config.json (+ quantization_config)")

    # ---- 权重 ----
    save_sharded(out, dst, int(args.max_shard_gib * 2**30))

    # ---- 附属文件 ----
    for fn in os.listdir(src):
        if fn in ("qmodel.pt", "compensators.pt", "ranks.json", "config.json"):
            continue
        if fn.endswith((".py", ".json", ".txt", ".model", ".tiktoken", ".jinja")):
            s = os.path.join(src, fn)
            if os.path.isfile(s):
                shutil.copy(s, os.path.join(dst, fn))
    print(f"      附属文件已复制 -> {dst}")
    print("完成。运行时记得加 --trust-remote-code (配置里 model_type='deepseek'，"
          "vLLM 靠它选标准 MHA 分支)。")


if __name__ == "__main__":
    main()
