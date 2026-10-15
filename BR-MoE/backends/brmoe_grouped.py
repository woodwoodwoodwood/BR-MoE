"""BR-MoE 的 grouped int3 MoE 后端（实验性）。

把 MoE 层的逐专家循环换成 **一个 launch 覆盖全部专家** 的 grouped GEMM：

    每层 = 2 次 grouped GEMM + 融合 silu(gate)*up
    y = Σ_k w_k · ( silu(x @ W1[e_k]ᵀ) * (x @ W3[e_k]ᵀ) ) @ W2[e_k]ᵀ

实现复用 `kernels/triton_int3/int3_moe`（免编译、triton 实现），与逐专家循环相比
主要是省掉每 token ~500 次的小 kernel launch。

支持特性：
  * 支持混合位宽 (2-bit / 3-bit / 4-bit) 专家：2/3-bit 专家直接打包入 3-bit 主平面，
    4-bit 专家通过稀疏 1-bit MSB 扩展平面 100% 无损恢复
  * 支持混合低秩补偿 (rank = 0, 16, 32, 64, 128, 256, 512, 1024)：通过 CSR 变秩切片
    (comp_offsets, w13_v, w13_u, w2_v, w2_u) 在 silu_mul 前后精确叠加低秩补偿项
  * N(=2I) 与 K 必须是 64 的倍数；K 是 32 的倍数；group_size 是 32 的倍数
  * 首次启用需要把权重解包→重新打包，结果缓存到磁盘

用法：
    from BR_MoE.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend="brmoe_grouped")
"""

import os
import sys
import time

import torch

_TRITON_MOE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "triton_int3"
)


def _import_triton_moe():
    if _TRITON_MOE_DIR not in sys.path:
        sys.path.insert(0, _TRITON_MOE_DIR)
    from int3_moe.ops import fused_moe_int3, pack_moe_weights

    return fused_moe_int3, pack_moe_weights


def _log(verbose, *a):
    if verbose:
        print("[brmoe_grouped]", *a, flush=True)


# ---------------------------------------------------------------------------
# 形状 / 补偿器检查
# ---------------------------------------------------------------------------

def _layer_meta(layer):
    meta = getattr(layer, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _get_expert_proj_triple(exp):
    """Return (gate_name, up_name, down_name) for an expert module."""
    if hasattr(exp, "gate_proj") and hasattr(exp, "up_proj") and hasattr(exp, "down_proj"):
        return ("gate_proj", "up_proj", "down_proj")
    if hasattr(exp, "w1") and hasattr(exp, "w3") and hasattr(exp, "w2"):
        return ("w1", "w3", "w2")
    return ("gate_proj", "up_proj", "down_proj")


def check_layer_supported(experts, verbose=True):
    """检查一层 MoE 是否可以被 grouped 后端接管。返回 (ok, reason, info)。"""
    if len(experts) == 0:
        return False, "no experts", {}

    gate_name, up_name, down_name = _get_expert_proj_triple(experts[0])
    g0 = getattr(experts[0], gate_name, None)
    if g0 is None:
        return False, f"缺少 {gate_name}", {}
    out_feat = getattr(g0, "out_features", None)
    in_feat = getattr(g0, "in_features", None)
    if out_feat is None or in_feat is None:
        return False, "layer 尚未加载 (缺少 in/out_features)", {}

    I, K = int(out_feat), int(in_feat)
    meta = _layer_meta(g0)
    gs = meta.get("group_size", None)
    nbits = meta.get("nbits", None)

    expert_bits = []
    expert_ranks = []
    w13_bits_list = []
    w2_bits_list = []
    w13_ranks_list = []
    w2_ranks_list = []
    for exp in experts:
        eb = 0
        er = 0
        b13_e = 0
        b2_e = 0
        r13_e = 0
        r2_e = 0
        for proj in (gate_name, up_name, down_name):
            lay = getattr(exp, proj, None)
            if lay is None:
                return False, f"缺少 {proj}", {}
            lmeta = _layer_meta(lay)
            b = lmeta.get("nbits", None)
            if b not in (2, 3, 4):
                return False, f"{proj} nbits={b} 不在支持的 {2, 3, 4} 范围内", {}
            if lmeta.get("group_size", None) != gs:
                return False, f"{proj} group_size 不一致", {}
            eb = max(eb, int(b))
            if proj in (gate_name, up_name):
                b13_e = max(b13_e, int(b))
            else:
                b2_e = int(b)
            u = getattr(lay, "U", None)
            v = getattr(lay, "V", None)
            if u is not None and v is not None:
                r_cur = int(v.shape[0])
                er = max(er, r_cur)
                if proj in (gate_name, up_name):
                    r13_e = max(r13_e, r_cur)
                else:
                    r2_e = r_cur
        expert_bits.append(eb)
        expert_ranks.append(er)
        w13_bits_list.append(b13_e)
        w2_bits_list.append(b2_e)
        w13_ranks_list.append(r13_e)
        w2_ranks_list.append(r2_e)

    is_uniform_tensor_bits = (len(set(w13_bits_list)) == 1 and len(set(w2_bits_list)) == 1)
    is_uniform_tensor_ranks = (len(set(w13_ranks_list)) == 1 and len(set(w2_ranks_list)) == 1)
    info = dict(
        E=len(experts),
        I=I,
        K=K,
        group_size=gs,
        nbits=nbits if len(set(expert_bits)) == 1 else "mixed",
        expert_bits=expert_bits,
        w13_bits_list=w13_bits_list,
        w2_bits_list=w2_bits_list,
        is_uniform_tensor_bits=is_uniform_tensor_bits,
        w13_bits=w13_bits_list[0] if is_uniform_tensor_bits else None,
        w2_bits=w2_bits_list[0] if is_uniform_tensor_bits else None,
        num_4bit_experts=sum(1 for b in expert_bits if b == 4),
        expert_ranks=expert_ranks,
        w13_ranks_list=w13_ranks_list,
        w2_ranks_list=w2_ranks_list,
        is_uniform_tensor_ranks=is_uniform_tensor_ranks,
        w13_rank=w13_ranks_list[0] if is_uniform_tensor_ranks else None,
        w2_rank=w2_ranks_list[0] if is_uniform_tensor_ranks else None,
        total_comp_rank=sum(expert_ranks),
        max_comp_rank=max(expert_ranks) if expert_ranks else 0,
        proj_triple=(gate_name, up_name, down_name),
    )
    # 是否带 per-group 浮点零点: 有则走无损搬运路径, 否则退回对称重新量化
    info["has_zero"] = ("zero" in meta) and (meta.get("zero") is not None)
    if "scale" not in meta or meta.get("scale") is None:
        return False, "缺少 scale", info
    if gs is None or (gs % 32) != 0:
        return False, f"group_size={gs} 必须是 32 的倍数", info
    if (2 * I) % 64 != 0 or K % 64 != 0:
        return False, f"2I={2 * I} / K={K} 必须是 64 的倍数", info
    if K % 32 != 0 or I % 32 != 0:
        return False, f"K={K} / I={I} 必须是 32 的倍数", info
    return True, "ok", info


# ---------------------------------------------------------------------------
# 权重打包
# ---------------------------------------------------------------------------

def _extract_quantized(lay, gs):
    """取 BRMoELinear 的量化三元组: (int 值 [n,k], scale [n,k//gs], zero [n,k//gs])。

    BR-MoE 的语义: W = (q - z) * s, q ∈ [0, 2^nbits - 1]

    存储布局 (已用 bench/diag_quant_shapes.py 在真实 checkpoint 上确认):
      * W_q       : (n*k/gs/10, gs) —— 量化时 reshape([-1, gs]) 之后没有再 reshape 回 (n,k)
      * scale/zero: (n*k/gs, 1)    —— 逐 (输出通道, 组)
      * unpack 会 padding 到 10 的倍数, 需按 n*k/gs 截断
    """
    from BR_MoE.core.quantize import Quantizer

    meta = lay.meta
    n, k = int(meta["shape"][0]), int(meta["shape"][1])
    groups = k // gs
    rows = n * groups
    q = Quantizer.unpack[meta["packing"]](lay.W_q, dtype=torch.int32)[:rows].reshape(n, k)
    s = meta["scale"].reshape(n, groups).to(torch.float16)
    z = meta["zero"].reshape(n, groups).to(torch.float16)
    return q, s, z


def _pack_one_layer_lossless(experts, info, device, verbose=True):
    """无损打包: 直接搬运 BRMoE 的 2/3/4-bit 量化值 + 每组 scale/zero + 低秩补偿器。

    * Per-3D-Tensor 模式 (is_uniform_tensor_bits=True, 即同层 w13 共享统一位宽 b_13 in {2,3,4},
      w2 共享统一位宽 b_2 in {2,3,4}):
        - w13_q 打包为规整 3D 张量 [E, 2I, (K // 32) * b_13], 单一静态大 Kernel (W_BITS=b_13) 发射!
        - w2_q  打包为规整 3D 张量 [E, K, (I // 32) * b_2],  单一静态大 Kernel (W_BITS=b_2) 发射!
        - 若同层补偿秩一致 (is_uniform_tensor_ranks=True): 直接堆叠为规整 3D 补偿矩阵
          w1_v/w3_v [E, r_13, K], w1_u/w3_u [E, I, r_13], w2_v [E, r_2, I], w2_u [E, K, r_2]。
    * Per-Expert 异构兜底模式 (同层内不同专家位宽不同):
        - 使用 CSR 位平面 3D 张量 w13_q [B_tot, 2I, K//32], w2_q [B_tot, K, I//32] + bit_offsets [E+1]。
    """
    from int3_moe.packing import pack_uniform_bit, pack_varbit_experts

    E, I, K, gs = info["E"], info["I"], info["K"], info["group_size"]

    q13_list = []
    q2_list = []
    s13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    s2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)
    z13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    z2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)

    gate_name, up_name, down_name = info.get("proj_triple", ("gate_proj", "up_proj", "down_proj"))
    expert_bits = info.get("expert_bits", [3] * E)
    expert_ranks = info.get("expert_ranks", [0] * E)
    w13_bits_list = info.get("w13_bits_list", expert_bits)
    w2_bits_list = info.get("w2_bits_list", expert_bits)
    is_uniform_tensor_bits = info.get(
        "is_uniform_tensor_bits",
        len(set(w13_bits_list)) == 1 and len(set(w2_bits_list)) == 1,
    )
    w13_ranks_list = info.get("w13_ranks_list", expert_ranks)
    w2_ranks_list = info.get("w2_ranks_list", expert_ranks)
    is_uniform_tensor_ranks = info.get(
        "is_uniform_tensor_ranks",
        len(set(w13_ranks_list)) == 1 and len(set(w2_ranks_list)) == 1,
    )
    total_rank = sum(expert_ranks)
    max_rank = max(expert_ranks) if expert_ranks else 0

    comp_offsets = [0]
    vg_list, ug_list = [], []
    vu_list, uu_list = [], []
    vd_list, ud_list = [], []
    w1_v_3d, w1_u_3d, w3_v_3d, w3_u_3d = [], [], [], []
    w2_v_3d, w2_u_3d = [], []

    for e, exp in enumerate(experts):
        pq, ps, pz = [], [], []
        for proj in (gate_name, up_name):
            q, s, z = _extract_quantized(getattr(exp, proj), gs)
            pq.append(q.to(device))
            ps.append(s.to(device))
            pz.append(z.to(device))
        q13_e = torch.cat(pq, dim=0)               # [2I, K]
        q13_list.append(q13_e)
        s13[e] = torch.cat(ps, dim=0).t()          # [K//gs, 2I]
        z13[e] = torch.cat(pz, dim=0).t()

        qd, sd, zd = _extract_quantized(getattr(exp, down_name), gs)
        qd = qd.to(device)
        q2_list.append(qd)                         # [K, I]
        s2[e] = sd.to(device).t()                  # [I//gs, K]
        z2[e] = zd.to(device).t()

        re = expert_ranks[e]
        comp_offsets.append(comp_offsets[-1] + re)
        if re > 0:
            def _get_uv(proj_name, out_dim, in_dim, target_r):
                lay = getattr(exp, proj_name)
                u = getattr(lay, "U", None)
                v = getattr(lay, "V", None)
                v_pad = torch.zeros((target_r, in_dim), dtype=torch.float16, device=device)
                ut_pad = torch.zeros((target_r, out_dim), dtype=torch.float16, device=device)
                if u is not None and v is not None and v.shape[0] > 0:
                    r_cur = min(target_r, int(v.shape[0]))
                    v_pad[:r_cur] = v[:r_cur].to(device=device, dtype=torch.float16)
                    ut_pad[:r_cur] = u[:, :r_cur].to(device=device, dtype=torch.float16).t()
                return v_pad, ut_pad

            vg, ug = _get_uv(gate_name, I, K, re)
            vu, uu = _get_uv(up_name, I, K, re)
            vd, ud = _get_uv(down_name, K, I, re)
            vg_list.append(vg)
            ug_list.append(ug)
            vu_list.append(vu)
            uu_list.append(uu)
            vd_list.append(vd)
            ud_list.append(ud)
            if is_uniform_tensor_ranks:
                r13 = w13_ranks_list[0]
                r2 = w2_ranks_list[0]
                if r13 > 0:
                    vg13, ug13 = _get_uv(gate_name, I, K, r13)
                    vu13, uu13 = _get_uv(up_name, I, K, r13)
                    w1_v_3d.append(vg13)
                    w1_u_3d.append(ug13.t().contiguous())
                    w3_v_3d.append(vu13)
                    w3_u_3d.append(uu13.t().contiguous())
                if r2 > 0:
                    vd2, ud2 = _get_uv(down_name, K, I, r2)
                    w2_v_3d.append(vd2)
                    w2_u_3d.append(ud2.t().contiguous())

    if is_uniform_tensor_bits:
        b13 = int(w13_bits_list[0])
        b2 = int(w2_bits_list[0])
        q13 = torch.stack(q13_list, dim=0)
        q2 = torch.stack(q2_list, dim=0)
        out = {
            "w13_q": pack_uniform_bit(q13.reshape(E * 2 * I, K), b13, transposed=False)
            .reshape(E, 2 * I, (K // 32) * b13),
            "w2_q": pack_uniform_bit(q2.reshape(E * K, I), b2, transposed=False)
            .reshape(E, K, (I // 32) * b2),
            "w13_bits": b13,
            "w2_bits": b2,
            "s13": s13, "s2": s2, "z13": z13, "z2": z2,
            "group_size": gs, "layout": "int3", "w_transposed": False,
            "lossless": True,
        }
    else:
        w13_q, bit_offsets = pack_varbit_experts(q13_list, w13_bits_list, transposed=False)
        w2_q, _ = pack_varbit_experts(q2_list, w2_bits_list, transposed=False)
        out = {
            "w13_q": w13_q,
            "w2_q": w2_q,
            "bit_offsets": bit_offsets,
            "s13": s13, "s2": s2, "z13": z13, "z2": z2,
            "group_size": gs, "layout": "varbit", "w_transposed": False,
            "lossless": True,
        }
    if total_rank > 0:
        if is_uniform_tensor_ranks:
            if w1_v_3d:
                out["w1_v"] = torch.stack(w1_v_3d, dim=0).contiguous()  # [E, r_13, K]
                out["w1_u"] = torch.stack(w1_u_3d, dim=0).contiguous()  # [E, I, r_13]
                out["w3_v"] = torch.stack(w3_v_3d, dim=0).contiguous()  # [E, r_13, K]
                out["w3_u"] = torch.stack(w3_u_3d, dim=0).contiguous()  # [E, I, r_13]
                out["w13_rank"] = int(w13_ranks_list[0])
            if w2_v_3d:
                out["w2_v"] = torch.stack(w2_v_3d, dim=0).contiguous()  # [E, r_2, I]
                out["w2_u"] = torch.stack(w2_u_3d, dim=0).contiguous()  # [E, K, r_2]
                out["w2_rank"] = int(w2_ranks_list[0])
        else:
            out["comp_offsets"] = torch.tensor(comp_offsets, dtype=torch.int32, device=device)
            out["w13_v"] = torch.stack([torch.cat(vg_list, dim=0), torch.cat(vu_list, dim=0)], dim=0).contiguous()
            out["w13_u"] = torch.stack([torch.cat(ug_list, dim=0), torch.cat(uu_list, dim=0)], dim=0).contiguous()
            out["w2_v"] = torch.cat(vd_list, dim=0).contiguous()
            out["w2_u"] = torch.cat(ud_list, dim=0).contiguous()
            out["max_comp_rank"] = int(max_rank)
    return out


def _pack_one_layer(experts, info, device, verbose=True):
    """打包一层专家权重: 优先无损路径, 缺 zero 时退回 fp16 重量化(有二次量化误差)。"""
    if info.get("has_zero", False):
        try:
            return _pack_one_layer_lossless(experts, info, device, verbose=verbose)
        except Exception as e:
            _log(verbose, f"  无损打包失败({type(e).__name__}: {e}), 退回 fp16 重量化")

    _, pack_moe_weights = _import_triton_moe()
    E, I, K, gs = info["E"], info["I"], info["K"], info["group_size"]
    gate_name, up_name, down_name = info.get("proj_triple", ("gate_proj", "up_proj", "down_proj"))
    w13 = torch.empty(E, 2 * I, K, dtype=torch.float16, device=device)
    w2 = torch.empty(E, K, I, dtype=torch.float16, device=device)
    for e, exp in enumerate(experts):
        # 注意: dequantize() 会修改 meta (删除 zero/scale), 每层只能调用一次
        w13[e] = torch.cat([getattr(exp, gate_name).dequantize(), getattr(exp, up_name).dequantize()], dim=0)
        w2[e] = getattr(exp, down_name).dequantize()
    return pack_moe_weights(w13, w2, gs)


def _to_device(packed, device):
    out = {}
    for k, v in packed.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _to_cpu(packed):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in packed.items()}


# ---------------------------------------------------------------------------
# 数值自检：grouped vs 原 BRMoELinear 逐专家
# ---------------------------------------------------------------------------

@torch.no_grad()
def _verify_layer(experts, packed, info, device, topk, verbose=True, tol=0.05):
    """用同一份权重对比 grouped 与逐专家 BRMoELinear 的输出。"""
    fused_moe_int3, _ = _import_triton_moe()
    E, I, K = info["E"], info["I"], info["K"]

    M = 8
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(M, K, generator=g) * 0.1).half().to(device)
    ti = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(M)]).to(device)
    tw = torch.full((M, topk), 1.0 / topk, dtype=torch.float16, device=device)

    y_grp = fused_moe_int3(x, tw, ti, packed, fast=True).float()

    y_ref = torch.zeros(M, K, dtype=torch.float32, device=device)
    gate_name, up_name, down_name = info.get("proj_triple", ("gate_proj", "up_proj", "down_proj"))
    for e in range(E):
        hit = (ti == e)
        rows = hit.any(dim=1)
        if not bool(rows.any()):
            continue
        xr = x[rows]
        h = torch.cat([getattr(experts[e], gate_name).matmul(xr), getattr(experts[e], up_name).matmul(xr)], dim=-1)
        gg, uu = h[:, :I], h[:, I:]
        act = (gg * torch.sigmoid(gg)) * uu
        y = getattr(experts[e], down_name).matmul(act)
        for t in range(topk):
            m = hit[rows, t]
            y_ref[rows] += torch.where(m[:, None], y.float() * tw[rows, t:t + 1],
                                       torch.zeros_like(y_ref[rows]))

    err = (y_grp - y_ref).abs().max().item()
    ref_scale = y_ref.abs().max().item()
    ok = err <= tol
    _log(verbose, f"  [verify] max|grouped - per-expert| = {err:.4f} "
                  f"(ref max={ref_scale:.3f}, tol={tol}) -> {'OK' if ok else 'FAIL'}")
    return ok, err


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def patch_moe_to_grouped(model, save_dir=None, cache_name="brmoe_grouped_cache.pt",
                         topk=None, verify=True, free_original=True, verbose=True,
                         limit=None):
    """把所有 MoE 层的前向替换成 grouped int3 kernel。

    Args:
        model: 已 from_compressed 加载的模型 (expert 为 BRMoELinear)
        save_dir: 打包结果缓存目录 (默认用 model.save_dir)
        verify: 是否做数值自检 (强烈建议开)
        free_original: 成功后释放原来的 int3 权重 (省显存)
    Returns:
        被接管的层数
    """
    fused_moe_int3, _ = _import_triton_moe()

    moe_layers = [
        (n, m)
        for n, m in model.named_modules()
        if hasattr(m, "experts") and hasattr(m, "gate") and isinstance(getattr(m, "experts"), torch.nn.ModuleList)
    ]
    if not moe_layers:
        raise RuntimeError("未找到包含 experts + gate 的 MoE 层, brmoe_grouped 后端不适用")

    _log(verbose, f"发现 {len(moe_layers)} 个 MoE 层")

    save_dir = save_dir or getattr(model, "save_dir", None)
    cache_path = os.path.join(save_dir, cache_name) if save_dir else None
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location="cpu")
            _log(verbose, f"命中打包缓存 {cache_path} ({len(cache)} 层)")
        except Exception as e:
            _log(verbose, f"缓存读取失败({e}), 将重新打包")
            cache = {}

    t_all = time.time()
    patched, skipped = 0, []
    new_cache = dict(cache)

    cfg_topk = getattr(getattr(model, "config", None), "num_experts_per_tok", None)

    for idx, (name, moe) in enumerate(moe_layers):
        if limit is not None and idx >= limit:
            _log(verbose, f"limit={limit}: 只接管前 {limit} 层(用于快速验证)")
            break
        experts = moe.experts
        ok, reason, info = check_layer_supported(experts, verbose=verbose)
        if not ok:
            skipped.append((name, reason))
            _log(verbose, f"[{idx}] {name}: 跳过 ({reason})")
            continue

        n_topk = (
            topk
            or getattr(moe, "num_experts_per_tok", None)
            or getattr(moe, "top_k", None)
            or cfg_topk
            or 6
        )
        gate_name, up_name, down_name = info.get("proj_triple", ("gate_proj", "up_proj", "down_proj"))
        device = getattr(experts[0], gate_name).W_q.device

        if name in cache:
            packed = _to_device(cache[name], device)
        else:
            t0 = time.time()
            packed = _pack_one_layer(experts, info, device, verbose=verbose)
            _log(verbose, f"[{idx}] {name}: 打包完成 {time.time() - t0:.1f}s "
                          f"(E={info['E']}, I={info['I']}, K={info['K']}, gs={info['group_size']})")
            new_cache[name] = _to_cpu(packed)

        if verify:
            ok_v, err = _verify_layer(experts, packed, info, device, n_topk, verbose=verbose)
            if not ok_v:
                skipped.append((name, f"数值自检未通过 (err={err:.4f})"))
                _log(verbose, f"[{idx}] {name}: 自检未通过, 保留原后端")
                continue

        # ---- 替换前向 (支持 DeepseekMoE.moe_infer, Unfused3DExpertsModule.forward, 以及标准 SparseMoeBlock.forward) ----
        if hasattr(moe, "moe_infer"):
            def grouped_moe_infer(self, x, flat_expert_indices, flat_expert_weights,
                                  _packed=packed, _topk=n_topk):
                n_tok = x.shape[0]
                ti = flat_expert_indices.view(n_tok, _topk)
                tw = flat_expert_weights.view(n_tok, _topk)
                return fused_moe_int3(x, tw, ti, _packed, fast=True)

            moe.moe_infer = grouped_moe_infer.__get__(moe, type(moe))
        elif type(experts).__name__ == "Unfused3DExpertsModule":
            def grouped_3d_experts_forward(self, hidden_states, top_k_index=None, top_k_weights=None, *a,
                                           _packed=packed, **kw):
                orig_shape = hidden_states.shape
                x2 = hidden_states.view(-1, orig_shape[-1])
                out = fused_moe_int3(x2, top_k_weights.to(x2.dtype), top_k_index.to(torch.int32), _packed, fast=True)
                return out.view(orig_shape)

            experts.forward = grouped_3d_experts_forward.__get__(experts, type(experts))
        else:
            def grouped_sparse_block_forward(self, hidden_states, _packed=packed, _topk=n_topk):
                orig_shape = hidden_states.shape
                hidden_dim = orig_shape[-1]
                flat_x = hidden_states.view(-1, hidden_dim)
                router_logits = self.gate(flat_x)
                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]
                routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, _topk, dim=-1)
                if getattr(self, "norm_topk_prob", True):
                    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                routing_weights = routing_weights.to(flat_x.dtype)
                final_hidden = fused_moe_int3(flat_x, routing_weights, selected_experts.to(torch.int32), _packed, fast=True)

                for sh_attr in ("shared_expert", "shared_experts"):
                    if hasattr(self, sh_attr) and getattr(self, sh_attr) is not None:
                        sh_out = getattr(self, sh_attr)(flat_x)
                        if hasattr(self, "shared_expert_gate") and self.shared_expert_gate is not None:
                            sh_out = torch.sigmoid(self.shared_expert_gate(flat_x)) * sh_out
                        final_hidden = final_hidden + sh_out
                return final_hidden.view(orig_shape), router_logits

            moe.forward = grouped_sparse_block_forward.__get__(moe, type(moe))

        moe._brmoe_grouped_packed = packed
        moe._brmoe_grouped_info = info
        patched += 1

        # ---- 释放原来的 int3 权重 ----
        if free_original:
            for exp in experts:
                for proj in (gate_name, up_name, down_name):
                    lay = getattr(exp, proj, None)
                    if lay is None:
                        continue
                    lay.W_q = None
                    lay.meta = None
                    lay.U = None
                    lay.V = None
            torch.cuda.empty_cache()

    if cache_path and new_cache and len(new_cache) > len(cache):
        try:
            torch.save(new_cache, cache_path)
            _log(verbose, f"打包结果已缓存到 {cache_path}")
        except Exception as e:
            _log(verbose, f"缓存写入失败: {e}")

    _log(verbose, f"接管 {patched}/{len(moe_layers)} 个 MoE 层, 用时 {time.time() - t_all:.1f}s")
    if skipped:
        _log(verbose, f"跳过 {len(skipped)} 层:")
        for n, r in skipped[:8]:
            _log(verbose, f"    {n}: {r}")
        if len(skipped) > 8:
            _log(verbose, f"    ... 其余 {len(skipped) - 8} 层")
    return patched
