"""BR-MoE 的 grouped int3 MoE 后端（实验性）。

把 MoE 层的逐专家循环换成 **一个 launch 覆盖全部专家** 的 grouped GEMM：

    每层 = 2 次 grouped GEMM + 融合 silu(gate)*up
    y = Σ_k w_k · ( silu(x @ W1[e_k]ᵀ) * (x @ W3[e_k]ᵀ) ) @ W2[e_k]ᵀ

实现复用 `kernels/triton_int3/int3_moe`（免编译、triton 实现），与逐专家循环相比
主要是省掉每 token ~500 次的小 kernel launch。

限制（不满足时会拒绝启用并给出原因）：
  * **仅支持无低秩补偿 (rank=0) 的 MoE 层** —— grouped kernel 不含 U/V 补偿项
  * N(=2I) 与 K 必须是 64 的倍数；K 是 32 的倍数；group_size 是 32 的倍数
  * 首次启用需要把 int3 权重 解包→重新打包（较慢），结果缓存到磁盘

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


def check_layer_supported(experts, verbose=True):
    """检查一层 MoE 是否可以被 grouped 后端接管。返回 (ok, reason, info)。"""
    if len(experts) == 0:
        return False, "no experts", {}

    g0 = experts[0].gate_proj
    out_feat = getattr(g0, "out_features", None)
    in_feat = getattr(g0, "in_features", None)
    if out_feat is None or in_feat is None:
        return False, "layer 尚未加载 (缺少 in/out_features)", {}

    I, K = int(out_feat), int(in_feat)
    meta = _layer_meta(g0)
    gs = meta.get("group_size", None)
    nbits = meta.get("nbits", None)

    # 补偿器: rank>0 时 grouped kernel 无法表达 (x@V)@U
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lay = getattr(experts[0], proj)
        if getattr(lay, "U", None) is not None and getattr(lay, "V", None) is not None:
            return False, f"{proj} 带低秩补偿 (U/V), grouped kernel 不支持", {}

    info = dict(E=len(experts), I=I, K=K, group_size=gs, nbits=nbits)
    # 是否带 per-group 浮点零点: 有则走无损搬运路径, 否则退回对称重新量化
    info["has_zero"] = ("zero" in meta) and (meta.get("zero") is not None)
    if "scale" not in meta or meta.get("scale") is None:
        return False, "缺少 scale", info
    if nbits != 3:
        return False, f"nbits={nbits} != 3", info
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
    """无损打包: 直接搬运 BRMoE 的量化值 + 每组 scale/zero, 不做二次量化。

    内核反量化是 (q - Z) * S, 与 BR-MoE 的 (q - z) * s 语义一致, 因此没有精度损失。
    """
    from int3_moe.packing import pack_int3

    E, I, K, gs = info["E"], info["I"], info["K"], info["group_size"]

    q13 = torch.empty(E, 2 * I, K, dtype=torch.int32, device=device)
    q2 = torch.empty(E, K, I, dtype=torch.int32, device=device)
    s13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    s2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)
    z13 = torch.empty(E, K // gs, 2 * I, dtype=torch.float16, device=device)
    z2 = torch.empty(E, I // gs, K, dtype=torch.float16, device=device)

    for e, exp in enumerate(experts):
        pq, ps, pz = [], [], []
        for proj in ("gate_proj", "up_proj"):
            q, s, z = _extract_quantized(getattr(exp, proj), gs)
            pq.append(q)
            ps.append(s)
            pz.append(z)
        q13[e] = torch.cat(pq, dim=0)              # [2I, K]
        s13[e] = torch.cat(ps, dim=0).t()          # [K//gs, 2I]
        z13[e] = torch.cat(pz, dim=0).t()

        qd, sd, zd = _extract_quantized(exp.down_proj, gs)
        q2[e] = qd                                 # [K, I]
        s2[e] = sd.t()                             # [I//gs, K]
        z2[e] = zd.t()

    return {
        "w13_q": pack_int3(q13.reshape(E * 2 * I, K), transposed=False)
        .reshape(E, 2 * I, K // 32 * 3),
        "w2_q": pack_int3(q2.reshape(E * K, I), transposed=False)
        .reshape(E, K, I // 32 * 3),
        "s13": s13, "s2": s2, "z13": z13, "z2": z2,
        "group_size": gs, "layout": "int3", "w_transposed": False,
        "lossless": True,
    }


def _pack_one_layer(experts, info, device, verbose=True):
    """打包一层专家权重: 优先无损路径, 缺 zero 时退回 fp16 重量化(有二次量化误差)。"""
    if info.get("has_zero", False):
        try:
            return _pack_one_layer_lossless(experts, info, device, verbose=verbose)
        except Exception as e:
            _log(verbose, f"  无损打包失败({type(e).__name__}: {e}), 退回 fp16 重量化")

    _, pack_moe_weights = _import_triton_moe()
    E, I, K, gs = info["E"], info["I"], info["K"], info["group_size"]
    w13 = torch.empty(E, 2 * I, K, dtype=torch.float16, device=device)
    w2 = torch.empty(E, K, I, dtype=torch.float16, device=device)
    for e, exp in enumerate(experts):
        # 注意: dequantize() 会修改 meta (删除 zero/scale), 每层只能调用一次
        w13[e] = torch.cat([exp.gate_proj.dequantize(), exp.up_proj.dequantize()], dim=0)
        w2[e] = exp.down_proj.dequantize()
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
    for e in range(E):
        hit = (ti == e)
        rows = hit.any(dim=1)
        if not bool(rows.any()):
            continue
        xr = x[rows]
        h = torch.cat([experts[e].gate_proj.matmul(xr), experts[e].up_proj.matmul(xr)], dim=-1)
        gg, uu = h[:, :I], h[:, I:]
        act = (gg * torch.sigmoid(gg)) * uu
        y = experts[e].down_proj.matmul(act)
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

    moe_layers = [(n, m) for n, m in model.named_modules()
                  if type(m).__name__ == "DeepseekMoE"]
    if not moe_layers:
        raise RuntimeError("未找到 DeepseekMoE 层, brmoe_grouped 后端不适用")

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

        n_topk = topk or getattr(moe, "num_experts_per_tok", 6)
        device = experts[0].gate_proj.W_q.device

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

        # ---- 替换前向 ----
        def grouped_moe_infer(self, x, flat_expert_indices, flat_expert_weights,
                              _packed=packed, _topk=n_topk):
            n_tok = x.shape[0]
            ti = flat_expert_indices.view(n_tok, _topk)
            tw = flat_expert_weights.view(n_tok, _topk)
            return fused_moe_int3(x, tw, ti, _packed, fast=True)

        moe.moe_infer = grouped_moe_infer.__get__(moe, type(moe))
        moe._brmoe_grouped_packed = packed
        moe._brmoe_grouped_info = info
        patched += 1

        # ---- 释放原来的 int3 权重 ----
        if free_original:
            for exp in experts:
                for proj in ("gate_proj", "up_proj", "down_proj"):
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
