"""int3 MoE 的端到端算子: 权重打包 + fused MoE 前向 + 对照参考实现。

MoE MLP 结构 (与 Mixtral / Qwen-MoE 一致):
    gate = x @ W1[e]^T          [n, I]
    up   = x @ W3[e]^T          [n, I]
    h    = silu(gate) * up      [n, I]
    y    = h @ W2[e]^T          [n, K]
把 gate/up 在输出维拼接成 w13 [E, 2I, K], 于是整层只需两次 grouped GEMM。

两条路径 (可 A/B 对比):
    fast=True  : Triton align (零 host 同步) + 超发 GEMM + 融合激活
    fast=False : torch align (argsort/scatter + .item()) + torch 激活  ← 优化前基线
"""

import torch
from math import gcd as _gcd
from torch import Tensor

from .packing import (quantize_int3_symmetric, pack_int3, unpack_int3,
                      pack_int3_slots4, unpack_int3_slots4)
from .align import moe_align_block_size as align_torch
from .align_triton import moe_align_block_size_triton, _BUF
from .kernel import int3_moe_gemm, silu_mul


# ---------------------------------------------------------------------------
# 权重打包
# ---------------------------------------------------------------------------

def pack_moe_weights(w13: Tensor, w2: Tensor, group_size: int, layout: str = "int3",
                     transposed: bool = False):
    """把 fp16 的 MoE 权重打包。

    Args:
        w13: [E, 2I, K] fp16  (gate 与 up 在输出维拼接)
        w2:  [E, K, I]  fp16
        layout: "int3" = 稠密 3.0 bit/weight; "int4" = 4-bit 槽位 (4.0 bpw, 解包更便宜)
    Returns:
        dict: w13_q / s13 / w2_q / s2 / group_size / layout
              int3: w13_q [E,2I,K//32*3], w2_q [E,K,I//32*3]
              int4: w13_q [E,2I,K//8],    w2_q [E,K,I//8]
    """
    E, twoI, K = w13.shape
    assert w2.shape == (E, K, twoI // 2), f"w2 形状应为 {(E, K, twoI//2)}"
    I = twoI // 2

    if layout == "fp16":
        # ---- 不量化的对照路径 ----
        # 直接存 fp16 权重, 按 [E, K_red, N] (K-major) 以便固定 k 行时 n 连续、载入合并。
        # 不做量化 -> 本不需要 scale; 这里填 1.0 只是为了让形状校验通过
        # (kernel 的 fp16 分支根本不读 scale)。group_size 取 gcd(K,I) 只为占位。
        g = _gcd(K, I)
        return {
            "w13_q": w13.transpose(1, 2).contiguous(),      # [E, K, 2I]
            "w2_q": w2.transpose(1, 2).contiguous(),        # [E, I, K]
            "s13": torch.ones(E, K // g, twoI, dtype=torch.float16, device=w13.device),
            "s2": torch.ones(E, I // g, K, dtype=torch.float16, device=w2.device),
            "group_size": g,
            "layout": "fp16",
            "w_transposed": True,
        }

    assert K % group_size == 0 and I % group_size == 0
    packer = pack_int3_slots4 if layout == "int4" else pack_int3
    words = (lambda Kk: Kk // 8) if layout == "int4" else (lambda Kk: Kk // 32 * 3)

    def _q(w, gs):
        """[N, Kk] -> 打包成 [Kpack, N] (K-major, 访存合并), 以及 scale。"""
        N, Kk = w.shape
        # scale = amax/3, 因为对称 int3 的可表示范围是 [-4s, +3s]
        amax = w.reshape(N, Kk // gs, gs).abs().amax(dim=2).clamp_min(1e-8)
        s = (amax / 3.0).to(torch.float16)
        q = quantize_int3_symmetric(w.to(torch.float16), s, gs)
        return packer(q.reshape(N, Kk), transposed=transposed), s

    def _regroup(p, Edim, Ndim):
        """[Kpack, Edim*Ndim] -> [Edim, Kpack, Ndim]。"""
        Kpack = p.shape[0]
        return p.reshape(Kpack, Edim, Ndim).permute(1, 0, 2).contiguous()

    w13_q, s13 = _q(w13.reshape(E * twoI, K), group_size)
    w2_q, s2 = _q(w2.reshape(E * K, I), group_size)
    if transposed:
        w13_out = _regroup(w13_q, E, twoI)          # [E, Kpack(K), 2I]
        w2_out = _regroup(w2_q, E, K)               # [E, Kpack(I), K]
    else:
        w13_out = w13_q.reshape(E, twoI, words(K))
        w2_out = w2_q.reshape(E, K, words(I))
    return {
        "w13_q": w13_out,
        "s13": s13.reshape(E, K // group_size, twoI),
        "w2_q": w2_out,
        "s2": s2.reshape(E, I // group_size, K),
        "group_size": group_size,
        "layout": layout,
        # K-major ([E,Kpack,N]) 与 N-major ([E,N,Kpack]) 实测速度中性 (0.93~1.01x):
        # k 循环会把整行 word 读完, 所以 N-major 的"分散小访问"最终并没有放大总流量。
        # 默认用更简单的 N-major; 需要时传 transposed=True 开启 K-major。
        "w_transposed": transposed,
    }


def dequant_int3(packed: Tensor, scales: Tensor, K: int, group_size: int,
                 layout: str = "int3", transposed: bool = True) -> Tensor:
    """解包 + 反量化回 fp16 -> [E, N, K] fp16。

    packed 为 [E, Kpack, N] (K-major) 或 [E, N, Kpack]。
    """
    E, d1, d2 = packed.shape
    if transposed:
        N, Kpack = d2, d1
        p = packed.permute(0, 2, 1).reshape(-1, Kpack)      # -> [E*N, Kpack]
    else:
        N, Kpack = d1, d2
        p = packed.reshape(-1, Kpack)
    q = unpack_int3_slots4(p, K, transposed=False) if layout == "int4" \
        else unpack_int3(p, K, transposed=False)
    q = q.reshape(E, N, K).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=1)   # [E, K, N]
    return ((q - 4) * s.transpose(1, 2)).to(torch.float16)


# ---------------------------------------------------------------------------
# 工作区缓存: 避免每层重复分配
# ---------------------------------------------------------------------------

class _WS:
    def __init__(self):
        self.cache = {}

    def get(self, max_post, twoI, I, M, K, device):
        key = (max_post, twoI, I, M, K, str(device))
        if key not in self.cache:
            self.cache[key] = dict(
                inter=torch.empty(max_post, twoI, dtype=torch.float16, device=device),
                act=torch.empty(max_post, I, dtype=torch.float16, device=device),
                out32=torch.zeros(M, K, dtype=torch.float32, device=device),
            )
        return self.cache[key]


_WS_CACHE = _WS()


# ---------------------------------------------------------------------------
# 主路径: fused MoE (两次 grouped GEMM)
# ---------------------------------------------------------------------------

def fused_moe_int3(
    x: Tensor,                 # [M, K] fp16
    topk_weights: Tensor,      # [M, top_k]
    topk_ids: Tensor,          # [M, top_k] int64, 每个 token 的 top_k 个专家 (互不相同)
    packed: dict,
    *,
    fast: bool = True,
    block_size: int = None,            # align 补齐粒度; None = slot
    block_m: int = 64,
    block_n: int = 64,
    slot: int = None,                  # 每个专家子块的行数; None = block_m (旧行为)
    block_k: int = 32,
    num_warps: int = 2,               # 实测最优 (见 kernel.py 注释)
    num_stages: int = 1,
    out_dtype: torch.dtype = torch.float16,
) -> Tensor:
    w13_q, s13 = packed["w13_q"], packed["s13"]
    w2_q, s2 = packed["w2_q"], packed["s2"]
    group_size = packed["group_size"]
    layout = packed.get("layout", "int3")
    layout4 = layout == "int4"
    layout16 = layout == "fp16"
    direct4 = packed.get("direct4", False)
    w_transposed = packed.get("w_transposed", False)

    E = w13_q.shape[0]
    twoI = w13_q.shape[2] if w_transposed else w13_q.shape[1]
    I = twoI // 2
    M, K = x.shape
    top_k = topk_ids.shape[1]
    num_valid = M * top_k

    nwords = (lambda Kk: Kk // 8) if layout4 else (lambda Kk: Kk // 32 * 3)
    if layout16:                           # [E, K, N] fp16
        assert w13_q.shape == (E, K, twoI), w13_q.shape
        assert w2_q.shape == (E, I, K), w2_q.shape
    elif w_transposed:
        assert w13_q.shape == (E, nwords(K), twoI), w13_q.shape
        assert w2_q.shape == (E, nwords(I), K), w2_q.shape
    else:
        assert w13_q.shape == (E, twoI, nwords(K)), w13_q.shape
        assert w2_q.shape == (E, K, nwords(I)), w2_q.shape
    assert s13.shape == (E, K // group_size, twoI), s13.shape
    assert s2.shape == (E, I // group_size, K), s2.shape

    tw = topk_weights.reshape(-1).to(torch.float32)

    # slot = 每个专家子块的行数, 同时也是 align 的补齐粒度 (即 expert_ids 的粒度):
    #   slot == block_m: 一个 block 一个专家 (旧行为)
    #   slot <  block_m: align 只按 slot 行补齐, 一个 block 装 block_m//slot 个不同专家
    #                    -> padding 浪费随 slot 而非 block_m 增长。
    #                    小 batch/decode (每专家 1~几行) 时这是数量级的差别。
    eff_slot = slot or block_m
    assert 16 <= eff_slot <= block_m and block_m % eff_slot == 0 and eff_slot % 16 == 0, \
        f"slot={eff_slot} 必须是 16 的倍数且整除 block_m={block_m}"
    if block_size is not None:
        assert block_size == eff_slot, "block_size 与 slot 必须一致 (align 粒度)"

    if fast:
        # ---- 优化 1: Triton align, 零 host 同步 ----
        sti, eid, meta, buf = moe_align_block_size_triton(
            topk_ids, E, eff_slot, flat_values=tw)
        route_w_sorted = buf["sv"]
        # 网格上界按 block_m 计 (eff_slot < block_m 时末块可能只覆盖部分行 -> kernel 内做行掩码)
        gm = (buf["max_post"] + block_m - 1) // block_m
        meta_arg, npost, max_post = meta, None, buf["max_post"]
    else:
        # ---- 优化前基线: torch align (含 .item() 同步) ----
        sti, eid, npost, route_w_sorted = align_torch(topk_ids, E, eff_slot, flat_values=tw)
        gm = (npost + block_m - 1) // block_m
        meta_arg, max_post = None, npost

    ws = _WS_CACHE.get(max_post, twoI, I, M, K, x.device)

    # ---- 第一层: A 做 gather, 输出落到"排序后行号" ----
    int3_moe_gemm(
        x, w13_q, s13, sti, eid, npost, num_valid,
        group_size=group_size, a_gather=True, add=False,
        out=ws["inter"], meta=meta_arg, grid_m=gm,
        block_m=block_m, block_n=block_n, block_k=block_k, slot=eff_slot,
        layout4=layout4, layout16=layout16, w_transposed=w_transposed, direct4=direct4,
        num_warps=num_warps, num_stages=num_stages,
    )

    # ---- 优化 3: 融合激活 silu(gate)*up ----
    if fast:
        silu_mul(ws["inter"], ws["act"], meta, I, max_post)
        act = ws["act"]
    else:
        g, u = ws["inter"][:npost, :I], ws["inter"][:npost, I:]
        act = (g.float() * torch.sigmoid(g.float()) * u.float()).to(torch.float16).contiguous()

    # ---- 第二层: A 连续读, 输出 scatter 回原 token 并按路由权重累加 ----
    ws["out32"].zero_()
    int3_moe_gemm(
        act, w2_q, s2, sti, eid, npost, num_valid,
        group_size=group_size, route_w=route_w_sorted,
        out=ws["out32"], a_gather=False, add=True,
        meta=meta_arg, grid_m=gm,
        block_m=block_m, block_n=block_n, block_k=block_k, slot=eff_slot,
        layout4=layout4, layout16=layout16, w_transposed=w_transposed, direct4=direct4,
        num_warps=num_warps, num_stages=num_stages,
    )
    return ws["out32"].to(out_dtype)


# ---------------------------------------------------------------------------
# 参考实现: 逐专家循环 (BR-MoE 的现状做法), 用 dequant 后的 fp16 权重精确计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def ref_fused_moe(
    x: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    packed: dict,
) -> Tensor:
    E = packed["w13_q"].shape[0]
    group_size = packed["group_size"]
    layout = packed.get("layout", "int3")
    w_transposed = packed.get("w_transposed", False)
    M, K = x.shape
    if layout == "fp16":
        # 不量化: 权重本来就是 fp16, 直接转置回 [E, N, K] 供逐专家 matmul
        W13 = packed["w13_q"].transpose(1, 2).float()      # [E, 2I, K]
        W2 = packed["w2_q"].transpose(1, 2).float()        # [E, K, I]
        I = W2.shape[2]
    else:
        Kw = packed["w2_q"].shape[1] if w_transposed else packed["w2_q"].shape[2]
        I = (Kw // 3 * 32) if layout == "int3" else (Kw * 8)
        W13 = dequant_int3(packed["w13_q"], packed["s13"], K, group_size, layout, w_transposed).float()
        W2 = dequant_int3(packed["w2_q"], packed["s2"], I, group_size, layout, w_transposed).float()
    xf = x.float()
    rwf = topk_weights.float()

    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    for e in range(E):
        hit = (topk_ids == e)                     # [M, top_k]
        rows = hit.any(dim=1)
        if not bool(rows.any()):
            continue
        h = xf[rows] @ W13[e].t()                 # [n, 2I]
        g, u = h[:, :I], h[:, I:]
        act = (g * torch.sigmoid(g)) * u
        y = act @ W2[e].t()                       # [n, K]
        for t in range(hit.shape[1]):
            m = hit[rows, t]
            out[rows] += torch.where(m[:, None], y * rwf[rows, t:t + 1], torch.zeros_like(y))
    return out.to(x.dtype)


__all__ = ["pack_moe_weights", "dequant_int3", "fused_moe_int3", "ref_fused_moe"]
