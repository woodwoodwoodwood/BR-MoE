"""int3 MoE 的端到端算子: 权重打包 + fused MoE 前向 + 对照参考实现。

MoE MLP 结构 (与 Mixtral / Qwen-MoE 一致):
    gate = x @ W1[e]^T          [n, I]
    up   = x @ W3[e]^T          [n, I]
    h    = silu(gate) * up      [n, I]
    y    = h @ W2[e]^T          [n, K]
把 gate/up 在输出维拼接成 w13 [E, 2I, K], 于是整层只需两次 grouped GEMM。

两条路径 (可 A/B 对比):
    fast=True  : 小 batch 用逐路由 GEMV, 其余用 Triton align + grouped GEMM
    fast=False : torch align (argsort/scatter + .item()) + torch 激活  ← 优化前基线
"""

import torch
from math import gcd as _gcd
from torch import Tensor

from .packing import (quantize_int3_symmetric, pack_int3, unpack_int3,
                      pack_int3_slots4, unpack_int3_slots4)
from .align import moe_align_block_size as align_torch
from .align_triton import moe_align_block_size_triton, _BUF
from .kernel import int3_moe_gemm, silu_mul, routed_int3_gemv, silu_mul_routes


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
    # s13/s2 的朝向必须**先 reshape 再 permute**。
    # _q 返回的 s 形状是 [E*twoI, K//gs] (行=输出通道, 列=k 分组)。kernel 期望
    # [E, K//gs, twoI] (索引方式: S[expert, kk//GS, offs_n])。
    # 直接 `.reshape(E, K//gs, twoI)` 是把同一块内存重新解释 -> 元素顺序被打乱:
    #   s13[e, g, n] 读到位置 e*(K/gs)*twoI + g*twoI + n
    #   正确的 s_orig[e*twoI+n, g] 在位置  e*twoI*(K/gs) + n*(K/gs) + g
    #   两者相等要求 g*twoI + n == n*(K/gs) + g, 一般 不成立。
    # 后果: 大部分 (g,n) 组合拿到别的组的 scale, 反量化值错得离谱 (实测 3.3x 超界)。
    return {
        "w13_q": w13_out,
        "s13": s13.reshape(E, twoI, K // group_size).permute(0, 2, 1).contiguous(),
        "w2_q": w2_out,
        "s2": s2.reshape(E, K, I // group_size).permute(0, 2, 1).contiguous(),
        "group_size": group_size,
        "layout": layout,
        # K-major ([E,Kpack,N]) 与 N-major ([E,N,Kpack]) 实测速度中性 (0.93~1.01x):
        # k 循环会把整行 word 读完, 所以 N-major 的"分散小访问"最终并没有放大总流量。
        # 默认用更简单的 N-major; 需要时传 transposed=True 开启 K-major。
        "w_transposed": transposed,
    }


def dequant_int3(packed: Tensor, scales: Tensor, K: int, group_size: int,
                 layout: str = "int3", transposed: bool = True,
                 zeros: Tensor = None) -> Tensor:
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
    if zeros is None:
        return ((q - 4) * s.transpose(1, 2)).to(torch.float16)
    # 注意: zeros 也要转置。
    #   scales/zeros 布局是 [E, K//gs, N] -> repeat_interleave -> [E, K, N] -> ^T -> [E, N, K]
    #   原来这里漏了 z 的 .transpose(1, 2), 于是 (q - z) 两个操作数分别停在
    #   [E, N, K] 和 [E, K, N], N!=K 时直接 RuntimeError:
    #     "The size of tensor a (2048) must match the size of tensor b (2816) at dim 2"
    #   对称路径 (zeros is None) 走的是常量 4, 所以没暴露 —— 也就是说**只有真实模型
    #   用的那条非对称路径是坏的**, 而且唯一的调用方 ref_fused_moe 全仓库没人调用。
    z = zeros.to(torch.float32).repeat_interleave(group_size, dim=1).transpose(1, 2)
    return ((q - z) * s.transpose(1, 2)).to(torch.float16)


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
                # GEMV 的 split-K 部分和走 fp32 atomic -> 中间 buffer 必须 fp32
                inter32=torch.zeros(max_post, twoI, dtype=torch.float32, device=device),
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
    gemv: bool = None,                 # None: direct INT3 GEMV for small decode batches
) -> Tensor:
    w13_q, s13 = packed["w13_q"], packed["s13"]
    w2_q, s2 = packed["w2_q"], packed["s2"]
    z13, z2 = packed.get("z13"), packed.get("z2")   # 非对称量化时的每组浮点零点
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

    sm = torch.cuda.get_device_capability(x.device) if fast and layout == "int3" else None
    if gemv is None:
        # A100 (sm_80): M=8 的 split-K GEMV 在随机路由 micro 中较快，但端到端
        # job 38920 比 grouped GEMM 基线慢 (路由相关时 TC 去重占优)，只在 M<=4 用。
        # 5090 (sm_120): GEMV 全胜且几乎打到带宽墙 (job 38938: M=1 19.8us/
        # 1145 GB/s, M=4 51.5us/1766 GB/s≈98% 峰值)。随机路由 micro 里 GEMV 赢到
        # M=24 (job 38944), 但 e2e 相关路由下 bs=16 回退 (38946: 7.30->7.95ms,
        # GEMV 逐路由对读权重, TC 去重) -> 保守取 M<=8。bs>=16 的根治要靠去重的
        # grouped GEMV, 不是扩阈值。
        gemv = (fast and layout == "int3" and slot is None and block_size is None
                and ((sm == (8, 0) and M <= 4) or (sm == (8, 6) and M <= 8)
                     or (sm == (12, 0) and M <= 8)))
    if gemv:
        assert layout == "int3" and fast, "direct GEMV requires fast=True and INT3 weights"
        ws = _WS_CACHE.get(num_valid, twoI, I, M, K, x.device)
        ids = topk_ids.reshape(-1).contiguous()
        # GEMV 旋钮, A100 实测 (bench/micro_moe.py --gemv-sweep, job 38913):
        #   生产默认 (block_n=32, warps=2) 从未调过; 扫描结果:
        #     K-major + block_n=64 全面占优: M=1 88->77us, M=4 256->185us, M=8 453->304us
        #     warps=4 只在 M<=2 有小幅优势; stages/groups 基本中性 -> 保持默认
        g_bn = 64 if (w_transposed or M >= 4) else 32
        # K-major split-K 扫描 (job 38918) 中，M=1/2 的 2 warps 均快于 4 warps。
        g_warps = 2
        # split-K: M 越小 grid 前两维的 program 越少 (M=1 仅 6x44=264 个, 108 核的
        # A100 每 SM 不到 2.5 个 -> 延迟受限)。拆 K 提并发, fp32 atomic 归约。
        # 档位来自扫描 (job 38918, K-major + bn=64): M<=4 拆 8 (M=1: 88->49us),
        # M<=8 拆 4, 再大 GEMV 本身就不占优了。w2 的 K=1408 只有 11 个 128-块,
        # 少拆一档。
        ks13 = 8 if M <= 4 else (4 if M <= 8 else (2 if M <= 16 else 1))
        ks2 = max(1, ks13 // 2)
        ws["inter32"].zero_()
        routed_int3_gemv(x, w13_q, s13, z13, ids, None, ws["inter32"],
                         top_k, group_size, w_transposed=w_transposed,
                         block_n=g_bn, num_warps=g_warps, ksplit=ks13)
        silu_mul_routes(ws["inter32"], ws["act"], I, num_valid)
        ws["out32"].zero_()
        routed_int3_gemv(ws["act"], w2_q, s2, z2, ids, tw, ws["out32"],
                         top_k, group_size, add=True, w_transposed=w_transposed,
                         block_n=g_bn, num_warps=g_warps, ksplit=ks2)
        return ws["out32"].to(out_dtype)

    # Decode still benefits from the smallest tensor-core tile once GEMV stops
    # being attractive.  The caller may override the tile explicitly.
    if (sm == (8, 0) and 4 < M <= 32 and slot is None and block_size is None
            and block_m == 64):
        slot = block_m = 16
        if M <= 16 and num_stages == 1:
            num_stages = 3

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
        layout4=layout4, layout16=layout16, zeros=z13,
        w_transposed=w_transposed, direct4=direct4,
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
        layout4=layout4, layout16=layout16, zeros=z2,
        w_transposed=w_transposed, direct4=direct4,
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
        W13 = dequant_int3(packed["w13_q"], packed["s13"], K, group_size, layout,
                           w_transposed, zeros=packed.get("z13")).float()
        W2 = dequant_int3(packed["w2_q"], packed["s2"], I, group_size, layout,
                          w_transposed, zeros=packed.get("z2")).float()
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
