"""int3 grouped MoE 的 CUDA 路径: tile 级融合 kernel + torch 做的 gather/scatter。

小 M 分派: M <= gemv_max_m 时**不进 CUDA kernel**, 直接委托 Triton GEMV
(fused_moe_int3 自动选 GEMV)。5090 实测 GEMV 在小 M 全胜 (M=1: 25 vs 49us,
M=8: 119 vs 160us) —— Marlin block 的固定成本 (4 级流水填充 / 256 线程 /
96KB smem) 在小 M 摊不薄, split-K 也救不了 (38997: ks>1 单调变慢)。
GEMV 吃的是 Triton 布局权重 (packed), 与 CUDA kernel 的 Marlin 布局 (pk)
不同源, 所以两个都要传。

结构 (与 Triton 路径 ops.py::fused_moe_int3 平行):
    x [M, K]
      -> align (复用 Triton align, slot=16, 零 host 同步)
      -> index_select 把 x 按 sorted_token_ids 聚到 sorted 空间 (pad 行 clamp,
         产出垃圾行, scatter 时用 where 按 meta[0]=num_post 掩掉)
      -> mul_3bit_moe(w13)  -> inter [rows, 2I]
      -> silu*up           -> act  [rows, I]
      -> mul_3bit_moe(w2)  -> out_sorted [rows, K]
      -> where(行有效) + index_add_ 散回 [M, K] (乘路由权重)

split-K (ksplit>1): kernel 的 grid.y 维, 每块只算 K 的 1/ksplit, 部分和走
fp32 atomic 归约 (缓冲每次调用前 zero_, 图安全)。小 M 下 grid.x 只有百来个
block 且每块串行全 K, 拆 K 砍关键路径 (与 GEMV 的 split-K 同理)。

CUDA Graph 安全: 全程无 host 同步 (num_post 只在 device 上读)。
"""
import os
import sys

import torch

from .repack import repack_moe  # noqa: F401  (re-export)


def _fused_moe_int3():
    """惰性导入 Triton 路径 (小 M 的 GEMV 在它内部自动选择)。"""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "triton_int3")
    d = os.path.abspath(d)
    if d not in sys.path:
        sys.path.insert(0, d)
    from int3_moe.ops import fused_moe_int3
    return fused_moe_int3


class _WS:
    def __init__(self):
        self.cache = {}

    def get(self, rows, max_post, K, twoI, I, device):
        key = (rows, max_post, K, twoI, I, str(device))
        if key not in self.cache:
            self.cache[key] = dict(
                a_sorted=torch.empty(rows, K, dtype=torch.float16, device=device),
                inter=torch.empty(rows, twoI, dtype=torch.float16, device=device),
                act=torch.empty(rows, I, dtype=torch.float16, device=device),
                out_sorted=torch.empty(rows, K, dtype=torch.float16, device=device),
                idx=torch.zeros(rows, dtype=torch.int64, device=device),
                rw=torch.zeros(rows, dtype=torch.float32, device=device),
                arange=torch.arange(rows, device=device),
            )
        return self.cache[key]


_WS_CACHE = _WS()


@torch.no_grad()
def fused_moe_int3_cuda(x, topk_weights, topk_ids, pk, ext,
                        out_dtype=torch.float16, ksplit=1, ksplit2=None,
                        packed=None, gemv_max_m=None, cfg=None):
    """pk 是 repack_moe 的输出; ext 是构建好的 brmoe_moe_int3 扩展模块。

    ksplit / ksplit2: 两级 GEMM 各自的 split-K 段数 (ksplit2 默认 = ksplit)。
    >1 时该级部分和走 fp32 atomic 归约, kernel 不再写 fp16 输出。

    cfg: (thread_n, thread_k, stages) —— CUDA kernel 的 tile/流水配置;
    None = kernel 默认 (128,128,4)。扫描结果见 bench/sweep_moe_cuda_cfg.py。

    packed: Triton 布局的 packed dict (ops.fused_moe_int3 的入参同款)。
    给了它且 M <= gemv_max_m 时直接委托 GEMV, 不进 CUDA kernel (见模块 docstring)。
    gemv_max_m=None = 按架构自选: sm_80 (A100) 取 2, 其余 (5090 等) 取 8。
    注意微观与 e2e 的口径差: 微观 (随机路由) A100 上 CUDA 从 M=2 就赢 (39002),
    但 e2e (相关路由 + gather/scatter 固定开销) M=2 时 CUDA 反而输 21%
    (39026 vs 39003: 6.86 vs 5.67 ms), M=4 起才赢 -> sm_80 取 2。
    """
    if gemv_max_m is None:
        cap = torch.cuda.get_device_capability(x.device)
        gemv_max_m = 2 if cap == (8, 0) else 8
    M0 = x.shape[0]
    if packed is not None and M0 <= gemv_max_m:
        return _fused_moe_int3()(x, topk_weights, topk_ids, packed, fast=True,
                                 out_dtype=out_dtype)

    from int3_moe.align_triton import moe_align_block_size_triton

    M, K = x.shape
    top_k = topk_ids.shape[1]
    E = pk["B13_1"].shape[0]
    twoI = pk["B13_1"].shape[2]
    I = twoI // 2

    # ---- align (slot=16 = CUDA kernel 的 m-tile 行数) ----
    sti, eid, meta, buf = moe_align_block_size_triton(
        topk_ids, E, 16, flat_values=topk_weights.reshape(-1))
    max_post = sti.numel()
    m_blocks_max = eid.numel()
    rows = m_blocks_max * 16                    # >= max_post
    ws = _WS_CACHE.get(rows, max_post, K, twoI, I, x.device)

    # ---- gather: x -> sorted 空间; [max_post, rows) 的行指到 0 (会被掩掉) ----
    ws["idx"][:max_post].copy_(sti.to(torch.int64).clamp_(0, M - 1))
    ws["rw"][:max_post].copy_(buf["sv"])
    torch.index_select(x, 0, ws["idx"], out=ws["a_sorted"])

    ks1 = max(1, int(ksplit))
    ks2 = ks1 if ksplit2 is None else max(1, int(ksplit2))
    _cfg_kw = (dict(thread_n=cfg[0], thread_k=cfg[1], stages=cfg[2])
               if cfg is not None else {})

    # ---- GEMM 1: w13 -> [rows, 2I] ----
    if ks1 > 1:
        if "inter32" not in ws:
            ws["inter32"] = torch.empty(rows, twoI, dtype=torch.float32,
                                        device=x.device)
        ws["inter32"].zero_()          # 图安全: memset 可被捕获
        ext.mul_3bit_moe(ws["a_sorted"], pk["B13_1"], pk["B13_2"], ws["inter"],
                         pk["s13"], pk["z13"], eid, meta[0:1], m_blocks_max,
                         C32=ws["inter32"], k_splits=ks1, **_cfg_kw)
        g, u = ws["inter32"][:, :I], ws["inter32"][:, I:]
    else:
        ext.mul_3bit_moe(ws["a_sorted"], pk["B13_1"], pk["B13_2"], ws["inter"],
                         pk["s13"], pk["z13"], eid, meta[0:1], m_blocks_max,
                         **_cfg_kw)
        g = ws["inter"][:, :I].float()
        u = ws["inter"][:, I:].float()
    # ---- silu * up (fp32 算, 落回 fp16) ----
    ws["act"].copy_((g * torch.sigmoid(g) * u).to(torch.float16))
    # ---- GEMM 2: w2 -> [rows, K] ----
    if ks2 > 1:
        if "outs32" not in ws:
            ws["outs32"] = torch.empty(rows, K, dtype=torch.float32,
                                       device=x.device)
        ws["outs32"].zero_()
        ext.mul_3bit_moe(ws["act"], pk["B2_1"], pk["B2_2"], ws["out_sorted"],
                         pk["s2"], pk["z2"], eid, meta[0:1], m_blocks_max,
                         C32=ws["outs32"], k_splits=ks2, **_cfg_kw)
        out_f = ws["outs32"]
    else:
        ext.mul_3bit_moe(ws["act"], pk["B2_1"], pk["B2_2"], ws["out_sorted"],
                         pk["s2"], pk["z2"], eid, meta[0:1], m_blocks_max,
                         **_cfg_kw)
        out_f = ws["out_sorted"].float()

    # ---- scatter: sorted -> 原 token, 乘路由权重, where 掩掉 pad 行 ----
    # (pad 行可能是 inf/nan 垃圾, 必须用 where 选而不是乘 0)
    row_ok = (ws["arange"] < meta[0]).unsqueeze(1)
    contrib = torch.where(row_ok, out_f * ws["rw"][:, None],
                          torch.zeros((), device=x.device))
    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    out.index_add_(0, ws["idx"], contrib)
    return out.to(out_dtype)
