"""moe_align_block_size 的 Triton 实现 (零 host 同步)。

为什么要重写: 原 torch 版 (align.py) 有 argsort / bincount / scatter / repeat_interleave
外加一次 `.item()` 同步, 实测是 **0.63~0.70 ms 的固定开销**, 在 decode 时占整层 38%。

设计要点:
    * 3 个 kernel, 全部无 host 同步:
        count    -> 统计每专家 token 数 (atomic 到哨兵槽, 无 mask)
        scan     -> 单 program: padded / 前缀和 / 填哨兵 / 写 num_post
        scatter  -> **每个专家一个 program, 用 cumsum 求"专家内序号"** (无 atomic)
        blockids -> 由 offset 反推每 block 的 expert id (向量化比较, 无循环)
    * 所有 buffer 按**静态上界**分配 (num_post <= total + min(E,total)*(BS-1)),
      所以 host 永远不需要知道 num_post; GEMM 用超发 + 提前退出来适配真实值
    * sorted_token_ids 存**原 token 下标** (范围 [0, M)), pad 位填哨兵 total

为什么 scatter 不用 atomic:
    实测 "masked lane 上 atomic_add 的返回值未定义" 会导致 dst 变成垃圾地址
    (compute-sanitizer 定位到 _scatter_kernel 在有效 lane 上写出界)。
    改成每专家一个 program + tl.cumsum 求排他前缀和之后:
        - 地址由构造保证合法 (dst < num_post)
        - 无 race, 结果**确定性**(同一输入每次输出完全一致), 便于复现
    代价是并行度只有 E 个 program, 但这部分总工作量极小 (E × total 个元素)。
"""

import torch
import triton
import triton.language as tl

# total 超过这个值就退回 torch 版 (单 program 的 cumsum 需要 BLOCK >= total)
BLOCK_TOTAL_MAX = 32768


@triton.jit
def _count_kernel(EXPERTS, COUNTS, total,
                  E: tl.constexpr, BLOCK: tl.constexpr):
    """统计每个专家的 token 数。越界 lane 落到哨兵槽 E (地址恒合法)。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    e = tl.load(EXPERTS + offs, mask=m, other=0)
    tl.atomic_add(COUNTS + tl.where(m, e, E), 1)     # 无 mask, 无返回值依赖


@triton.jit
def _count_histogram_kernel(EXPERTS, COUNTS, total,
                            E: tl.constexpr, BINS: tl.constexpr, BLOCK: tl.constexpr):
    """Combine repeated expert IDs inside each CTA before global atomics."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ex = tl.load(EXPERTS + offs, offs < total, other=E)
    counts = tl.histogram(ex, BINS)
    expert = tl.arange(0, BINS)
    tl.atomic_add(COUNTS + expert, counts, expert < E, sem='relaxed')


@triton.jit
def _scan_kernel(COUNTS, OFFSET, META, SORTED_TOKENS, SORTED_VALUES,
                 sentinel,
                 E: tl.constexpr, BS: tl.constexpr,
                 BLOCK_E: tl.constexpr, BLOCK_FILL: tl.constexpr):
    """单 program: 算 padded / 排他前缀和 / 填哨兵 / 发布 num_post。"""
    e = tl.arange(0, BLOCK_E)
    em = e < E
    cnt = tl.load(COUNTS + e, mask=em, other=0)
    # ★ 必须在这里把计数复位: _count_kernel 是 atomic_add 累加的,
    #   不复位的话同一配置第二次调用时 cnt 会翻倍 -> num_post 被放大到超出
    #   预分配的 max_post -> 越界写 + 后续调用结果全错。
    #   (这个 bug 只在"同一进程内用相同 shapes 连续调用"时才暴露, 单次调用是对的)
    tl.store(COUNTS + e, 0, mask=em)
    padded = tl.where(em, ((cnt + BS - 1) // BS) * BS, 0)
    off = tl.cumsum(padded, axis=0) - padded          # 排他前缀和
    tl.store(OFFSET + e, off, mask=em)

    num_post = tl.sum(padded, axis=0)
    tl.store(META + 0, num_post)
    tl.store(META + 1, (num_post + BS - 1) // BS)

    # 先把 [0, num_post) 全填哨兵/0, scatter 再覆盖有效位 -> 不需要 host 侧 fill
    f = tl.arange(0, BLOCK_FILL)
    start = 0
    while start < num_post:
        fm = start + f < num_post
        tl.store(SORTED_TOKENS + start + f, sentinel, mask=fm)
        tl.store(SORTED_VALUES + start + f, 0.0, mask=fm)
        start += BLOCK_FILL


@triton.jit
def _scatter_by_expert_kernel(EXPERTS, VALUES, OFFSET, SORTED_TOKENS, SORTED_VALUES,
                              total, ROUTE_POS,
                              TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    """每个专家一个 program: cumsum 求"专家内序号", 计数排序落位。

    dst = OFFSET[e] + rank, rank 是该 expert 在该 program 内的排他前缀计数,
    因此 dst ∈ [OFFSET[e], OFFSET[e]+count[e]) ⊂ [0, num_post), 地址由构造保证合法。
    """
    e = tl.program_id(0)
    base = tl.load(OFFSET + e)

    i = tl.arange(0, BLOCK)
    inb = i < total
    ex = tl.load(EXPERTS + i, mask=inb, other=-1)
    sel = (ex == e) & inb
    sel_i = sel.to(tl.int32)
    rank = tl.cumsum(sel_i, axis=0) - sel_i            # 排他前缀和 = 前面同专家的个数

    dst = tl.where(sel, base + rank, 0)
    token = i // TOP_K                                 # (token, k) 展平下标 -> token
    tl.store(SORTED_TOKENS + dst, token, mask=sel)
    if ROUTE_POS is not None:
        # Each valid original route has exactly one owner expert/program.
        tl.store(ROUTE_POS + i, dst, mask=sel)
    if VALUES is not None:
        tl.store(SORTED_VALUES + dst, tl.load(VALUES + i, mask=inb, other=0.0), mask=sel)


@triton.jit
def _blockids_kernel(OFFSET, EXPERT_IDS, META,
                     E: tl.constexpr, BS: tl.constexpr,
                     BLOCK_E: tl.constexpr, BLOCK_B: tl.constexpr):
    """由各专家起始 block 反推 expert_ids (向量化比较, 无需循环)。"""
    num_blocks = tl.load(META + 1)
    b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    bm = b < num_blocks
    e = tl.arange(0, BLOCK_E)
    em = e < E
    start = tl.load(OFFSET + e, mask=em, other=1 << 24) // BS    # 各专家起始 block
    ids = tl.sum((start[None, :] <= b[:, None]).to(tl.int32), axis=1) - 1
    tl.store(EXPERT_IDS + b, ids, mask=bm)


# ---------------------------------------------------------------------------

def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class _Buf:
    """按 (total, E, BS, device, top_k) 缓存 buffer, 避免每层重复分配。"""

    def __init__(self):
        self.cache = {}

    def get(self, total, E, BS, device, top_k):
        key = (total, E, BS, str(device), top_k)
        if key not in self.cache:
            # 紧上界: sum_e ceil(c_e/BS)*BS <= total + k*(BS-1), k = 有 token 的专家数 <= min(E,total)
            #   M=1  (total=2, E=8): 506 -> 128   (正好等于真实值)
            #   M=512(total=1024):  1536 -> 1528 (真实值约 1088)
            k = min(E, total)
            max_post = max(1, total + k * (BS - 1))
            max_blocks = (max_post + BS - 1) // BS
            self.cache[key] = dict(
                counts=torch.zeros(E + 1, dtype=torch.int32, device=device),
                offset=torch.empty(E + 1, dtype=torch.int32, device=device),
                meta=torch.zeros(2, dtype=torch.int32, device=device),
                sti=torch.empty(max_post, dtype=torch.int32, device=device),
                sv=torch.empty(max_post, dtype=torch.float32, device=device),
                eid=torch.empty(max_blocks, dtype=torch.int32, device=device),
                max_post=max_post,
                max_blocks=max_blocks,
            )
        return self.cache[key]


_BUF = _Buf()
_FALLBACK_WARNED = set()


def moe_align_block_size_triton(topk_ids, num_experts, block_size,
                                flat_values=None, cache=None, return_route_positions=False,
                                histogram=False, scatter_warps=4):
    """返回 (sti, eid, meta, buffers)。

    meta 是 device 上的 [num_post, num_blocks] int32, 供 GEMM 超发启动时读取。
    sti 长度是静态上界; 真实长度 = meta[0] (不需要同步给 host)。
    return_route_positions=True 时 buf['route_pos'][original_route] 给出 sorted 行。
    此选项要求合法专家 ID；插件入口会预先规整负 ID。每次调用重建所有位置。
    histogram / scatter_warps 为显式 prefill 调参项；未指定时保留原路径。

    total 过大 (见 BLOCK_TOTAL_MAX) 时自动退回 torch 版以保证正确性。
    """
    E = num_experts
    total = topk_ids.numel()
    top_k = topk_ids.shape[1]
    if flat_values is not None:
        flat_values = flat_values.contiguous()

    if total > BLOCK_TOTAL_MAX:
        if return_route_positions:
            raise ValueError('route positions require the Triton alignment size range')
        from .align import moe_align_block_size as _torch_align
        if total not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(total)
            print(f"[align_triton] total={total} > {BLOCK_TOTAL_MAX}, 退回 torch 实现")
        sti, eid, npost, sv = _torch_align(topk_ids, E, block_size, flat_values=flat_values)
        meta = torch.tensor([npost, (npost + block_size - 1) // block_size],
                            dtype=torch.int32, device=topk_ids.device)
        return sti, eid, meta, dict(max_post=sti.numel(), max_blocks=meta[1].item())

    buf = (cache or _BUF).get(total, E, block_size, topk_ids.device, top_k)
    if return_route_positions and 'route_pos' not in buf:
        buf['route_pos'] = torch.empty(total, dtype=torch.int32, device=topk_ids.device)
    experts = topk_ids.reshape(-1).to(torch.int32).contiguous()

    block_e = _next_pow2(E)
    BLOCK = 1024
    if histogram:
        _count_histogram_kernel[(triton.cdiv(total, BLOCK),)](
            experts, buf['counts'], total, E=E, BINS=_next_pow2(E+1), BLOCK=BLOCK)
    else:
        _count_kernel[(triton.cdiv(total, BLOCK),)](
            experts, buf["counts"], total, E=E, BLOCK=BLOCK)
    _scan_kernel[(1,)](
        buf["counts"], buf["offset"], buf["meta"], buf["sti"], buf["sv"], total,
        E=E, BS=block_size, BLOCK_E=block_e, BLOCK_FILL=4096,
    )
    _scatter_by_expert_kernel[(E,)](
        experts, flat_values, buf["offset"], buf["sti"], buf["sv"], total,
        buf['route_pos'] if return_route_positions else None,
        TOP_K=top_k, BLOCK=_next_pow2(total), num_warps=scatter_warps,
    )
    BLOCK_B = 128
    _blockids_kernel[(triton.cdiv(buf["max_blocks"], BLOCK_B),)](
        buf["offset"], buf["eid"], buf["meta"],
        E=E, BS=block_size, BLOCK_E=block_e, BLOCK_B=BLOCK_B,
    )
    return buf["sti"], buf["eid"], buf["meta"], buf


__all__ = ["moe_align_block_size_triton"]
