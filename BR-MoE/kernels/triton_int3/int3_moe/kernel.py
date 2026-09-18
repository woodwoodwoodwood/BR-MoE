"""int3 MoE expert grouped GEMM —— Triton 实现。

相对 BR-MoE 的 CUDA 版 (Marlin 血统) 的取舍:

  保留:
    * 3.0 bit/weight 的稠密打包 (32 个权重 -> 3 个 uint32), 位技巧与 BR-MoE 相同
    * 权重常驻量化态, 在寄存器里反量化成 fp16 后走张量核 (W3A16)
    * per-group scale (group_size 64 / 128)

  丢掉 (Triton 不需要):
    * Marlin 的 _perm 重排 —— 那是为了配合 ldmatrix 的 smem swizzle; Triton 自己管 layout
    * cp.async 多级流水 / 全局原子 barrier / stream-K 调度 —— Triton 自动流水, 且
      grouped 场景每个 block 归属一个 expert, 不需要工作窃取
    * 双 mma 结构 (frag_b0/frag_b1) —— Triton 的 tl.dot 自己切 mma

  新增 (MoE 特有):
    * A 侧 gather: 用 sorted_token_ids 做行间接寻址, 不搬 token
    * B 侧 per-expert 权重: expert_ids[seg] 选专家
    * C 侧 scatter + 路由权重 + pad 行丢弃, 用 fp32 atomic_add 支持 top_k > 1 累加

多专家 per block (SLOT):
    一个 m-block 覆盖 `BLOCK_M` 行, 内部按 `SLOT` 切成 `NSEG = BLOCK_M // SLOT` 个
    子块, 每个子块归属一个**独立专家**, 各做一次 [SLOT, BLOCK_K] x [BLOCK_K, BLOCK_N],
    输出行互不重叠 -> 各自直接落盘 (不需要合并)。

    为什么需要它: align 会把每个专家的行数补齐到 `SLOT` 的整数倍, 而小 batch/decode
    场景下每个专家只有 1~几行, 所以
        padding 浪费 = SLOT / 每个专家的平均行数
    SLOT=64 (一个 block 一个专家) 时, M=8/topk=8/E=256 实测浪费 59x;
    改成 SLOT=16 (mma 允许的最小 M) 后浪费降到 15x, 同时一个 block 装 4 个专家、
    block 总数减少 4x (per-block 固定开销随之摊薄)。

    注意 SLOT < BLOCK_M 时 num_post 不再是 BLOCK_M 的整数倍 -> 末块是"部分有效",
    所以 SORTED_TOKENS 必须按行掩码读取 (越界行填哨兵 num_valid, 自然落进无效行)。

约束:
    * N (输出维度) 必须是 BLOCK_N 的倍数
    * K 必须是 32 的倍数; group_size 必须是 32 的倍数
    * SLOT 是 16 的倍数且整除 BLOCK_M
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _slot_dot(A, a_row, mask_m, wb, wstep, S, expert, offs_n,
              SA, K, GS, SS_E, SS_K,
              SLOT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              LAYOUT4: tl.constexpr, DIRECT4: tl.constexpr):
    """一个 (expert, SLOT 行) 子块: 返回 acc [SLOT, BLOCK_N] = A[行] @ W[expert]^T。

    权重在寄存器里反量化: 每 32 个权重压进 3 个 uint32, 解包靠"空位阶梯拼接"。
    """
    BK_G: tl.constexpr = BLOCK_K // 32       # 一个 k-tile 含几个 32 值组
    ik = tl.arange(0, 32)
    wid = (ik // 8)[:, None]                 # 落在 4 个"伪 word"中的哪个
    ii = ik % 8
    # 4 个值在低 half(bits 0-11), 4 个在高 half(bits 16-27)
    sh = ((ii % 4) * 3 + (ii // 4) * 16)[:, None]
    sh4 = (ii * 4)[:, None]                  # 4-bit 槽位布局的位移

    acc = tl.zeros((SLOT, BLOCK_N), dtype=tl.float32)
    # BLOCK_K 只作为外层步长, 内层静态展开成 BK_G 个 32 宽 group。
    # (不能用 tl.reshape/tl.view 把 BK_G 个 group 拼成一个 [BLOCK_K, N] tile ——
    #  Triton 2.1 没有 reshape, 而 tl.view 实测会重排元素 -> 数值错误 0.11 vs 6e-5)
    for k0 in range(0, K, BLOCK_K):
        for gg in tl.static_range(BK_G):
            kk = k0 + gg * 32
            if LAYOUT4:
                # 对照布局: 8 值/word 的 4-bit 槽位, 每 32 值组 = 4 个 word。
                # 解包只要 移位+掩码, 没有 int3 的"空位阶梯拼接"成本。
                if DIRECT4:
                    # 按 k 直接寻址它落在哪个 word -> 省掉 3 个 tl.where。
                    # 那 3 个 where 会被广播到 [32, BLOCK_N], 也就是 3 条指令/值,
                    # 是这条路径上最大的单项开销。代价是同一个 word 被 8 个 k 重复读,
                    # 但地址相同会被硬件合并、且全命中 L1。
                    wv = tl.load(wb + ((kk + ik) // 8)[:, None] * wstep)
                    q = (wv >> sh4) & 0xF
                else:
                    wo = wb + (kk // 8) * wstep
                    w0 = tl.load(wo)
                    w1 = tl.load(wo + wstep)
                    w2 = tl.load(wo + 2 * wstep)
                    w3 = tl.load(wo + 3 * wstep)
                    w = tl.where(wid == 1, w1, tl.where(wid == 2, w2, tl.where(wid == 3, w3, w0)))
                    q = (w >> sh4) & 0xF             # [32, BLOCK_N]
            else:
                # 稠密 int3: 每 32 值组 = 3 个 word (含空位阶梯拼接)
                wo = wb + (kk // 32) * 3 * wstep
                w0 = tl.load(wo)
                w1 = tl.load(wo + wstep)
                w2 = tl.load(wo + 2 * wstep)

                # 三字的空位 (每 half bits 12-15) 拼成 12-bit 位流 -> 恰好 4 个 3-bit 字段
                lo = ((w0 >> 12) & 0xF) | (((w1 >> 12) & 0xF) << 4) | (((w2 >> 12) & 0xF) << 8)
                hi = ((w0 >> 28) & 0xF) | (((w1 >> 28) & 0xF) << 4) | (((w2 >> 28) & 0xF) << 8)
                # 看成第 4 个"伪 word": 低 half 放 v24-v27, 高 half 放 v28-v31
                w3 = lo | (hi << 16)

                w = tl.where(wid == 1, w1, tl.where(wid == 2, w2, tl.where(wid == 3, w3, w0)))
                q = (w >> sh) & 0x7                  # [32, BLOCK_N], 取值 [0,7]

            sc = tl.load(S + expert * SS_E + (kk // GS) * SS_K + offs_n)   # [BLOCK_N]
            b = ((q - 4).to(tl.float16)) * sc[None, :]   # 反量化: (q-4)*scale

            a = tl.load(
                A + a_row[:, None] * SA + (kk + ik)[None, :],
                mask=mask_m[:, None], other=0.0,
            )
            # 注意: Triton 2.1 的 tl.dot 不接受累加器参数 (第 3 个位置是 allow_tf32)
            acc += tl.dot(a, b)
    return acc


@triton.jit
def _slot_dot_fp16(A, a_row, mask_m, W, expert, offs_n, SA, SW_E, SW_K, K,
                   SLOT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """fp16 权重 (不量化) 的对照路径: 直接载入 [BLOCK_K, BLOCK_N] 权重块, 无解包。

    用来在同一条 grouped kernel 结构下量出"不量化"的代价 (权重字节 2.67x int3)。
    权重按 [E, K, N] 存 -> 固定 k 行下 n 连续, 载入完全合并。
    """
    ik = tl.arange(0, BLOCK_K)
    acc = tl.zeros((SLOT, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        b = tl.load(W + expert * SW_E + (k0 + ik)[:, None] * SW_K + offs_n[None, :])
        a = tl.load(A + a_row[:, None] * SA + (k0 + ik)[None, :],
                    mask=mask_m[:, None], other=0.0)
        acc += tl.dot(a, b)
    return acc


@triton.jit
def int3_moe_gemm_kernel(
    A,              # [* , K] fp16 输入激活 (行 stride = SA)
    Wp,             # [E, N, K//32*3] int32 打包后的 int3 权重
    S,              # [E, K//GS, N] fp16 每组 scale
    ROUTE_W,        # [num_post] fp16 路由权重 (MUL_W 时使用)
    SORTED_TOKENS,  # [num_post] int32 每行的原 token 下标 (pad 行 = num_valid 哨兵)
    EXPERT_IDS,     # [ceil(num_post / SLOT)] int32 每个 SLOT 行子块归属的专家
    C,              # [M_out, N] fp16 (ADD=False) 或 fp32 (ADD=True, 需先清零)
    META,           # [2] int32 = [num_post, num_blocks]; HAS_META=False 时是哨兵指针
    num_valid,
    num_post_arg,   # HAS_META=False 时的真实 num_post (HAS_META=True 时改从 META 读)
    N, K,
    SA,             # A 的行 stride
    SW_N, SW_E,     # Wp 的 n 方向 stride / 专家方向 stride
    SS_K, SS_N, SS_E,   # S 的三个 stride
    SW_K,           # LAYOUT16 时 Wp 的 k 方向 stride ([E, K, N] 布局)
    BLOCK_M: tl.constexpr,
    SLOT: tl.constexpr,      # 每个子块的行数 (16/32/64), 整除 BLOCK_M; =BLOCK_M 时退化为"一专家一 block"
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # 一次处理 BLOCK_K 个 k (32 的倍数)
    GS: tl.constexpr,        # group_size
    A_GATHER: tl.constexpr,  # A 的行号是否要经 sorted_token_ids 重映射
    ADD: tl.constexpr,       # 输出用 atomic_add 累加 (top_k>1) 还是直接写
    MUL_W: tl.constexpr,     # 是否乘路由权重
    HAS_META: tl.constexpr,  # True 时从 META 读 num_post 并提前退出 (支持超发启动)
    LAYOUT4: tl.constexpr,   # True = 4-bit 槽位布局 (8 值/word), False = 稠密 int3
    W_T: tl.constexpr,       # True = 权重按 K-major [Kpack, N] 存
    LAYOUT16: tl.constexpr,  # True = 不量化 fp16 权重 (对照路径, 走 _slot_dot_fp16)
    DIRECT4: tl.constexpr,   # True = 4-bit 槽位按 k 直接寻址 (省掉 3 个 tl.where)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    NSEG: tl.constexpr = BLOCK_M // SLOT     # 一个 block 装几个专家子块

    # 超发启动: host 不知道 num_post, 网格按静态上界给, 越界的 program 立刻退出。
    # 这样彻底避免了 align 之后的 host 同步 (原 torch 版 .item() 的开销来源之一)
    if HAS_META:
        if pid_m * BLOCK_M >= tl.load(META + 0):
            return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, SLOT)

    for g in tl.static_range(NSEG):
        seg = pid_m * NSEG + g               # 该子块在 padded 空间的"SLOT 行块"序号
        base_row = seg * SLOT
        offs_m = base_row + offs_j
        # SLOT < BLOCK_M 时 num_post 不是 BLOCK_M 的倍数 -> 末块部分越界。
        # 越界行填哨兵 num_valid, 于是 mask_m 自动为 False (A 读 0、C 不写),
        # 同时保证 SORTED_TOKENS/ROUTE_W 的读取不越界。
        if HAS_META:
            num_post = tl.load(META + 0)
        else:
            num_post = num_post_arg
        inb = offs_m < num_post
        seg_inb = base_row < num_post        # 该子块至少有一行有效
        # ★ expert_ids 只写到 num_blocks-1 (num_blocks 是 num_post//SLOT 向上取整),
        #   而 grid 是按 BLOCK_M 向上取整的 -> 最后一个 block 里可能有整块越界的子块
        #   (seg >= num_blocks)。那里 expert_ids 是 torch.empty 的脏值, 不掩码的话
        #   expert*SW_E 会飞到天外 -> 非法访存。掩码后落到专家 0 (权重地址恒定合法),
        #   且这些行全被 mask_m 挡掉, 不影响结果。
        expert = tl.load(EXPERT_IDS + seg, mask=seg_inb, other=0)
        tok = tl.load(SORTED_TOKENS + offs_m, mask=inb, other=num_valid)
        mask_m = tok < num_valid             # 有效行

        # ---- 权重基址 ----
        # W_T=True  (K-major [Kpack, N]): 固定 word 下 n 连续
        # W_T=False (N-major [N, Kpack]): 相邻 n 相隔 Kpack 个 word
        # (实测两者速度中性: k 循环会把整行 word 读完, 分散小访问最终并没有放大总流量)
        base_e = Wp + expert * SW_E
        if W_T:
            wb = base_e + offs_n             # [BLOCK_N]
            wstep = N                        # 相邻 word 相隔 N 个 int32
        else:
            wb = base_e + offs_n[None, :] * SW_N     # [1, BLOCK_N]
            wstep = 1

        # 两个 GEMM 的行映射正好互为反面:
        #   A_GATHER=True  (第一层, 输入是原始激活): A 要按 token gather, 输出按"排序后行号"落盘
        #                  -> 中间结果 [num_post, 2I] 每行独立, 不需要 atomic
        #   A_GATHER=False (第二层, 输入是中间结果): A 按排序行号连续读, 输出要 scatter 回原 token
        #                  -> 多个专家可能写同一行, 需要 atomic_add
        a_row = tok if A_GATHER else offs_m
        c_row = offs_m if A_GATHER else tok

        if LAYOUT16:
            acc = _slot_dot_fp16(A, a_row, mask_m, Wp, expert, offs_n,
                                 SA, SW_E, SW_K, K, SLOT, BLOCK_N, BLOCK_K)
        else:
            acc = _slot_dot(A, a_row, mask_m, wb, wstep, S, expert, offs_n,
                            SA, K, GS, SS_E, SS_K, SLOT, BLOCK_N, BLOCK_K,
                            LAYOUT4, DIRECT4)

        if MUL_W:
            # pad 行的路由权重是 0, 所以用 inb (越界掩码) 就够; 且这些行 C 也不会写
            rw = tl.load(ROUTE_W + offs_m, mask=inb, other=0.0)
            acc = acc * rw[:, None]

        c_ptr = C + c_row[:, None] * N + offs_n[None, :]
        if ADD:
            tl.atomic_add(c_ptr, acc, mask=mask_m[:, None])
        else:
            tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def silu_mul_kernel(
    INTER,          # [*, 2I] fp16, 前半是 gate, 后半是 up
    OUT,            # [*, I] fp16
    META,           # [2] int32, META[0] = num_post
    I,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """优化 3: silu(gate) * up 的融合激活, 同样支持超发 + 提前退出。

    比 torch 版本省掉: fp32 中间张量、三次 elementwise kernel、以及对无效行的计算。
    """
    pid_m = tl.program_id(0)
    row0 = pid_m * BLOCK_M
    if row0 >= tl.load(META + 0):
        return
    num_post = tl.load(META + 0)
    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m[:, None] < num_post) & (offs_i[None, :] < I)

    base = offs_m[:, None] * (2 * I) + offs_i[None, :]
    g = tl.load(INTER + base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(INTER + base + I, mask=mask, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g) * u).to(tl.float16)
    tl.store(OUT + offs_m[:, None] * I + offs_i[None, :], y, mask=mask)


# ---------------------------------------------------------------------------
# 发射器
# ---------------------------------------------------------------------------

def int3_moe_gemm(
    a: torch.Tensor,             # [M_in, K] fp16; A_GATHER=False 时按行连续处理
    w_packed: torch.Tensor,      # [E, N, K//32*3] int32
    scales: torch.Tensor,        # [E, K//GS, N] fp16
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: int,
    num_valid: int,
    *,
    group_size: int = 128,
    route_w: torch.Tensor = None,     # [num_post] fp16
    out: torch.Tensor = None,         # [M_out, N] fp16 (ADD=False) / fp32 (ADD=True)
    out_m: int = None,
    a_gather: bool = False,
    add: bool = False,
    meta: torch.Tensor = None,        # [2] int32 device 张量 -> 用超发启动, 零 host 同步
    grid_m: int = None,               # meta 不为 None 时给静态上界 (max_blocks)
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,                # 32 的倍数, 且必须整除 group_size
    slot: int = None,                 # 每个专家子块的行数; None = block_m (旧行为)
    layout4: bool = False,            # True = 4-bit 槽位布局 (8 值/word)
    layout16: bool = False,           # True = fp16 权重 [E, K, N] (不量化对照路径)
    w_transposed: bool = False,       # True = 权重按 [E, Kpack, N] (K-major) 存
    direct4: bool = False,            # True = 4-bit 槽位按 k 直接寻址 (去掉 3 个 tl.where)
    # 实测最优 (T4): 64x64 tile 用 2 warp 更合适 (每线程 64 个累加器 -> ILP 更高),
    # stages=1 省掉多缓冲 smem -> 占用率更高。相比 (4,3) 快 1.5~2.6x
    num_warps: int = 2,
    num_stages: int = 1,
) -> torch.Tensor:
    """一次 grouped GEMM: 所有专家的 (token, expert) 对在一个 launch 里算完。

    expert_ids 的粒度必须与 slot 一致:
        slot == block_m -> expert_ids[pid_m] 是每个 m-block 的专家 (旧行为)
        slot <  block_m -> expert_ids[seg]  是每个 slot 行的专家, 即 align 要用
                            block_size=slot 跑, 这样一个 block 里能装多个专家
    """
    if slot is None:
        slot = block_m
    assert 16 <= slot <= block_m and block_m % slot == 0 and slot % 16 == 0, \
        f"slot={slot} 必须是 16 的倍数且整除 block_m={block_m}"

    E = w_packed.shape[0]
    sw_k = 0
    if layout16:                           # [E, K, N] fp16 (不量化)
        K, N = w_packed.shape[1], w_packed.shape[2]
        sw_k = w_packed.stride(1)
        sw_n = 0
    elif w_transposed:                     # [E, Kpack, N]
        Kpack, N = w_packed.shape[1], w_packed.shape[2]
        K = Kpack * 8 if layout4 else Kpack // 3 * 32
        sw_n = 0
    else:                                  # [E, N, Kpack]
        N, Kpack = w_packed.shape[1], w_packed.shape[2]
        K = Kpack * 8 if layout4 else Kpack // 3 * 32
        sw_n = w_packed.stride(1)
    assert layout16 or w_packed.dtype == torch.int32, "Wp 需为 int32 (按位运算)"
    assert w_packed.is_contiguous() and scales.is_contiguous()
    assert a.dtype == torch.float16, "激活需为 fp16"
    assert N % block_n == 0, f"N={N} 必须是 BLOCK_N={block_n} 的倍数"
    assert K % 32 == 0 and group_size % 32 == 0
    assert block_k % 32 == 0 and group_size % block_k == 0, \
        f"block_k={block_k} 必须是 32 的倍数且整除 group_size={group_size}"
    assert a.shape[1] == K, f"A 的 K={a.shape[1]} 与权重 K={K} 不符"
    assert scales.shape == (E, K // group_size, N), f"scale 形状应为 {(E, K//group_size, N)}, 实际 {tuple(scales.shape)}"

    if out is None:
        dtype = torch.float32 if add else torch.float16
        out = torch.zeros((out_m or num_tokens_post_pad, N), dtype=dtype, device=a.device)
    if add:
        assert out.dtype == torch.float32, "累加模式输出需为 fp32 (Triton 的 atomic_add 不支持 fp16)"

    if meta is not None:
        assert grid_m is not None, "用 meta 启动时必须给 grid_m (静态上界)"
        gm, meta_arg, has_meta = grid_m, meta, True
        num_post_arg = 0
    else:
        # align 保证 num_post 是 slot 的整数倍 (不必是 block_m 的倍数)
        assert num_tokens_post_pad % slot == 0, \
            f"num_post={num_tokens_post_pad} 必须是 slot={slot} 的倍数 (align 要用 block_size=slot)"
        # SLOT < BLOCK_M 时末块可能只覆盖部分行 -> 用 cdiv (行掩码在 kernel 里做)
        gm = (num_tokens_post_pad + block_m - 1) // block_m
        meta_arg, has_meta = sorted_token_ids, False
        num_post_arg = num_tokens_post_pad

    grid = (gm, N // block_n)
    int3_moe_gemm_kernel[grid](
        a, w_packed, scales,
        route_w if route_w is not None else a,
        sorted_token_ids, expert_ids, out,
        meta_arg, num_valid, num_post_arg, N, K,
        a.stride(0), sw_n, w_packed.stride(0),
        scales.stride(1), scales.stride(2), scales.stride(0), sw_k,
        BLOCK_M=block_m, SLOT=slot, BLOCK_N=block_n, BLOCK_K=block_k, GS=group_size,
        A_GATHER=a_gather, ADD=add, MUL_W=route_w is not None,
        HAS_META=has_meta, LAYOUT4=layout4, W_T=w_transposed, LAYOUT16=layout16,
        DIRECT4=direct4,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def silu_mul(inter: torch.Tensor, out: torch.Tensor, meta: torch.Tensor, I: int,
             max_rows: int, block_m: int = 64, block_i: int = 128, num_warps: int = 4):
    """融合激活: out = silu(inter[:, :I]) * inter[:, I:], 支持超发 + 提前退出。"""
    grid = (triton.cdiv(max_rows, block_m), triton.cdiv(I, block_i))
    silu_mul_kernel[grid](inter, out, meta, I,
                          BLOCK_M=block_m, BLOCK_I=block_i, num_warps=num_warps)
    return out


__all__ = ["int3_moe_gemm_kernel", "int3_moe_gemm"]
