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
def _routed_int3_gemv(A, W, S, Z, W_MSB, MSB_MAP, BIT_OFFSETS, IDS, ROUTE_W, C,
                      K: tl.constexpr, N: tl.constexpr, TOP_K: tl.constexpr,
                      SW_E: tl.constexpr, SW_N: tl.constexpr, SS_E: tl.constexpr,
                      SS_K: tl.constexpr, SZ_E: tl.constexpr, SZ_K: tl.constexpr,
                      SMSB_E: tl.constexpr, SMSB_N: tl.constexpr,
                      GS: tl.constexpr, W_T: tl.constexpr, HAS_ZERO: tl.constexpr,
                      HAS_MSB: tl.constexpr, LAYOUT4: tl.constexpr, VARBIT: tl.constexpr,
                      W_BITS: tl.constexpr,
                      ADD: tl.constexpr, BLOCK_N: tl.constexpr, GROUPS: tl.constexpr):
    """One routed token per program: GEMV avoids the 16-row tensor-core padding.

    grid = (routes, cdiv(N, BLOCK_N), KSPLIT)。第三维是 split-K: 每个 program 只
    算 K 的一段, 部分和用 fp32 atomic_add 归约 (所以 **C 必须是 fp32 且预先清零**,
    ADD 与非 ADD 都一样)。decode 的 M 很小, grid 前两维只有几百个 program
    (M=1: 6x44=264, 108 核的 A100 每 SM 不到 2.5 个), 拆 K 是提并行度最近的一刀。
    """
    route = tl.program_id(0)
    token = route // TOP_K
    expert = tl.load(IDS + route)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ni = n < N
    # ---- split-K: 本 program 负责的 k 区间 [kb, ke) ----
    ks = tl.program_id(2)
    nks = tl.num_programs(2)
    STEP: tl.constexpr = 32 * GROUPS
    per = tl.cdiv(tl.cdiv(K, STEP), nks) * STEP
    kb = ks * per
    ke = tl.minimum(kb + per, K)
    if kb >= ke:
        return
    if VARBIT:
        s_b = tl.load(BIT_OFFSETS + expert)
        e_b = tl.load(BIT_OFFSETS + expert + 1)
        nbits = e_b - s_b
        if nbits <= 0:
            return
    if HAS_MSB:
        msb_e = tl.load(MSB_MAP + expert)
    group = tl.arange(0, GROUPS)[:, None]
    k = tl.arange(0, 32)[None, :]
    wid = (k // 8)[:, :, None]
    ii = k % 8
    shift = ((ii % 4) * 3 + (ii // 4) * 16)[:, :, None]
    shift4 = (ii * 4)[:, :, None]
    acc = tl.full((BLOCK_N,), 0, tl.float32)
    for kk in range(kb, ke, 32 * GROUPS):
        valid_group = kk + group * 32 < ke
        valid_weight = valid_group & ni[None, :]
        if VARBIT:
            if W_T:
                base_b = W + s_b * SW_E + ((kk // 32 + group) * N) + n[None, :]
            else:
                base_b = W + s_b * SW_E + n[None, :] * SW_N + (kk // 32 + group)
            w0 = tl.load(base_b, valid_weight, other=0)
            w1 = tl.load(base_b + SW_E, valid_weight, other=0)
            q = ((w0[:, None, :] >> k[:, :, None]) & 1) | (((w1[:, None, :] >> k[:, :, None]) & 1) << 1)
            if nbits >= 3:
                w2 = tl.load(base_b + 2 * SW_E, valid_weight, other=0)
                q = q | (((w2[:, None, :] >> k[:, :, None]) & 1) << 2)
            if nbits >= 4:
                w3 = tl.load(base_b + 3 * SW_E, valid_weight, other=0)
                q = q | (((w3[:, None, :] >> k[:, :, None]) & 1) << 3)
        elif LAYOUT4:
            if W_T:
                base = W + expert * SW_E + ((kk // 32 + group) * 4 * N) + n[None, :]
                step = N
            else:
                base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 4
                step = 1
            w0 = tl.load(base, valid_weight, other=0)
            w1 = tl.load(base + step, valid_weight, other=0)
            w2 = tl.load(base + 2 * step, valid_weight, other=0)
            w3 = tl.load(base + 3 * step, valid_weight, other=0)
            w = tl.where(wid == 1, w1[:, None, :],
                         tl.where(wid == 2, w2[:, None, :],
                                  tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
            q = (w >> shift4) & 15
        elif W_BITS == 2:
            if W_T:
                base = W + expert * SW_E + ((kk // 32 + group) * 2 * N) + n[None, :]
                step = N
            else:
                base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 2
                step = 1
            w0 = tl.load(base, valid_weight, other=0)
            w1 = tl.load(base + step, valid_weight, other=0)
            q = ((w0[:, None, :] >> k[:, :, None]) & 1) | (((w1[:, None, :] >> k[:, :, None]) & 1) << 1)
        elif W_BITS == 4:
            if W_T:
                base = W + expert * SW_E + ((kk // 32 + group) * 4 * N) + n[None, :]
                step = N
            else:
                base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 4
                step = 1
            w0 = tl.load(base, valid_weight, other=0)
            w1 = tl.load(base + step, valid_weight, other=0)
            w2 = tl.load(base + 2 * step, valid_weight, other=0)
            w3 = tl.load(base + 3 * step, valid_weight, other=0)
            q = (
                ((w0[:, None, :] >> k[:, :, None]) & 1)
                | (((w1[:, None, :] >> k[:, :, None]) & 1) << 1)
                | (((w2[:, None, :] >> k[:, :, None]) & 1) << 2)
                | (((w3[:, None, :] >> k[:, :, None]) & 1) << 3)
            )
        else:
            if W_T:
                base = W + expert * SW_E + ((kk // 32 + group) * 3 * N) + n[None, :]
                step = N
            else:
                base = W + expert * SW_E + n[None, :] * SW_N + (kk // 32 + group) * 3
                step = 1
            w0 = tl.load(base, valid_weight, other=0)
            w1 = tl.load(base + step, valid_weight, other=0)
            w2 = tl.load(base + 2 * step, valid_weight, other=0)
            lo = ((w0 >> 12) & 15) | (((w1 >> 12) & 15) << 4) | (((w2 >> 12) & 15) << 8)
            hi = ((w0 >> 28) & 15) | (((w1 >> 28) & 15) << 4) | (((w2 >> 28) & 15) << 8)
            w3 = lo | (hi << 16)
            w = tl.where(wid == 1, w1[:, None, :],
                         tl.where(wid == 2, w2[:, None, :],
                                  tl.where(wid == 3, w3[:, None, :], w0[:, None, :])))
            q = (w >> shift) & 7
            if HAS_MSB:
                if msb_e >= 0:
                    if W_T:
                        msb_base = W_MSB + msb_e * SMSB_E + ((kk // 32 + group) * N) + n[None, :]
                    else:
                        msb_base = W_MSB + msb_e * SMSB_E + n[None, :] * SMSB_N + (kk // 32 + group)
                    w_msb = tl.load(msb_base, valid_weight, other=0)
                    q = q | (((w_msb[:, None, :] >> k[:, :, None]) & 1) << 3)
        sc = tl.load(S + expert * SS_E + ((kk + group * 32) // GS) * SS_K + n[None, :],
                     valid_weight, other=0)
        if HAS_ZERO:
            zc = tl.load(Z + expert * SZ_E + ((kk + group * 32) // GS) * SZ_K + n[None, :],
                         valid_weight, other=0)
            b = (q.to(tl.float16) - zc[:, None, :]) * sc[:, None, :]
        else:
            b = (q - 4).to(tl.float16) * sc[:, None, :]
        a = tl.load(A + (route if ADD else token) * K + kk + group * 32 + k,
                    valid_group, other=0)
        acc += tl.sum(tl.sum(a[:, :, None].to(tl.float32) * b.to(tl.float32), axis=1), axis=0)
    if ADD:
        rw = tl.load(ROUTE_W + route).to(tl.float32)
        tl.atomic_add(C + token * N + n, acc * rw, ni)
    else:
        tl.atomic_add(C + route * N + n, acc, ni)


@triton.jit
def _silu_mul_routes(INTER, OUT, I: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i < I
    g = tl.load(INTER + route * (2 * I) + i, mask, other=0).to(tl.float32)
    u = tl.load(INTER + route * (2 * I) + I + i, mask, other=0).to(tl.float32)
    y = (g * tl.sigmoid(g) * u).to(tl.float16)
    tl.store(OUT + route * I + i, y, mask)


def routed_int3_gemv(a, w, scales, zeros, ids, route_w, out, top_k, group_size,
                     *, add=False, w_transposed=False, block_n=32, num_warps=2,
                     num_stages=1, groups=4, ksplit=1,
                     w_msb=None, msb_map=None, bit_offsets=None, layout4=False, w_bits=3):
    """Launch the direct INT2/INT3/INT4 or CSR VarBit GEMV for the two MoE projections.

    `out` 必须是 **fp32 且预先清零** —— split-K (grid 第三维) 的部分和用
    atomic_add 归约, 非 ADD 路径也不例外。ksplit=1 时与旧行为等价 (除输出 dtype)。
    """
    varbit = bit_offsets is not None
    E = scales.shape[0]
    if w_transposed:
        n = w.shape[2]
        k = (w.shape[1] * 32) if varbit else ((w.shape[1] * 8) if layout4 else (w.shape[1] // w_bits * 32))
        sw_n = 1
    else:
        n = w.shape[1]
        k = (w.shape[2] * 32) if varbit else ((w.shape[2] * 8) if layout4 else (w.shape[2] // w_bits * 32))
        sw_n = w.stride(1)
    assert a.shape[1] == k and scales.shape == (E, k // group_size, n)
    assert out.dtype == torch.float32, "split-K 的 atomic 归约要求 fp32 输出"
    has_zero = zeros is not None
    z = zeros if has_zero else scales
    has_msb = (w_msb is not None) and (msb_map is not None) and (not layout4) and (not varbit)
    w_msb_arg = w_msb if has_msb else w
    msb_map_arg = msb_map if has_msb else ids
    bit_offsets_arg = bit_offsets if varbit else ids
    smsb_e = w_msb.stride(0) if has_msb else 0
    smsb_n = (1 if w_transposed else w_msb.stride(1)) if has_msb else 0
    _routed_int3_gemv[(ids.numel(), triton.cdiv(n, block_n), ksplit)](
        a, w, scales, z, w_msb_arg, msb_map_arg, bit_offsets_arg, ids,
        route_w if route_w is not None else a, out,
        K=k, N=n, TOP_K=top_k, SW_E=w.stride(0), SW_N=sw_n,
        SS_E=scales.stride(0), SS_K=scales.stride(1),
        SZ_E=z.stride(0), SZ_K=z.stride(1),
        SMSB_E=smsb_e, SMSB_N=smsb_n,
        GS=group_size, W_T=w_transposed, HAS_ZERO=has_zero,
        HAS_MSB=has_msb, LAYOUT4=layout4, VARBIT=varbit, W_BITS=w_bits,
        ADD=add, BLOCK_N=block_n, GROUPS=groups,
        num_warps=num_warps, num_stages=num_stages,
    )


def silu_mul_routes(inter, out, i, routes):
    _silu_mul_routes[(routes, triton.cdiv(i, 256))](
        inter, out, I=i, BLOCK=256, num_warps=4)


@triton.jit
def _slot_dot(A, a_row, mask_m, wb, wstep, sw_e, nbits, msb_wb, msb_wstep, msb_e,
              S, Z, expert, offs_n,
              SA, K, GS, SS_E, SS_K, SZ_E, SZ_K,
              SLOT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              LAYOUT4: tl.constexpr, DIRECT4: tl.constexpr, HAS_ZERO: tl.constexpr,
              HAS_MSB: tl.constexpr, VARBIT: tl.constexpr, W_BITS: tl.constexpr):
    """一个 (expert, SLOT 行) 子块: 返回 acc [SLOT, BLOCK_N] = A[行] @ W[expert]^T。

    权重在寄存器里反量化:
      * VARBIT=True: 从 CSR 位平面 3D 张量 [B_tot, ...] 按 nbits in {2, 3, 4} 动态读取 b_e 个位平面
      * VARBIT=False: 每 32 个权重压进 3 个 uint32 (或带 1-bit MSB 平面)
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
            if VARBIT:
                wo = wb + (kk // 32) * wstep
                w0 = tl.load(wo)
                w1 = tl.load(wo + sw_e)
                q = ((w0 >> ik[:, None]) & 1) | (((w1 >> ik[:, None]) & 1) << 1)
                if nbits >= 3:
                    w2 = tl.load(wo + 2 * sw_e)
                    q = q | (((w2 >> ik[:, None]) & 1) << 2)
                if nbits >= 4:
                    w3 = tl.load(wo + 3 * sw_e)
                    q = q | (((w3 >> ik[:, None]) & 1) << 3)
            elif LAYOUT4:
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
            elif W_BITS == 2:
                # Per-3D-Tensor 统一 2-bit: 每 32 值组 = 2 个连续 word, 静态展开零分支
                wo = wb + (kk // 32) * 2 * wstep
                w0 = tl.load(wo)
                w1 = tl.load(wo + wstep)
                q = ((w0 >> ik[:, None]) & 1) | (((w1 >> ik[:, None]) & 1) << 1)
            elif W_BITS == 4:
                # Per-3D-Tensor 统一 4-bit: 每 32 值组 = 4 个连续 word, 静态展开零分支
                wo = wb + (kk // 32) * 4 * wstep
                w0 = tl.load(wo)
                w1 = tl.load(wo + wstep)
                w2 = tl.load(wo + 2 * wstep)
                w3 = tl.load(wo + 3 * wstep)
                q = (
                    ((w0 >> ik[:, None]) & 1)
                    | (((w1 >> ik[:, None]) & 1) << 1)
                    | (((w2 >> ik[:, None]) & 1) << 2)
                    | (((w3 >> ik[:, None]) & 1) << 3)
                )
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
                if HAS_MSB:
                    if msb_e >= 0:
                        w_msb = tl.load(msb_wb + (kk // 32) * msb_wstep)
                        q = q | (((w_msb >> ik[:, None]) & 1) << 3)

            sc = tl.load(S + expert * SS_E + (kk // GS) * SS_K + offs_n)   # [BLOCK_N]
            if HAS_ZERO:
                # 非对称量化: 每组有独立的浮点零点 -> (q - z) * s  (BR-MoE 的原始量化格式)
                zc = tl.load(Z + expert * SZ_E + (kk // GS) * SZ_K + offs_n)  # [BLOCK_N]
                b = (q.to(tl.float16) - zc[None, :]) * sc[None, :]
            else:
                b = ((q - 4).to(tl.float16)) * sc[None, :]   # 对称: (q-4)*scale

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
    Wp,             # [E, N, K//32*3] 或 [B_tot, N, K//32] int32 打包后的权重
    S,              # [E, K//GS, N] fp16 每组 scale
    Z,              # [E, K//GS, N] fp16 每组 zero (非对称量化; HAS_ZERO=False 时不读)
    W_MSB,          # [E_4bit, N, K//32] (或 K-major) int32 4-bit 专家的 MSB 平面
    MSB_MAP,        # [E] int32: 2/3-bit 专家为 -1, 4-bit 专家为 0..E_4bit-1
    BIT_OFFSETS,    # [E + 1] int32: VARBIT=True 时专家 e 的位平面区间 [s_b, e_b)
    ROUTE_W,        # [num_post] fp16 路由权重 (MUL_W 时使用)
    SORTED_TOKENS,  # [num_post] int32 每行的原 token 下标 (pad 行 = num_valid 哨兵)
    EXPERT_IDS,     # [ceil(num_post / SLOT)] int32 每个 SLOT 行子块归属的专家
    C,              # [M_out, N] fp16 (ADD=False) 或 fp32 (ADD=True, 需先清零)
    META,           # [2] int32 = [num_post, num_blocks]; HAS_META=False 时是哨兵指针
    num_valid,
    num_post_arg,   # HAS_META=False 时的真实 num_post (HAS_META=True 时改从 META 读)
    N, K,
    SA,             # A 的行 stride
    SW_N, SW_E,     # Wp 的 n 方向 stride / 专家(或位平面)方向 stride
    SS_K, SS_N, SS_E,   # S 的三个 stride
    SZ_E, SZ_K,     # Z 的专家方向 / group 方向 stride
    SMSB_E, SMSB_N, # W_MSB 的专家方向 / n 方向 stride
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
    HAS_ZERO: tl.constexpr,  # True = 权重带 per-group 浮点零点 (非对称量化)
    HAS_MSB: tl.constexpr,   # True = 含 4-bit 专家的 1-bit MSB 扩展平面
    VARBIT: tl.constexpr,    # True = CSR 位平面 3D 张量 [B_tot, ...] + BIT_OFFSETS [E + 1]
    W_BITS: tl.constexpr,    # Per-3D-Tensor 统一位宽 (2, 3, 或 4): 编译期常量, 单一大 Kernel 发射
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

        if VARBIT:
            s_b = tl.load(BIT_OFFSETS + expert)
            e_b = tl.load(BIT_OFFSETS + expert + 1)
            nbits = e_b - s_b
            base_e = Wp + s_b * SW_E
        else:
            nbits = 3
            base_e = Wp + expert * SW_E

        if W_T:
            wb = base_e + offs_n             # [BLOCK_N]
            wstep = N                        # 相邻 word 相隔 N 个 int32
        else:
            wb = base_e + offs_n[None, :] * SW_N     # [1, BLOCK_N]
            wstep = 1

        if HAS_MSB:
            msb_e = tl.load(MSB_MAP + expert, mask=seg_inb, other=-1)
            base_msb_e = W_MSB + tl.maximum(msb_e, 0) * SMSB_E
            if W_T:
                msb_wb = base_msb_e + offs_n
                msb_wstep = N
            else:
                msb_wb = base_msb_e + offs_n[None, :] * SMSB_N
                msb_wstep = 1
        else:
            msb_e = -1
            msb_wb = wb
            msb_wstep = wstep

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
            acc = _slot_dot(A, a_row, mask_m, wb, wstep, SW_E, nbits, msb_wb, msb_wstep, msb_e,
                            S, Z, expert, offs_n,
                            SA, K, GS, SS_E, SS_K, SZ_E, SZ_K, SLOT, BLOCK_N, BLOCK_K,
                            LAYOUT4, DIRECT4, HAS_ZERO, HAS_MSB, VARBIT, W_BITS)

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
# CSR 变秩低秩补偿 (Mixed-Rank Low-Rank Compensator) Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _csr_comp_v_w13_kernel(
    A,              # [M, K] fp16
    W13_V,          # [2, R_tot, K] fp16 (0: gate V, 1: up V)
    OFFSETS,        # [E + 1] int32
    FLAT_IDS,       # [R] int32
    H_OUT,          # [2, R, MAX_R] fp32
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    R_TOT: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    pid_rb = tl.program_id(1)
    r_base = pid_rb * BLOCK_R
    expert = tl.load(FLAT_IDS + route)
    s_e = tl.load(OFFSETS + expert)
    e_e = tl.load(OFFSETS + expert + 1)
    r_e = e_e - s_e
    if r_base >= r_e:
        return

    tok = route // TOP_K
    offs_r = r_base + tl.arange(0, BLOCK_R)
    r_mask = offs_r < r_e
    row_v = (s_e + offs_r).to(tl.int64)

    acc_g = tl.zeros((BLOCK_R,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_R,), dtype=tl.float32)
    offs_k0 = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k0
        k_mask = offs_k < K
        a_vec = tl.load(A + tok * K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        vg = tl.load(
            W13_V + row_v[:, None] * K + offs_k[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        vu = tl.load(
            W13_V + R_TOT * K + row_v[:, None] * K + offs_k[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_g += tl.sum(vg * a_vec[None, :], axis=1)
        acc_u += tl.sum(vu * a_vec[None, :], axis=1)

    R_routes = tl.num_programs(0)
    tl.store(H_OUT + route * MAX_R + offs_r, acc_g, mask=r_mask)
    tl.store(H_OUT + R_routes * MAX_R + route * MAX_R + offs_r, acc_u, mask=r_mask)


@triton.jit
def _csr_comp_u_w13_kernel(
    H_IN,           # [2, R, MAX_R] fp32
    W13_U,          # [2, R_tot, I] fp16 (0: gate U^T, 1: up U^T)
    OFFSETS,        # [E + 1] int32
    FLAT_IDS,       # [R] int32
    SORTED_TOKENS,  # [max_rows] int32 (SORTED_MODE=True 时使用)
    META,           # [2] int32 (SORTED_MODE=True 时使用)
    INTER,          # [*, 2*I] fp16 或 fp32
    R_routes,
    I: tl.constexpr,
    R_TOT: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SORTED_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    if SORTED_MODE:
        if row >= tl.load(META + 0):
            return
        route = tl.load(SORTED_TOKENS + row)
        if route >= R_routes:
            return
    else:
        route = row

    expert = tl.load(FLAT_IDS + route)
    s_e = tl.load(OFFSETS + expert)
    e_e = tl.load(OFFSETS + expert + 1)
    r_e = e_e - s_e
    if r_e <= 0:
        return

    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < I

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
    offs_r0 = tl.arange(0, BLOCK_R)
    for r0 in range(0, r_e, BLOCK_R):
        offs_r = r0 + offs_r0
        r_mask = offs_r < r_e
        row_u = (s_e + offs_r).to(tl.int64)
        hg = tl.load(H_IN + route * MAX_R + offs_r, mask=r_mask, other=0.0)
        hu = tl.load(H_IN + R_routes * MAX_R + route * MAX_R + offs_r, mask=r_mask, other=0.0)
        ug = tl.load(
            W13_U + row_u[:, None] * I + offs_n[None, :],
            mask=r_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        uu = tl.load(
            W13_U + R_TOT * I + row_u[:, None] * I + offs_n[None, :],
            mask=r_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_g += tl.sum(hg[:, None] * ug, axis=0)
        acc_u += tl.sum(hu[:, None] * uu, axis=0)

    base_inter = INTER + row * (2 * I)
    cur_g = tl.load(base_inter + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    cur_u = tl.load(base_inter + I + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(base_inter + offs_n, (cur_g + acc_g).to(INTER.dtype.element_ty), mask=n_mask)
    tl.store(base_inter + I + offs_n, (cur_u + acc_u).to(INTER.dtype.element_ty), mask=n_mask)


@triton.jit
def _csr_comp_v_w2_kernel(
    ACT,            # [*, I] fp16
    W2_V,           # [R_tot, I] fp16
    OFFSETS,        # [E + 1] int32
    FLAT_IDS,       # [R] int32
    SORTED_TOKENS,  # [max_rows] int32 (SORTED_MODE=True 时使用)
    META,           # [2] int32 (SORTED_MODE=True 时使用)
    H_OUT,          # [R, MAX_R] fp32
    R_routes,
    I: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SORTED_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    if SORTED_MODE:
        if row >= tl.load(META + 0):
            return
        route = tl.load(SORTED_TOKENS + row)
        if route >= R_routes:
            return
    else:
        route = row

    pid_rb = tl.program_id(1)
    r_base = pid_rb * BLOCK_R
    expert = tl.load(FLAT_IDS + route)
    s_e = tl.load(OFFSETS + expert)
    e_e = tl.load(OFFSETS + expert + 1)
    r_e = e_e - s_e
    if r_base >= r_e:
        return

    offs_r = r_base + tl.arange(0, BLOCK_R)
    r_mask = offs_r < r_e
    row_v = (s_e + offs_r).to(tl.int64)

    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    offs_k0 = tl.arange(0, BLOCK_K)
    for k0 in range(0, I, BLOCK_K):
        offs_k = k0 + offs_k0
        k_mask = offs_k < I
        act_vec = tl.load(ACT + row * I + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        vd = tl.load(
            W2_V + row_v[:, None] * I + offs_k[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vd * act_vec[None, :], axis=1)

    tl.store(H_OUT + route * MAX_R + offs_r, acc, mask=r_mask)


@triton.jit
def _csr_comp_u_w2_kernel(
    H_IN,           # [R, MAX_R] fp32
    W2_U,           # [R_tot, K] fp16
    OFFSETS,        # [E + 1] int32
    FLAT_IDS,       # [R] int32
    FLAT_WTS,       # [R] fp16 / fp32
    OUT32,          # [M, K] fp32
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    route = tl.program_id(0)
    expert = tl.load(FLAT_IDS + route)
    s_e = tl.load(OFFSETS + expert)
    e_e = tl.load(OFFSETS + expert + 1)
    r_e = e_e - s_e
    if r_e <= 0:
        return

    tok = route // TOP_K
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < K

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    offs_r0 = tl.arange(0, BLOCK_R)
    for r0 in range(0, r_e, BLOCK_R):
        offs_r = r0 + offs_r0
        r_mask = offs_r < r_e
        row_u = (s_e + offs_r).to(tl.int64)
        hd = tl.load(H_IN + route * MAX_R + offs_r, mask=r_mask, other=0.0)
        ud = tl.load(
            W2_U + row_u[:, None] * K + offs_n[None, :],
            mask=r_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(hd[:, None] * ud, axis=0)

    rw = tl.load(FLAT_WTS + route).to(tl.float32)
    tl.atomic_add(OUT32 + tok * K + offs_n, acc * rw, mask=n_mask)


def apply_csr_comp_w13(
    a: torch.Tensor,
    w13_v: torch.Tensor,
    w13_u: torch.Tensor,
    comp_offsets: torch.Tensor,
    flat_ids: torch.Tensor,
    inter: torch.Tensor,
    top_k: int,
    max_rank: int,
    *,
    sorted_tokens: torch.Tensor = None,
    meta: torch.Tensor = None,
    max_rows: int = None,
    h_buf: torch.Tensor = None,
):
    """在 silu_mul 之前将 gate_proj / up_proj 的 CSR 低秩补偿累加进 inter。"""
    if max_rank <= 0 or w13_v is None or w13_v.shape[1] == 0:
        return
    R_routes = flat_ids.numel()
    K = a.shape[1]
    R_tot = w13_v.shape[1]
    I = w13_u.shape[2]
    block_r = 16 if max_rank <= 16 else 32
    block_k = 128
    block_n = 64

    if h_buf is None or h_buf.shape[0] < 2 * R_routes * max_rank:
        h = torch.empty((2, R_routes, max_rank), dtype=torch.float32, device=a.device)
    else:
        h = h_buf[: 2 * R_routes * max_rank].view(2, R_routes, max_rank)

    _csr_comp_v_w13_kernel[(R_routes, triton.cdiv(max_rank, block_r))](
        a, w13_v, comp_offsets, flat_ids, h,
        K=K, TOP_K=top_k, R_TOT=R_tot, MAX_R=max_rank,
        BLOCK_R=block_r, BLOCK_K=block_k, num_warps=4,
    )

    sorted_mode = sorted_tokens is not None
    rows_grid = max_rows if sorted_mode else R_routes
    st_arg = sorted_tokens if sorted_mode else flat_ids
    meta_arg = meta if sorted_mode else flat_ids
    _csr_comp_u_w13_kernel[(rows_grid, triton.cdiv(I, block_n))](
        h, w13_u, comp_offsets, flat_ids, st_arg, meta_arg, inter,
        R_routes, I=I, R_TOT=R_tot, MAX_R=max_rank,
        BLOCK_R=block_r, BLOCK_N=block_n, SORTED_MODE=sorted_mode,
        num_warps=4,
    )


def apply_csr_comp_w2(
    act: torch.Tensor,
    w2_v: torch.Tensor,
    w2_u: torch.Tensor,
    comp_offsets: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_wts: torch.Tensor,
    out32: torch.Tensor,
    top_k: int,
    max_rank: int,
    *,
    sorted_tokens: torch.Tensor = None,
    meta: torch.Tensor = None,
    max_rows: int = None,
    h_buf: torch.Tensor = None,
):
    """在 silu_mul 之后将 down_proj 的 CSR 低秩补偿乘路由权重并累加进 out32。"""
    if max_rank <= 0 or w2_v is None or w2_v.shape[0] == 0:
        return
    R_routes = flat_ids.numel()
    I = w2_v.shape[1]
    K = w2_u.shape[1]
    block_r = 16 if max_rank <= 16 else 32
    block_k = 128
    block_n = 64

    if h_buf is None or h_buf.shape[0] < R_routes * max_rank:
        h = torch.empty((R_routes, max_rank), dtype=torch.float32, device=act.device)
    else:
        h = h_buf[: R_routes * max_rank].view(R_routes, max_rank)

    sorted_mode = sorted_tokens is not None
    rows_grid = max_rows if sorted_mode else R_routes
    st_arg = sorted_tokens if sorted_mode else flat_ids
    meta_arg = meta if sorted_mode else flat_ids
    _csr_comp_v_w2_kernel[(rows_grid, triton.cdiv(max_rank, block_r))](
        act, w2_v, comp_offsets, flat_ids, st_arg, meta_arg, h,
        R_routes, I=I, MAX_R=max_rank,
        BLOCK_R=block_r, BLOCK_K=block_k, SORTED_MODE=sorted_mode,
        num_warps=4,
    )

    _csr_comp_u_w2_kernel[(R_routes, triton.cdiv(K, block_n))](
        h, w2_u, comp_offsets, flat_ids, flat_wts, out32,
        K=K, TOP_K=top_k, MAX_R=max_rank,
        BLOCK_R=block_r, BLOCK_N=block_n,
        num_warps=4,
    )


def apply_3d_comp_w13(
    a: torch.Tensor,
    w1_v: torch.Tensor,
    w1_u: torch.Tensor,
    w3_v: torch.Tensor,
    w3_u: torch.Tensor,
    flat_ids: torch.Tensor,
    inter: torch.Tensor,
    top_k: int,
    *,
    sorted_tokens: torch.Tensor = None,
    max_rows: int = None,
):
    """Per-3D-Tensor 规整 3D 低秩补偿 (gate & up):
    w1_v, w3_v: [E, r_13, K], w1_u, w3_u: [E, I, r_13]。
    全程静态形状 index_select + torch.bmm, 零 host 同步, 100% 兼容 CUDA Graph。
    """
    if w1_v is None or w1_v.ndim != 3 or w1_v.shape[1] == 0:
        return
    M, K = a.shape
    R = flat_ids.numel()
    idx = flat_ids.long()
    x_rep = a.to(w1_v.dtype).unsqueeze(1).expand(M, top_k, K).reshape(R, K, 1)

    h1 = torch.bmm(w1_v.index_select(0, idx), x_rep)           # [R, r_13, 1]
    y1 = torch.bmm(w1_u.index_select(0, idx), h1).squeeze(-1)  # [R, I]
    h3 = torch.bmm(w3_v.index_select(0, idx), x_rep)           # [R, r_13, 1]
    y3 = torch.bmm(w3_u.index_select(0, idx), h3).squeeze(-1)  # [R, I]
    y13 = torch.cat([y1, y3], dim=-1).to(inter.dtype)          # [R, 2I]

    if sorted_tokens is None:
        inter[:R].add_(y13)
    else:
        rows = max_rows if max_rows is not None else sorted_tokens.numel()
        y13_pad = torch.zeros((R + 1, y13.shape[1]), dtype=inter.dtype, device=inter.device)
        y13_pad[:R].copy_(y13)
        st = sorted_tokens[:rows].long().clamp(min=0, max=R)
        inter[:rows].add_(y13_pad.index_select(0, st))


def apply_3d_comp_w2(
    act: torch.Tensor,
    w2_v: torch.Tensor,
    w2_u: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_wts: torch.Tensor,
    out32: torch.Tensor,
    top_k: int,
    *,
    sorted_tokens: torch.Tensor = None,
    max_rows: int = None,
):
    """Per-3D-Tensor 规整 3D 低秩补偿 (down):
    w2_v: [E, r_2, I], w2_u: [E, K, r_2]。
    全程静态形状 index_select + torch.bmm, 零 host 同步, 100% 兼容 CUDA Graph。
    """
    if w2_v is None or w2_v.ndim != 3 or w2_v.shape[1] == 0:
        return
    R = flat_ids.numel()
    M = R // top_k
    I = w2_v.shape[2]
    K = w2_u.shape[1]
    idx = flat_ids.long()

    if sorted_tokens is None:
        act_routes = act[:R].to(w2_v.dtype).unsqueeze(-1)      # [R, I, 1]
    else:
        rows = max_rows if max_rows is not None else sorted_tokens.numel()
        act_pad = torch.zeros((R + 1, I), dtype=w2_v.dtype, device=act.device)
        st = sorted_tokens[:rows].long().clamp(min=0, max=R)
        act_pad.scatter_(0, st.unsqueeze(-1).expand(rows, I), act[:rows].to(w2_v.dtype))
        act_routes = act_pad[:R].unsqueeze(-1)                 # [R, I, 1]

    h2 = torch.bmm(w2_v.index_select(0, idx), act_routes)      # [R, r_2, 1]
    y2 = torch.bmm(w2_u.index_select(0, idx), h2).squeeze(-1)  # [R, K]
    y2_weighted = (y2.float() * flat_wts.float().unsqueeze(-1)).view(M, top_k, K).sum(dim=1)
    out32.add_(y2_weighted.to(out32.dtype))


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
    zeros: torch.Tensor = None,       # [E, K//GS, N] fp16, 非对称量化的每组零点
    w_transposed: bool = False,       # True = 权重按 [E, Kpack, N] (K-major) 存
    direct4: bool = False,            # True = 4-bit 槽位按 k 直接寻址 (去掉 3 个 tl.where)
    w_msb: torch.Tensor = None,       # [E_4bit, N, K//32] (或 K-major) int32
    msb_map: torch.Tensor = None,     # [E] int32
    bit_offsets: torch.Tensor = None, # [E + 1] int32: CSR 位平面切片偏移
    w_bits: int = 3,                  # Per-3D-Tensor 统一位宽 (2, 3, 或 4)
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

    varbit = bit_offsets is not None
    E = scales.shape[0]
    sw_k = 0
    if layout16:                           # [E, K, N] fp16 (不量化)
        K, N = w_packed.shape[1], w_packed.shape[2]
        sw_k = w_packed.stride(1)
        sw_n = 0
    elif w_transposed:                     # [E/B_tot, Kpack, N]
        Kpack, N = w_packed.shape[1], w_packed.shape[2]
        K = (Kpack * 32) if varbit else (Kpack * 8 if layout4 else Kpack // w_bits * 32)
        sw_n = 0
    else:                                  # [E/B_tot, N, Kpack]
        N, Kpack = w_packed.shape[1], w_packed.shape[2]
        K = (Kpack * 32) if varbit else (Kpack * 8 if layout4 else Kpack // w_bits * 32)
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

    has_zero = zeros is not None
    if has_zero:
        assert zeros.shape == scales.shape, (
            f"zeros 形状应与 scales 一致 {(E, K // group_size, N)}, "
            f"实际 {tuple(zeros.shape)}"
        )
    # HAS_ZERO=False 时内核不会读该指针, 用 scales 占位即可
    z_arg = zeros if has_zero else scales
    sz_e = zeros.stride(0) if has_zero else 0
    sz_k = zeros.stride(1) if has_zero else 0

    has_msb = (w_msb is not None) and (msb_map is not None) and (not layout4) and (not layout16) and (not varbit)
    w_msb_arg = w_msb if has_msb else w_packed
    msb_map_arg = msb_map if has_msb else expert_ids
    bit_offsets_arg = bit_offsets if varbit else expert_ids
    smsb_e = w_msb.stride(0) if has_msb else 0
    smsb_n = (0 if w_transposed else w_msb.stride(1)) if has_msb else 0

    grid = (gm, N // block_n)
    int3_moe_gemm_kernel[grid](
        a, w_packed, scales, z_arg,
        w_msb_arg, msb_map_arg, bit_offsets_arg,
        route_w if route_w is not None else a,
        sorted_token_ids, expert_ids, out,
        meta_arg, num_valid, num_post_arg, N, K,
        a.stride(0), sw_n, w_packed.stride(0),
        scales.stride(1), scales.stride(2), scales.stride(0),
        sz_e, sz_k, smsb_e, smsb_n, sw_k,
        BLOCK_M=block_m, SLOT=slot, BLOCK_N=block_n, BLOCK_K=block_k, GS=group_size,
        A_GATHER=a_gather, ADD=add, MUL_W=route_w is not None,
        HAS_META=has_meta, LAYOUT4=layout4, W_T=w_transposed, LAYOUT16=layout16,
        DIRECT4=direct4, HAS_ZERO=has_zero, HAS_MSB=has_msb, VARBIT=varbit, W_BITS=w_bits,
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


__all__ = [
    "int3_moe_gemm_kernel",
    "int3_moe_gemm",
    "routed_int3_gemv",
    "silu_mul_routes",
    "silu_mul",
    "apply_csr_comp_w13",
    "apply_csr_comp_w2",
    "apply_3d_comp_w13",
    "apply_3d_comp_w2",
]
