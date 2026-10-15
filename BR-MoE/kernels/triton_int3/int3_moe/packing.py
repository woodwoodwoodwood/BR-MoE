"""int3 (真 3-bit) 权重的打包 / 解包 —— 32 个权重压进 3 个 uint32, 零 bit 浪费。

============================================================================
布局 (与 BR-MoE / Marlin 的位技巧相同, 但去掉了 perm, 因为 Triton 不需要 ldmatrix)
============================================================================

记一组 32 个权重 (沿 K 连续) 为 v0..v31, 均为 int8, 取值 [0,7]
(已做对称偏移: q = clamp(round(w/s) + 4, 0, 7), 反量化时再减 4)。

  word0   bits 0-11 : v0  v1  v2  v3      (每个 3 bit, 从 bit0 起, 间隔 3)
          bits 16-27: v4  v5  v6  v7
          --- 空位 ---
          bits 12-14: v24
          bit  15   : v25 的 bit0
          bits 28-30: v28
          bit  31   : v29 的 bit0

  word1   bits 0-11 : v8  v9  v10 v11
          bits 16-27: v12 v13 v14 v15
          --- 空位 ---
          bits 12-13: v25 的高 2 位 (bit2..1)
          bits 14-15: v26 的低 2 位 (bit1..0)
          bits 28-29: v29 的高 2 位
          bits 30-31: v30 的低 2 位

  word2   bits 0-11 : v16 v17 v18 v19
          bits 16-27: v20 v21 v22 v23
          --- 空位 ---
          bit  12   : v26 的 bit2
          bits 13-15: v27
          bit  28   : v30 的 bit2
          bits 29-31: v31

为什么 v25/v26/v29/v30 要"劈开"跨 word 存放: 每个 16-bit half 只有 4 bit 空位,
3 个 word 的空位拼起来才够 8 个值 (24 bit)。按 4-4-4 阶梯错位存放后,
kernel 侧用 (w0>>12) | (w1>>12)<<4 | (w2>>12)<<8 就能拼出一条 12-bit 位流,
正好切成 4 个 3-bit 字段, 且顺序是 v24 v25 v26 v27 —— 反量化可以复用同一套掩码。

kernel 侧更省事的等价写法: 把这条位流当作"第 4 个 word"
    word3 = stream_lo | (stream_hi << 16)
然后 4 个 word 用完全相同的公式取 8 个值:
    k_in_group = 0..31,  wid = k_in_group // 8,  i = k_in_group % 8
    value = (word[wid] >> ((i%4)*3 + (i//4)*16)) & 0x7
"""

import torch
from torch import Tensor


def quantize_int3_symmetric(w: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """把 fp16 权重量化成 [0,7] 的无符号 int3。

    Args:
        w:          [N, K] fp16/int8, 原始权重 (沿 K 分组)
        scale:      [N, K // group_size] fp16, 每组的 scale
        group_size: 量化分组大小
    Returns:
        [N, K] int32, 取值 [0,7]  (反量化: (q - 4) * scale)
    """
    N, K = w.shape
    assert K % group_size == 0, "K 必须能被 group_size 整除"
    assert scale.shape == (N, K // group_size), f"scale 形状应为 {(N, K//group_size)}"
    s = scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    q = torch.round(w.to(torch.float32) / s).to(torch.int32)
    q = torch.clamp(q + 4, 0, 7)
    return q


def pack_int3_slots4(q: Tensor, transposed: bool = False) -> Tensor:
    """对照布局: 把 [0,7] 的 3-bit 值塞进 4-bit 槽位 (8 值/uint32, 4.0 bpw)。

    量化精度仍是 3-bit, 只是**存储**多浪费 1 bit/值, 换来 kernel 侧几乎为零的解包成本
    (只需 >> (4*i) & 0xF, 不需要 int3 那套"空位阶梯拼接")。
    用途: 量化"省 25% 权重显存"值不值"多花的整数指令"。
    """
    N, K = q.shape
    assert K % 8 == 0, "K 必须是 8 的倍数"
    v = q.to(torch.int64).reshape(N, K // 8, 8)
    out = torch.zeros(N, K // 8, dtype=torch.int64, device=q.device)
    for i in range(8):
        out |= (v[..., i] & 0xF) << (4 * i)
    out = out.to(torch.int32)
    return out.t().contiguous() if transposed else out


def unpack_int3_slots4(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_int3_slots4 的逆: [N, K//8] int32 -> [N, K] int32。"""
    if transposed:
        p = p.t().contiguous()
    N = p.shape[0]
    v = p.to(torch.int64).reshape(N, K // 8, 1)
    i = torch.arange(8, dtype=torch.int64, device=p.device)
    q = (v >> (4 * i)) & 0xF                      # [N, K//8, 8]
    return q.reshape(N, K).to(torch.int32)


def pack_int3(q: Tensor, transposed: bool = False) -> Tensor:
    """把 [0,7] 的 int3 值打包: [N, K] -> [N, K//32*3] int32 (或转置)。

    transposed=True 时返回 [K//32*3, N]，即 K-major。
    为什么需要它: kernel 一个 (m-block, n-block) 要对固定的若干 k 组
    读一整排 n 的 word。若按 [N, Kpack] 存, 相邻 n 相隔 Kpack 个 int32
    (本工程为 192 B), 每个 4 B 元素都要单独拉一条 128 B cache line ->
    过取 32 倍, 有效带宽只有 ~9 GB/s。转置后 n 连续, 访存完全合并。
    """
    assert q.dtype in (torch.int32, torch.int64), "请传入整数张量"
    N, K = q.shape
    assert K % 32 == 0, "K 必须是 32 的倍数"
    v = q.to(torch.int64).reshape(N, K // 32, 32)

    def g(i):
        return v[..., i] & 0x7

    w0 = (
        (g(0) << 0) | (g(1) << 3) | (g(2) << 6) | (g(3) << 9)
        | (g(4) << 16) | (g(5) << 19) | (g(6) << 22) | (g(7) << 25)
        | (g(24) << 12) | ((g(25) & 0x1) << 15)
        | (g(28) << 28) | ((g(29) & 0x1) << 31)
    )
    w1 = (
        (g(8) << 0) | (g(9) << 3) | (g(10) << 6) | (g(11) << 9)
        | (g(12) << 16) | (g(13) << 19) | (g(14) << 22) | (g(15) << 25)
        | ((g(25) & 0x6) << 11) | ((g(26) & 0x3) << 14)
        | ((g(29) & 0x6) << 27) | ((g(30) & 0x3) << 30)
    )
    w2 = (
        (g(16) << 0) | (g(17) << 3) | (g(18) << 6) | (g(19) << 9)
        | (g(20) << 16) | (g(21) << 19) | (g(22) << 22) | (g(23) << 25)
        | ((g(26) & 0x4) << 10) | (g(27) << 13)
        | ((g(30) & 0x4) << 26) | (g(31) << 29)
    )
    out = torch.stack([w0, w1, w2], dim=-1).to(torch.int32).reshape(N, K // 32 * 3)
    return out.t().contiguous() if transposed else out


def unpack_int3(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_int3 的逆运算: [N, K//32*3] int32 -> [N, K] int32 ([0,7])。"""
    if transposed:
        p = p.t().contiguous()
    N = p.shape[0]
    assert p.shape[1] == K // 32 * 3
    KB = K // 32
    p = p.to(torch.int64).reshape(N, KB, 3)
    w0, w1, w2 = p[..., 0], p[..., 1], p[..., 2]
    out = torch.empty(N, KB, 32, dtype=torch.int64, device=p.device)

    # 常规 24 个值: 每个 word 出 8 个 (低 half bits 0-11, 高 half bits 16-27)
    for i in range(4):
        out[..., 0 + i] = (w0 >> (3 * i)) & 0x7
        out[..., 4 + i] = (w0 >> (16 + 3 * i)) & 0x7
        out[..., 8 + i] = (w1 >> (3 * i)) & 0x7
        out[..., 12 + i] = (w1 >> (16 + 3 * i)) & 0x7
        out[..., 16 + i] = (w2 >> (3 * i)) & 0x7
        out[..., 20 + i] = (w2 >> (16 + 3 * i)) & 0x7

    # 溢出的 8 个值: 3 个 word 的空位拼成 12-bit 位流, 再切 4 个 3-bit
    lo = ((w0 >> 12) & 0xF) | (((w1 >> 12) & 0xF) << 4) | (((w2 >> 12) & 0xF) << 8)
    hi = ((w0 >> 28) & 0xF) | (((w1 >> 28) & 0xF) << 4) | (((w2 >> 28) & 0xF) << 8)
    for i in range(4):
        out[..., 24 + i] = (lo >> (3 * i)) & 0x7
        out[..., 28 + i] = (hi >> (3 * i)) & 0x7

    return out.reshape(N, K).to(torch.int32)


def pack_uint1_msb(q: Tensor, transposed: bool = False) -> Tensor:
    """把 4-bit 量化值 q in [0, 15] 的最高位 ((q >> 3) & 1) 打包成 1-bit 位平面:
    [N, K] -> [N, K // 32] int32 (或 transposed=True 时 [K // 32, N])。

    配合 pack_int3(q & 0x7) 使用, 可使混合 2/3/4-bit 专家层:
      - 2-bit / 3-bit 专家仅占 3.0 bit/weight (无需 MSB 位平面);
      - 4-bit 专家用 3.0 bit 主平面 + 1.0 bit MSB 位平面 = 4.0 bit/weight 100% 无损恢复:
        q = q_lo3 | (b_msb << 3).
    """
    assert q.dtype in (torch.int32, torch.int64), "请传入整数张量"
    N, K = q.shape
    assert K % 32 == 0, "K 必须是 32 的倍数"
    msb = ((q.to(torch.int64) >> 3) & 0x1).reshape(N, K // 32, 32)
    shifts = torch.arange(32, dtype=torch.int64, device=q.device)
    out = torch.sum(msb << shifts, dim=-1).to(torch.int32)
    return out.t().contiguous() if transposed else out


def unpack_uint1_msb(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_uint1_msb 的逆运算: [N, K // 32] int32 -> [N, K] int32 (取值 {0, 1})。"""
    if transposed:
        p = p.t().contiguous()
    N = p.shape[0]
    assert p.shape[1] == K // 32
    v = p.to(torch.int64).unsqueeze(-1)  # [N, K // 32, 1]
    shifts = torch.arange(32, dtype=torch.int64, device=p.device)
    bits = (v >> shifts) & 0x1           # [N, K // 32, 32]
    return bits.reshape(N, K).to(torch.int32)


def pack_bitplanes(q: Tensor, nbits: int, transposed: bool = False) -> Tensor:
    """将单个专家的 nbits (2/3/4/8) 量化整数矩阵 [N, K] 按 1-bit 位平面无损打包:
    返回形状:
      * transposed=False (N-major): [nbits, N, K // 32] int32
      * transposed=True  (K-major): [nbits, K // 32, N] int32

    核心优势:
      每个位平面固定形状为 [N, K // 32] (恰好 1.0 bit/weight)。
      因此同一层内不同位宽 b_e in {2, 3, 4} 的专家可以沿第 0 维 (位平面维)
      直接拼接成单个规则 3D 张量 [sum(b_e), N, K // 32], 配合 bit_offsets[E + 1]
      实现真正的零浪费异构位宽打包 (2-bit 占 2.0 bpw, 3-bit 占 3.0 bpw, 4-bit 占 4.0 bpw)。
    """
    assert q.dtype in (torch.int32, torch.int64), "请传入整数张量"
    assert nbits in (2, 3, 4, 8), f"不支持的 nbits={nbits}"
    N, K = q.shape
    assert K % 32 == 0, "K 必须是 32 的倍数"
    if nbits == 0:
        shape = (0, K // 32, N) if transposed else (0, N, K // 32)
        return torch.empty(shape, dtype=torch.int32, device=q.device)

    v = q.to(torch.int64).reshape(N, K // 32, 32)
    shifts = torch.arange(32, dtype=torch.int64, device=q.device)
    planes = []
    for b in range(nbits):
        bit_b = (v >> b) & 0x1                                   # [N, K // 32, 32]
        word_b = torch.sum(bit_b << shifts, dim=-1).to(torch.int32)  # [N, K // 32]
        planes.append(word_b.t().contiguous() if transposed else word_b)
    return torch.stack(planes, dim=0).contiguous()


def unpack_bitplanes(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_bitplanes 的逆运算: [nbits, N, K // 32] (或 K-major) int32 -> [N, K] int32。"""
    nbits = p.shape[0]
    if transposed:
        p = p.permute(0, 2, 1).contiguous()
    N = p.shape[1]
    assert p.shape[2] == K // 32
    shifts = torch.arange(32, dtype=torch.int64, device=p.device)
    out = torch.zeros((N, K // 32, 32), dtype=torch.int64, device=p.device)
    for b in range(nbits):
        wb = p[b].to(torch.int64).unsqueeze(-1)                  # [N, K // 32, 1]
        out |= ((wb >> shifts) & 0x1) << b
    return out.reshape(N, K).to(torch.int32)


def pack_varbit_experts(
    q_list: list[Tensor],
    expert_bits: list[int],
    transposed: bool = False,
) -> tuple[Tensor, Tensor]:
    """将 E 个不同位宽 (b_e in {2, 3, 4}) 的专家权重 [N, K] 打包进单个 3D 张量:
      * w_packed: [B_tot, N, K // 32] (或 K-major [B_tot, K // 32, N]) int32, B_tot = sum(expert_bits)
      * bit_offsets: [E + 1] int32, 专家 e 的位平面切片为 bit_offsets[e] : bit_offsets[e + 1]
    """
    assert len(q_list) == len(expert_bits) and len(q_list) > 0
    device = q_list[0].device
    offsets = [0]
    chunks = []
    for q_e, b_e in zip(q_list, expert_bits):
        b_int = int(b_e)
        offsets.append(offsets[-1] + b_int)
        chunks.append(pack_bitplanes(q_e, b_int, transposed=transposed))
    w_packed = torch.cat(chunks, dim=0).contiguous()
    bit_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return w_packed, bit_offsets


def unpack_varbit_experts(
    w_packed: Tensor,
    bit_offsets: Tensor,
    K: int,
    transposed: bool = False,
) -> Tensor:
    """将 CSR 位平面 3D 张量 [B_tot, N, K // 32] 解包回 [E, N, K] int32。"""
    E = bit_offsets.numel() - 1
    out_list = []
    for e in range(E):
        s_b = int(bit_offsets[e].item())
        e_b = int(bit_offsets[e + 1].item())
        out_list.append(unpack_bitplanes(w_packed[s_b:e_b], K, transposed=transposed))
    return torch.stack(out_list, dim=0)


def pack_int2(q: Tensor, transposed: bool = False) -> Tensor:
    """将 [N, K] 的 2-bit 无符号整数张量 (取值 0..3) 紧密打包为 [N, (K // 32) * 2] int32。
    每 32 个沿 K 连续的权重打包为相邻的 2 个 int32 字 (w0 存 bit-0 平面, w1 存 bit-1 平面)。
    当 transposed=True 时返回 K-major [(K // 32) * 2, N] int32，使单个静态大 Kernel (W_BITS=2)
    在沿 N 合并访存时仅需连续读取 wo 与 wo + wstep 两个字。
    """
    assert q.dtype in (torch.int32, torch.int64), "请传入整数张量"
    N, K = q.shape
    assert K % 32 == 0, "K 必须是 32 的倍数"
    v = q.to(torch.int64).reshape(N, K // 32, 32)
    assert int(v.min()) >= 0 and int(v.max()) <= 3, "q 超出 [0, 3]"
    shifts = torch.arange(32, dtype=torch.int64, device=q.device)
    w0 = torch.sum(((v >> 0) & 0x1) << shifts, dim=-1).to(torch.int32)
    w1 = torch.sum(((v >> 1) & 0x1) << shifts, dim=-1).to(torch.int32)
    out = torch.stack([w0, w1], dim=-1).reshape(N, (K // 32) * 2)
    return out.t().contiguous() if transposed else out


def unpack_int2(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_int2 的逆运算: [N, (K // 32) * 2] -> [N, K] int32。"""
    if transposed:
        p = p.t().contiguous()
    N = p.shape[0]
    assert p.shape[1] == (K // 32) * 2
    w = p.to(torch.int64).reshape(N, K // 32, 2)
    shifts = torch.arange(32, dtype=torch.int64, device=p.device)
    b0 = (w[:, :, 0:1] >> shifts) & 0x1
    b1 = (w[:, :, 1:2] >> shifts) & 0x1
    out = b0 | (b1 << 1)
    return out.reshape(N, K).to(torch.int32)


def pack_int4(q: Tensor, transposed: bool = False) -> Tensor:
    """将 [N, K] 的 4-bit 无符号整数张量 (取值 0..15) 紧密打包为 [N, (K // 32) * 4] int32。
    每 32 个沿 K 连续的权重打包为相邻的 4 个 int32 字 (w0..w3 存 bit-0..3 平面)。
    当 transposed=True 时返回 K-major [(K // 32) * 4, N] int32，供单个静态大 Kernel (W_BITS=4) 直接访存。
    """
    assert q.dtype in (torch.int32, torch.int64), "请传入整数张量"
    N, K = q.shape
    assert K % 32 == 0, "K 必须是 32 的倍数"
    v = q.to(torch.int64).reshape(N, K // 32, 32)
    assert int(v.min()) >= 0 and int(v.max()) <= 15, "q 超出 [0, 15]"
    shifts = torch.arange(32, dtype=torch.int64, device=q.device)
    words = [
        torch.sum(((v >> b) & 0x1) << shifts, dim=-1).to(torch.int32)
        for b in range(4)
    ]
    out = torch.stack(words, dim=-1).reshape(N, (K // 32) * 4)
    return out.t().contiguous() if transposed else out


def unpack_int4(p: Tensor, K: int, transposed: bool = False) -> Tensor:
    """pack_int4 的逆运算: [N, (K // 32) * 4] -> [N, K] int32。"""
    if transposed:
        p = p.t().contiguous()
    N = p.shape[0]
    assert p.shape[1] == (K // 32) * 4
    w = p.to(torch.int64).reshape(N, K // 32, 4)
    shifts = torch.arange(32, dtype=torch.int64, device=p.device)
    out = (
        ((w[:, :, 0:1] >> shifts) & 0x1)
        | (((w[:, :, 1:2] >> shifts) & 0x1) << 1)
        | (((w[:, :, 2:3] >> shifts) & 0x1) << 2)
        | (((w[:, :, 3:4] >> shifts) & 0x1) << 3)
    )
    return out.reshape(N, K).to(torch.int32)


def pack_uniform_bit(q: Tensor, nbits: int, transposed: bool = False) -> Tensor:
    """按给定全张量统一位宽 nbits in {2, 3, 4} 打包单个 [N, K] 矩阵为 [N, (K // 32) * nbits]。"""
    if nbits == 2:
        return pack_int2(q, transposed=transposed)
    if nbits == 3:
        return pack_int3(q, transposed=transposed)
    if nbits == 4:
        return pack_int4(q, transposed=transposed)
    raise ValueError(f"Unsupported uniform tensor bit-width: {nbits}")


def unpack_uniform_bit(p: Tensor, K: int, nbits: int, transposed: bool = False) -> Tensor:
    """pack_uniform_bit 的逆运算。"""
    if nbits == 2:
        return unpack_int2(p, K, transposed=transposed)
    if nbits == 3:
        return unpack_int3(p, K, transposed=transposed)
    if nbits == 4:
        return unpack_int4(p, K, transposed=transposed)
    raise ValueError(f"Unsupported uniform tensor bit-width: {nbits}")


def int3_storage_bits() -> float:
    """实际存储位宽 (无浪费)。"""
    return 96.0 / 32.0


__all__ = [
    "quantize_int3_symmetric",
    "pack_int2",
    "unpack_int2",
    "pack_int3",
    "unpack_int3",
    "pack_int4",
    "unpack_int4",
    "pack_uniform_bit",
    "unpack_uniform_bit",
    "pack_int3_slots4",
    "unpack_int3_slots4",
    "pack_uint1_msb",
    "unpack_uint1_msb",
    "pack_bitplanes",
    "unpack_bitplanes",
    "pack_varbit_experts",
    "unpack_varbit_experts",
    "int3_storage_bits",
]
