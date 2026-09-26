"""把 Triton 路径的 int3 打包权重转成 brmoe/Marlin CUDA kernel 的布局。

布局链 (对每专家的 [N, K] 量化值):
    unpack_int3            [E, N, Kpack] -> [E, N, K] int (0..7)
    转置                   -> [E, K, N]
    16x16 tile permute     reshape(E, K//16, 16, N//16, 16).permute(0,1,3,2,4)
    Marlin _perm           每 32 列按 _perm 交错
    位打包                 32 值 -> q1 (2 字) + q2 (1 字), 含跨字位 (值 24..31)

零点折叠: 我们的格式是 w = (q - z) * s (fp16 per-group 零点, 减法),
CUDA kernel 是 w = q*s + z' (__hfma2) -> z' = -z*s, 离线折叠。

scales/zeros 走 get_scale_perm 重排 (与 brmoe pack() 相同)。
"""
import torch


# ---- Marlin 置换表 (逐行复制自 BR-MoE/kernels/brmoe/__init__.py::_get_perms,
#      内联以避免 import 已编译扩展) ----
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = torch.tensor(perm).reshape((-1, 8))
    interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    perm = perm[:, interleave].flatten()
    return perm


_perm = _get_perms()


def get_scale_perm(groupsize: int) -> torch.Tensor:
    """groupsize 必须是 8 的倍数; 8 x (groupsize/8) 列优先布局。"""
    if groupsize % 8 != 0:
        raise ValueError("groupsize must be a multiple of 8")
    rows = groupsize // 8
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(rows)])
    return torch.tensor(scale_perm)


def _bitpack_32(v: torch.Tensor):
    """v: [..., 32] int64 (0..7) -> q1 [..., 2] int32, q2 [..., 1] int32。

    逐位复刻 brmoe/__init__.py pack() 里的 numpy 循环 (值 24..31 跨字)。
    """
    z = torch.zeros_like(v[..., 0])
    q1w0, q1w1, q2 = z.clone(), z.clone(), z.clone()
    for j in range(4):
        q1w0 |= v[..., 0 + j] << (3 * j)
        q1w0 |= v[..., 4 + j] << (16 + 3 * j)
        q1w1 |= v[..., 8 + j] << (3 * j)
        q1w1 |= v[..., 12 + j] << (16 + 3 * j)
        q2 |= v[..., 16 + j] << (3 * j)
        q2 |= v[..., 20 + j] << (16 + 3 * j)
    q1w0 |= v[..., 24] << 12
    q1w0 |= v[..., 28] << 28
    q1w0 |= (v[..., 25] & 0x1) << 15
    q1w0 |= (v[..., 29] & 0x1) << 31
    q1w1 |= (v[..., 25] & 0x6) << 11
    q1w1 |= (v[..., 26] & 0x3) << 14
    q1w1 |= (v[..., 29] & 0x6) << 27
    q1w1 |= (v[..., 30] & 0x3) << 30
    q2 |= (v[..., 26] & 0x4) << 10
    q2 |= (v[..., 30] & 0x4) << 26
    q2 |= v[..., 27] << 13
    q2 |= v[..., 31] << 29
    q1 = torch.stack([q1w0, q1w1], dim=-1)
    return q1.to(torch.int32), q2.unsqueeze(-1).to(torch.int32)


def repack_one(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, gs: int):
    """单专家: q [N, K] int (0..7), s/z [K//gs, N] fp16 (我们的布局)
    -> B1 [K//16, N] int32, B2 [K//16, N//2] int32, s' [K//gs, N], z' 同 s'。
    """
    N, K = q.shape
    assert K % 16 == 0 and N % 16 == 0
    # ---- 权重: [N, K] -> [K, N] -> tile permute -> _perm -> 位打包 ----
    w = q.t().contiguous()                                       # [K, N]
    w = w.reshape(K // 16, 16, N // 16, 16).permute(0, 2, 1, 3)  # [K//16, N//16, 16, 16]
    w = w.reshape(K // 16, N * 16)
    w = w.reshape(K // 16, -1, _perm.numel())[:, :, _perm]       # Marlin 交错
    w = w.reshape(K // 16, -1).to(torch.int64)
    q1, q2 = _bitpack_32(w.reshape(K // 16, -1, 32))
    B1 = q1.reshape(K // 16, N).contiguous()                     # [K//16, 2*N/2 = N]
    B2 = q2.reshape(K // 16, N // 2).contiguous()

    # ---- scale/zero: 折叠零点后按 scale_perm 重排 ----
    sp = get_scale_perm(gs).to(s.device)
    zf = (-(z.to(torch.float32) * s.to(torch.float32))).to(torch.float16)
    s2 = s.reshape(-1, sp.numel())[:, sp].reshape(K // gs, N).contiguous()
    z2 = zf.reshape(-1, sp.numel())[:, sp].reshape(K // gs, N).contiguous()
    return B1, B2, s2, z2


@torch.no_grad()
def repack_moe(packed: dict):
    """Triton packed dict -> CUDA packed dict (全部专家堆叠, 带专家维)。

    输入 (tools/brmoe_int3_vllm/kernel.py::build_packed 的同构):
        w13_q [E, 2I, Kpack] int32 (N-major), s13/z13 [E, K//gs, 2I] fp16
        w2_q  [E, K, Ipack] int32,             s2/z2  [E, I//gs, K] fp16
    输出:
        B13_1 [E, K//16, 2I]  B13_2 [E, K//16, I]   s13/z13 [E, K//gs, 2I]
        B2_1  [E, I//16, K]   B2_2  [E, I//16, K//2]  s2/z2  [E, I//gs, K]
    """
    from int3_moe.packing import unpack_int3

    gs = int(packed["group_size"])
    E = packed["w13_q"].shape[0]
    Kpack = packed["w13_q"].shape[2]
    K = Kpack // 3 * 32
    Ipack = packed["w2_q"].shape[2]
    I = Ipack // 3 * 32

    out = {}
    for tag, wq, sk, zk, N, Kk in (
        ("13", packed["w13_q"], packed["s13"], packed["z13"], 2 * I, K),
        ("2", packed["w2_q"], packed["s2"], packed["z2"], K, I),
    ):
        q = unpack_int3(wq.reshape(E * N, Kk // 32 * 3), Kk)     # [E*N, Kk]
        q = q.reshape(E, N, Kk)
        B1s, B2s, ss, zs = [], [], [], []
        for e in range(E):
            B1, B2, s2, z2 = repack_one(q[e], sk[e], zk[e], gs)
            B1s.append(B1); B2s.append(B2); ss.append(s2); zs.append(z2)
        out[f"B{tag}_1"] = torch.stack(B1s)
        out[f"B{tag}_2"] = torch.stack(B2s)
        out[f"s{tag}"] = torch.stack(ss)
        out[f"z{tag}"] = torch.stack(zs)
    out["group_size"] = gs
    return out
