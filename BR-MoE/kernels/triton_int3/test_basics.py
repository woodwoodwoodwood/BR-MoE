"""int3 打包 / 量化 / MoE 对齐 的基础自检 (纯 torch, 不需要 Triton)。

    python3 test_basics.py
"""

import sys
import torch

sys.path.insert(0, ".")
from int3_moe.packing import quantize_int3_symmetric, pack_int3, unpack_int3, int3_storage_bits
from int3_moe.align import moe_align_block_size, check_align


def test_roundtrip():
    for (N, K) in [(64, 128), (256, 4096), (33, 32), (1, 32), (7, 96)]:
        q = torch.randint(0, 8, (N, K), dtype=torch.int32)
        p = pack_int3(q)
        assert p.shape == (N, K // 32 * 3), p.shape
        assert torch.equal(unpack_int3(p, K), q), f"往返不一致 {(N, K)}"
    print(f"[OK] pack/unpack 往返 (5 种形状), 存储 {int3_storage_bits():.2f} bit/weight")


def test_field_isolation():
    """每个 3-bit 字段只被一个值占据 —— 用"只在该位置放 7"的方式验证。"""
    N, K = 4, 64
    for k in range(K):
        q = torch.zeros(N, K, dtype=torch.int32)
        q[:, k] = 7
        p = pack_int3(q)
        assert torch.equal(unpack_int3(p, K), q), f"k={k} 字段冲突"
    print(f"[OK] {K} 个位置逐一位隔离, 无字段重叠/串扰")


def test_quantize():
    """对称 int3 的可表示范围是 [-4s, +3s] 而不是正负对称的。

    q = clamp(round(w/s) + 4, 0, 7) 的零点是 4, 所以能表示的是
        (q-4)*s ∈ {-4s, -3s, ..., +3s}
    => 取 scale = amax/3 才能让 |w| <= amax 的权重全部不被截断 (取 amax/4 会截断正值)。
    如果权重分布严重不对称, 应该改用带零点 (Layer3bitWithZeros / 非对称) 的格式。
    """
    torch.manual_seed(0)
    N, K, GS = 64, 256, 128
    w = torch.randn(N, K, dtype=torch.float32) * 0.02
    amax = w.reshape(N, K // GS, GS).abs().amax(dim=2).clamp_min(1e-6)
    s_naive = amax / 4.0          # 经典对称做法 -> 会截断
    s_ok = amax / 3.0             # 适配 [-4s,+3s] 的正确做法

    for s, tag in ((s_naive, "amax/4(会截断)"), (s_ok, "amax/3(正确)")):
        q = quantize_int3_symmetric(w, s, GS)
        assert q.min() >= 0 and q.max() <= 7, "量化值越界"
        p = pack_int3(q)
        assert torch.equal(unpack_int3(p, K), q), "量化后往返不一致"
        dq = (unpack_int3(p, K) - 4).float() * s.repeat_interleave(GS, dim=1)
        err = (dq - w).abs()
        half = s.repeat_interleave(GS, dim=1) * 0.5
        clipped = int((err > half + 1e-6).sum().item())
        print(f"     scale={tag:14s} max|err|={err.max():.5f}  超半格的元素={clipped}")

    q = quantize_int3_symmetric(w, s_ok, GS)
    p = pack_int3(q)
    dq = (unpack_int3(p, K) - 4).float() * s_ok.repeat_interleave(GS, dim=1)
    err = (dq - w).abs()
    assert (err <= s_ok.repeat_interleave(GS, dim=1) * 0.5 + 1e-6).all(), "scale=amax/3 时误差应 <= 半格"
    print(f"[OK] int3 对称量化 (scale=amax/3): max|err|={err.max():.5f}, 均值 scale={s_ok.mean():.5f}")


def rand_topk(M, E, TOPK, gen=None):
    """模拟真实 router 的 top-k 输出 (每个 token 的 TOPK 个 expert 互不相同)。"""
    logits = torch.randn(M, E, generator=gen)
    return logits.topk(TOPK, dim=1).indices.to(torch.long)


def test_align_uniform():
    g = torch.Generator().manual_seed(1)
    E, M, TOPK, BS = 8, 197, 2, 64
    topk_ids = rand_topk(M, E, TOPK, g)
    check_align(topk_ids, E, BS)
    sti, eid, npost = moe_align_block_size(topk_ids, E, BS)
    cnt = torch.bincount(topk_ids.reshape(-1), minlength=E)
    print(f"[OK] align (随机路由): M={M} topk={TOPK} E={E} -> num_post={npost}, "
          f"blocks={eid.numel()}, pad 行={int((sti >= M*TOPK).sum())}, 各专家 token 数={cnt.tolist()}")


def test_align_edge():
    """极端分布: 全部 token 挤进 1 个专家 (其余专家 0 token) / 均匀分布。"""
    E, M, TOPK, BS = 4, 65, 2, 64
    # 全走 expert 0: 130 个 pair -> 对齐到 192, 3 个 block 全是 expert 0
    topk_ids = torch.zeros(M, TOPK, dtype=torch.long)
    check_align(topk_ids, E, BS)
    sti, eid, npost = moe_align_block_size(topk_ids, E, BS)
    assert (npost, eid.tolist()) == (192, [0, 0, 0]), (npost, eid.tolist())
    assert int((sti >= M * TOPK).sum()) == 192 - 130, "pad 行数不对"

    # 均匀分布: 每个专家分到 1 个 block
    g = torch.Generator().manual_seed(7)
    M2 = E * 32
    topk_ids = rand_topk(M2, E, TOPK, g)
    check_align(topk_ids, E, BS)
    _, eid2, npost2 = moe_align_block_size(topk_ids, E, BS)
    print(f"[OK] align (极端分布): 全挤 1 专家 -> num_post=192/eid=[0,0,0]/pad={192-130}; "
          f"{E} 专家均匀 -> num_post={npost2}/blocks={eid2.numel()}")


def test_align_bs16():
    """block_size 必须是 kernel BLOCK_M 的倍数, 验证 16/32/64/128 都能工作。"""
    g = torch.Generator().manual_seed(2)
    E, M, TOPK = 6, 100, 3
    topk_ids = rand_topk(M, E, TOPK, g)
    for bs in (16, 32, 64, 128):
        check_align(topk_ids, E, bs)
    print("[OK] align block_size = 16 / 32 / 64 / 128")


if __name__ == "__main__":
    test_roundtrip()
    test_field_isolation()
    test_quantize()
    test_align_uniform()
    test_align_edge()
    test_align_bs16()
    print("\n=== 基础测试全部通过 ===")
