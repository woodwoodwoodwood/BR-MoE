"""int3 MoE grouped GEMM 的数值正确性 + 性能对照。

    python3 test_kernel.py            # 正确性
    python3 test_kernel.py --bench    # 附性能对照
"""

import sys
import time
import torch

sys.path.insert(0, ".")
from int3_moe.align import moe_align_block_size
from int3_moe.align_triton import moe_align_block_size_triton
from int3_moe.kernel import int3_moe_gemm
from int3_moe.ops import pack_moe_weights, dequant_int3, fused_moe_int3, ref_fused_moe

DEV = "cuda"
torch.manual_seed(0)


def make_moe(E, M, K, I, topk, group_size, scale=0.02):
    w13 = (torch.randn(E, 2 * I, K, device=DEV) * scale).half()
    w2 = (torch.randn(E, K, I, device=DEV) * scale).half()
    packed = pack_moe_weights(w13, w2, group_size)
    x = (torch.randn(M, K, device=DEV) * 0.5).half()
    logits = torch.randn(M, E, device=DEV)
    tw, ti = logits.softmax(-1).topk(topk, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).float()
    return x, tw, ti.to(torch.int64), packed, w13, w2


def test_single_grouped_gemm():
    """第一层 grouped GEMM 单独对照: 与"逐专家 fp16 矩阵乘"比。"""
    E, M, K, N, topk, GS = 8, 96, 256, 128, 2, 128
    w = (torch.randn(E, N, K, device=DEV) * 0.03).half()
    x = (torch.randn(M, K, device=DEV) * 0.5).half()
    logits = torch.randn(M, E, device=DEV)
    _, ti = logits.topk(topk, dim=-1)
    ti = ti.to(torch.int64)

    amax = w.reshape(E * N, K // GS, GS).abs().amax(-1).clamp_min(1e-8)
    from int3_moe.packing import quantize_int3_symmetric, pack_int3
    s = (amax / 3.0).half().reshape(E, K // GS, N)
    wq = pack_int3(quantize_int3_symmetric(w.reshape(E * N, K), s.reshape(E * N, K // GS), GS)
                   ).reshape(E, N, K // 32 * 3)

    sti, eid, npost = moe_align_block_size(ti, E, 64)
    # sorted_token_ids 里装的是"原 token 下标" (范围 [0, M)), 不是 (token, k) 对下标
    assert int(sti[sti < M * topk].max()) < M, "sorted_token_ids 应为 token 下标"
    # 这里用的是 pack_int3 直接产出的 N-major [E, N, Kpack] 布局
    out = int3_moe_gemm(x, wq, s, sti, eid, npost, M * topk,
                        group_size=GS, a_gather=True, add=False, out_m=npost,
                        w_transposed=False)

    # 参考: 解包权重 -> 逐专家 fp16 matmul, 按 sorted 行号排列
    wd = dequant_int3(wq, s, K, GS, transposed=False).float()
    ref = torch.zeros(npost, N, device=DEV)
    for r in range(npost):
        t = int(sti[r])
        if t >= M * topk:
            continue
        e = int(eid[r // 64])
        ref[r] = x[t].float() @ wd[e].t()

    err = (out.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    print(f"[GEMM] grouped (M={M}, K={K}, N={N}, E={E}, topk={topk}, npost={npost})")
    print(f"       max|err|={err.max():.5f}  最大相对误差={(err/denom).max():.4f}  "
          f"参考 max|y|={ref.abs().max():.3f}")
    assert err.max() < 0.15, f"误差过大 {err.max()}"
    return err.max().item()


def test_fused_moe():
    """整层 fused MoE 对照 (fast / slow 两条路径) 与逐专家循环 (BR-MoE 现状) 比。"""
    E, M, K, I, topk, GS = 8, 128, 512, 1024, 2, 128
    x, tw, ti, packed, w13, w2 = make_moe(E, M, K, I, topk, GS)
    yref = ref_fused_moe(x, tw, ti, packed)
    print(f"[FUSED] E={E} M={M} K={K} I={I} topk={topk} GS={GS}  参考 max|y|={yref.float().abs().max():.3f}")
    worst = 0.0
    for fast in (False, True):
        y = fused_moe_int3(x, tw, ti, packed, fast=fast)
        # fast 路径用 atomic_add 累加, 行内顺序不确定 -> 不能要求 bit-exact
        err = (y.float() - yref.float()).abs()
        denom = yref.float().abs().clamp_min(1e-2)
        tag = "fast(Triton align+融合激活)" if fast else "slow(torch align)"
        print(f"        {tag:26s} max|err|={err.max():.5f}  最大相对误差={(err/denom).max():.4f}")
        assert err.max() < 0.3, f"fast={fast} 误差过大 {err.max()}"
        worst = max(worst, err.max().item())
    return worst


def test_edge_cases():
    """边界情况, 两条路径都验证。"""
    E, K, I, GS = 8, 256, 256, 128
    cases = []
    for M, topk, mode in [(64, 1, "topk1"), (64, 3, "topk3"), (17, 2, "M小"),
                          (200, 2, "M非对齐"), (1, 2, "M=1")]:
        x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
        yr = ref_fused_moe(x, tw, ti, packed)
        for fast in (False, True):
            y = fused_moe_int3(x, tw, ti, packed, fast=fast)
            e = (y.float() - yr.float()).abs().max().item()
            tag = "fast" if fast else "slow"
            assert e < 0.3, f"{mode} {tag} 误差过大 {e}"
            cases.append((f"{mode}/{tag}", M, topk, e))
    # 单专家全收 (所有 token 都挤到 expert 0)
    M, topk = 64, 2
    x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
    ti = torch.zeros_like(ti)
    yr = ref_fused_moe(x, tw, ti, packed)
    for fast in (False, True):
        y = fused_moe_int3(x, tw, ti, packed, fast=fast)
        e = (y.float() - yr.float()).abs().max().item()
        assert e < 0.3, f"单专家全收 fast={fast} 误差过大 {e}"
        cases.append((f"单专家全收/{'fast' if fast else 'slow'}", M, topk, e))
    for name, M, topk, e in cases:
        print(f"[EDGE] {name:20s} M={M:4d} topk={topk}  max|err|={e:.5f}")
    return cases


def per_expert_int3(x, tw, ti, packed, block_size=64, GS=128):
    """基线 1: 逐专家调用同一个 int3 kernel (不做跨专家分组, 等价于 BR-MoE 现状的调度方式)。"""
    E = packed["w13_q"].shape[0]
    K = x.shape[1]
    I = packed["w2_q"].shape[2] // 3 * 32
    M, topk = ti.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    for e in range(E):
        idx = (ti == e).nonzero(as_tuple=False)
        n_e = idx.shape[0]
        if n_e == 0:
            continue
        n_pad = (n_e + block_size - 1) // block_size * block_size
        sti = torch.full((n_pad,), n_e, dtype=torch.int32, device=DEV)   # 哨兵 = n_e
        sti[:n_e] = idx[:, 0].to(torch.int32)
        eid = torch.zeros(n_pad // block_size, dtype=torch.int32, device=DEV)
        rw = torch.zeros(n_pad, dtype=torch.float32, device=DEV)
        rw[:n_e] = tw[idx[:, 0], idx[:, 1]]

        w1 = packed["w13_q"][e:e + 1].contiguous()
        s1 = packed["s13"][e:e + 1].contiguous()
        inter = int3_moe_gemm(x, w1, s1, sti, eid, n_pad, n_e, group_size=GS,
                              a_gather=True, add=False, out_m=n_pad)
        g, u = inter[:, :I], inter[:, I:]
        act = (g.float() * torch.sigmoid(g.float()) * u.float()).half().contiguous()
        w2 = packed["w2_q"][e:e + 1].contiguous()
        s2 = packed["s2"][e:e + 1].contiguous()
        int3_moe_gemm(act, w2, s2, sti, eid, n_pad, n_e, group_size=GS,
                      route_w=rw, out=out, a_gather=False, add=True)
    return out


def per_expert_fp16(x, tw, ti, W13, W2, I):
    """基线 2: 反量化成 fp16 权重后逐专家 matmul (BR-MoE 的 dequant-then-matmul 路线)。"""
    E = W13.shape[0]
    K = x.shape[1]
    M = x.shape[0]
    out = torch.zeros(M, K, dtype=torch.float32, device=DEV)
    xf = x.float()
    for e in range(E):
        hit = (ti == e)
        rows = hit.any(1)
        if not bool(rows.any()):
            continue
        h = xf[rows] @ W13[e].t().float()
        g, u = h[:, :I], h[:, I:]
        act = ((g * torch.sigmoid(g)) * u)
        y = act @ W2[e].t().float()
        for t in range(hit.shape[1]):
            m = hit[rows, t]
            out[rows] += torch.where(m[:, None], y * tw[rows, t:t + 1].float(),
                                     torch.zeros_like(y))
    return out


def test_repeat_calls():
    """回归测试: 同一进程内用相同 shapes 反复调用。

    这个用例是必需的 —— 曾经因为 align 的计数累加器没复位, 同一配置第二次调用起
    结果就全错 (num_post 被放大到超出预分配缓冲)。单次调用是发现不了的。
    """
    E, M, K, I, topk, GS = 8, 64, 512, 1024, 2, 128
    x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
    yr = ref_fused_moe(x, tw, ti, packed)
    worst = 0.0
    for i in range(6):
        y = fused_moe_int3(x, tw, ti, packed)
        e = (y.float() - yr.float()).abs().max().item()
        worst = max(worst, e)
        assert e < 0.3, f"第 {i+1} 次调用结果错误 (max|err|={e})"
    # buffer 复用 + 不同 block_k / 两种 align 交替, 确认不互相污染
    for bk in (64, 32, 64, 32):
        y = fused_moe_int3(x, tw, ti, packed, block_k=bk)
        e = (y.float() - yr.float()).abs().max().item()
        assert e < 0.3, f"block_k={bk} 结果错误 (max|err|={e})"
    for fast in (False, True, False, True):
        y = fused_moe_int3(x, tw, ti, packed, fast=fast)
        e = (y.float() - yr.float()).abs().max().item()
        assert e < 0.3, f"fast={fast} 结果错误 (max|err|={e})"
    print(f"[REPEAT] 连续 6 次 + bk 交替 4 次 + align 交替 4 次, max|err|={worst:.5f}")
    return worst


def test_slot_variants():
    """回归测试: 多专家 per block (block_m > slot)。

    覆盖两个容易踩的坑:
      1. slot < block_m 时 num_post 不是 block_m 的整数倍 -> 末块只有部分行有效,
         必须有行掩码。曾经因为"整块越界"的子块去读未初始化的 expert_ids
         (它只写到 num_blocks-1), 导致 expert*SW_E 飞出边界 -> 非法访存。
      2. 每个专家只有 1 行时, slot=16 会让一个 block 里装下 4 个不同专家。
    """
    # (E, M, K, I, topk, GS, block_m, slot) —— 特意混入 num_post % block_m != 0 的组合
    cases = [
        (256, 8, 1024, 512, 8, 128, 64, 16),    # 64 对散到 59 个专家, num_post=944 (944%64=48)
        (256, 32, 1024, 512, 8, 128, 64, 16),   # num_post=2560 (2560%64=0)
        (256, 32, 1024, 512, 8, 128, 64, 32),
        (8, 128, 512, 1024, 2, 128, 64, 16),
        (8, 128, 512, 1024, 2, 128, 32, 16),
        (8, 1, 512, 1024, 2, 128, 64, 16),      # 极小 M
    ]
    worst = 0.0
    for E, M, K, I, topk, GS, bm, sl in cases:
        x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
        yr = ref_fused_moe(x, tw, ti, packed)
        e = None
        for _ in range(3):                      # 重复调用: 覆盖 align buffer 复用
            y = fused_moe_int3(x, tw, ti, packed, block_m=bm, slot=sl)
            e = (y.float() - yr.float()).abs().max().item()
            assert e < 0.3, f"E={E} M={M} block_m={bm} slot={sl}: max|err|={e}"
        worst = max(worst, e)
        print(f"[SLOT] E={E:>3} M={M:>3} topk={topk} block_m={bm:>2} slot={sl:>2}  max|err|={e:.5f}")
    return worst


def bench():
    """grouped int3 vs 逐专家 int3 vs 逐专家 fp16 (dequant 后 matmul)。"""
    E, K, I, GS, topk = 8, 2048, 4096, 128, 2
    print("\n" + "=" * 96)
    print(f"性能对照 [T4 15GB, Triton fp16 mma]  E={E} K={K} I={I} topk={topk} group={GS}")
    print("=" * 96)
    hdr = (f"{'M':>6} {'场景':>8} | {'① grouped fast':>14} {'② grouped slow':>14} "
           f"{'③ 逐专家 int3':>14} {'④ 逐专家 fp16':>14} | {'①/②':>5} {'③/①':>6} {'④/①':>6}")
    print(hdr)
    print("-" * 108)
    for M, tag in [(1, "decode"), (8, "小 batch"), (64, "中 batch"), (512, "prefill")]:
        x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
        tr = packed.get("w_transposed", False)
        W13 = dequant_int3(packed["w13_q"], packed["s13"], K, GS, transposed=tr).float()
        W2 = dequant_int3(packed["w2_q"], packed["s2"], I, GS, transposed=tr).float()

        def timeit(fn, iters=5, reps=4):
            """多轮取最小: T4 会随温度降频, 取 min 才能避免"后测的更慢"这种顺序偏差。"""
            best = float("inf")
            for _ in range(reps):
                fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t0) / iters * 1e3)
            return best

        t_fast = timeit(lambda: fused_moe_int3(x, tw, ti, packed, fast=True))
        t_slow = timeit(lambda: fused_moe_int3(x, tw, ti, packed, fast=False))
        t2 = timeit(lambda: per_expert_int3(x, tw, ti, packed), iters=3)
        t3 = timeit(lambda: per_expert_fp16(x, tw, ti, W13, W2, I), iters=3)
        print(f"{M:>6} {tag:>8} | {t_fast:>14.3f} {t_slow:>14.3f} {t2:>14.3f} {t3:>14.3f} | "
              f"{t_slow/t_fast:>4.2f}x {t2/t_fast:>5.2f}x {t3/t_fast:>5.2f}x")
    print("-" * 108)
    print("单位: 毫秒 / 1 个 MoE 层 (含两次 GEMM + 激活)")


def bench_blockk():
    """优化 2 的效果: BLOCK_K 越大越能摊薄循环开销 (注意内层仍是 32 宽 group)。"""
    E, K, I, GS, topk = 8, 2048, 4096, 128, 2
    print("\n" + "=" * 78)
    print(f"优化 2: BLOCK_K 扫描 [T4]  E={E} K={K} I={I} group={GS}")
    print("=" * 78)
    print(f"{'M':>6} {'场景':>8} | {'bk=32':>9} {'bk=64':>9} {'bk=128':>9} | {'64/32':>7} {'128/32':>7}")
    print("-" * 78)

    def timeit(fn, iters=5, reps=4):
        best = float("inf")
        for _ in range(reps):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / iters * 1e3)
        return best

    for M, tag in [(1, "decode"), (8, "小 batch"), (64, "中 batch"), (512, "prefill")]:
        x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
        yr = ref_fused_moe(x, tw, ti, packed)
        ts = {}
        for bk in (32, 64, 128):
            # 顺带校验每个配置都正确 (防止"快但算错")
            y = fused_moe_int3(x, tw, ti, packed, block_k=bk)
            err = (y.float() - yr.float()).abs().max().item()
            assert err < 0.3, f"M={M} block_k={bk} 结果错误 {err}"
            ts[bk] = timeit(lambda: fused_moe_int3(x, tw, ti, packed, block_k=bk))
        print(f"{M:>6} {tag:>8} | {ts[32]:>9.3f} {ts[64]:>9.3f} {ts[128]:>9.3f} | "
              f"{ts[32]/ts[64]:>6.2f}x {ts[32]/ts[128]:>6.2f}x")
    print("-" * 78)


def bench_layouts():
    """三路对照: int3 (3.0bpw) vs int4 槽位 (4.0bpw) vs fp16 (16bpw)。

    三者跑**同一套 grouped kernel**(同一 launch 结构 / 同一 align / 同一融合激活),
    只换权重的存储与解码方式, 所以差别纯粹来自"每个权重占多少字节 + 解码多贵":

      int3  稠密 3-bit: 32 值压进 3 个 uint32, 解码要"空位阶梯拼接"
      int4  4-bit 槽位: 8 值/uint32, 解码只要 (w>>(4i))&0xF
            -> 这也是"真 int4"的准确速度代理: 两者都是 8 值/uint32、同样的解码指令,
               差别只在量化取值 (影响精度, 不影响速度)
      fp16  不量化: 直接载入 fp16 权重 (权重字节 2.67x int3), 无解码

    量化误差以"真 fp16"为基准 (同一个 x / 同样的路由), 报 max|y - y_fp16|。
    """
    E, K, I, GS, topk = 8, 2048, 4096, 128, 2
    print("\n" + "=" * 96)
    print(f"三路对照 [T4 15GB, 同一 grouped kernel]  E={E} K={K} I={I} topk={topk} group={GS}")
    print("=" * 96)

    def timeit(fn, iters=5, reps=4):
        best = float("inf")
        for _ in range(reps):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / iters * 1e3)
        return best

    print(f"{'M':>5} {'场景':>8} | {'int3 3.0bpw':>12} {'int4 4.0bpw':>12} {'fp16 16bpw':>12} | "
          f"{'int4/int3':>10} {'fp16/int3':>10} | {'int3 量化误差':>13}")
    print("-" * 96)
    for M, tag in [(1, "decode"), (8, "小 batch"), (64, "中 batch"), (512, "prefill")]:
        x, tw, ti, p3, w13, w2 = make_moe(E, M, K, I, topk, GS)
        pk = {"int3": p3,
              "int4": pack_moe_weights(w13, w2, GS, layout="int4"),
              "fp16": pack_moe_weights(w13, w2, GS, layout="fp16")}
        y16 = ref_fused_moe(x, tw, ti, pk["fp16"]).float()     # 真 fp16 = 精度基准
        t, qerr = {}, {}
        for name, p in pk.items():
            y = fused_moe_int3(x, tw, ti, p)
            qerr[name] = (y.float() - y16).abs().max().item()
            t[name] = timeit(lambda: fused_moe_int3(x, tw, ti, p))
        print(f"{M:>5} {tag:>8} | {t['int3']:>12.3f} {t['int4']:>12.3f} {t['fp16']:>12.3f} | "
              f"{t['int4']/t['int3']:>9.2f}x {t['fp16']/t['int3']:>9.2f}x | "
              f"{qerr['int3']:>13.5f}  (int4 {qerr['int4']:.5f})")
    print("-" * 96)
    mb = {}
    x, tw, ti, p3, w13, w2 = make_moe(E, 1, K, I, topk, GS)
    for name, p in (("int3", p3),
                    ("int4", pack_moe_weights(w13, w2, GS, layout="int4")),
                    ("fp16", pack_moe_weights(w13, w2, GS, layout="fp16"))):
        mb[name] = (p["w13_q"].numel() + p["w2_q"].numel()) * p["w13_q"].element_size() / 2**20
    print(f"每层权重字节: int3 = {mb['int3']:.1f} MiB, int4 = {mb['int4']:.1f} MiB, "
          f"fp16 = {mb['fp16']:.1f} MiB  (int4/int3 = {mb['int4']/mb['int3']:.2f}x, "
          f"fp16/int3 = {mb['fp16']/mb['int3']:.2f}x)")
    print("(int4 槽位与 int3 用同一套量化值 -> 量化误差应完全相同)")


def bench_layout():
    """优化 4: 稠密 int3 (3.0 bpw) vs 4-bit 槽位 (4.0 bpw)。

    两者量化精度完全一样, 数值必须一致; 差别只有
      int3  : 权重省 25% 显存, 但解包要多做"空位阶梯拼接" (~11 条整数指令/值)
      slot4 : 权重多 33% 显存, 解包只要 移位+掩码 (~5 条/值)
    用来回答: 在这张卡上, 省下的带宽够不够抵掉多花的算力。
    """
    E, K, I, GS, topk = 8, 2048, 4096, 128, 2
    print("\n" + "=" * 84)
    print(f"优化 4: 权重布局对照 [T4]  E={E} K={K} I={I} group={GS}")
    print("=" * 84)
    print(f"{'M':>6} {'场景':>8} | {'int3 3.0bpw':>12} {'slot4 4.0bpw':>13} | "
          f"{'int3/slot4':>11} | 数值一致性")
    print("-" * 84)

    def timeit(fn, iters=5, reps=4):
        best = float("inf")
        for _ in range(reps):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / iters * 1e3)
        return best

    for M, tag in [(1, "decode"), (8, "小 batch"), (64, "中 batch"), (512, "prefill")]:
        x, tw, ti, _, w13, w2 = make_moe(E, M, K, I, topk, GS)
        p3 = pack_moe_weights(w13, w2, GS, layout="int3")
        p4 = pack_moe_weights(w13, w2, GS, layout="int4")
        mb3 = p3["w13_q"].numel() * 4 / 2**20 + p3["w2_q"].numel() * 4 / 2**20
        mb4 = p4["w13_q"].numel() * 4 / 2**20 + p4["w2_q"].numel() * 4 / 2**20
        # 同一套量化 -> 两种布局的输出必须一致 (这是很强的等价性检验)
        y3 = fused_moe_int3(x, tw, ti, p3)
        y4 = fused_moe_int3(x, tw, ti, p4)
        same = (y3.float() - y4.float()).abs().max().item()
        t3 = timeit(lambda: fused_moe_int3(x, tw, ti, p3))
        t4 = timeit(lambda: fused_moe_int3(x, tw, ti, p4))
        print(f"{M:>6} {tag:>8} | {t3:>12.3f} {t4:>13.3f} | {t4/t3:>10.2f}x | "
              f"|Δ|={same:.6f} {'OK' if same < 1e-3 else 'FAIL'}")
        assert same < 1e-3, f"两种布局数值不一致 {same}"
    print("-" * 84)
    print(f"权重显存: int3 = {mb3:.1f} MiB, slot4 = {mb4:.1f} MiB (slot4 多 {mb4/mb3-1:.0%})")


def bench_breakdown():
    """定位优化收益来源: torch align vs Triton align, 以及两条完整路径。"""
    E, K, I, GS, topk = 8, 2048, 4096, 128, 2
    print("\n" + "=" * 104)
    print("优化收益拆解 (毫秒/层)")
    print("=" * 104)
    print(f"{'M':>6} | {'align(torch)':>13} {'align(Triton)':>14} {'加速':>7} | "
          f"{'fused slow':>11} {'fused fast':>11} {'加速':>7}")
    print("-" * 104)

    def timeit(fn, iters=10, reps=4):
        best = float("inf")
        for _ in range(reps):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / iters * 1e3)
        return best

    for M in (1, 64, 512):
        x, tw, ti, packed, _, _ = make_moe(E, M, K, I, topk, GS)
        twf = tw.reshape(-1).float()

        t_align_torch = timeit(
            lambda: moe_align_block_size(ti, E, 64, flat_values=twf))
        t_align_tri = timeit(
            lambda: moe_align_block_size_triton(ti, E, 64, flat_values=twf))

        t_slow = timeit(lambda: fused_moe_int3(x, tw, ti, packed, fast=False))
        t_fast = timeit(lambda: fused_moe_int3(x, tw, ti, packed, fast=True))

        print(f"{M:>6} | {t_align_torch:>13.3f} {t_align_tri:>14.3f} "
              f"{t_align_torch/t_align_tri:>6.1f}x | "
              f"{t_slow:>11.3f} {t_fast:>11.3f} {t_slow/t_fast:>6.2f}x")
    print("-" * 104)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}  "
          f"triton={__import__('triton').__version__}\n")
    test_single_grouped_gemm()
    print()
    test_fused_moe()
    print()
    test_edge_cases()
    print()
    test_slot_variants()
    print()
    test_repeat_calls()
    print("\n=== 数值正确性测试全部通过 ===")
    if "--bench" in sys.argv:
        bench()
        bench_blockk()
        bench_layout()
        bench_layouts()
        bench_breakdown()
