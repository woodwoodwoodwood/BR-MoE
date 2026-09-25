"""MoE kernel 的测量回路: 把"优化 kernel"从猜测变成实测。

定位
----
int3 MoE 在 vLLM 里曾经 bs=1 慢 fp16 1.62x (A100 graph, 38901)。本脚本把
`fused_moe_int3` 的真实路径拆成可扫的维度, 逐项验证/证伪假设。实测结论
(2026-09-26, A100 + 5090, 详见各阶段输出与 README 的性能表):

  * num_stages/流水线: 证伪。stages=1 在 M>=8 就是最优, 不是瓶颈。
  * slot/block_m=64 写死: 证实。decode 的 padding 浪费, 换 16 有 1.18~1.34x。
  * 三 layout 同耗时但字节差 4.57x: 既不是访存也不是解包受限 -> 结构性浪费。
  * GEMV + K-major + split-K: 最终方案。5090 上 M=4 达 1766 GB/s (~98% 峰值)。

功能
----
    阶段 0   现状基线 (插件默认配置) + int3/int4(d)/fp16 多 layout 对照
    阶段 0a  自动选择模式 (生产口径: 按 M 自动 / 强制 GEMV / 强制张量核心)
    阶段 0a2 数值交叉校验 (--check): 反量化金标准 vs GEMV/TC/自动 三路
    阶段 0b  cuBLAS 参照 (--ref bmm): 无 padding 的"有用工作量"基准
    阶段 0c  w_transposed 对照 (--wtrans): N-major vs K-major, 含 TC/GEMV 数值
    阶段 0d  GEMV 旋钮扫描 (--gemv-sweep): block_n x ksplit x warps
    阶段 A/B/A2/C  TC 路径的 launch/tile 扫描

用法 (必须在有 GPU 的分配里跑; 也可 sbatch bench/micro_moe_*.slurm)
------------------------------------------------------------------
    python bench/micro_moe.py --ms 1,8,32 --layouts int3,int4d --check
    python bench/micro_moe.py --ms 1,2,4,8 --gemv-sweep --check --stage none
    python bench/micro_moe.py --ms 16,32 --quick        # TC 路径扫描
"""
import argparse
import itertools
import sys
import os

import torch

# 直接复用 vLLM 插件加载 kernel 的方式, 保证测的就是真实路径。
# brmoe_int3_vllm 是 editable 安装 (tools/ 是其包根), 这里再兜一层以防环境没装。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))
from brmoe_int3_vllm.kernel import get_fused_moe_int3  # noqa: E402

# 真实模型尺寸 (deepseek-moe-16b-base / brmoe-3bit-vllm)
E_DEF, K_DEF, I_DEF, GS_DEF, TOPK_DEF = 64, 2048, 1408, 64, 6


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------

def bench_eager(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep * 1e3      # us


def bench_graph(fn, warmup=3, rep=30):
    """CUDA Graph 捕获后测 replay —— 与端到端 benchmark 同口径 (无 CPU 发射开销)。"""
    g = torch.cuda.CUDAGraph()
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(rep):
                fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / 5 / rep * 1e3
    except Exception as exc:                  # noqa: BLE001
        print(f"        [graph 捕获失败: {type(exc).__name__}: {str(exc)[:80]}]")
        return None


# ---------------------------------------------------------------------------
# 权重 / 路由
# ---------------------------------------------------------------------------

def per_expert_bytes(E, K, I, gs, layout="int3"):
    """一个专家被激活时, kernel 需要读的权重字节。**必须按 layout 分别算**。

    教训: 第一版对所有 layout 都返回 int3 的字节数, 于是 fp16 对照行的 GB/s
    被低估了 4.57 倍 (M=1 实际 ~629 GB/s, 却报成 137)。三种 layout 读的字节
    差好几倍, 用同一个分母就是在自欺。
    """
    if layout == "fp16":
        return ((2 * I) * K + K * I) * 2              # 不量化 -> 无 scale/zero
    words = (lambda Kk: Kk // 8) if layout == "int4" else (lambda Kk: Kk // 32 * 3)
    w13 = (2 * I) * words(K) * 4                      # w13_q int32
    w2 = K * words(I) * 4
    sc13 = (K // gs) * (2 * I) * 2                    # fp16 scale
    sc2 = (I // gs) * K * 2
    return w13 + w2 + 2 * sc13 + 2 * sc2              # zero 与 scale 同大小


def make_weight_pair(E, K, I, device):
    """随机 fp16 权重 —— 三种 layout 共用同一份, 保证对比公平。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    w13 = (torch.randn(E, 2 * I, K, generator=g) * 0.02).half().to(device)
    w2 = (torch.randn(E, K, I, generator=g) * 0.02).half().to(device)
    return w13, w2


def pack_weights(w13, w2, gs, layout, device, transposed=False, direct4=False):
    """打包成 kernel 要的 dict。int3/int4 附带 z13/z2 (复现真实的非对称量化路径)。

    transposed=False -> [E, N, Kpack] (N-major)
    transposed=True  -> [E, Kpack, N] (K-major)  <-- 每 k-tile 的读在 n 方向连续
    direct4=True     -> 4-bit 槽位按 k 直接寻址, **去掉 3 个 tl.where** (只对 int4 有意义)
    """
    from int3_moe.ops import pack_moe_weights
    E, twoI, K = w13.shape
    I = twoI // 2
    packed = pack_moe_weights(w13, w2, gs, layout=layout, transposed=transposed)
    # direct4 在整条代码链里从来没有被设成 True 过 (ops.py:176 的默认值是 False,
    # linear_method.py:233 也写死 False)。kernel.py:74 的注释说那 3 个 tl.where
    # 是这条路径上最大的单项开销 —— 所以这个开关必须测。
    packed["direct4"] = bool(direct4)
    # 零点是**对称/非对称**的开关 (kernel 里走 has_zero 分支)。int4 也要给, 否则
    # int4 会顺便省掉一次 zero 加载, 对比就不公平了。
    if layout in ("int3", "int4"):
        # 对称量化等价于 zero=4 (见 ops.dequant_int3 的 zeros is None 分支)
        packed["z13"] = torch.full((E, K // gs, twoI), 4.0,
                                   dtype=torch.float16, device=device)
        packed["z2"] = torch.full((E, I // gs, K), 4.0,
                                  dtype=torch.float16, device=device)
    return packed


def make_routing(M, E, topk, device, seed=0):
    """均匀随机路由: 每个 token 选 topk 个互不相同的专家。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.argsort(torch.rand(M, E, generator=g), dim=1)[:, :topk]
    ids = ids.to(device=device, dtype=torch.int64).contiguous()
    w = torch.rand(M, topk, generator=g, device="cpu")
    w = (w / w.sum(1, keepdim=True)).half().to(device).contiguous()
    return w, ids


def make_ref_fn(x, topk_weights, topk_ids, w13, w2, I, K):
    """cuBLAS 参照: 只对**激活的**专家做一次 batched GEMM (torch.bmm)。

    为什么需要
    ----------
    实测 (A100, 2026-09-26) 三个 layout 的耗时几乎一样 (165~187 us), 但读的
    字节数差 4.57 倍 -> 这个 kernel **既不是访存受限, 也不是解包受限**。

    原因是 align 的 slot padding: slot=64 时**每个激活专家都补齐到 64 行**,
    哪怕它只有 1 个 token。于是 M=1 时:

        有用算力  6 专家 x 8.65M 参数 x 2     =   104 MFLOP
        实际算力  6 专家 x 64 行 x 8.65M x 2  =  6.64 GFLOP   (64x 浪费)

    fp16 走同一个 kernel, align 用同一个 slot -> **fp16 和 int3 算的是同一个
    矩阵**, 所以耗时相同。int3 那 4.57 倍的字节优势根本兑现不了。

    这个参照把 padding 拿掉 (只按每专家真实行数, 补到 m_max 做 batch 维),
    走 cuBLAS + 张量核心, 给出"有用工作量"的耗时基准。

    判读
    ----
      参照耗时 ~ 字节/峰值带宽  -> 带宽受限 -> int3 下界 = 参照 x (int3字节/fp16字节)
      参照耗时 >> 那个下界     -> 连 cuBLAS 在 M 小时都跑不满 -> 反超前提不成立
    """
    topk = topk_ids.shape[1]
    M = x.shape[0]
    E = w13.shape[0]

    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)             # 按专家分组 (稳定)
    cnt = torch.bincount(flat, minlength=E)
    off = torch.cumsum(cnt, 0) - cnt
    active = (cnt > 0).nonzero().flatten()
    n_active = int(active.numel())
    if n_active == 0:
        return None, 0, 0, 0
    m_max = max(int(cnt[active].max()), 1)

    rows = torch.arange(m_max, device=x.device)[None, :]
    cnt_a = cnt[active]
    valid = rows < cnt_a[:, None]                        # [n_active, m_max]
    gidx = torch.where(valid, off[active][:, None] + rows, 0)
    tok = order[gidx] // topk                            # [n_active, m_max] token id
    wsel = topk_weights.reshape(-1)[order][gidx].float()
    wpad = torch.where(valid, wsel, torch.zeros((), device=x.device))
    flat_tok = tok.reshape(-1).contiguous()

    # bmm 需要 [n_active, K, N]
    W13t = w13[active].transpose(1, 2).contiguous()      # [n_active, K, 2I]
    W2t = w2[active].transpose(1, 2).contiguous()        # [n_active, I, K]
    routed = torch.zeros(M, K, dtype=torch.float32, device=x.device)

    def fn():
        xi = x[tok]                                      # [n_active, m_max, K]
        inter = torch.bmm(xi, W13t)                      # [n_active, m_max, 2I]
        g, u = inter[..., :I], inter[..., I:]
        act = (g * torch.sigmoid(g) * u).to(x.dtype)
        oe = torch.bmm(act, W2t)                         # [n_active, m_max, K]
        routed.zero_()
        routed.index_add_(0, flat_tok,
                          (oe.float() * wpad[..., None]).reshape(-1, K))
        return routed.to(x.dtype)

    ref_bytes = n_active * ((2 * I) * K + K * I) * 2     # fp16, 只读激活专家
    return fn, n_active, m_max, ref_bytes


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_config(fused, x, tw, tid, packed, uniq, cfg, bytes_per_expert, use_graph):
    from int3_moe.ops import _WS_CACHE
    _WS_CACHE.cache.clear()          # 换 tilimg -> 工作区尺寸变了, 避免缓存膨胀
    kw = dict(cfg)
    cb = lambda: fused(x, tw, tid, packed, fast=True, **kw)   # noqa: E731
    try:
        us = bench_graph(cb) if use_graph else bench_eager(cb)
    except Exception as exc:                                  # noqa: BLE001
        print(f"        [失败 {type(exc).__name__}: {str(exc)[:90]}]")
        return None
    if us is None:
        return None
    gbs = uniq * bytes_per_expert / (us * 1e-6) / 1e9
    return us, gbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,8,32,512,2048",
                    help="token 数 (decode: 1/8/32, prefill: 512/2048)")
    ap.add_argument("--E", type=int, default=E_DEF)
    ap.add_argument("--K", type=int, default=K_DEF)
    ap.add_argument("--I", type=int, default=I_DEF)
    ap.add_argument("--gs", type=int, default=GS_DEF)
    ap.add_argument("--topk", type=int, default=TOPK_DEF)
    ap.add_argument("--peak-gbs", type=float, default=1935.0,
                    help="设备峰值带宽 GB/s (A100 80GB PCIe=1935, 5090=1792)")
    ap.add_argument("--eager", action="store_true", help="用 eager 计时而非 graph")
    ap.add_argument("--layouts", default="int3,fp16",
                    help="要对比的权重布局: int3 / int4 / int4d / fp16。"
                         "int4 = 4-bit 槽位 (4.0bpw); int4d = 再加 direct4, "
                         "即按 k 直接寻址、**去掉 3 个 tl.where** —— kernel.py:74 的注释说"
                         "那 3 个 where 是这条路径上最大的单项开销。"
                         "注意: 之前测 int4 时 direct4 没开, 等于两条一样的慢路径在对比")
    ap.add_argument("--ref", default="none", choices=["none", "bmm"],
                    help="bmm = 额外跑一个 cuBLAS 参照 (torch.bmm, 只算激活专家的"
                         "真实行数、不含 slot padding), 用来定「硬件最好能跑多快」")
    ap.add_argument("--check", action="store_true",
                    help="额外做数值交叉校验: 把 int3 权重反量化后当金标准, "
                         "逐条比对 GEMV / 张量核心 / 自动三条路径的输出 (新 kernel 必开)")
    ap.add_argument("--wtrans", action="store_true",
                    help="额外扫 w_transposed (权重存成 [E,Kpack,N] K-major)。"
                         "唯一还没在本机验证过的访存开关 —— 同一个 kernel 里 fp16 走 "
                         "K-major 能到 1094 GB/s, 而打包布局走 N-major 只有 228 GB/s")
    ap.add_argument("--gemv-sweep", action="store_true",
                    help="直接扫 routed_int3_gemv 的旋钮 (block_n/groups/warps/stages)。"
                         "ops.py 的 GEMV 分支全部用默认值 (32/4/2/1), 从未调过。 "
                         "GEMV 是延迟受限 (grid 仅 ~528 program, 16 次串行迭代, 无流水), "
                         "这些旋钮比 tile 参数更可能是杠杆")
    ap.add_argument("--quick", action="store_true", help="缩小搜索空间")
    ap.add_argument("--stage", default="all", choices=["all", "launch", "slot", "bn", "none"],
                    help="只跑某个阶段")
    args = ap.parse_args()
    layouts = [L.strip() for L in args.layouts.split(",") if L.strip()]
    # "int4d" = int4 槽位 + direct4。标签里的尾缀 d 控制这个开关。
    LAY = {L: ((L[:-1] if L.endswith("d") else L), L.endswith("d")) for L in layouts}

    if not torch.cuda.is_available():
        print("!! 需要 GPU")
        return 1
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"=== MoE kernel 扫描 on {prop.name} (sm_{prop.major}{prop.minor}) ===")
    print(f"    E={args.E} K={args.K} I={args.I} gs={args.gs} top_k={args.topk}")
    print(f"    计时: {'eager' if args.eager else 'CUDA Graph'}   "
          f"峰值带宽: {args.peak_gbs} GB/s")

    fused = get_fused_moe_int3()
    E, K, I, gs, topk = args.E, args.K, args.I, args.gs, args.topk
    for L in layouts:
        base_l, d4 = LAY[L]
        extra = "  (direct4: 无 tl.where)" if d4 else ""
        print(f"    每激活专家读 ({L:5}) : "
              f"{per_expert_bytes(E, K, I, gs, base_l)/1e6:8.2f} MB{extra}")
    print()

    # int4 / fp16 是同 kernel 的另外两条 layout 路径, 用来隔离"解包"这一项开销:
    #   fp16 -> 完全不解包 (但读 4.57x 字节)
    #   int4 -> 解包便宜很多 (但读 1.29x 字节)
    # 三种 layout 从同一份 fp16 原始权重打包 -> 权重内容完全相同, 只差表示形式
    W13, W2 = make_weight_pair(E, K, I, dev)
    packs = {L: pack_weights(W13, W2, gs, LAY[L][0], dev, direct4=LAY[L][1])
             for L in layouts}

    ms_list = [int(v) for v in args.ms.split(",")]
    results = {}

    for M in ms_list:
        x = (torch.randn(M, K, device=dev) * 0.1).half().contiguous()
        tw, tid = make_routing(M, E, topk, dev)
        uniq = int(torch.unique(tid).numel())
        print(f"########## M = {M}  (激活 {uniq}/{E} 个专家) ##########")

        base = dict(block_m=64, block_n=64, slot=64, block_k=32,
                    num_warps=2, num_stages=1)

        # ---- 阶段 0: 现状 (vLLM 插件实际走的配置) + 其它 layout 对照 ----
        print("  -- 现状基线 (插件现况 = slot64/block_m64/warps2/stages1) --")
        cur = {}
        for layout in layouts:
            r = run_config(fused, x, tw, tid, packs[layout], uniq,
                           base, per_expert_bytes(E, K, I, gs, LAY[layout][0]),
                           not args.eager)
            if r:
                us, gbs = r
                cur[layout] = us
                note = {"int3": "插件现况", "fp16": "不量化对照",
                        "int4": "4bit 槽位", "int4d": "4bit + 无 tl.where"
                        }.get(layout, "")
                print(f"     {layout:<5}({note:<16}) {us:8.2f} us   {gbs:7.1f} GB/s  "
                      f"({gbs/args.peak_gbs*100:5.1f}% 峰值)")
        print("       ^ fp16 行的字节数是 int3 的 4.57x; int4 是 1.29x。"
              "若 fp16/int4 的 GB/s 远高于 int3 -> 瓶颈在解包而非访存")

        # ---- 阶段 0a: int3 的「自动选择」模式 (ops.py:202-224) ----
        # 关键: 那套自动优化要求 `slot is None and block_size is None`。而上面那组
        # baseline **显式传了 slot=64/block_m=64**, 等于把它整条关掉 —— 所以历史
        # 扫描数据全部是「关掉优化之后」的, 不能用来评价这次的内核改动。
        #
        # 字节口径也不同: GEMV 每个 (token,expert) 路由对独立读一遍该专家权重
        # (kernel.py:54 "One routed token per program"), 所以按 num_valid 份算;
        # 张量核心路径按去重后的专家数 uniq 份算。两者不能直接比 GB/s。
        num_valid = M * topk
        variants = [
            ("slot64 强制(旧口径)", dict(slot=64, block_m=64), uniq),
            ("自动(生产现况)",      dict(),                    uniq),
            ("自动+强制GEMV",       dict(gemv=True),           num_valid),
            ("自动+强制张量核心",   dict(gemv=False),          uniq),
        ]
        print(f"  -- int3 自动选择模式 (num_valid={num_valid}, uniq={uniq}) --")
        auto_us = {}
        for name, kw, nreads in variants:
            r = run_config(fused, x, tw, tid, packs["int3"], nreads,
                           kw, per_expert_bytes(E, K, I, gs, "int3"), not args.eager)
            if r:
                us, gbs = r
                auto_us[name] = us
                how = ("GEMV" if kw.get("gemv") is True
                       else ("张量核心" if kw.get("gemv") is False else "按 M 自动"))
                print(f"     {name:<20} {us:8.2f} us   {gbs:7.1f} GB/s  "
                      f"({gbs/args.peak_gbs*100:5.1f}%)  [{how}, 读 {nreads} 份权重]")
        if "slot64 强制(旧口径)" in auto_us and "自动(生产现况)" in auto_us:
            a = auto_us["slot64 强制(旧口径)"]
            b = auto_us["自动(生产现况)"]
            print(f"       ==> 自动选择 vs 旧强制配置: {a/b:.2f}x  "
                  f"({a:.2f} -> {b:.2f} us)")
        if "自动+强制GEMV" in auto_us and "自动+强制张量核心" in auto_us:
            g = auto_us["自动+强制GEMV"]
            t = auto_us["自动+强制张量核心"]
            print(f"       ==> GEMV vs 张量核心: {t/g:.2f}x  "
                  f"({t:.2f} -> {g:.2f} us)")

        # ---- 阶段 0a2: 数值交叉校验 (新 kernel 必须过这关) ----
        # 整个项目的可信度建立在"逐位一致"上; 引入 GEMV 新路径后必须重新确认。
        if args.check:
            print("  -- 数值交叉校验 (金标准 = int3 权重反量化后跑 cuBLAS bmm) --")
            from int3_moe.ops import dequant_int3
            P = packs["int3"]
            W13d = dequant_int3(P["w13_q"], P["s13"], K, gs, layout="int3",
                                transposed=False, zeros=P.get("z13"))
            W2d = dequant_int3(P["w2_q"], P["s2"], I, gs, layout="int3",
                               transposed=False, zeros=P.get("z2"))
            gold_fn, _, _, _ = make_ref_fn(x, tw, tid, W13d, W2d, I, K)
            gold = gold_fn().float()
            scl = max(gold.abs().max().item(), 1e-9)
            print(f"     金标准 max|y|={scl:.3e}")
            for name, kw, _ in variants:
                try:
                    y = fused(x, tw, tid, packs["int3"], fast=True, **kw).float()
                except Exception as exc:                       # noqa: BLE001
                    print(f"     {name:<20} 运行失败 {type(exc).__name__}: {str(exc)[:60]}")
                    continue
                d = (y - gold).abs().max().item()
                rel = d / scl
                print(f"     {name:<20} max|diff|={d:.3e}  相对={rel:.2e}  "
                      f"{'OK' if rel < 5e-3 else '!! 偏差过大'}")
            del W13d, W2d

        # ---- 阶段 0b: cuBLAS 参照 (定"硬件最好能跑多快") ----
        if args.ref == "bmm":
            print("  -- cuBLAS 参照: torch.bmm, 只算激活专家的真实行数 (无 slot padding) --")
            ref_fn, n_act, m_max, ref_bytes = make_ref_fn(x, tw, tid, W13, W2, I, K)
            ref_us = None
            if ref_fn is not None:
                try:
                    ref_us = bench_eager(ref_fn) if args.eager else bench_graph(ref_fn)
                except Exception as exc:                      # noqa: BLE001
                    print(f"     [失败 {type(exc).__name__}: {str(exc)[:100]}]")
            if ref_us:
                gbs = ref_bytes / (ref_us * 1e-6) / 1e9
                floor = ref_bytes / 1e9 / args.peak_gbs * 1e6      # us @ 峰值带宽
                print(f"     cuBLAS fp16     {ref_us:8.2f} us   {gbs:7.1f} GB/s  "
                      f"({gbs/args.peak_gbs*100:5.1f}% 峰值)   "
                      f"n_active={n_act} m_max={m_max}")
                print(f"       纯带宽下界 {floor:8.2f} us  -> 参照偏离 {ref_us/floor:.2f}x")
                if "int3" in cur:
                    ratio = (per_expert_bytes(E, K, I, gs, "int3")
                             / per_expert_bytes(E, K, I, gs, "fp16"))
                    tgt = ref_us * ratio
                    print(f"       若 int3 达到同样效率 -> {tgt:8.2f} us "
                          f"(= 参照 x {ratio:.3f})")
                    print(f"       ==> int3 现况 {cur['int3']:8.2f} us, "
                          f"离该目标 {cur['int3']/tgt:.2f}x")

        # ---- 阶段 0c: w_transposed 对照 (只影响打包布局的访存模式) ----
        if args.wtrans:
            print("  -- w_transposed 对照: N-major [E,N,Kpack] vs K-major [E,Kpack,N] --")
            i3b = per_expert_bytes(E, K, I, gs, "int3")
            gold = None                       # 两种布局共享同一份 fp16 权重
            for wt in (False, True):
                p = pack_weights(W13, W2, gs, "int3", dev, transposed=wt)
                tag = "K-major" if wt else "N-major(现况)"
                # TC 路径 (旧口径 slot64) —— 历史上"布局中性"的结论只在这里成立
                r = run_config(fused, x, tw, tid, p, uniq, base, i3b, not args.eager)
                # GEMV 路径 (强制): 每 (token,expert) 对独立读权重 -> 按 num_valid 份。
                # GEMV 的 w 载入形状是 [GROUPS, BLOCK_N]: N-major 下相邻 lane 间隔
                # Kpack*4=768B (每 4B 拉一个 32B sector -> 最多 8x 浪费);
                # K-major 下 n 连续 -> 128B 合并。预期差别巨大。
                g = run_config(fused, x, tw, tid, p, M * topk, dict(gemv=True),
                               i3b, not args.eager)
                line = f"     {tag:<14}"
                if r:
                    line += f"  TC {r[0]:8.2f} us ({r[1]:6.1f} GB/s)"
                if g:
                    line += f"  GEMV {g[0]:8.2f} us ({g[1]:6.1f} GB/s)"
                print(line)
                # 数值校验: 转置布局必须产出与金标准一致的结果。
                # GEMV / TC / 自动三条路径都要过 —— 生产上 K-major 后 TC 路径
                # (M>GEMV 上限时) 也会走 W_T 分支, 漏验就是把错误带进 e2e。
                if args.check and g:
                    from int3_moe.ops import dequant_int3
                    if gold is None:
                        W13d = dequant_int3(p["w13_q"], p["s13"], K, gs,
                                            layout="int3", transposed=wt,
                                            zeros=p.get("z13"))
                        W2d = dequant_int3(p["w2_q"], p["s2"], I, gs,
                                           layout="int3", transposed=wt,
                                           zeros=p.get("z2"))
                        gold_fn, _, _, _ = make_ref_fn(x, tw, tid, W13d, W2d, I, K)
                        gold = gold_fn().float()
                        del W13d, W2d
                    scl = max(gold.abs().max().item(), 1e-9)
                    for vname, vkw in (("GEMV", dict(gemv=True)),
                                       ("TC", dict(base)),
                                       ("自动", dict())):
                        y = fused(x, tw, tid, p, fast=True, **vkw).float()
                        d = (y - gold).abs().max().item()
                        rel = d / scl
                        print(f"       数值: {vname}({tag}) vs 金标准 "
                              f"max|diff|={d:.3e} 相对={rel:.2e} "
                              f"{'OK' if rel < 5e-3 else '!! 偏差过大'}")

        # ---- 阶段 0d: GEMV 旋钮扫描 (block_n/groups/warps/stages) ----
        # ops.py 的 GEMV 分支写死默认 (block_n=32, groups=4, warps=2, stages=1)。
        # GEMV 是延迟受限 (M=1 时 grid 仅 6x88=528 program, 每 program 16 次串行
        # k 迭代, stages=1 无流水) —— 这些旋钮是最近的杠杆, 扫完不行再上 split-K。
        if args.gemv_sweep:
            from int3_moe.kernel import routed_int3_gemv, silu_mul_routes
            from int3_moe.ops import _WS_CACHE

            def gemv_fn(pack, bn, gr, w, s, ks):
                w13_q, s13, z13 = pack["w13_q"], pack["s13"], pack.get("z13")
                w2_q, s2, z2 = pack["w2_q"], pack["s2"], pack.get("z2")
                wt = pack.get("w_transposed", False)
                twoI = w13_q.shape[2] if wt else w13_q.shape[1]
                nv = M * topk
                twf = tw.reshape(-1).to(torch.float32)
                idsf = tid.reshape(-1).contiguous()
                ws = _WS_CACHE.get(nv, twoI, twoI // 2, M, K, x.device)

                def fn():
                    ws["inter32"].zero_()
                    routed_int3_gemv(x, w13_q, s13, z13, idsf, None, ws["inter32"],
                                     topk, gs, w_transposed=wt,
                                     block_n=bn, num_warps=w, num_stages=s,
                                     groups=gr, ksplit=ks)
                    silu_mul_routes(ws["inter32"], ws["act"], twoI // 2, nv)
                    ws["out32"].zero_()
                    routed_int3_gemv(ws["act"], w2_q, s2, z2, idsf, twf,
                                     ws["out32"], topk, gs, add=True,
                                     w_transposed=wt, block_n=bn,
                                     num_warps=w, num_stages=s, groups=gr,
                                     ksplit=ks)
                    return ws["out32"]
                return fn

            i3b = per_expert_bytes(E, K, I, gs, "int3") * (M * topk)
            for wt in (False, True):
                p = pack_weights(W13, W2, gs, "int3", dev, transposed=wt)
                tag = "K-major" if wt else "N-major"
                print(f"  -- GEMV 旋钮扫描 [{tag}] (生产默认 block_n=32 groups=4 "
                      f"warps=2 stages=1 ksplit=auto) --")
                best = None
                for bn, ks, w in itertools.product((32, 64), (1, 2, 4, 8), (2, 4)):
                    gr, s = 4, 1
                    try:
                        fn = gemv_fn(p, bn, gr, w, s, ks)
                        us = bench_eager(fn) if args.eager else bench_graph(fn)
                    except Exception as exc:                    # noqa: BLE001
                        print(f"     bn={bn} ksplit={ks} warps={w}  "
                              f"[失败 {type(exc).__name__}: {str(exc)[:60]}]")
                        continue
                    gbs = i3b / (us * 1e-6) / 1e9
                    mark = ""
                    if best is None or us < best[0]:
                        best = (us, (bn, gr, w, s, ks), fn)
                        mark = "  <-- 目前最好"
                    dflt = " (旧生产默认)" if (bn, ks, w) == (32, 1, 2) else ""
                    print(f"     bn={bn:<4}ksplit={ks} warps={w}  "
                          f"{us:8.2f} us   {gbs:7.1f} GB/s{dflt}{mark}")
                if best:
                    us, (bn, gr, w, s, ks), fn = best
                    print(f"     ==> [{tag}] 最优: bn={bn} ksplit={ks} warps={w} "
                          f"({us:.2f} us)")
                    if args.check:
                        from int3_moe.ops import dequant_int3
                        W13d = dequant_int3(p["w13_q"], p["s13"], K, gs,
                                            layout="int3", transposed=wt,
                                            zeros=p.get("z13"))
                        W2d = dequant_int3(p["w2_q"], p["s2"], I, gs,
                                           layout="int3", transposed=wt,
                                           zeros=p.get("z2"))
                        gold_fn, _, _, _ = make_ref_fn(x, tw, tid, W13d, W2d, I, K)
                        gold = gold_fn().float()
                        y = fn().float()
                        del W13d, W2d
                        d = (y - gold.reshape(-1, K).float()).abs().max().item()
                        rel = d / max(gold.abs().max().item(), 1e-9)
                        print(f"       数值: 最优配置 vs 金标准 max|diff|={d:.3e}"
                              f" 相对={rel:.2e} {'OK' if rel < 5e-3 else '!! 偏差过大'}")

        # ---- 阶段 A: num_warps × num_stages ----
        if args.stage in ("all", "launch"):
            print("  -- 阶段 A: num_warps x num_stages (slot=64, block_m=64, block_n=64) --")
            warps = [2, 4, 8] if not args.quick else [2, 4]
            stages = [1, 2, 3, 4] if not args.quick else [1, 3]
            best = None
            for w, s in itertools.product(warps, stages):
                cfg = dict(base, num_warps=w, num_stages=s)
                r = run_config(fused, x, tw, tid, packs["int3"], uniq, cfg, per_expert_bytes(E, K, I, gs, "int3"),
                               not args.eager)
                if r:
                    us, gbs = r
                    mark = ""
                    if best is None or us < best[0]:
                        best = (us, (w, s))
                        mark = "  <-- 目前最好"
                    print(f"     warps={w} stages={s}  {us:8.2f} us   "
                          f"{gbs:7.1f} GB/s  ({gbs/args.peak_gbs*100:5.1f}%){mark}")
            if best:
                print(f"     ==> launch 最优: num_warps={best[1][0]} num_stages={best[1][1]}"
                      f"  ({best[0]:.2f} us)")
                base["num_warps"], base["num_stages"] = best[1]

        # ---- 阶段 B: slot × block_m ----
        if args.stage in ("all", "slot"):
            print(f"  -- 阶段 B: slot x block_m (num_warps={base['num_warps']}, "
                  f"num_stages={base['num_stages']}) --")
            pairs = [(sl, bm) for bm in (16, 32, 64, 128)
                     for sl in (16, 32, 64) if sl <= bm and bm % sl == 0]
            if args.quick:
                pairs = [(16, 16), (16, 32), (16, 64), (64, 64)]
            best = None
            for sl, bm in pairs:
                cfg = dict(base, slot=sl, block_m=bm)
                r = run_config(fused, x, tw, tid, packs["int3"], uniq, cfg, per_expert_bytes(E, K, I, gs, "int3"),
                               not args.eager)
                if r:
                    us, gbs = r
                    mark = ""
                    if best is None or us < best[0]:
                        best = (us, (sl, bm))
                        mark = "  <-- 目前最好"
                    print(f"     slot={sl:<3} block_m={bm:<4}  {us:8.2f} us   "
                          f"{gbs:7.1f} GB/s  ({gbs/args.peak_gbs*100:5.1f}%){mark}")
            if best:
                print(f"     ==> tile 最优: slot={best[1][0]} block_m={best[1][1]}"
                      f"  ({best[0]:.2f} us)")
                base["slot"], base["block_m"] = best[1]

        # ---- 阶段 A2: 在**新 tile** 上重扫 warps × stages ----
        # 这是原扫描的结构性漏洞: 阶段 A 在 slot=64/block_m=64 上挑 launch 参数,
        # 阶段 B 又把它原样带到小 tile 上, 于是 (bm=16, stages>1) 从没被测过。
        # 而两者最优解本来就不同:
        #   大 tile (bm=64): MMA 够重, 自己掩盖得住访存延迟 -> stages=1 最佳 (省 smem)
        #   小 tile (bm=16): 每个 k-step 只有 3 个字要读, 没有多缓冲就完全暴露在
        #                    DRAM 延迟下。反推显示 bm=16 时每 block 要 16 us / 4.2 MFLOP,
        #                    只有 24 TFLOPS —— 这是**延迟受限**, 正是 stages 该起作用的地方。
        if args.stage in ("all", "launch") and (base["slot"], base["block_m"]) != (64, 64):
            print(f"  -- 阶段 A2: 在最优 tile (slot={base['slot']}, block_m={base['block_m']}) "
                  f"上重扫 launch --")
            warps = [2, 4, 8] if not args.quick else [2, 4]
            stages = [1, 2, 3, 4] if not args.quick else [1, 2, 3]
            best = None
            for w, s in itertools.product(warps, stages):
                cfg = dict(base, num_warps=w, num_stages=s)
                r = run_config(fused, x, tw, tid, packs["int3"], uniq, cfg,
                               per_expert_bytes(E, K, I, gs, "int3"), not args.eager)
                if r:
                    us, gbs = r
                    mark = ""
                    if best is None or us < best[0]:
                        best = (us, (w, s))
                        mark = "  <-- 目前最好"
                    print(f"     warps={w} stages={s}  {us:8.2f} us   "
                          f"{gbs:7.1f} GB/s  ({gbs/args.peak_gbs*100:5.1f}%){mark}")
            if best:
                print(f"     ==> 小 tile 的 launch 最优: num_warps={best[1][0]} "
                      f"num_stages={best[1][1]}  ({best[0]:.2f} us)")
                base["num_warps"], base["num_stages"] = best[1]

        # ---- 阶段 C: block_n ----
        if args.stage in ("all", "bn"):
            print(f"  -- 阶段 C: block_n (slot={base['slot']}, block_m={base['block_m']}) --")
            best = None
            for bn in (32, 64, 128, 256):
                if (2 * I) % bn or K % bn:
                    continue
                cfg = dict(base, block_n=bn)
                r = run_config(fused, x, tw, tid, packs["int3"], uniq, cfg, per_expert_bytes(E, K, I, gs, "int3"),
                               not args.eager)
                if r:
                    us, gbs = r
                    mark = ""
                    if best is None or us < best[0]:
                        best = (us, bn)
                        mark = "  <-- 目前最好"
                    print(f"     block_n={bn:<4}  {us:8.2f} us   {gbs:7.1f} GB/s  "
                          f"({gbs/args.peak_gbs*100:5.1f}%){mark}")
            if best:
                print(f"     ==> block_n 最优: {best[1]}  ({best[0]:.2f} us)")
                base["block_n"] = best[1]

        results[M] = (dict(base), uniq)
        print()

    # ---- 汇总 ----
    print("=" * 78)
    print("汇总: 各 M 的推荐配置")
    print("=" * 78)
    print(f"{'M':>6}  {'uniq_experts':>13}  {'slot':>5}{'block_m':>9}{'block_n':>9}"
          f"{'warps':>7}{'stages':>8}")
    for M, (cfg, uniq) in results.items():
        print(f"{M:>6}  {uniq:>13}  {cfg['slot']:>5}{cfg['block_m']:>9}{cfg['block_n']:>9}"
              f"{cfg['num_warps']:>7}{cfg['num_stages']:>8}")
    print()
    print("怎么读 (结论来自 2026-09-26 A100 的实测):")
    print("  * 阶段 A (num_warps/stages): 实测 stages=1 在 M>=8 就是最优, M=1 也只差 1.13x")
    print("    -> 流水线**不是**瓶颈。这条假设已被证伪, 不要再往这个方向调。")
    print("  * 阶段 B (slot/block_m): slot=64/block_m=64 (也就是插件现况) 在每一行都是")
    print("    最差的那个。换 slot=16/block_m=16 能拿 1.18~1.34x。")
    print("  * 阶段 B 之后一定要看阶段 A2: 抄 launch 参数到新 tile 上是不成立的。")
    print("    按耗时反推张量核心吞吐 (M=32, 59 专家):")
    print("       slot=64/bm=64  padding 后 3776 行 -> 65.4 GFLOP / 895us = 73 TFLOPS")
    print("       slot=16/bm=16  padding 后  944 行 -> 16.3 GFLOP / 670us = 24 TFLOPS")
    print("    大 tile 是 MMA 受限 (效率尚可但浪费 6.4x 算力), 小 tile 是**延迟受限**")
    print("    (每 block 16us / 4.2 MFLOP)。两者恰好抵消 -> 净收益只有 1.34x。")
    print("    小 tile 上的 stages>1 才是真正没验证过的组合。")
    print("  * 三个 layout 耗时几乎相同、字节却差 4.57x -> 既不是访存受限, 也不是解包")
    print("    受限。真正的上限是 slot padding: 每专家都算 64 行, 所以 fp16 和 int3 在")
    print("    算**同一个矩阵** (M=1 时 98.4% 的算力是 padding 浪费)。")
    print("  * --ref bmm 给出「有用工作量」的 cuBLAS 基准。判读:")
    print("      参照耗时 ≈ 纯带宽下界        -> 带宽受限, 反超有结构性空间")
    print("      参照耗时 >> 下界             -> 连 cuBLAS 在 M 小时都跑不满")
    print("  * 最终方案已落地 (2026-09-26): GEMV + K-major + split-K,")
    print("    5090 上 M<=8 走 GEMV (~98% 峰值带宽), 更大 M 走 TC。")
    print("    注意 GEMV/TC 的交叉点依赖路由重复度: 随机路由 micro 与 e2e")
    print("    (同 prompt 批, 路由相关) 结论可能相反 —— 以 e2e 为准。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
