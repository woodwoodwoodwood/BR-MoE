"""对比 repack_one (向量化) 与 brmoe 原版 pack() (numpy 参照) —— 必须逐位一致。

纯 CPU, 登录节点可跑:
    python bench/check_repack.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "triton_int3"))
sys.path.insert(0, os.path.join(ROOT, "BR-MoE", "kernels", "brmoe"))

from marlin_int3_moe.repack import repack_one  # noqa: E402
from brmoe import Layer3bitWithZeros           # noqa: E402  (原版参照)


def main():
    torch.manual_seed(0)
    N, K, gs = 256, 256, 64
    # 随机非对称量化参数
    s = (torch.rand(K // gs, N) * 0.02 + 0.005).half()     # [K//gs, N]
    zq = (torch.rand(K // gs, N) * 5 + 0.5).half()         # 量化单位的零点
    # 随机合法量化值
    q = torch.randint(0, 8, (N, K), dtype=torch.int32)     # [N, K] in [0,7]
    # 假量化权重: w = (q - z) * s
    s_b = s.repeat_interleave(gs, dim=0).t().float()       # [N, K]
    z_b = zq.repeat_interleave(gs, dim=0).t().float()
    Wdeq = ((q.float() - z_b) * s_b).half()                # [N, K]

    # ---- 我的向量化 repack ----
    B1a, B2a, sa, za = repack_one(q, s, zq, gs)

    # ---- 原版 numpy pack (参照) ----
    layer = Layer3bitWithZeros(K, N, groupsize=gs)
    lin = torch.nn.Linear(K, N, bias=False, dtype=torch.half)
    lin.weight.data = Wdeq.clone()
    # 原版语义: q = round((w - z_pack)/s) -> z_pack = -zq*s (见 repack.py 注释)
    z_pack = (-(zq.float() * s.float())).half()            # [K//gs, N]
    layer.pack(lin, scales=s.t(), zeros=z_pack.t())

    # ---- 逐项对比 ----
    def cmp(name, a, b):
        a = a.cpu() if a.is_cuda else a
        b = b.cpu() if b.is_cuda else b
        if a.shape != b.shape:
            print(f"  {name}: 形状不同 {tuple(a.shape)} vs {tuple(b.shape)} !!")
            return False
        same = torch.equal(a, b)
        if not same:
            d = (a.int() - b.int()).abs()
            idx = (d > 0).nonzero()
            print(f"  {name}: 不一致! 差异元素 {idx.shape[0]}/{a.numel()}, "
                  f"首个 {tuple(idx[0])}: {a[tuple(idx[0])]} vs {b[tuple(idx[0])]}")
        else:
            print(f"  {name}: 逐位一致 OK {tuple(a.shape)}")
        return same

    ok = True
    ok &= cmp("B1", B1a, layer.B1)
    ok &= cmp("B2", B2a, layer.B2)
    ok &= cmp("s", sa, layer.s)
    ok &= cmp("z", za, layer.z)
    print("==", "全部一致 OK" if ok else "有差异 !!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
