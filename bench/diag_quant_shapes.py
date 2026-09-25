"""打印 checkpoint 里一层权重的量化张量形状, 用于对齐 grouped 无损打包。

重点确认:
  * W_q (packed) 的二维形状到底是 (n/10, k) 还是 (n*k/gs/10, gs)
  * scale / zero 的元素个数与逻辑形状
  * unpack 之后的形状
"""

import torch

CKPT = "/mnt/4090/data/jianglei/models/MiLo/deepseek-3bit-3bit_rank0/qmodel.pt"

NAMES = [
    "model.layers.1.mlp.experts.0.gate_proj",   # in=K=2048, out=I=1408
    "model.layers.1.mlp.experts.0.down_proj",   # in=I=1408, out=K=2048
]


def main():
    from BR_MoE.core.quantize import Quantizer

    print(f"loading {CKPT} ...", flush=True)
    sd = torch.load(CKPT, map_location="cpu")
    print("loaded", flush=True)

    for name in NAMES:
        d = sd[name]
        print("=" * 70)
        print(name)
        for k in sorted(d.keys()):
            v = d[k]
            if torch.is_tensor(v):
                print(f"  {k:26s} {str(tuple(v.shape)):20s} {v.dtype}")
            else:
                print(f"  {k:26s} {v!r}")

        packing = d["packing"]
        packing = packing if isinstance(packing, str) else str(packing)
        Wq = d["W_q"]
        u = Quantizer.unpack[packing](Wq, dtype=torch.int16)
        print(f"  -> unpack(packing={packing}) shape = {tuple(u.shape)}")
        s = d["scale"]
        z = d["zero"]
        print(f"  -> scale numel={s.numel()}  zero numel={z.numel()}")
        gs = d["group_size"]
        gs = int(gs) if not torch.is_tensor(gs) else int(gs.flatten()[0])
        print(f"  -> group_size={gs}")
        print(f"  -> 若 W_q 是 (n*k/gs/10, gs): unpack 后 reshape 成 (n, k) 需要 "
              f"n*k/gs={u.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
