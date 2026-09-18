# Triton int3 Grouped GEMM (experimental backend)

A **Triton** implementation of the 3.0-bit MoE grouped GEMM, as a portable counterpart to the
CUDA / Marlin-lineage backend in `BR_MoE/kernels/brmoe`. No build step, no `setup.py` —
just `torch` + `triton`.

One MoE layer = **two grouped GEMMs + one fused activation**, all in Triton:

```
gate = x @ W1[e]^T          [n, I]
up   = x @ W3[e]^T          [n, I]
h    = silu(gate) * up      [n, I]        <- fused into one kernel
y    = h @ W2[e]^T          [n, K]
```

`gate`/`up` are concatenated along the output dim into `w13 [E, 2I, K]`, so a layer needs only
two launches. The whole path is **zero host-sync**: an over-launched grid plus a device-side
`meta` tensor (num_post) lets the kernel exit early instead of the host reading back a count.

## What is here

| File | Contents |
|---|---|
| `int3_moe/packing.py` | dense 3.0 bpw packing (32 weights → 3× `uint32`, no wasted bits) + a 4-bit-slot layout used as a control |
| `int3_moe/align.py` | `moe_align_block_size` reference implementation (torch) |
| `int3_moe/align_triton.py` | Triton align: count → scan → scatter → block-ids, all on device |
| `int3_moe/kernel.py` | the grouped GEMM (int3 / int4-slot / fp16 weight layouts) + fused `silu(gate)*up` |
| `int3_moe/ops.py` | `pack_moe_weights`, `fused_moe_int3`, `ref_fused_moe` (per-expert reference) |
| `test_basics.py` | packing / quantization / alignment self-checks |
| `test_kernel.py` | numerical correctness (incl. regression tests) + benchmarks |

`bench/baseline.txt` and `bench/final.txt` are raw benchmark logs (before/after the optimization
round described below), kept as evidence for the numbers quoted here.

## Quick start

```bash
cd BR_MoE/kernels/triton_int3
python test_basics.py            # packing / align checks
python test_kernel.py            # correctness suite
python test_kernel.py --bench    # correctness + all benchmarks
```

Minimal usage:

```python
import torch
from int3_moe.ops import pack_moe_weights, fused_moe_int3

E, twoI, K, I, GS, topk = 8, 8192, 2048, 4096, 128, 2
w13 = (torch.randn(E, twoI, K, device="cuda") * 0.02).half()
w2  = (torch.randn(E, K, I,  device="cuda") * 0.02).half()
packed = pack_moe_weights(w13, w2, GS)          # dense int3, 3.0 bit/weight

x  = torch.randn(64, K, device="cuda").half()
ti = torch.randint(0, E, (64, topk), device="cuda")          # top_k expert ids per token
tw = torch.full((64, topk), 1.0 / topk, device="cuda").half() # routing weights

y = fused_moe_int3(x, tw, ti, packed)           # [64, K]
```

## Measured results (Tesla T4 15 GB, Triton 2.1.0)

`E=8, K=2048, I=4096, topk=2, group=128`. Times are **ms per MoE layer** (2 GEMMs + activation).

### Weight layout comparison — same kernel, only the storage/decode differs

| M | int3 3.0 bpw | int4 4.0 bpw | fp16 16 bpw | int4/int3 | fp16/int3 |
|---|---|---|---|---|---|
| 1 | 0.773 | 0.807 | 0.630 | 1.04× | 0.81× |
| 8 | 1.594 | 1.724 | 1.466 | 1.08× | 0.92× |
| 64 | 2.241 | 2.865 | 2.445 | 1.28× | 1.09× |
| 512 | 6.585 | 7.387 | 6.152 | 1.12× | 0.93× |

Weight bytes per layer: **int3 = 72.0 MiB, int4 = 96.0 MiB (+33%), fp16 = 384.0 MiB (+5.33×)**.

The `int4 4.0 bpw` column is the 8-values-per-`uint32` slot layout: `(w >> 4i) & 0xF`. Its
**decode instruction count is identical** to dense int3 (4 word loads + 3 broadcast `tl.where`
+ shift + mask), so it is a valid *speed* proxy for a real 4-bit kernel — but **not an accuracy
proxy**, because it reuses the same 3-bit quantized values.

### vs. the per-expert loop (the current baseline)

| M | this kernel | per-expert int3 | per-expert fp16 |
|---|---|---|---|
| 1 | 0.724 | 3.173 (4.38×) | 2.638 (3.64×) |
| 8 | 1.876 | 5.190 (2.77×) | 8.081 (4.31×) |
| 64 | 2.233 | 6.177 (2.77×) | 10.176 (4.56×) |
| 512 | 5.645 | 9.533 (1.69×) | 17.028 (3.02×) |

## Findings

**What helped**

| Change | Effect |
|---|---|
| Triton align (count→scan→scatter→block-ids) + device-side `meta`, over-launched grid | align **3.3–3.5×** faster; removes all host sync |
| Fused `silu(gate)*up` (instead of 3 torch elementwise ops + an fp32 intermediate) | **1.13–2.21×** on the fused path |
| Padded-row granularity: pad to 16 rows instead of 64 (`block_size`) | **1.17–1.25×** in the small-M / many-expert regime |
| `num_warps=2, num_stages=1` (instead of 4/3) | **1.5–2.6×** |
| Fusing the two GEMMs + activation into one pass over the expert | 1.69–4.38× vs per-expert |

Overall: **1.9–2.4× faster than the initial version** of this kernel, **1.7–4.4× faster than the
per-expert loop**.

**What did *not* help** (measured, kept here so nobody repeats them)

| Change | Result |
|---|---|
| `BLOCK_K` 32 → 64/128 | 5–60% **slower**. The dequant:mma ratio does not depend on `BLOCK_K`, and a larger A tile eats the T4's 64 KB smem |
| `BLOCK_M` 64 → 128 | 4–5× **slower** (fp32 accumulator spills: 128×64 / 64 threads) |
| `BLOCK_N` 64 → 32 (2× more blocks) | 35–61% **slower** |
| Multi-expert per block (`block_m=64, slot=16`) | 7–23% **slower** than plain `block_m=slot=16`; the `NSEG` sub-tiles run *serially* inside one block instead of letting the SM schedule independent blocks |
| K-major weight layout `[E, Kpack, N]` | neutral (0.93–1.01×) — the k loop already consumes whole cache lines |
| Removing 3 of the ~8 decode ops (per-k word addressing instead of `tl.where`) | **1–3%** only |

**Bottleneck analysis.** Removing *all* decode (the fp16 path) changes the time by only 8–22%,
and removing 3/8 of the decode ops changes it by 1–3% — so the dequantized-ALU count is **not** the
critical path. Increasing work per block (`BLOCK_K`, `BLOCK_M`) always hurts, and increasing block
count (`BLOCK_N` down) hurts too, so the config is at a local optimum. The remaining serial cost is
the **per-iteration fixed overhead of the k loop** (address generation + load issue + dot issue),
which is identical for int3 and fp16 — which is why those two end up within ±20% of each other.

Practical consequence: on **sm_75** there is no `cp.async`, and Triton cannot software-pipeline an
operand that is *computed* (the dequantized B) rather than loaded, so the kernel never reaches the
bandwidth-bound regime where quantization's byte advantage pays off. On sm_80+ (or with a
hand-written kernel that pipelines the dequant) `int3` should reach the bandwidth limit and beat
fp16 by up to the byte ratio (5.33×).

**Memory is the real reason to use int3 here.** On the T4 the int3 and fp16 paths run at roughly
the same speed while int3 uses **5.33× less memory** — and for a 27 B model the fp16 weights
(≈52 GiB) do not fit in 15 GB at all, while the 3-bit ones (≈10 GiB) do.

## Notes

* Tested on a Tesla T4 (sm_75) with Triton 2.1.0 and CUDA 12.1. Requires sm_75 or newer.
* `block_n=64` is the sweet spot on this part; wider tiles are 2–3× slower.
* Absolute timings drift by up to ~2× between long runs (the T4 de-rates under sustained load),
  so always compare configurations **inside one process, alternating order**; the ratios above
  come from such interleaved measurements.
* The quantization used in `pack_moe_weights` is naive symmetric RTN (`scale = amax/3`), i.e. it
  is there to exercise the kernel, **not** a competitive quantizer. Accuracy numbers in the logs
  come from that RTN step, not from the kernel (whose numerical error is ~6e-5).
* Code comments are written in Chinese.
