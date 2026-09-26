"""复现 get_moe_cuda_ext 的加载流程, 但把异常打印出来 (插件里是裸 except 吞掉的)。

用法: python bench/check_ext.py
"""
import importlib.util
import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = os.path.join(ROOT, "BR-MoE", "kernels", "marlin_int3_moe")

cap = torch.cuda.get_device_capability(0)
tag = f"_sm{cap[0]}{cap[1]}"
cands = [f for f in os.listdir(d)
         if f.startswith("brmoe_moe_int3") and f.endswith(".so")]
pref = [f for f in cands if tag in f] or [f for f in cands if "_sm" not in f]
print(f"候选: {cands} -> 选中: {pref}")

if not pref:
    sys.exit("!! 没有候选 .so")

so = os.path.join(d, sorted(pref)[0])
print(f"加载 {so}")
try:
    spec = importlib.util.spec_from_file_location("brmoe_moe_int3", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = torch.device("cuda")
    A = torch.zeros(16, 128, dtype=torch.float16, device=dev)
    B1 = torch.zeros(1, 8, 128, dtype=torch.int32, device=dev)
    B2 = torch.zeros(1, 8, 64, dtype=torch.int32, device=dev)
    C = torch.zeros(16, 128, dtype=torch.float16, device=dev)
    s = torch.ones(1, 2, 128, dtype=torch.float16, device=dev)
    eid = torch.zeros(1, dtype=torch.int32, device=dev)
    meta = torch.tensor([16, 1], dtype=torch.int32, device=dev)
    mod.mul_3bit_moe(A, B1, B2, C, s, s, eid, meta[0:1], 1)
    torch.cuda.synchronize()
    print("探测调用 OK -> ext 可用")
except Exception:
    traceback.print_exc()
    sys.exit(1)
