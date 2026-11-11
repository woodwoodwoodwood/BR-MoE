"""验证 brmoe_int3::linear 的 custom_op 注册 —— 重点是 M 是否不再被特化。

不需要 GPU: fake / symbolic 路径全在 CPU 上就能跑。
"""
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import brmoe_int3_vllm.linear_method as L

K, N, GS = 2048, 2048, 64
M = 496                      # 故意用非 512 的值 —— 原来崩的就是它

print("=" * 72)
print("1. 算子是否注册")
print("=" * 72)
try:
    pkt = torch.ops.brmoe_int3.linear
    print("   ✅ torch.ops.brmoe_int3.linear 存在")
    print("   签名:", pkt.default._schema)
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}")
    raise SystemExit(1)

print()
print("=" * 72)
print("2. fake impl: 只描述形状, 不碰数据")
print("=" * 72)
from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode():
    fx = torch.empty(M, K, dtype=torch.float16)
    fqw = torch.empty(K // 32 * 3, N, dtype=torch.int32)
    fs = torch.empty(K // GS, N, dtype=torch.float16)
    fz = torch.empty(K // GS, N, dtype=torch.float16)
    fy = L.brmoe_int3_linear(fx, fqw, fs, fz, GS)
    want = (*fx.shape[:-1], N)
    ok = tuple(fy.shape) == want
    print(f"   假张量输出 shape = {tuple(fy.shape)}  dtype={fy.dtype}")
    print(f"   期望             = {want}")
    print(f"   -> {'✅ 一致' if ok else '❌ 不一致'}")

print()
print("=" * 72)
print("3. ★ 关键: symbolic 追踪下 M 是否保持符号化 (根因所在)")
print("=" * 72)
x = torch.empty(M, K, dtype=torch.float16)
qw = torch.empty(K // 32 * 3, N, dtype=torch.int32)
sc = torch.empty(K // GS, N, dtype=torch.float16)
zo = torch.empty(K // GS, N, dtype=torch.float16)


def f(x, qw, sc, zo):
    return L.brmoe_int3_linear(x, qw, sc, zo, GS)


gm = make_fx(f, tracing_mode="symbolic")(x, qw, sc, zo)

print("--- 追踪出的图 ---")
print(gm.code)

print("--- 各节点形状 ---")
for n in gm.graph.nodes:
    v = n.meta.get("val")
    shp = getattr(v, "shape", None)
    print(f"   {n.op:11} {str(n.target)[:44]:46} shape={shp}")

print()
print("--- 判定 ---")
code = gm.code
hard = "496" in code
print(f"   图中硬编码 496 ?  {'❌ 是 -> 仍会特化, 没修好' if hard else '✅ 否'}")

d0 = [n for n in gm.graph.nodes if n.op == "placeholder"][0].meta["val"].shape[0]
sym = not isinstance(d0, int)
print(f"   输入 dim0 = {d0!r} (type={type(d0).__name__})")
print(f"   -> {'✅ SymInt, 符号化保住了 (不同 M 不会触发守卫失败)' if sym else '❌ int, 被特化成常数'}")

n_nodes = len([n for n in gm.graph.nodes if n.op == "call_function"])
print(f"   call_function 节点数 = {n_nodes} "
      f"({'✅ 整块是单个不透明算子, dynamo 没穿进去' if n_nodes <= 3 else '⚠️ 图里被展开了'})")

print()
print("=" * 72)
print("4. opcheck (只跑 schema / fake 相关, 不执行真 kernel)")
print("=" * 72)
try:
    torch.library.opcheck(L._brmoe_int3_linear_op, (x, qw, sc, zo, GS),
                          test_utils=["test_schema"])
    print("   ✅ test_schema 通过")
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}")

print()
print("=" * 72)
print("结论")
print("=" * 72)
print("   符号化保持住 + 单个不透明算子 => CUDA Graph 捕获时不再有"
      " 'assert size == 512' 这类守卫")
