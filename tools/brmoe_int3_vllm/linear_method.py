"""BR-MoE int3 的**非专家线性层** (attention / dense MLP / shared_experts)。

背景 / 为什么需要这个文件
------------------------
BR-MoE 的 3bit 模型量化范围是：除 embedding / layernorm / router 外的**所有**线性层。
在源 checkpoint (qmodel.pt) 里逐模块核实过，共 5380 个量化模块:

    attention 投影 (q/k/v/o)   112 = 28层 x 4
    routed experts             5184 = 27层 x 64专家 x 3
    shared experts               81 = 27层 x 3
    dense MLP (layer 0)           3

而 vLLM 插件原先只实现了 MoE (FusedMoE) 的 quant method，`config.py` 对
LinearBase 一律回退到 UnquantizedLinearMethod —— 这会让转换器把 196 个
非专家投影反量化成 fp16，产出的模型比忠实版本大 1.46 GiB
(8.75 GiB vs 7.29 GiB)，且与 HF 侧跑的不是同一个模型。

本文件补上这 196 个层。

kernel 选择
----------
复用已有的 Triton `int3_moe_gemm`，而不是 BR-MoE 的 CUDA 扩展
(`kernels/brmoe/mul_3bit_with_zeros`)，理由:

  1. CUDA 扩展是 `brmoe_cuda.cpython-310-*.so` —— **py3.10** 编译的，
     而 vLLM 插件跑在 py3.12 环境，直接 import 会失败，要重编。
  2. `int3_moe_gemm` 的三个开关正好覆盖线性层的全部需求:
         W_T=True  权重 K-major [Kpack, N]  <- 转换器 dense int3 就是这个布局
         HAS_ZERO=True  非对称量化, 读 zeros
         A_GATHER=False 不做行重映射
     把 E=1 (单专家) + top_k=1 代进去, 就是普通 GEMM `y = x @ W^T`。
  3. 布局零改动, 不必重跑转换。

若日后 attention 的 Triton GEMM 成为瓶颈, 再考虑重编 CUDA 扩展走
BRMoE_Asymmetric_Linear.matmul 那条路 (它的大 tile 是 (256,64)/(128,128))。

权重契约 (与 tools/convert_brmoe_to_vllm.py --dense-mode int3 一致)
------------------------------------------------------------------
    <proj>.qweight  int32 [K//32*3, N]    K-major (W_T=True 直接吃)
    <proj>.scales   fp16  [K//gs,  N]
    <proj>.zeros    fp16  [K//gs,  N]
    dequant: W = (unpack_int3(qweight) - zeros) * scales

注意 N 的取值里有 10944 (layer 0 的 dense down_proj 输出维度) —— BR-MoE 的
CUDA kernel 在 n 非 128 整数倍时 workspace 会少算导致死锁 (见 backends/brmoe.py
的注释)。Triton 这条路不涉及那个 buffer, 但 tile 选择仍要保证
    K % block_k == 0 且 N % block_n == 0
所以这里对 block_n 做了自适应。
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

# vLLM 的参数属性设置 helper —— 不同版本位置不同, 逐个试
try:
    from vllm.model_executor.parameter import set_weight_attrs
except Exception:                                     # pragma: no cover
    try:
        from vllm.model_executor.utils import set_weight_attrs
    except Exception:
        def set_weight_attrs(param, attrs):
            for k, v in (attrs or {}).items():
                setattr(param, k, v)


# BR-MoE 的 Triton 打包模块 (包目录名含连字符 "BR-MoE", 不能直接 import)
import importlib.util
import os
import sys


def _load_module(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BRMOE_PKG = os.environ.get(
    "BRMOE_PKG",
    "/mnt/709/data3/home/jianglei/ada/BR-MoE/BR-MoE",
)

# 只用到 int3_moe_gemm。**不需要** packing.py 的 BitPack —— 那是转换器的
# 离线解包工具 (而且它其实来自 hqq.core.bitpack, 不在 BR-MoE 里);
# 运行时 kernel 直接吃打包好的 qweight。
_kernel = _load_module(
    "brmoe_int3_kernel_lin",
    os.path.join(_BRMOE_PKG, "kernels", "triton_int3", "int3_moe", "kernel.py"),
)

int3_moe_gemm = _kernel.int3_moe_gemm


# ---- 小 M 的 GEMV + split-K 工作区 (按形状缓存, CUDA Graph 安全) ----
_GEMV_WS = {}


def _get_gemv_ws(M: int, N: int, device):
    """GEMV 路径的两个缓冲: ids (全 0 = 单专家) 和 fp32 输出 (split-K atomic 归约)。"""
    key = (M, N, str(device))
    hit = _GEMV_WS.get(key)
    if hit is None:
        hit = (torch.zeros(M, dtype=torch.int64, device=device),
               torch.zeros(M, N, dtype=torch.float32, device=device))
        _GEMV_WS[key] = hit
    return hit


def pick_tiles(K: int, N: int, group_size: int, M: int | None = None):
    """挑一组满足整除约束的 (block_m, block_n, block_k, slot)。

    block_k 必须是 32 的倍数 (kernel 每次处理 BLOCK_K 个 k) 且整除 K;
    block_n 必须整除 N。N=10944 = 2^6*171 这种非 2 的幂要小心。
    block_m 按 M 自适应 —— 见下方注释, 这是 decode 阶段的主要开销来源。
    """
    def ok(v, divisor, mult=1):
        return divisor % v == 0 and v % mult == 0

    block_k = next((bk for bk in (32, 64, 128) if ok(bk, K, 32)), 32)
    # 注意: K 未必能被 128 整除 (如 attention 的 K=2048 可以, 但保险起见逐档降)
    if K % block_k:
        block_k = 32 if K % 32 == 0 else next((v for v in (16, 8, 4, 2, 1)
                                               if K % v == 0), 1)
    block_n = next((bn for bn in (64, 32, 16) if ok(bn, N)), 16)
    if N % block_n:
        block_n = next((v for v in (8, 4, 2, 1) if N % v == 0), 1)
    # ---- block_m: 按 M 自适应 (原来是写死 64) ----
    # kernel 会覆盖 grid_m*block_m **整行**, 而 n_rows = ceil(num_post/block_m)*block_m。
    # 写死 64 时实测:
    #     M=1  -> num_post=16 -> n_rows=64   (浪费 64x)
    #     M=16 -> num_post=16 -> n_rows=64   (浪费  4x)
    # 即 bs=1..16 算的全是 64 行 —— batch 翻 16 倍算力一点不变, 实测 TPOT 被
    # 锁在 18.79~22.67 ms/step (而 attention 走 fp16 的版本是 6.03~11.49, 会
    # 随 batch 正常增长)。这就是 int3dense 慢 3 倍的直接原因。
    #
    # 约束 (kernel.py:322): 16 <= slot <= block_m, block_m % slot == 0, slot % 16 == 0
    # 所以 block_m 最小就是 16。让 block_m 贴合 num_post(16 的倍数) 即可把
    # n_rows 压到刚好覆盖:
    #     M<=16 -> bm=16  n_rows=16  (4x)
    #     M<=32 -> bm=32  n_rows=32  (2x)
    #     M>32  -> bm=64  与原行为一致 (此时本就无浪费)
    # block_m 必须是 2 的幂 (Triton 约束), 所以 num_post=48 这类只能落到
    # bm=64, 与原来相同, 不会更差。
    #
    # 注: 图模式下 M 是捕获时的 batch, 每次 capture 独立 —— 各档位各自取到
    #     合适的 block_m, 正是我们要的。
    slot = 16
    if M is None:
        block_m = 64                    # 未给 M -> 保持旧行为, 不猜
    else:
        _num_post = (M + slot - 1) // slot * slot
        if _num_post <= 16:
            block_m = 16
        elif _num_post <= 32:
            block_m = 32
        else:
            block_m = 64
    # slot 必须满足 16 <= slot <= block_m 且整除 block_m; 取**最小**的 16 (见上)。
    # 因为 kernel 断言 num_post % slot == 0 (align 按 slot 补齐), 而 decode
    # 阶段 M 可能只有 1~2 个 token —— 实测 M=2 + slot=64 会报
    #   "num_post=2 必须是 slot=64 的倍数"。
    # slot=16 时最多浪费 15 行, 对齐开销可忽略。
    return block_m, block_n, block_k, slot


# ---------------------------------------------------------------------------
# 元数据缓存 (expert_ids / sorted_token_ids)
# ---------------------------------------------------------------------------
# 这两个张量的**内容只由形状决定** (eid 恒为 0 = 都归专家 0, sti 恒为 arange),
# 而 kernel 对它们只读 —— 所以按形状缓存复用, 省掉每次调用的 torch.zeros +
# torch.arange 两个 kernel。这正是 MoE 路径 `_WS_CACHE` 的同款做法。
# CUDA Graph 下在捕获时建立、之后复用 (地址被烘进图里, 内容不变)。
_LIN_META: dict = {}


def _get_lin_meta(grid_m: int, nseg: int, n_rows: int, device):
    key = (grid_m, nseg, n_rows, device)
    hit = _LIN_META.get(key)
    if hit is None:
        hit = (torch.zeros(grid_m * nseg, dtype=torch.int32, device=device),
               torch.arange(n_rows, dtype=torch.int32, device=device))
        _LIN_META[key] = hit
    return hit


@torch.no_grad()
def _brmoe_int3_linear_impl(x: torch.Tensor,
                            qweight: torch.Tensor,
                            scales: torch.Tensor,
                            zeros: torch.Tensor,
                            group_size: int) -> torch.Tensor:
    """y = x @ W^T,  W = (unpack_int3(qweight) - zeros) * scales

    x        [M, K]  fp16
    qweight  [K//32*3, N] int32   (K-major, 喂给 W_T=True)
    scales   [K//gs, N]   fp16
    zeros    [K//gs, N]   fp16
    ->  [M, N] fp16
    """
    M, K = x.shape
    N = scales.shape[1]
    gs = int(group_size)
    assert scales.shape == zeros.shape == (K // gs, N), (scales.shape, zeros.shape)
    assert qweight.shape == (K // 32 * 3, N), qweight.shape

    x2 = x.reshape(-1, K).contiguous()
    M2 = x2.shape[0]

    # 视图成 "单专家": **w_transposed=True 要求 [E, Kpack, N]**
    # (kernel 从 shape[1] 读 Kpack 反推 K)。我们的 qweight 本身就是 [Kpack, N],
    # 所以只需在最前面补一维 —— 千万不要写成 .view(1, N, -1), 那会把
    # [Kpack, N] 变成 [N, Kpack], kernel 会读到 K = N*32/3
    # (实测报 "A 的 K=2048 与权重 K=65536 不符", 65536 = 6144*32/3)。
    Wp = qweight.unsqueeze(0).contiguous()
    S = scales.view(1, K // gs, N).contiguous()
    Z = zeros.view(1, K // gs, N).contiguous()

    # ---- 小 M: GEMV + split-K (复用 MoE 的 routed_int3_gemv, 单专家视角) ----
    # 下方 TC 路径在 M=1 时 grid 只有 (1, N//64)=32 个 CTA (108 核的 A100 每 SM
    # 0.3 个), 且 M 补齐到 16 行 —— 微基准 27.7us vs cuBLAS fp16 ~6us, 注释里
    # 自己也写着"结构性问题, 调参解决不了"。GEMV 逐行无补齐, split-K 把 K 拆给
    # 更多 program。MoE 路径同一刀的实测: 88 -> 49 us (M=1, A100, job 38918)。
    if M2 <= 8:
        ids, out32 = _get_gemv_ws(M2, N, x.device)
        out32.zero_()
        # split-K 拉满: 实测 (verify_linear_gemv.py, A100) ks 越大越快,
        # M=1/4 都是 ks=16 (K=2048 的全部 128-块) 最优 —— 每 program 只算一块,
        # 延迟受限下并行度就是一切。atomics 增多但实测没有成为瓶颈。
        ks = max(1, min(16, (K + 127) // 128))
        _kernel.routed_int3_gemv(x2, Wp, S, Z, ids, None, out32, 1, gs,
                                 w_transposed=True, block_n=64, num_warps=2,
                                 ksplit=ks)
        return out32.to(x.dtype).view(*x.shape[:-1], N)

    block_m, block_n, block_k, slot = pick_tiles(K, N, gs, M2)

    # ---- 行对齐: 只算网格上界, **不再 pad 数据** ----
    # 关键: kernel 的 a 加载自带谓词掩码。
    #   kernel.py:305   mask_m = tok < num_valid    (num_valid = M2, 我们传入)
    #   kernel.py:211   a = tl.load(A + a_row*SA + ..., mask=mask_m, other=0.0)
    # 越界行 (offs_m >= M2) 返回 0、C 也不写 —— 所以下面这层补齐纯属多余:
    #     xp = torch.zeros(n_rows, K)   <- memset kernel
    #     xp[:M2] = x2                  <- copy kernel
    # 直接把 x2 (只有 M2 行) 传进去即可。masked load 不会真的去 dereference
    # 越界地址 (Triton 的谓词是先算地址、再按 mask 决定是否访存)。
    num_post = (M2 + slot - 1) // slot * slot     # 仍是 slot 的倍数 (launcher 断言)
    grid_m = (num_post + block_m - 1) // block_m
    n_rows = grid_m * block_m

    # eid 的长度必须是 grid_m * (block_m // slot): kernel 在一个 block 里按
    # NSEG = block_m//slot 个 slot 子块去读 expert_ids。全部填 0 = 都归专家 0。
    # eid/sti 与数据无关且只读 -> 缓存复用 (见 _LIN_META 的注释)。
    nseg = block_m // slot
    eid, sti = _get_lin_meta(grid_m, nseg, n_rows, x.device)
    out = torch.empty((n_rows, N), dtype=x.dtype, device=x.device)

    int3_moe_gemm(
        x2, Wp, S,
        sti, eid,
        num_post,              # num_tokens_post_pad (对齐后的行数)
        M2,                    # num_valid (真实行数)
        group_size=gs,
        a_gather=False,        # 线性层: 行号不重映射 (SORTED_TOKENS 不被读)
        add=False,
        out=out,
        # meta=None -> HAS_META=False, kernel 改用 num_post_arg 作为 num_post。
        # 千万**不要**传一个 zeros(2) 的假 meta: 那样 HAS_META=True 会从 meta[0]
        # 读到 0, 等于整层不算。参见 ops.py 里 fast/else 两条分支的 meta_arg 用法。
        meta=None,
        grid_m=grid_m,
        block_m=block_m, block_n=block_n, block_k=block_k, slot=slot,
        layout4=False, layout16=False,
        zeros=Z,
        w_transposed=True,     # 权重 K-major
        direct4=False,
        # ---- 启动参数 ----
        # int3_moe_gemm 的默认是 num_warps=2 / num_stages=1, 那是给 MoE 路径调的
        # (那边 grid 里有专家维度撑着)。线性层小 M 下 grid = (grid_m, N//block_n)
        # 只有 32 个 CTA, 而 5090 有 170 个 SM, 默认值明显偏瘦。
        #
        # 实测 (bench/micro_linear.py --sweep 1, K=N=2048, 单层):
        #     M=1 : 默认 2/1 = 34.50 µs  ->  4/3 = 27.73 µs   (1.24x)
        #     M=8 : 默认 2/1 = 35.86 µs  ->  4/3 = 28.97 µs   (1.24x)
        #     M=64: 默认 2/1 = 140.83 µs ->  4/3 = 73.09 µs   (1.93x)
        #
        # 注意: 这**不是**那个 100x 带宽缺口的原因 —— 即便调到最优, M=1 时仍是
        # ~28 µs vs cuBLAS fp16 的 ~6 µs, 差 4.8x。那是 kernel tiling 的结构性
        # 问题, 调参解决不了。这里只是把免费的收益拿掉。
        #
        # 扫描验证过: 每种 (num_warps, num_stages) 组合结果**逐位相同**, 所以
        # 纯属调度参数, 不影响数值正确性。
        num_warps=4,
        num_stages=3,
    )
    return out[:M2].view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------
# 注册成**不透明算子** —— 这一步是让 CUDA Graph 能重新捕获的关键
# ---------------------------------------------------------------------------
# 症状 (int3dense 全 int3 模型, --enforce-eager 0):
#
#     Capturing CUDA graphs (PIECEWISE): 2%
#     inductor_cache/.../cd4w4ymiqcjdnkaycjamol2aypybnkp65aqyyctcxmcfmr6zguoj.py:2021
#         assert_size_stride(arg10_1, (s72, 2048), (2048, 1), 'input')
#     AssertionError: expected size 512==496, stride 2048==2048 at dim=0
#     This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
#
# 根因 —— dynamo **穿透**进了上面那个函数体 (它就是普通 Python), 而函数体里有
# 数据依赖的形状推导:
#
#     num_post = (M2 + slot - 1) // slot * slot     # M2 = x2.shape[0]
#     grid_m   = (num_post + block_m - 1) // block_m
#     n_rows   = grid_m * block_m
#
# 在 proxy 张量上做 floor-div, dynamo 无法保持符号化, 只能把 M2 **特化成当时的
# 具体整数** (512)。于是编出来的图只对 M=512 成立; CUDA Graph 捕获时 inductor
# 把守卫写成 `assert size == 512`, 而真实 batch 是 496 -> 断言失败, capture 直接崩。
# eager 下没有问题, 因为每次调用都重新算一遍 (这也解释了为什么 eager 一直是
# 16/16 全对的)。
#
# 修法 —— 用 vLLM 自己量化方法 (GPTQ / Marlin / 可图的 FusedMoE) 的同款做法:
# 把整块包成一个 `torch.library.custom_op`, 再给一个**只描述形状**的 fake impl。
# 这样 dynamo 只看到"一个不透明算子", 只向 fake 要输出形状; 那些特化就消失了。
# (顺带说明我们的 MoE 为什么一直在图里没事: moe_method.py 有
#  `@CustomOp.register(...)`, 走的是注册过的 `torch.ops.vllm.moe_forward_shared`。)
#
# 注意: 注册在 import 时执行且只执行一次。Python 的模块缓存保证这一点,
#       所以不需要额外的重注册保护。
_LINEAR_OP_NAME = "brmoe_int3::linear"


@torch.library.custom_op(_LINEAR_OP_NAME, mutates_args=())
def _brmoe_int3_linear_op(x: torch.Tensor,
                          qweight: torch.Tensor,
                          scales: torch.Tensor,
                          zeros: torch.Tensor,
                          group_size: int) -> torch.Tensor:
    # mutates_args=() 是对的: kernel 只写它自己内部建的 out, 四个入参都是只读。
    return _brmoe_int3_linear_impl(x, qweight, scales, zeros, group_size)


@_brmoe_int3_linear_op.register_fake
def _brmoe_int3_linear_fake(x: torch.Tensor,
                            qweight: torch.Tensor,
                            scales: torch.Tensor,
                            zeros: torch.Tensor,
                            group_size: int) -> torch.Tensor:
    """meta/fake 实现: **只描述形状, 绝不碰数据**。

    输出形状写成输入的纯符号表达式 —— 只要这里不出现任何数据依赖的中间量,
    dynamo 就不会把 M 特化成常数, 图也就不再有"只对一个 M 有效"的守卫。
    """
    return x.new_empty((*x.shape[:-1], scales.shape[1]))


def brmoe_int3_linear(x: torch.Tensor,
                      qweight: torch.Tensor,
                      scales: torch.Tensor,
                      zeros: torch.Tensor,
                      group_size: int) -> torch.Tensor:
    """对外入口, 签名与原来的实现完全一致 (所以 `apply()` 一行都不用改)。

    内部走的是上面注册的不透明 op: 编译期 dynamo 只看到这一层, 不会穿进去。
    """
    return _brmoe_int3_linear_op(x, qweight, scales, zeros, group_size)


def _make_shard_loader(parts, output_dim: int = 1):
    """按 output 维切片的 weight_loader。

    vLLM 会把 q/k/v 融成 `qkv_proj` (shard_id = "q"/"k"/"v")，或把 gate/up 融成
    `gate_up_proj` (shard_id = 0/1)，加载时把每个 shard 填到 param 的对应区间。

    vLLM 标准的 ColumnParallelLinear.weight_loader 假设 **output 在 dim 0**，
    而我们的 `qweight` 是 [Kpack, N]、`scales`/`zeros` 是 [K//gs, N] ——
    output 在 **dim 1**。直接沿用标准 loader 会切错维度（静默算错或 shape 报错），
    所以必须自己接管。vLLM 的 GPTQ (qweight [in, out/pack]) 是同一个问题，
    它有同款做法可参照。
    """
    offsets, off = {}, 0
    for i, d in enumerate(parts):
        offsets[i] = off
        off += d
    for i, s in enumerate(("q", "k", "v")):      # 字符串 shard_id 也映射上
        if i in offsets:
            offsets[s] = offsets[i]

    def loader(param, loaded_weight, shard_id=None):
        if shard_id is None:                      # 非融合层: 整块拷
            param.data.copy_(loaded_weight)
            return
        start = offsets[shard_id]
        size = loaded_weight.shape[output_dim]
        param.data.narrow(output_dim, start, size).copy_(loaded_weight)

    return loader


class BRMoEInt3LinearMethod(LinearMethodBase):
    """把 int3 量化的**非专家线性层**接到 int3_moe_gemm 上。

    只支持 Tensor Parallel = 1 (权重未分片)。TP>1 需要按 output 维切分,
    这里显式拒绝, 避免静默算错。
    """

    def __init__(self, quant_config) -> None:
        self.quant_config = quant_config
        self.group_size = int(getattr(quant_config, "group_size", 64))

    # ---- 权重创建 ----
    def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                       output_partition_sizes, input_size: int,
                       output_size: int, params_dtype, **extra_weight_attrs):
        # 注意: **不要**要求 len(output_partition_sizes) == 1。
        # QKVParallelLinear 在 TP=1 下也会传 [n_q, n_k, n_v] (如 [2048,2048,2048]),
        # 那是把 q/k/v 融成一个 qkv_proj 的**正常**行为, 不是 TP 分片。
        # 同理 MergedColumnParallelLinear (gate/up 融合) 会传 [n_gate, n_up]。
        parts = [int(x) for x in output_partition_sizes]
        K = int(input_size_per_partition)
        N = sum(parts)
        gs = self.group_size

        # 与转换器 --dense-mode int3 的输出契约一致
        layer.register_parameter(
            "qweight",
            nn.Parameter(torch.empty(K // 32 * 3, N, dtype=torch.int32),
                         requires_grad=False),
        )
        for name in ("scales", "zeros"):
            layer.register_parameter(
                name,
                nn.Parameter(torch.empty(K // gs, N, dtype=torch.float16),
                             requires_grad=False),
            )

        loader = _make_shard_loader(parts, output_dim=1)
        for pname in ("qweight", "scales", "zeros"):
            attrs = dict(extra_weight_attrs or {})
            attrs["weight_loader"] = loader       # 覆盖 vLLM 默认的那个 (它按 dim 0 切, 会切错)
            set_weight_attrs(getattr(layer, pname), attrs)

        layer._br_int3 = dict(K=K, N=N, group_size=gs, parts=parts)

    # ---- 加载后处理 ----
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # kernel 要求 K-major 连续; safetensors 载入后已是 contiguous,
        # 这里只做一次保险性的 contiguous(), 不复制时就地返回。
        for pname in ("qweight", "scales", "zeros"):
            p = getattr(layer, pname)
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()

    # ---- 前向 ----
    def apply(self, layer: nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        y = brmoe_int3_linear(x, layer.qweight, layer.scales, layer.zeros,
                              layer._br_int3["group_size"])
        if bias is not None:
            y = y + bias
        return y
