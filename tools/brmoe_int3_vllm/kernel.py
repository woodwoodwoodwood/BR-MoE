"""把 BR-MoE 的 Triton int3 grouped MoE 包装成 vLLM 侧可调用的算子。

复用 `BR-MoE/kernels/triton_int3/int3_moe/ops.py::fused_moe_int3`，不做任何改动。

关于 CUDA Graph
---------------
`fused_moe_int3(fast=True)` 走 `moe_align_block_size_triton`，该实现**零 host 同步**
（`.item()` 只出现在 `fast=False` 的对照基线路径里），所以可以被 CUDA Graph 捕获。
工作区 (`_WS_CACHE`) 与 align 缓冲区 (`align_triton._BUF`) 都按 key 缓存，首次调用
分配、之后复用 —— vLLM 在捕获前会做 warmup，届时缓存已建立。
如果实测捕获异常，用 `--enforce-eager` 先跑通。
"""

import os
import sys
import threading

import torch

_LOCK = threading.Lock()
_FUSED = None
_IMPORT_ERR = None


def _kernels_dir() -> str:
    """<repo>/BR-MoE/kernels/triton_int3"""
    here = os.path.dirname(os.path.abspath(__file__))          # .../tools/brmoe_int3_vllm
    repo = os.path.dirname(os.path.dirname(here))              # <repo>
    return os.path.join(repo, "BR-MoE", "kernels", "triton_int3")


def get_fused_moe_int3():
    """惰性导入 (triton 导入较慢，也会拖慢 vLLM 启动)。"""
    global _FUSED, _IMPORT_ERR
    if _FUSED is None:
        with _LOCK:
            if _FUSED is None:
                d = _kernels_dir()
                if not os.path.isdir(d):
                    raise ImportError(f"找不到 BR-MoE 的 int3 kernel 目录: {d}")
                if d not in sys.path:
                    sys.path.insert(0, d)
                try:
                    from int3_moe.ops import fused_moe_int3
                except Exception as e:                          # pragma: no cover
                    _IMPORT_ERR = e
                    raise ImportError(
                        f"导入 int3_moe.ops 失败 ({type(e).__name__}: {e})。"
                        f" 需要 triton>=2.1 且 GPU 算力 >= sm_75。"
                    ) from e
                _FUSED = fused_moe_int3
    return _FUSED


def build_packed(layer, group_size: int) -> dict:
    """从 `RoutedExperts` 上的参数组装 `fused_moe_int3` 需要的 packed dict。

    布局: checkpoint 里是 N-major ([E, N, Kpack]); 但 moe_method 的
    `process_weights_after_loading` 会把 w13_q/w2_q 转置成 K-major 并置
    `layer.w_transposed = True` (GEMV 访存合并, 实测 +5~19%), 所以这里从
    layer 上读标记, 不再写死:

        N-major: w13_q [E, 2I, K//32*3]      w2_q [E, K, I//32*3]
        K-major: w13_q [E, K//32*3, 2I]      w2_q [E, I//32*3, K]
        scale/zero 两种布局同形: s13/z13 [E, K//gs, 2I]  s2/z2 [E, I//gs, K]
    """
    return {
        "w13_q": layer.w13_q,
        "s13": layer.w13_s,
        "z13": layer.w13_z,      # 非对称量化的每组浮点零点
        "w2_q": layer.w2_q,
        "s2": layer.w2_s,
        "z2": layer.w2_z,
        "group_size": group_size,
        "layout": "int3",
        "w_transposed": getattr(layer, "w_transposed", False),
    }


@torch.no_grad()
def brmoe_int3_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer,
    group_size: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """一次 fused MoE: 小 batch 用 GEMV, 其余用 grouped GEMM。"""
    fused = get_fused_moe_int3()

    # kernel 期望 int64 的 expert id (docstring 明确)
    if topk_ids.dtype != torch.int64:
        topk_ids = topk_ids.to(torch.int64)
    # x 必须是 2D [M, K]
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])

    topk_ids, topk_weights = _sanitize_routing(topk_ids, topk_weights)

    return fused(
        x,
        topk_weights,
        topk_ids,
        build_packed(layer, group_size),
        fast=True,                                   # Triton align, 零 host 同步
        out_dtype=out_dtype if out_dtype is not None else x.dtype,
    )


_DEBUG_DONE = False


def _sanitize_routing(topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """把 vLLM 的无效专家槽位（-1）规整掉。

    为什么需要: BR-MoE 的 align 走 `torch.bincount(expert_ids)`（align.py），
    而 bincount 要求**非负**；vLLM 在 padding / expert_map 场景下会用 -1 标记
    无效专家（见 vllm `moe_align_block_size.py` 的文档），直接传进去会报
    `bincount only supports 1-d non-negative integral inputs`。

    处理: 把 -1 换成 0（合法专家），同时把该槽位的路由权重清零，
    使它对加权和无贡献。全程用张量算子，不做 host 同步，CUDA Graph 安全。
    """
    global _DEBUG_DONE
    if os.environ.get("BRMOE_DEBUG") and not _DEBUG_DONE:
        _DEBUG_DONE = True
        print(
            f"[brmoe_int3] topk_ids shape={tuple(topk_ids.shape)} "
            f"dtype={topk_ids.dtype} min={int(topk_ids.min())} "
            f"max={int(topk_ids.max())} | topk_weights {topk_weights.dtype}",
            flush=True,
        )

    neg = topk_ids < 0
    # torch.where 不做同步；不能 in-place 改（会污染 vLLM 的调度器状态）
    topk_ids = torch.where(neg, torch.zeros((), dtype=topk_ids.dtype,
                                            device=topk_ids.device), topk_ids)
    topk_weights = torch.where(neg, torch.zeros((), dtype=topk_weights.dtype,
                                               device=topk_weights.device),
                               topk_weights)
    return topk_ids, topk_weights
