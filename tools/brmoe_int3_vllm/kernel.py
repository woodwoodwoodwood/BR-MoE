"""把 BR-MoE 的 Triton int3 grouped MoE 包装成 vLLM 侧可调用的算子。

按形状分派 CUDA MoE、Triton grouped GEMM 与 GEMV。A100 的 DeepSeek-MoE
prefill 使用额外校准的路径；BRMOE_PREFILL_BACKEND=legacy 可回退到之前的分派。

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


# ---- grouped MoE CUDA kernel (tile 级融合, M 中间区间) ----
_EXT = None
_EXT_TRIED = False


def get_moe_cuda_ext():
    """懒加载 marlin_int3_moe 扩展; 加载失败 (未编译/架构不匹配) 返回 None。"""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        import importlib.util
        import torch as _t
        # marlin_int3_moe 是 triton_int3 的**同级**目录 (kernels/ 下), 不是子目录
        # —— 这里曾经多拼了一层, FileNotFoundError 被裸 except 吞掉,
        # CUDA 路径因此从未启用 (39028 的 traceback 实证)。
        d = os.path.abspath(os.path.join(_kernels_dir(), "..", "marlin_int3_moe"))
        # 按 GPU 架构选 .so: 优先 brmoe_moe_int3_sm<cc>.*.so, 否则用无后缀默认版
        cap = _t.cuda.get_device_capability(0)
        tag = f"_sm{cap[0]}{cap[1]}"
        cands = [f for f in os.listdir(d)
                 if f.startswith("brmoe_moe_int3") and f.endswith(".so")]
        pref = [f for f in cands if tag in f] or [f for f in cands
                                                  if "_sm" not in f]
        if not pref:
            return None
        so = os.path.join(d, sorted(pref)[0])
        spec = importlib.util.spec_from_file_location("brmoe_moe_int3", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 架构不匹配要到 kernel 启动才炸; 用一个极小调用探测
        # 形状约束: prob_k/prob_n 必须是 128 的倍数 (16*thread_k/n_blocks)。
        # 注意: 探测 K 曾经是 64 -> 每次都被 ERR_PROB_SHAPE 拒掉, 而裸 except
        # 把 RuntimeError 吞成 ext=None -> CUDA 路径长期静默未启用
        # (check_ext.py 可复现)。K=128 才是合法探测形状。
        dev = _t.device("cuda")
        A = _t.zeros(16, 128, dtype=_t.float16, device=dev)
        B1 = _t.zeros(1, 8, 128, dtype=_t.int32, device=dev)
        B2 = _t.zeros(1, 8, 64, dtype=_t.int32, device=dev)
        C = _t.zeros(16, 128, dtype=_t.float16, device=dev)
        s = _t.ones(1, 2, 128, dtype=_t.float16, device=dev)
        eid = _t.zeros(1, dtype=_t.int32, device=dev)
        meta = _t.tensor([16, 1], dtype=_t.int32, device=dev)
        mod.mul_3bit_moe(A, B1, B2, C, s, s, eid, meta[0:1], 1)
        _t.cuda.synchronize()
        _EXT = mod
    except Exception:
        import traceback
        traceback.print_exc()   # 不能再静默吞掉 (probe 形状 bug 就是这么藏了几周)
        _EXT = None
    return _EXT


def build_cuda_packed(layer, group_size: int):
    """从 N-major 原件构建 Marlin 布局副本 (必须在 K-major 转置之前调用)。"""
    if get_moe_cuda_ext() is None:
        return None
    get_fused_moe_int3()   # 确保 triton_int3 已在 sys.path (repack 依赖 int3_moe.packing)
    kd = os.path.abspath(os.path.join(_kernels_dir(), ".."))   # kernels/ 包根
    if kd not in sys.path:
        sys.path.insert(0, kd)
    from marlin_int3_moe.repack import repack_moe
    packed_n = {
        "w13_q": layer.w13_q, "s13": layer.w13_s, "z13": layer.w13_z,
        "w2_q": layer.w2_q, "s2": layer.w2_s, "z2": layer.w2_z,
        "group_size": group_size,
    }
    return repack_moe(packed_n)


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

    # Experimental opt-in, calibrated with full MoE + full-INT3 e2e on sm_120.
    # Different-prompt traffic and sm_80 require their own dispatch calibration.
    # Fused CUDA glue takes priority when both experiment switches are enabled.
    use_fused_cuda = (os.environ.get("BRMOE_CUDA_FUSE") in ("1", "atomic")
                      and getattr(layer, "brmoe_cuda_packed", None) is not None
                      and get_moe_cuda_ext() is not None)
    if (os.environ.get("BRMOE_GROUPED_GEMV") == "1"
            and not use_fused_cuda
            and 4 <= x.shape[0] <= 16
            and torch.cuda.get_device_capability(x.device) == (12, 0)):
        from int3_moe.grouped_gemv import fused_moe_grouped_gemv, grouped_gemv_config
        return fused_moe_grouped_gemv(
            x, topk_weights, topk_ids, build_packed(layer, group_size),
            out_dtype=out_dtype, **grouped_gemv_config(x.shape[0]))

    # ---- 分派 (单一入口 fused_moe_int3_cuda, 内部再按 M 二次分派) ----
    global _PATH_LOGGED
    if os.environ.get("BRMOE_DEBUG") and not _PATH_LOGGED:
        _PATH_LOGGED = True
        pk_ = getattr(layer, "brmoe_cuda_packed", None)
        print(f"[brmoe_int3] dispatch: M={x.shape[0]} "
              f"cuda_packed={'有' if pk_ is not None else '无'} "
              f"ext={'有' if get_moe_cuda_ext() is not None else '无'}",
              flush=True)
    #   M <= gemv_max_m : GEMV (fused_moe_int3_cuda 内部委托; None 时按架构:
    #                     sm_120 原版→8 / 融合→2；sm_80→2)
    #   gemv_max_m < M <= 512 : CUDA tile 级融合 kernel (5090 实测 1.85~1.95x 于 TC)
    #   A100 / GS64 / DeepSeek: 256<=M<=512 可用 CUDA 32 行 tile（需重编扩展与 fusion=1）；
    #                        M>512 用下方专用 grouped TC + FP32 top-k 归约。
    #   其他 M > 512  : 原 Triton grouped GEMM。
    # 数值: verify_moe_cuda.py 全量校验通过 (39016, 5090+A100); 依赖修复:
    #       with_zeros kernel 的 s/z 组偏移 + 插件探测形状/路径 (见 git log)。
    M = x.shape[0]
    pk = getattr(layer, "brmoe_cuda_packed", None)
    # Calibrated on the full-INT3 DeepSeek-MoE E64/K2048/I1408/top-k6 shape.
    # Keep other models/devices on their existing dispatch until measured.
    from .prefill import prefill_enabled
    prefill = (M >= 256 and group_size == 64 and x.dtype == torch.float16
               and x.shape[1] == 2048 and topk_ids.shape[1] == 6
               and layer.w13_s.shape[0] == 64 and layer.w13_s.shape[-1] == 2816
               and getattr(layer, 'w_transposed', False)
               and torch.cuda.get_device_capability(x.device) == (8, 0)
               and prefill_enabled())
    if prefill and M > 512 and topk_ids.numel() <= 32768:
        from int3_moe.grouped_tc import fused_moe_int3_tc
        return fused_moe_int3_tc(
            x, topk_weights, topk_ids, build_packed(layer, group_size),
            block_m=128, block_n=128, block_k=64, num_warps=4, num_stages=3,
            reduce_topk=True, fast_align=True,
            out_dtype=out_dtype if out_dtype is not None else x.dtype)
    if pk is not None and M <= 512:
        ext = get_moe_cuda_ext()
        if ext is not None:
            kd = os.path.abspath(os.path.join(_kernels_dir(), ".."))
            if kd not in sys.path:
                sys.path.insert(0, kd)
            from marlin_int3_moe.moe_cuda import fused_moe_int3_cuda
            larger_tile = (prefill and os.environ.get('BRMOE_CUDA_FUSE') == '1'
                           and getattr(ext, 'supports_moe_thread_m', False))
            return fused_moe_int3_cuda(
                x, topk_weights, topk_ids, pk, ext,
                out_dtype=out_dtype if out_dtype is not None else x.dtype,
                packed=build_packed(layer, group_size),   # 小 M 时内部走 GEMV
                gemv_max_m=None,
                tile_m=32 if larger_tile else 16,
                fast_align=larger_tile)   # 按架构与融合开关选择阈值

    return fused(
        x,
        topk_weights,
        topk_ids,
        build_packed(layer, group_size),
        fast=True,                                   # Triton align, 零 host 同步
        out_dtype=out_dtype if out_dtype is not None else x.dtype,
    )


_DEBUG_DONE = False
_PATH_LOGGED = False


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
