"""MoE 的 token 对齐 —— moe_align_block_size 的 torch 实现。

grouped GEMM 的前提: 把 (token, expert) 对按 expert 排序, 让每个 expert 的 token
数量向上对齐到 block_size 的倍数, 这样 kernel 里"一个 BLOCK_M 只属于一个 expert"
才成立 (可以直接用 expert_ids[pid_m] 取权重)。

产出:
    sorted_token_ids:      [num_tokens_post_pad] int32
                           前 num_valid 个有效位置是 token 下标; pad 位置填哨兵 (哨兵值
                           习惯取 num_valid, 让 kernel 用 `tok < num_valid` 一次判掉)
                           注意: 同一个 token 会在 top_k 个 expert 里各出现一次
    expert_ids:            [num_tokens_post_pad // block_size] int32, 每个 m-block 的 expert
    num_tokens_post_pad:   padded 后的总行数
"""

import torch
from torch import Tensor
from typing import Tuple
from collections import Counter


@torch.no_grad()
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    flat_values: Tensor = None,
) -> Tuple[Tensor, Tensor, int]:
    """把 topk_ids 按 expert 排序 + 对齐到 block_size。

    Args:
        topk_ids: [M, top_k] int32/int64, 每个 token 选中的 expert 下标
        num_experts: 专家总数
        block_size: 对齐粒度, 必须能被 kernel 的 BLOCK_M 整除 (通常 64)
        flat_values: 可选, [M*top_k] 与 topk_ids 展平后一一对应的标量序列
                     (例如路由权重), 会按同一布局 scatter 出去
    Returns:
        (sorted_token_ids, expert_ids, num_tokens_post_pad)
        若传了 flat_values, 额外返回 sorted_values [num_tokens_post_pad] (pad 位为 0)
    """
    device = topk_ids.device
    topk_ids = topk_ids.to(torch.int64)
    M, top_k = topk_ids.shape
    num_valid = M * top_k

    flat = topk_ids.reshape(-1)                                  # (token, k) 展平
    order = torch.argsort(flat, stable=True)                     # 按 expert 稳定排序
    sorted_experts = flat[order]
    sorted_tokens = torch.div(order, top_k, rounding_mode="floor")  # 每个 pair 的原 token 下标

    counts = torch.bincount(sorted_experts, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    num_tokens_post_pad = int(padded.sum().item())

    # 每个 expert 的 block 区间起点 (按 padded 计)
    pad_starts = torch.cumsum(padded, 0) - padded
    # 每个 expert 在"排序后序列"里的起点 (按真实数量计)
    cnt_starts = torch.cumsum(counts, 0) - counts

    # 有效 token 的落位: expert 的 padded 起点 + 该 expert 内的序号
    within = torch.arange(num_valid, device=device) - cnt_starts[sorted_experts]
    dst = pad_starts[sorted_experts] + within

    sorted_token_ids = torch.full((num_tokens_post_pad,), num_valid, dtype=torch.int32, device=device)
    sorted_token_ids.scatter_(0, dst, sorted_tokens.to(torch.int32))

    # 每个 m-block 归属的 expert
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), padded // block_size
    ).to(torch.int32)

    if flat_values is not None:
        assert flat_values.shape == (num_valid,), f"flat_values 形状应为 {(num_valid,)}"
        sv = torch.zeros(num_tokens_post_pad, dtype=flat_values.dtype, device=device)
        sv.scatter_(0, dst, flat_values[order])
        return sorted_token_ids, expert_ids, num_tokens_post_pad, sv

    return sorted_token_ids, expert_ids, num_tokens_post_pad


@torch.no_grad()
def check_align(topk_ids: Tensor, num_experts: int, block_size: int) -> None:
    """自检: 排序正确性 + pad 位置 + expert_ids 一致性。"""
    sorted_token_ids, expert_ids, num_post = moe_align_block_size(topk_ids, num_experts, block_size)
    M, top_k = topk_ids.shape
    num_valid = M * top_k

    assert num_post % block_size == 0
    assert expert_ids.numel() == num_post // block_size

    valid = sorted_token_ids[sorted_token_ids < num_valid]
    assert valid.numel() == num_valid, "有效 token 数不对"
    # (token, expert) 对的多重集必须与输入完全一致 (允许同一 token 重复命中同一 expert:
    # 那种情况下 kernel 会对同一行做多次加权累加, 语义上正是路由想要的)
    seen = Counter()
    for blk in range(expert_ids.numel()):
        e = int(expert_ids[blk])
        seg = sorted_token_ids[blk * block_size : (blk + 1) * block_size]
        for t in seg.tolist():
            if t >= num_valid:
                continue
            seen[(t, e)] += 1
    expect = Counter()
    for t in range(M):
        for k in range(top_k):
            expect[(t, int(topk_ids[t, k]))] += 1
    assert seen == expect, "排序后的 (token, expert) 多重集与输入不匹配"

    # pad 行必须落在每个 block 的尾部 (不能夹在有效行中间), 否则 kernel 的 mask 会漏算
    for blk in range(expert_ids.numel()):
        seg = sorted_token_ids[blk * block_size : (blk + 1) * block_size]
        real = (seg < num_valid)
        assert not real.numel() or bool(
            (torch.nonzero(real, as_tuple=False).flatten().diff() == 1).all()
        ), "有效行不连续"


__all__ = ["moe_align_block_size", "check_align"]
