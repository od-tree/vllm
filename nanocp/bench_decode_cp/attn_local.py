"""Paged GQA decode through the already-built FA2 extension.

Head exchange stays off (``src_rank=-1``). A CP tile plan may still be
passed: it is the same CTA list head-TP uses, with both KV heads local.
This file does not modify the head-TP operator. The returned LSE is the
segment's already-combined LSE, normalized to ``[16, B]``.
"""

from __future__ import annotations

import torch

from merge_cp import normalize_lse_to_head_major

Q = 16
SCALE = 256 ** -0.5


def _ops():
    import vllm_flash_attn._vllm_fa2_C  # noqa: F401

    return torch.ops._vllm_fa2_C


def empty_partial(q: torch.Tensor):
    """O = 0, LSE = -inf. Used when a segment has no tokens."""
    batch = q.shape[0]
    out = torch.zeros_like(q)
    lse = torch.full((Q, batch), float("-inf"), dtype=torch.float32, device=q.device)
    return out, lse


def paged_fwd(q, k, v, block_table, seqused, cu_q, cu_k, out, max_seqlen_k,
              num_splits=0, tile_plan=None):
    """FA2 and its intra-rank combine with caller-owned buffers."""
    if tile_plan is None:
        tiles, row_splits, max_splits = None, None, 0
    else:
        tiles, row_splits, max_splits = tile_plan
    # Head exchange stays off. The last three arguments are the optional CTA list.
    return _ops().varlen_fwd(
        q, k, v, out, cu_q, cu_k, seqused, None, block_table, None,
        1, int(max_seqlen_k), 0.0, SCALE, True, False, -1, -1, 0.0, False,
        int(num_splits), None,
        None, 0, 0, -1, *([None] * 16), 0, 0, 0, tiles, row_splits, int(max_splits),
    )


def decode_offsets(batch, device):
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    return cu_q, torch.zeros_like(cu_q)


def attn_local(q, k, v, block_table, seqused, num_splits: int = 0,
               max_seqlen_k: int | None = None, tile_plan=None, offsets=None,
               may_have_empty_rows=True):
    """Run one decode step.

    ``q`` is ``[B, 16, 256]`` bf16. Returns ``(out, lse)`` with ``out`` the
    same shape and ``lse`` float32 ``[16, B]``. ``num_splits < 1`` asks FA2
    for its occupancy heuristic, the same switch as the head-TP baseline.
    Both the single-split host transpose and the split-KV combine write that
    ``[16, B]`` order (head ``h`` is KV head ``h // 8``, group ``h % 8``).
    ``max_seqlen_k`` should be passed from the CPU length list inside a timed
    region so this function does not sync on ``seqused``.
    """
    batch = q.shape[0]
    if batch == 0:
        return q, torch.empty(Q, 0, dtype=torch.float32, device=q.device)
    if max_seqlen_k is None:
        max_seqlen_k = int(seqused.max().item())
    if max_seqlen_k <= 0:
        return empty_partial(q)
    if q.is_cuda:
        torch.cuda.set_device(q.device)
    device = q.device
    cu_q, cu_k = offsets if offsets is not None else decode_offsets(batch, device)
    out = torch.empty_like(q)
    ret = paged_fwd(q, k, v, block_table, seqused, cu_q, cu_k, out, max_seqlen_k,
                    num_splits, tile_plan)
    if not isinstance(ret, (tuple, list)) or len(ret) < 2 or ret[1] is None:
        raise RuntimeError(f"varlen_fwd did not return an LSE (got {type(ret)})")
    lse = normalize_lse_to_head_major(ret[1], Q, batch)
    if tuple(lse.shape) != (Q, batch):
        raise RuntimeError(f"normalized LSE shape {tuple(lse.shape)} != {(Q, batch)}")
    # A row with seqused 0 must not contribute, even when another row sets max_seqlen_k.
    if may_have_empty_rows:
        empty = seqused.view(batch) == 0
        out.masked_fill_(empty.view(batch, 1, 1), 0)
        lse = lse.masked_fill(empty.view(1, batch), float("-inf"))
    return out, lse
