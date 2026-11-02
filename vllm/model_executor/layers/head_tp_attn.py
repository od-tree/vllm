# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager head-sharded decode for one Full Attention layer.

Remote rows do not enter ``Attention``. Their partial outputs are concatenated
on the owner before the gate and ``o_proj``. The first kernel is PyTorch SDPA;
it reads the KV cache layout the selected backend actually stores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.head_tp import (
    HeadTPBatch,
    HeadTPShard,
    HeadTPStep,
    exchange_schedule,
    request_token_rows,
)


class HeadTPTransport(Protocol):
    def send(self, dst: int, payload: dict[str, Any]) -> None: ...

    def recv(self, src: int) -> dict[str, Any]: ...


class DPHeadTPTransport:
    """Point-to-point over the existing data-parallel group."""

    def send(self, dst: int, payload: dict[str, Any]) -> None:
        get_dp_group().send_tensor_dict(payload, dst=dst)

    def recv(self, src: int) -> dict[str, Any]:
        received = get_dp_group().recv_tensor_dict(src=src)
        if received is None:
            raise RuntimeError(
                f"head collaboration recv from rank {src} returned nothing"
            )
        return received


def resolve_kv_layout(
    cache: torch.Tensor, num_kv_heads: int, head_dim: int
) -> tuple[str, int]:
    """Return ``(layout, block_size)`` for a stored KV cache.

    Both the FlashAttention and Triton forwards in this tree store
    ``[num_blocks, num_kv_heads, block_size, 2 * head_dim]``. The Triton
    kernel then transposes before writing. A token-major packed tensor is
    accepted when that is the shape actually passed in.
    """

    if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
        raise ValueError(
            "unsupported kv cache shape "
            f"{tuple(cache.shape)} for head_dim {head_dim}"
        )
    if cache.shape[1] == num_kv_heads:
        return "packed_head_major", int(cache.shape[2])
    if cache.shape[2] == num_kv_heads:
        return "packed_token_major", int(cache.shape[1])
    raise ValueError(
        "kv cache shape "
        f"{tuple(cache.shape)} does not place {num_kv_heads} heads on "
        "axis 1 or axis 2"
    )


def write_kv_range(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_range: tuple[int, int],
    *,
    num_kv_heads: int,
    head_dim: int,
    block_size: int | None = None,
) -> None:
    """Write one KV-head interval of the current tokens.

    ``k`` and ``v`` are ``[rows, kv_per, head_dim]`` and already cover
    ``kv_range`` only. Padding slots (``< 0``) are skipped.
    """

    if k.shape[0] == 0:
        return
    layout, inferred_block = resolve_kv_layout(cache, num_kv_heads, head_dim)
    block = inferred_block if block_size is None else block_size
    kv_start, kv_end = kv_range
    if k.shape[1] != kv_end - kv_start or v.shape[1] != kv_end - kv_start:
        raise ValueError(
            f"k/v heads {k.shape[1]}/{v.shape[1]} do not match range {kv_range}"
        )
    slots = slot_mapping.detach().to("cpu").tolist()
    for row, slot in enumerate(slots):
        if slot < 0:
            continue
        block_id = int(slot) // block
        offset = int(slot) % block
        if layout == "packed_head_major":
            cache[block_id, kv_start:kv_end, offset, :head_dim] = k[row]
            cache[block_id, kv_start:kv_end, offset, head_dim:] = v[row]
        else:
            cache[block_id, offset, kv_start:kv_end, :head_dim] = k[row]
            cache[block_id, offset, kv_start:kv_end, head_dim:] = v[row]


def _gather_heads(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    kv_range: tuple[int, int],
    *,
    layout: str,
    block_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_start, kv_end = kv_range
    kv_per = kv_end - kv_start
    if seq_len == 0:
        empty = cache.new_empty(0, kv_per, head_dim)
        return empty, empty
    positions = torch.arange(seq_len, device=block_table.device)
    block_ids = block_table[positions // block_size].to(torch.long)
    offsets = positions % block_size
    blocks = cache.index_select(0, block_ids.to(cache.device))
    index = offsets.to(cache.device).view(-1, 1, 1, 1).expand(
        seq_len, kv_per, 1, head_dim
    )
    if layout == "packed_head_major":
        k_all = blocks[:, kv_start:kv_end, :, :head_dim]
        v_all = blocks[:, kv_start:kv_end, :, head_dim:]
    else:
        k_all = blocks[:, :, kv_start:kv_end, :head_dim].permute(0, 2, 1, 3)
        v_all = blocks[:, :, kv_start:kv_end, head_dim:].permute(0, 2, 1, 3)
    k = k_all.gather(2, index).squeeze(2)
    v = v_all.gather(2, index).squeeze(2)
    return k, v


def _sdpa_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``q`` is ``[q_per, dim]``. ``k`` and ``v`` are ``[seq, kv_per, dim]``."""

    q_per = q.shape[0]
    kv_per = k.shape[1]
    if kv_per == 0 or q_per % kv_per != 0:
        raise ValueError(f"cannot group {q_per} Q heads onto {kv_per} KV heads")
    group = q_per // kv_per
    if group != 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    q_b = q.reshape(1, q_per, q.shape[-1]).unsqueeze(2)
    k_b = k.permute(1, 0, 2).unsqueeze(0)
    v_b = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=scale, is_causal=False)
    return out.squeeze(0).squeeze(1)


def eager_gqa_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_range: tuple[int, int],
    scale: float,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int | None = None,
) -> torch.Tensor:
    """SDPA over one KV-head interval.

    ``q`` is ``[rows, q_per, head_dim]``. ``seq_lens[i]`` is the number of
    valid cache positions visible to row ``i``, including the token just
    written. Returns ``[rows, q_per, head_dim]``.
    """

    del slot_mapping  # the new token is already in the cache at seq_lens
    layout, inferred_block = resolve_kv_layout(cache, num_kv_heads, head_dim)
    block = inferred_block if block_size is None else block_size
    outputs = []
    lengths = seq_lens.detach().to("cpu").tolist()
    for row, seq_len in enumerate(lengths):
        k, v = _gather_heads(
            cache,
            block_table,
            int(seq_len),
            kv_range,
            layout=layout,
            block_size=block,
            head_dim=head_dim,
        )
        outputs.append(_sdpa_gqa(q[row], k, v, scale))
    if not outputs:
        return q.new_empty(0, q.shape[1], head_dim)
    return torch.stack(outputs, dim=0)


def concat_q_heads(parts: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate shard outputs in KV-head order. ``[rows, heads, dim]``."""

    return torch.cat(parts, dim=1)


def exchange_head_outputs(
    step: HeadTPStep,
    rank: int,
    dp_size: int,
    outgoing: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    compute: Callable[[str, int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    transport: HeadTPTransport,
) -> dict[tuple[str, int], torch.Tensor]:
    """Trade partial Q/K/V for partial attention outputs.

    Owners send even when the row count is 0, so a helper blocked on that
    shard still completes and the later expert-parallel collective lines up.
    """

    pending: dict[tuple[str, int], torch.Tensor] = {}
    received: dict[tuple[str, int], torch.Tensor] = {}
    for phase, owner, helper, req_id, shard_index in exchange_schedule(step, dp_size):
        key = (req_id, shard_index)
        if phase == "fwd":
            if rank == owner:
                q, k, v = outgoing[key]
                transport.send(
                    helper,
                    {
                        "req_id": req_id,
                        "rows": q.shape[0],
                        "q": q,
                        "k": k,
                        "v": v,
                    },
                )
            elif rank == helper:
                message = transport.recv(owner)
                q = message["q"]
                if message["rows"] == 0:
                    pending[key] = q.new_empty(0, q.shape[1], q.shape[2])
                else:
                    pending[key] = compute(
                        req_id, shard_index, q, message["k"], message["v"]
                    )
        elif rank == helper:
            transport.send(
                owner,
                {
                    "req_id": req_id,
                    "rows": pending[key].shape[0],
                    "out": pending[key],
                },
            )
        elif rank == owner:
            message = transport.recv(helper)
            received[key] = message["out"]
    return received


def _slice_heads(
    tensor: torch.Tensor,
    rows: torch.Tensor,
    head_range: tuple[int, int],
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    selected = tensor.index_select(0, rows).view(-1, num_heads, head_dim)
    start, end = head_range
    return selected[:, start:end, :]


def _partial_for(batch: HeadTPBatch, req_id: str, layer_name: str):
    if layer_name:
        found = batch.partial_caches.get(f"{req_id}/{layer_name}")
        if found is not None:
            return found
    return batch.partial_caches.get(req_id)


def _helper_slots(
    partial: Any,
    num_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Slots for the new rows.

    A pre-sized mapping is the history length including those rows. A cache
    filled by the prefix load stores only the history, so the new rows start
    at ``seq_len``.
    """

    if partial.slot_mapping.shape[0] == num_rows:
        return partial.slot_mapping, partial.seq_len
    history = partial.seq_len
    slots = torch.arange(
        history, history + num_rows, dtype=torch.int64, device=device
    )
    return slots, history + num_rows


def _visible_seq_lens(
    seq_len: int, num_rows: int, device: torch.device
) -> torch.Tensor:
    """Length visible to each new token, ending at ``seq_len``."""

    if num_rows == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    start = seq_len - num_rows + 1
    return torch.arange(start, seq_len + 1, device=device, dtype=torch.int32)


def _empty_qkv(
    like: torch.Tensor, q_per: int, kv_per: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        like.new_empty(0, q_per, head_dim),
        like.new_empty(0, kv_per, head_dim),
        like.new_empty(0, kv_per, head_dim),
    )


def _owner_shard_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: list[int],
    shard: HeadTPShard,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    q_per = shard.q_range[1] - shard.q_range[0]
    kv_per = shard.kv_range[1] - shard.kv_range[0]
    if not rows:
        return q.new_empty(0, q_per, head_dim)
    row_index = torch.tensor(rows, dtype=torch.long, device=q.device)
    q_part = _slice_heads(q, row_index, shard.q_range, num_heads, head_dim)
    k_part = _slice_heads(k, row_index, shard.kv_range, num_kv_heads, head_dim)
    v_part = _slice_heads(v, row_index, shard.kv_range, num_kv_heads, head_dim)
    slots = slot_mapping.index_select(0, row_index)
    write_kv_range(
        cache,
        slots,
        k_part,
        v_part,
        shard.kv_range,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    return eager_gqa_decode(
        q_part,
        cache,
        slots,
        shard.kv_range,
        scale,
        block_table=block_table,
        seq_lens=_visible_seq_lens(seq_len, len(rows), q.device),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


def run_mixed_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    step: HeadTPStep,
    batch: HeadTPBatch,
    local_attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cache: torch.Tensor,
    local_metadata: Any,
    local_slot_mapping: dict[str, torch.Tensor] | None,
    transport: HeadTPTransport | None = None,
    layer_name: str = "",
) -> torch.Tensor:
    """Local rows use ``local_attn``. Owner rows keep shard 0 and collect the rest.

    ``q`` is ``[tokens, num_heads * head_dim]``. The result has that same
    shape, with heads in the original order, so the existing gate and
    ``o_proj`` can run unchanged.
    """

    if batch.rank < 0 or batch.dp_size < 1:
        raise ValueError("head collaboration batch is missing its DP rank")
    attn_output = q.new_zeros(q.shape[0], num_heads * head_dim)
    local_rows = batch.local_rows
    if local_rows.numel():
        if local_metadata is None or local_slot_mapping is None:
            raise RuntimeError(
                "local head-collaboration rows have no subset attention metadata"
            )
        q_local = q.index_select(0, local_rows)
        k_local = k.index_select(0, local_rows)
        v_local = v.index_select(0, local_rows)
        ctx = get_forward_context()
        local_ctx = replace(
            ctx,
            attn_metadata=local_metadata,
            slot_mapping=local_slot_mapping,
        )
        with override_forward_context(local_ctx):
            local_out = local_attn(q_local, k_local, v_local)
        attn_output.index_copy_(0, local_rows, local_out)

    shard0_out: dict[str, torch.Tensor] = {}
    outgoing: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for req_id, plan in step.plans.items():
        if plan.state != "active" or plan.owner_rank != batch.rank:
            continue
        rows = (
            request_token_rows(batch.query_start_loc, batch.remote_req_indexes[req_id])
            if req_id in batch.remote_req_indexes
            else []
        )
        shard0 = plan.shards[0]
        if rows:
            req_index = batch.remote_req_indexes[req_id]
            shard0_out[req_id] = _owner_shard_forward(
                q,
                k,
                v,
                rows,
                shard0,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=scale,
                cache=cache,
                slot_mapping=batch.slot_mapping,
                block_table=batch.block_table[req_index],
                seq_len=int(batch.seq_lens[req_index].item()),
                block_size=batch.block_size,
            )
        else:
            shard0_out[req_id] = q.new_empty(
                0, shard0.q_range[1] - shard0.q_range[0], head_dim
            )
        row_index = torch.tensor(rows, dtype=torch.long, device=q.device)
        for shard_index, shard in enumerate(plan.shards):
            if shard_index == 0:
                continue
            if rows:
                outgoing[(req_id, shard_index)] = (
                    _slice_heads(q, row_index, shard.q_range, num_heads, head_dim),
                    _slice_heads(k, row_index, shard.kv_range, num_kv_heads, head_dim),
                    _slice_heads(v, row_index, shard.kv_range, num_kv_heads, head_dim),
                )
            else:
                outgoing[(req_id, shard_index)] = _empty_qkv(
                    q,
                    shard.q_range[1] - shard.q_range[0],
                    shard.kv_range[1] - shard.kv_range[0],
                    head_dim,
                )

    def _helper_compute(
        req_id: str,
        shard_index: int,
        q_part: torch.Tensor,
        k_part: torch.Tensor,
        v_part: torch.Tensor,
    ) -> torch.Tensor:
        plan = step.plans[req_id]
        shard = plan.shards[shard_index]
        partial = _partial_for(batch, req_id, layer_name)
        if partial is None:
            raise RuntimeError(
                f"rank {batch.rank} has no KV cache for helper request {req_id}"
            )
        slots, visible_end = _helper_slots(partial, q_part.shape[0], q_part.device)
        write_kv_range(
            partial.cache,
            slots,
            k_part,
            v_part,
            shard.kv_range,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=batch.block_size,
        )
        return eager_gqa_decode(
            q_part,
            partial.cache,
            slots,
            shard.kv_range,
            scale,
            block_table=partial.block_table,
            seq_lens=_visible_seq_lens(visible_end, q_part.shape[0], q_part.device),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=batch.block_size,
        )

    link = transport if transport is not None else DPHeadTPTransport()
    received = exchange_head_outputs(
        step,
        batch.rank,
        batch.dp_size,
        outgoing,
        _helper_compute,
        link,
    )
    for req_id, plan in step.plans.items():
        if plan.state != "active" or plan.owner_rank != batch.rank:
            continue
        if req_id not in batch.remote_req_indexes:
            continue
        rows = request_token_rows(
            batch.query_start_loc, batch.remote_req_indexes[req_id]
        )
        if not rows:
            continue
        parts = [shard0_out[req_id]]
        for shard_index, _shard in enumerate(plan.shards):
            if shard_index == 0:
                continue
            parts.append(received[(req_id, shard_index)])
        merged = concat_q_heads(parts).reshape(len(rows), num_heads * head_dim)
        row_index = torch.tensor(rows, dtype=torch.long, device=q.device)
        attn_output.index_copy_(0, row_index, merged)
    return attn_output
