# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request head collaboration for a single decode step.

Static tensor parallel stays 1. These objects describe which KV-head groups
of one request move to other data-parallel ranks for Full Attention only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

HeadTPState = Literal["pending_load", "active", "local"]
HeadTPKVSource = Literal["cpu_pool", "gpu_copy", "none"]


@dataclass
class HeadTPShard:
    """One rank's Q and KV head interval. ``end`` is exclusive."""

    rank: int
    q_range: tuple[int, int]
    kv_range: tuple[int, int]


@dataclass
class HeadTPPlan:
    """Head split for one request. ``shards[0]`` is always the owner."""

    req_id: str
    owner_rank: int
    degree: int
    shards: list[HeadTPShard]
    state: HeadTPState
    kv_source: HeadTPKVSource
    visible_len: int
    predicted_remaining: int

    def __post_init__(self) -> None:
        if len(self.shards) != self.degree:
            raise ValueError(
                f"{self.req_id}: degree {self.degree} but {len(self.shards)} shards"
            )
        if self.shards[0].rank != self.owner_rank:
            raise ValueError(
                f"{self.req_id}: owner rank {self.owner_rank} does not match "
                f"shards[0] rank {self.shards[0].rank}"
            )


@dataclass
class HeadTPStep:
    """Plans visible to every DP rank for this step.

    ``active`` is true when any rank has a plan in ``active``. Every rank must
    see the same value, because Full Attention then does a point-to-point
    exchange before the expert-parallel collective.
    """

    plans: dict[str, HeadTPPlan] = field(default_factory=dict)
    active: bool = False


@dataclass
class HeadTPPartialCache:
    """KV pages a helper holds for one request, outside the prefix hash table."""

    cache: torch.Tensor
    block_table: torch.Tensor
    seq_len: int
    slot_mapping: torch.Tensor


@dataclass
class HeadTPBatch:
    """This rank's view of the current batch, in flattened token order."""

    rank: int
    dp_size: int
    req_ids: list[str]
    query_start_loc: list[int]
    local_rows: torch.Tensor
    remote_req_indexes: dict[str, int]
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    block_size: int
    partial_caches: dict[str, HeadTPPartialCache] = field(default_factory=dict)


@dataclass
class HeadTPForwardState:
    """What ``execute_model`` passes into the forward context."""

    force_eager: bool = False
    step: HeadTPStep | None = None
    local_attn_metadata: Any = None
    local_slot_mapping: dict[str, torch.Tensor] | None = None
    batch: HeadTPBatch | None = None


_PARTIAL_CACHES: dict[str, HeadTPPartialCache] = {}


def register_head_tp_partial_cache(req_id: str, cache: HeadTPPartialCache) -> None:
    """Record helper KV for ``req_id``. The prefix stage fills this."""

    _PARTIAL_CACHES[req_id] = cache


def drop_head_tp_partial_cache(req_id: str) -> None:
    _PARTIAL_CACHES.pop(req_id, None)


def head_tp_partial_caches() -> dict[str, HeadTPPartialCache]:
    return _PARTIAL_CACHES


def head_tp_is_active(step: HeadTPStep | None) -> bool:
    return step is not None and step.active


def make_shards(
    num_q_heads: int,
    num_kv_heads: int,
    degree: int,
    owner_rank: int,
    dp_size: int,
) -> list[HeadTPShard]:
    """Split Q/KV heads into ``degree`` contiguous groups.

    The owner keeps group 0, so the KV heads it wrote during prefill stay
    put. Later groups go to the next data-parallel ranks, wrapping around.
    """

    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Q heads {num_q_heads} must be a positive multiple of "
            f"KV heads {num_kv_heads}"
        )
    if (
        degree < 2
        or degree > num_kv_heads
        or num_kv_heads % degree != 0
        or degree > dp_size
    ):
        raise ValueError(
            f"degree {degree} is not legal for {num_kv_heads} KV heads "
            f"and data_parallel_size {dp_size}"
        )
    kv_per = num_kv_heads // degree
    q_per = kv_per * (num_q_heads // num_kv_heads)
    shards: list[HeadTPShard] = []
    for group in range(degree):
        shards.append(
            HeadTPShard(
                rank=(owner_rank + group) % dp_size,
                q_range=(group * q_per, (group + 1) * q_per),
                kv_range=(group * kv_per, (group + 1) * kv_per),
            )
        )
    return shards


def make_plan(
    req_id: str,
    num_q_heads: int,
    num_kv_heads: int,
    degree: int,
    owner_rank: int,
    dp_size: int,
    *,
    state: HeadTPState,
    kv_source: HeadTPKVSource,
    visible_len: int,
    predicted_remaining: int,
) -> HeadTPPlan:
    shards = make_shards(num_q_heads, num_kv_heads, degree, owner_rank, dp_size)
    return HeadTPPlan(
        req_id=req_id,
        owner_rank=owner_rank,
        degree=degree,
        shards=shards,
        state=state,
        kv_source=kv_source,
        visible_len=visible_len,
        predicted_remaining=predicted_remaining,
    )


def exchange_schedule(
    step: HeadTPStep, dp_size: int
) -> list[tuple[str, int, int, str, int]]:
    """Deadlock-free send order for active plans.

    Each entry is ``(phase, owner_rank, helper_rank, req_id, shard_index)``.
    ``phase`` is ``"fwd"`` when the owner sends Q/K/V, then ``"bwd"`` when
    the helper sends the partial attention output. One pair communicates
    per entry, in owner-rank order, so blocking sends cannot cross.
    """

    by_owner: dict[int, list[HeadTPPlan]] = {}
    for req_id in sorted(step.plans):
        plan = step.plans[req_id]
        if plan.state != "active":
            continue
        by_owner.setdefault(plan.owner_rank, []).append(plan)
    schedule: list[tuple[str, int, int, str, int]] = []
    for phase in ("fwd", "bwd"):
        for owner in range(dp_size):
            for plan in by_owner.get(owner, []):
                for shard_index, shard in enumerate(plan.shards):
                    if shard_index == 0:
                        continue
                    schedule.append(
                        (phase, owner, shard.rank, plan.req_id, shard_index)
                    )
    return schedule


def split_batch_rows(
    step: HeadTPStep,
    rank: int,
    req_ids: list[str],
    query_start_loc: list[int],
) -> tuple[list[int], list[int], dict[str, int]]:
    """Classify this rank's batch rows.

    Returns local request indexes, flattened local token rows, and
    ``req_id -> request index`` for rows this rank owns and splits.
    """

    local_reqs: list[int] = []
    local_rows: list[int] = []
    remote_reqs: dict[str, int] = {}
    for req_index, req_id in enumerate(req_ids):
        plan = step.plans.get(req_id)
        start = query_start_loc[req_index]
        end = query_start_loc[req_index + 1]
        if (
            plan is not None
            and plan.state == "active"
            and plan.owner_rank == rank
        ):
            remote_reqs[req_id] = req_index
            continue
        local_reqs.append(req_index)
        local_rows.extend(range(start, end))
    return local_reqs, local_rows, remote_reqs


def request_token_rows(query_start_loc: list[int], req_index: int) -> list[int]:
    return list(range(query_start_loc[req_index], query_start_loc[req_index + 1]))


def _compact_query_start(
    req_indexes: list[int],
    query_start_loc: list[int],
    like: torch.Tensor,
) -> torch.Tensor:
    values = [0]
    for req_index in req_indexes:
        width = query_start_loc[req_index + 1] - query_start_loc[req_index]
        values.append(values[-1] + width)
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def _index_select_if(
    value: torch.Tensor | None,
    indexes: torch.Tensor,
) -> torch.Tensor | None:
    if value is None or value.ndim == 0:
        return value
    if indexes.numel() and value.shape[0] < int(indexes.max()) + 1:
        return value
    return value.index_select(0, indexes.to(device=value.device))


def slice_attention_metadata(
    metadata: Any,
    *,
    local_req_indexes: list[int],
    local_rows: list[int],
    query_start_loc: list[int],
) -> Any:
    """Copy one attention metadata object onto the local rows only.

    The copy drops the captured scheduler metadata so the eager kernel
    builds its own launch. The original object is left unchanged, including
    any persistent buffer it views.
    """

    if not (
        hasattr(metadata, "query_start_loc")
        and hasattr(metadata, "slot_mapping")
        and (
            hasattr(metadata, "block_table")
            or hasattr(metadata, "block_table_tensor")
        )
    ):
        return metadata
    if not local_req_indexes:
        return metadata

    cloned = copy.copy(metadata)
    req_index = torch.tensor(local_req_indexes, dtype=torch.long)
    row_index = torch.tensor(local_rows, dtype=torch.long)
    token_counts = [
        query_start_loc[i + 1] - query_start_loc[i] for i in local_req_indexes
    ]
    qsl = _compact_query_start(
        local_req_indexes, query_start_loc, metadata.query_start_loc
    )
    cloned.query_start_loc = qsl
    if hasattr(cloned, "query_start_loc_cpu"):
        cloned.query_start_loc_cpu = qsl.detach().to("cpu")
    if hasattr(cloned, "num_reqs"):
        cloned.num_reqs = len(local_req_indexes)
    if hasattr(cloned, "num_actual_tokens"):
        cloned.num_actual_tokens = len(local_rows)
    if hasattr(cloned, "max_query_len"):
        cloned.max_query_len = max(token_counts)
    for name in (
        "seq_lens",
        "block_table",
        "block_table_tensor",
        "rswa_prefix_lens",
        "dcp_context_kv_lens",
        "seq_lens_cpu_upper_bound",
    ):
        if hasattr(cloned, name):
            setattr(cloned, name, _index_select_if(getattr(cloned, name), req_index))
    seq_lens = getattr(cloned, "seq_lens", None)
    if hasattr(cloned, "max_seq_len") and seq_lens is not None and seq_lens.numel():
        cloned.max_seq_len = int(seq_lens.max().item())
    if hasattr(cloned, "slot_mapping"):
        cloned.slot_mapping = _index_select_if(cloned.slot_mapping, row_index)
    causal = getattr(cloned, "causal", None)
    if isinstance(causal, torch.Tensor):
        cloned.causal = _index_select_if(causal, req_index)
    if hasattr(cloned, "scheduler_metadata"):
        cloned.scheduler_metadata = None
    if hasattr(cloned, "max_num_splits"):
        cloned.max_num_splits = 0
    if hasattr(cloned, "sdpa_attn_masks"):
        cloned.sdpa_attn_masks = None
    mm_ranges = getattr(cloned, "mm_prefix_query_range_tensor", None)
    if isinstance(mm_ranges, torch.Tensor):
        cloned.mm_prefix_query_range_tensor = _index_select_if(mm_ranges, row_index)
    return cloned


def slice_attention_metadata_dict(
    attn_metadata: Any,
    *,
    local_req_indexes: list[int],
    local_rows: list[int],
    query_start_loc: list[int],
) -> Any:
    """Slice Full Attention metadata. Other layers keep the original object."""

    if not isinstance(attn_metadata, dict):
        return slice_attention_metadata(
            attn_metadata,
            local_req_indexes=local_req_indexes,
            local_rows=local_rows,
            query_start_loc=query_start_loc,
        )
    sliced: dict[str, Any] = {}
    seen: dict[int, Any] = {}
    for name, metadata in attn_metadata.items():
        key = id(metadata)
        if key not in seen:
            seen[key] = slice_attention_metadata(
                metadata,
                local_req_indexes=local_req_indexes,
                local_rows=local_rows,
                query_start_loc=query_start_loc,
            )
        sliced[name] = seen[key]
    return sliced


def slice_slot_mappings(
    slot_mapping: dict[str, torch.Tensor] | None,
    local_rows: list[int],
    num_tokens: int,
) -> dict[str, torch.Tensor] | None:
    if slot_mapping is None:
        return None
    if not local_rows:
        return {}
    row_index = torch.tensor(local_rows, dtype=torch.long)
    sliced: dict[str, torch.Tensor] = {}
    seen: dict[int, torch.Tensor] = {}
    for name, mapping in slot_mapping.items():
        key = mapping.data_ptr()
        if key not in seen:
            if mapping.ndim >= 1 and mapping.shape[0] >= num_tokens:
                seen[key] = mapping.index_select(0, row_index.to(mapping.device))
            else:
                seen[key] = mapping
        sliced[name] = seen[key]
    return sliced


def prepare_head_tp_batch(
    step: HeadTPStep | None,
    input_batch: InputBatch,
    *,
    rank: int,
    dp_size: int,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    attn_metadata: Any = None,
    slot_mapping_by_layer: dict[str, torch.Tensor] | None = None,
    partial_caches: dict[str, HeadTPPartialCache] | None = None,
) -> HeadTPForwardState:
    """Build the forward-context payload for an active head-collaboration step."""

    if not head_tp_is_active(step):
        return HeadTPForwardState()
    assert step is not None
    num_reqs = input_batch.num_reqs
    req_ids = list(input_batch.req_ids[:num_reqs])
    query_start_np = input_batch.query_start_loc_np[: num_reqs + 1]
    query_start = [int(x) for x in query_start_np]
    local_reqs, local_rows, remote_reqs = split_batch_rows(
        step, rank, req_ids, query_start
    )
    device = slot_mapping.device
    local_row_tensor = torch.tensor(local_rows, dtype=torch.long, device=device)
    seq_lens = input_batch.seq_lens[:num_reqs]
    num_tokens = query_start[-1] if query_start else 0
    local_metadata = None
    local_slots = None
    if local_reqs:
        local_metadata = slice_attention_metadata_dict(
            attn_metadata,
            local_req_indexes=local_reqs,
            local_rows=local_rows,
            query_start_loc=query_start,
        )
        local_slots = slice_slot_mappings(
            slot_mapping_by_layer, local_rows, num_tokens
        )
    batch = HeadTPBatch(
        rank=rank,
        dp_size=dp_size,
        req_ids=req_ids,
        query_start_loc=query_start,
        local_rows=local_row_tensor,
        remote_req_indexes=remote_reqs,
        slot_mapping=slot_mapping,
        block_table=block_table[:num_reqs],
        seq_lens=seq_lens,
        block_size=block_size,
        partial_caches=(
            dict(head_tp_partial_caches())
            if partial_caches is None
            else dict(partial_caches)
        ),
    )
    return HeadTPForwardState(
        force_eager=True,
        step=step,
        batch=batch,
        local_attn_metadata=local_metadata,
        local_slot_mapping=local_slots,
    )
