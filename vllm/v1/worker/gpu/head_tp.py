# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between the GPU model runner and head collaboration."""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.head_tp_attn import DPHeadTPTransport
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.head_tp import (
    HeadTPForwardState,
    HeadTPLoadTracker,
    HeadTPStep,
    head_tp_is_active,
    prepare_head_tp_batch,
)
from vllm.v1.head_tp_kv import exchange_prefix_loads
from vllm.v1.kv_cache_interface import is_full_attention_spec, iter_layer_specs
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.utils import AttentionGroup


def _full_attention_group(
    attn_groups: list[list[AttentionGroup]],
) -> tuple[int, AttentionGroup] | None:
    for group_index, groups in enumerate(attn_groups):
        for group in groups:
            if is_full_attention_spec(group.kv_cache_spec):
                return group_index, group
    return None


def prepare_head_tp_forward(
    scheduler_output: SchedulerOutput,
    input_batch: InputBatch,
    attn_metadata: Any,
    slot_mappings_by_layer: dict[str, torch.Tensor] | None,
    block_tables: tuple[torch.Tensor, ...] | None,
    attn_groups: list[list[AttentionGroup]],
    rank: int,
    dp_size: int,
) -> HeadTPForwardState:
    """Subset local-row metadata for an active step. Otherwise return inactive."""

    step = scheduler_output.head_tp_step
    if not head_tp_is_active(step):
        return HeadTPForwardState()
    found = _full_attention_group(attn_groups)
    if (
        found is None
        or block_tables is None
        or slot_mappings_by_layer is None
        or attn_metadata is None
    ):
        raise RuntimeError(
            "active head collaboration requires full-attention metadata"
        )
    group_index, group = found
    spec = next(iter_layer_specs(group.kv_cache_spec))
    block_size = group.kernel_block_size or spec.block_size
    return prepare_head_tp_batch(
        step,
        input_batch,
        rank=rank,
        dp_size=dp_size,
        slot_mapping=slot_mappings_by_layer[group.layer_names[0]],
        block_table=block_tables[group_index],
        block_size=block_size,
        attn_metadata=attn_metadata,
        slot_mapping_by_layer=slot_mappings_by_layer,
    )


def head_tp_load_tracker(runner: Any) -> HeadTPLoadTracker:
    tracker = getattr(runner, "_head_tp_loads", None)
    if tracker is None:
        tracker = HeadTPLoadTracker(runner.dp_rank)
        runner._head_tp_loads = tracker
    return tracker


def observe_head_tp_loads(
    runner: Any, scheduler_output: SchedulerOutput
) -> tuple[list[str], list[str]]:
    """Open or finish copy spans for this step's committed token lengths."""

    step = scheduler_output.head_tp_step
    owner_lens: dict[str, int] = {}
    if step is not None:
        req_index = runner.req_states.req_id_to_index
        computed = runner.req_states.num_computed_tokens_np
        for req_id, plan in step.plans.items():
            if plan.owner_rank != runner.dp_rank:
                continue
            index = req_index.get(req_id)
            if index is None:
                owner_lens[req_id] = plan.visible_len
            else:
                owner_lens[req_id] = int(computed[index])
    return head_tp_load_tracker(runner).observe(step, owner_lens)


def release_head_tp_loads(runner: Any, req_ids: set[str]) -> None:
    tracker = getattr(runner, "_head_tp_loads", None)
    if tracker is not None and req_ids:
        tracker.release(req_ids)


def register_head_tp_kv_source(
    runner: Any,
    source: str,
    caches: dict[str, torch.Tensor],
) -> None:
    """Register a block-indexed KV tensor for ``cpu_pool`` or ``gpu_copy``.

    The tensor uses the attention backend layout. This does not insert blocks
    into the prefix hash table.
    """

    stored = getattr(runner, "_head_tp_kv_sources", None)
    if stored is None:
        stored = {}
        runner._head_tp_kv_sources = stored
    stored[source] = caches


def run_prefix_loads(runner: Any, step: HeadTPStep | None) -> dict[str, int]:
    """Copy pending KV spans, then let the owner mark them done.

    Returns bytes written onto helpers, summed on the owner. Missing source
    tensors fail the span instead of writing a prefix-hash block.
    """

    if step is None or not step.plans:
        return {}
    if not any(plan.state == "pending_load" for plan in step.plans.values()):
        return {}
    found = _full_attention_kv(runner)
    if found is None:
        return {}
    layers, gpu_caches, num_kv_heads, head_dim, block_size = found
    sources: dict[str, dict[str, torch.Tensor]] = {
        "gpu_copy": gpu_caches,
    }
    _ensure_cpu_pool_source(runner)
    extra = getattr(runner, "_head_tp_kv_sources", None) or {}
    for name, caches in extra.items():
        sources[name] = caches
    block_ids = _owner_block_ids(runner, step, block_size)
    transport = _prefix_transport(runner)
    if transport is None:
        return {}
    return exchange_prefix_loads(
        step,
        runner.dp_rank,
        dp_size=runner.dp_size,
        tracker=head_tp_load_tracker(runner),
        transport=transport,
        sources=sources,
        block_ids=block_ids,
        layers=layers,
        backend="",
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


def _ensure_cpu_pool_source(runner: Any) -> None:
    extra = getattr(runner, "_head_tp_kv_sources", None) or {}
    if "cpu_pool" in extra:
        return
    caches = _connector_cpu_caches()
    if caches:
        register_head_tp_kv_source(runner, "cpu_pool", caches)


def _connector_cpu_caches() -> dict[str, torch.Tensor] | None:
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return None
        group = get_kv_transfer_group()
    except Exception:
        return None
    worker = getattr(group, "connector_worker", None)
    caches = getattr(worker, "head_tp_cpu_caches", None)
    if not caches:
        return None
    return caches


def _prefix_transport(runner: Any):
    injected = getattr(runner, "_head_tp_transport", None)
    if injected is not None:
        return injected
    try:
        from vllm.distributed.parallel_state import get_dp_group

        get_dp_group()
    except Exception:
        return None
    return DPHeadTPTransport()


def _full_attention_kv(
    runner: Any,
) -> tuple[list[str], dict[str, torch.Tensor], int, int, int] | None:
    found = _full_attention_group(runner.attn_groups)
    if found is None:
        return None
    _, group = found
    spec = next(iter_layer_specs(group.kv_cache_spec))
    ctx = runner.compilation_config.static_forward_context
    caches: dict[str, torch.Tensor] = {}
    for name in group.layer_names:
        layer = ctx[name] if isinstance(ctx, dict) else None
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.ndim == 0:
            return None
        caches[name] = cache
    if not caches:
        return None
    block_size = int(group.kernel_block_size or spec.block_size)
    return (
        list(group.layer_names),
        caches,
        int(spec.num_kv_heads),
        int(spec.head_size),
        block_size,
    )


def _owner_block_ids(
    runner: Any, step: HeadTPStep, block_size: int
) -> dict[str, list[int]]:
    override = getattr(runner, "_head_tp_block_ids", None) or {}
    ids: dict[str, list[int]] = {}
    for req_id, plan in step.plans.items():
        if plan.owner_rank != runner.dp_rank or plan.state != "pending_load":
            continue
        if plan.kv_source == "cpu_pool":
            cpu_ids = getattr(step, "cpu_block_ids", {}).get(req_id)
            if cpu_ids:
                ids[req_id] = [int(block) for block in cpu_ids]
            continue
        if req_id in override:
            ids[req_id] = [int(block) for block in override[req_id]]
            continue
        row = _block_row(runner, req_id)
        if row is None:
            continue
        need = 0
        span = head_tp_load_tracker(runner).pending_span(req_id)
        if span is not None and block_size > 0:
            need = cdiv(span[1], block_size)
        ids[req_id] = row if need == 0 else row[:need]
    return ids


def _block_row(runner: Any, req_id: str) -> list[int] | None:
    index = runner.req_states.req_id_to_index.get(req_id)
    tables = getattr(getattr(runner, "block_tables", None), "block_tables", None)
    if index is None or not tables:
        return None
    found = _full_attention_group(runner.attn_groups)
    if found is None:
        return None
    group_index, _ = found
    if group_index >= len(tables):
        return None
    row = tables[group_index].gpu[index].detach().to("cpu").tolist()
    return [int(block) for block in row]
