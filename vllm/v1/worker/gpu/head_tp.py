# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between the GPU model runner and head collaboration."""

from __future__ import annotations

from typing import Any

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.head_tp import HeadTPForwardState, head_tp_is_active, prepare_head_tp_batch
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
