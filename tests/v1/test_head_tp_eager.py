# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from vllm import SamplingParams
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.layers.head_tp_attn import (
    concat_q_heads,
    eager_gqa_decode,
    exchange_head_outputs,
    resolve_kv_layout,
    run_mixed_head_attention,
    write_kv_range,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.head_tp import (
    HeadTPBatch,
    HeadTPPartialCache,
    HeadTPStep,
    exchange_schedule,
    head_tp_is_active,
    make_plan,
    make_shards,
    slice_attention_metadata,
    split_batch_rows,
)
from vllm.v1.request import Request


def test_qwen35_shards_keep_owner_on_the_first_kv_head():
    shards = make_shards(16, 2, 2, owner_rank=0, dp_size=2)
    assert [(s.rank, s.q_range, s.kv_range) for s in shards] == [
        (0, (0, 8), (0, 1)),
        (1, (8, 16), (1, 2)),
    ]


def test_degree_must_divide_kv_heads_and_fit_the_dp_group():
    with pytest.raises(ValueError):
        make_shards(16, 2, 3, owner_rank=0, dp_size=4)
    with pytest.raises(ValueError):
        make_shards(16, 2, 1, owner_rank=0, dp_size=2)
    with pytest.raises(ValueError):
        make_shards(16, 8, 4, owner_rank=0, dp_size=2)
    shards = make_shards(16, 8, 4, owner_rank=1, dp_size=4)
    assert [s.kv_range for s in shards] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert [s.q_range for s in shards] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert [s.rank for s in shards] == [1, 2, 3, 0]


def test_inactive_step_stays_on_local_attention():
    assert not head_tp_is_active(None)
    assert not head_tp_is_active(HeadTPStep())
    request = Request.from_engine_core_request(
        EngineCoreRequest(
            request_id="request-1",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        block_hasher=None,
    )
    assert request.head_tp_plan is None
    assert request.predicted_remaining is None
    assert request.prefix_cpu_hit_tokens == 0
    assert SchedulerOutput.make_empty().head_tp_step is None


def test_exchange_schedule_is_owner_ordered():
    step = HeadTPStep(
        plans={
            "b": make_plan(
                "b", 16, 2, 2, 1, 2,
                state="active",
                kv_source="none",
                visible_len=4,
                predicted_remaining=8,
            ),
            "a": make_plan(
                "a", 16, 2, 2, 0, 2,
                state="active",
                kv_source="none",
                visible_len=4,
                predicted_remaining=8,
            ),
        },
        active=True,
    )
    schedule = exchange_schedule(step, dp_size=2)
    fwd = [item for item in schedule if item[0] == "fwd"]
    bwd = [item for item in schedule if item[0] == "bwd"]
    assert fwd == [
        ("fwd", 0, 1, "a", 1),
        ("fwd", 1, 0, "b", 1),
    ]
    assert [item[0] for item in schedule].index("bwd") == len(fwd)
    assert [(item[1], item[2], item[3]) for item in bwd] == [
        (0, 1, "a"),
        (1, 0, "b"),
    ]


def test_mutual_exchange_returns_each_partial_to_its_owner():
    step = HeadTPStep(
        plans={
            "a": make_plan(
                "a", 16, 2, 2, 0, 2,
                state="active",
                kv_source="none",
                visible_len=1,
                predicted_remaining=1,
            ),
            "b": make_plan(
                "b", 16, 2, 2, 1, 2,
                state="active",
                kv_source="none",
                visible_len=1,
                predicted_remaining=1,
            ),
        },
        active=True,
    )
    qa = torch.ones(1, 2, 2)
    qb = torch.full((1, 2, 2), 3.0)
    outgoing = {
        0: {("a", 1): (qa, qa, qa)},
        1: {("b", 1): (qb, qb, qb)},
    }
    queues: dict[tuple[int, int], queue.Queue] = {
        (0, 1): queue.Queue(),
        (1, 0): queue.Queue(),
    }

    class _QueueTransport:
        def __init__(self, rank: int) -> None:
            self.rank = rank

        def send(self, dst: int, payload: dict) -> None:
            queues[(self.rank, dst)].put(payload)

        def recv(self, src: int) -> dict:
            return queues[(src, self.rank)].get(timeout=10)

    received: dict[int, dict] = {}
    errors: list[BaseException] = []

    def _run(rank: int) -> None:
        try:
            received[rank] = exchange_head_outputs(
                step,
                rank,
                2,
                outgoing[rank],
                lambda _req, _shard, q, _k, _v: q + 1,
                _QueueTransport(rank),
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert not any(thread.is_alive() for thread in threads)
    torch.testing.assert_close(received[0][("a", 1)], qa + 1)
    torch.testing.assert_close(received[1][("b", 1)], qb + 1)


@dataclass
class _AttnMeta:
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 4


def test_slicing_local_metadata_drops_the_remote_row():
    metadata = _AttnMeta(
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        slot_mapping=torch.tensor([10, 20, 30], dtype=torch.int32),
        block_table=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
        seq_lens=torch.tensor([6, 9, 4], dtype=torch.int32),
        num_actual_tokens=3,
        max_query_len=1,
        max_seq_len=9,
        scheduler_metadata=torch.tensor([1, 2, 3]),
    )
    cloned = slice_attention_metadata(
        metadata,
        local_req_indexes=[0, 2],
        local_rows=[0, 2],
        query_start_loc=[0, 1, 2, 3],
    )
    assert metadata.slot_mapping.tolist() == [10, 20, 30]
    assert metadata.scheduler_metadata is not None
    assert cloned.slot_mapping.tolist() == [10, 30]
    assert cloned.block_table.tolist() == [[1, 2], [5, 6]]
    assert cloned.seq_lens.tolist() == [6, 4]
    assert cloned.query_start_loc.tolist() == [0, 1, 2]
    assert cloned.num_actual_tokens == 2
    assert cloned.max_seq_len == 6
    assert cloned.scheduler_metadata is None
    assert cloned.max_num_splits == 0
    local_reqs, local_rows, remote = split_batch_rows(
        HeadTPStep(
            plans={
                "mid": make_plan(
                    "mid", 16, 2, 2, 0, 2,
                    state="active",
                    kv_source="none",
                    visible_len=6,
                    predicted_remaining=1,
                )
            },
            active=True,
        ),
        rank=0,
        req_ids=["left", "mid", "right"],
        query_start_loc=[0, 1, 2, 3],
    )
    assert local_reqs == [0, 2]
    assert local_rows == [0, 2]
    assert remote == {"mid": 1}


def _history_cache(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
) -> torch.Tensor:
    cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, 2 * head_dim, dtype=torch.float32
    )
    cache.normal_()
    return cache


def _fill_request(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
    head_dim: int,
) -> None:
    """Put history in positions ``[0, seq_len)`` and clear the last slot."""

    layout, _ = resolve_kv_layout(cache, cache.shape[1], head_dim)
    assert layout == "packed_head_major"
    for pos in range(seq_len - 1):
        block_id = int(block_table[pos // block_size])
        offset = pos % block_size
        cache[block_id, :, offset, :] = torch.randn_like(cache[block_id, :, offset, :])
    block_id = int(block_table[(seq_len - 1) // block_size])
    offset = (seq_len - 1) % block_size
    cache[block_id, :, offset, :].zero_()


def _slot(block_table: torch.Tensor, seq_len: int, block_size: int) -> torch.Tensor:
    pos = seq_len - 1
    block_id = int(block_table[pos // block_size])
    return torch.tensor([block_id * block_size + pos % block_size], dtype=torch.int64)


def _decode_all_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    slots = _slot(block_table, seq_len, block_size)
    write_kv_range(
        cache,
        slots,
        k.view(1, num_kv_heads, head_dim),
        v.view(1, num_kv_heads, head_dim),
        (0, num_kv_heads),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    out = eager_gqa_decode(
        q.view(1, -1, head_dim),
        cache,
        slots,
        (0, num_kv_heads),
        scale,
        block_table=block_table,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    return out.reshape(1, -1)


def test_write_kv_range_leaves_the_other_heads_in_place():
    head_dim = 4
    cache = _history_cache(1, 2, 4, head_dim)
    before = cache[:, 1].clone()
    k = torch.randn(1, 1, head_dim)
    v = torch.randn(1, 1, head_dim)
    write_kv_range(
        cache,
        torch.tensor([1]),
        k,
        v,
        (0, 1),
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=4,
    )
    torch.testing.assert_close(cache[:, 1], before)
    torch.testing.assert_close(cache[0, 0, 1, :head_dim], k[0, 0])
    torch.testing.assert_close(cache[0, 0, 1, head_dim:], v[0, 0])


def test_sharded_decode_matches_full_heads_for_degree_four():
    num_q, num_kv, head_dim, degree = 16, 8, 4, 4
    block_size, seq_len = 4, 6
    scale = head_dim**-0.5
    shards = make_shards(num_q, num_kv, degree, owner_rank=0, dp_size=4)
    base = _history_cache(2, num_kv, block_size, head_dim)
    block_table = torch.tensor([0, 1], dtype=torch.int32)
    _fill_request(base, block_table, seq_len, block_size, head_dim)
    q = torch.randn(num_q * head_dim)
    k = torch.randn(num_kv * head_dim)
    v = torch.randn(num_kv * head_dim)
    reference = _decode_all_heads(
        q, k, v, base.clone(), block_table, seq_len,
        num_kv_heads=num_kv, head_dim=head_dim, scale=scale, block_size=block_size,
    )
    parts = []
    for shard in shards:
        cache = base.clone()
        slots = _slot(block_table, seq_len, block_size)
        q_part = q.view(num_q, head_dim)[shard.q_range[0] : shard.q_range[1]]
        k_part = k.view(num_kv, head_dim)[shard.kv_range[0] : shard.kv_range[1]]
        v_part = v.view(num_kv, head_dim)[shard.kv_range[0] : shard.kv_range[1]]
        write_kv_range(
            cache, slots, k_part.unsqueeze(0), v_part.unsqueeze(0), shard.kv_range,
            num_kv_heads=num_kv, head_dim=head_dim, block_size=block_size,
        )
        parts.append(
            eager_gqa_decode(
                q_part.unsqueeze(0), cache, slots, shard.kv_range, scale,
                block_table=block_table,
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                num_kv_heads=num_kv, head_dim=head_dim, block_size=block_size,
            )
        )
    sharded = concat_q_heads(parts).reshape(1, -1)
    torch.testing.assert_close(sharded, reference, atol=1e-5, rtol=1e-5)


class _InlineHelper:
    def __init__(self, compute) -> None:
        self.compute = compute
        self.out: torch.Tensor | None = None

    def send(self, dst: int, payload: dict) -> None:
        del dst
        q = payload["q"]
        if payload["rows"] == 0:
            self.out = q.new_empty(0, q.shape[1], q.shape[2])
            return
        self.out = self.compute(q, payload["k"], payload["v"])

    def recv(self, src: int) -> dict:
        del src
        assert self.out is not None
        return {"req_id": "mid", "rows": self.out.shape[0], "out": self.out}


def test_mixed_batch_matches_full_attention_after_gate_and_output_proj():
    """One local row, one owner row, one local row. Helper sees only its Q heads."""

    num_q, num_kv, head_dim = 16, 2, 8
    block_size, seq_len, hidden = 4, 6, 8
    scale = head_dim**-0.5
    torch.manual_seed(0)
    base = _history_cache(6, num_kv, block_size, head_dim)
    block_tables = torch.tensor(
        [[0, 1], [2, 3], [4, 5]], dtype=torch.int32
    )
    for row in range(3):
        _fill_request(base, block_tables[row], seq_len, block_size, head_dim)
    q = torch.randn(3, num_q * head_dim)
    k = torch.randn(3, num_kv * head_dim)
    v = torch.randn(3, num_kv * head_dim)
    gate = torch.randn(3, num_q * head_dim)
    weight = torch.randn(hidden, num_q * head_dim)

    reference_cache = base.clone()
    reference_rows = [
        _decode_all_heads(
            q[i], k[i], v[i], reference_cache, block_tables[i], seq_len,
            num_kv_heads=num_kv, head_dim=head_dim, scale=scale, block_size=block_size,
        )
        for i in range(3)
    ]
    reference = torch.cat(reference_rows, dim=0)
    reference_hidden = F.linear(reference * torch.sigmoid(gate), weight)

    owner_cache = base.clone()
    helper_cache = base.clone()
    plan = make_plan(
        "mid", num_q, num_kv, 2, 0, 2,
        state="active", kv_source="none", visible_len=seq_len, predicted_remaining=4,
    )
    step = HeadTPStep(plans={"mid": plan}, active=True)
    helper_slot = _slot(block_tables[1], seq_len, block_size)

    def _helper(q_part: torch.Tensor, k_part: torch.Tensor, v_part: torch.Tensor):
        assert q_part.shape == (1, num_q // 2, head_dim)
        assert k_part.shape[1] == 1
        write_kv_range(
            helper_cache, helper_slot, k_part, v_part, plan.shards[1].kv_range,
            num_kv_heads=num_kv, head_dim=head_dim, block_size=block_size,
        )
        return eager_gqa_decode(
            q_part, helper_cache, helper_slot, plan.shards[1].kv_range, scale,
            block_table=block_tables[1],
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            num_kv_heads=num_kv, head_dim=head_dim, block_size=block_size,
        )

    local_slots = torch.tensor(
        [
            int(_slot(block_tables[0], seq_len, block_size)),
            int(_slot(block_tables[2], seq_len, block_size)),
        ]
    )
    local_meta = _AttnMeta(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        slot_mapping=local_slots,
        block_table=block_tables[[0, 2]],
        seq_lens=torch.tensor([seq_len, seq_len], dtype=torch.int32),
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=seq_len,
    )

    def _local_attn(q_local, k_local, v_local):
        ctx = get_forward_context()
        assert ctx.attn_metadata["attn"].num_actual_tokens == q_local.shape[0]
        assert ctx.slot_mapping["attn"].shape[0] == 2
        outs = []
        for row, req_index in enumerate((0, 2)):
            outs.append(
                _decode_all_heads(
                    q_local[row], k_local[row], v_local[row], owner_cache,
                    block_tables[req_index], seq_len,
                    num_kv_heads=num_kv, head_dim=head_dim, scale=scale,
                    block_size=block_size,
                )
            )
        return torch.cat(outs, dim=0)

    batch = HeadTPBatch(
        rank=0,
        dp_size=2,
        req_ids=["left", "mid", "right"],
        query_start_loc=[0, 1, 2, 3],
        local_rows=torch.tensor([0, 2], dtype=torch.long),
        remote_req_indexes={"mid": 1},
        slot_mapping=torch.tensor(
            [int(_slot(block_tables[i], seq_len, block_size)) for i in range(3)]
        ),
        block_table=block_tables,
        seq_lens=torch.tensor([seq_len, seq_len, seq_len], dtype=torch.int32),
        block_size=block_size,
        partial_caches={
            "mid": HeadTPPartialCache(
                cache=helper_cache,
                block_table=block_tables[1],
                seq_len=seq_len,
                slot_mapping=helper_slot,
            )
        },
    )
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={"attn": local_meta},
        slot_mapping={"attn": local_slots},
        head_tp_step=step,
        head_tp_batch=batch,
        head_tp_local_attn_metadata={"attn": local_meta},
        head_tp_local_slot_mapping={"attn": local_slots},
    )
    with override_forward_context(ctx):
        mixed = run_mixed_head_attention(
            q, k, v,
            num_heads=num_q,
            num_kv_heads=num_kv,
            head_dim=head_dim,
            scale=scale,
            step=step,
            batch=batch,
            local_attn=_local_attn,
            cache=owner_cache,
            local_metadata={"attn": local_meta},
            local_slot_mapping={"attn": local_slots},
            transport=_InlineHelper(_helper),
        )
    mixed_hidden = F.linear(mixed * torch.sigmoid(gate), weight)
    torch.testing.assert_close(mixed_hidden, reference_hidden, atol=1e-5, rtol=1e-5)
    assert mixed_hidden.shape[0] == 3
