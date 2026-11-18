# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.head_tp import (
    HeadTPLoadTracker,
    HeadTPPartialCache,
    HeadTPStep,
    head_tp_partial_caches,
    make_plan,
    register_head_tp_partial_cache,
)
from vllm.v1.head_tp_sched import (
    HeadTPProfile,
    LengthBucketEMA,
    MIN_COLLAB_TOKENS,
    RECHECK_INTERVAL,
    decode_gathered_plans,
    encode_local_plans,
    exchange_head_tp_step,
    legal_degrees,
    publish_head_tp_step,
    select_kv_source,
)
from vllm.v1.request import Request


def _request(req_id: str, prompt_len: int = 8, max_tokens: int = 128) -> Request:
    request = Request.from_engine_core_request(
        EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=list(range(prompt_len)),
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        block_hasher=None,
    )
    request.num_computed_tokens = prompt_len
    return request


def _profile() -> HeadTPProfile:
    return HeadTPProfile(
        full_attn_s={0: 10.0},
        attn_s={(0, 2): 4.0, (0, 4): 2.0},
        load_s={
            ("gpu_copy", 0, 2): 1.0,
            ("gpu_copy", 0, 4): 1.0,
            ("cpu_pool", 0, 2): 1.0,
        },
    )


def _publish(request: Request, **overrides):
    args = dict(
        rank=0,
        dp_size=2,
        num_q_heads=16,
        num_kv_heads=2,
        block_size=16,
        partner_queue_limit=8,
        profile=_profile(),
        lengths=LengthBucketEMA(),
        peer_free=[8, 8],
        peer_running=[0, 0],
        peer_attn_us=[0, 0],
        cpu_hit_tokens={request.request_id: None},
        has_gpu_blocks={request.request_id: True},
    )
    args.update(overrides)
    return publish_head_tp_step(
        {request.request_id: request},
        {request.request_id: 1},
        **args,
    )


def test_legal_degrees_skip_values_that_do_not_divide_kv_heads():
    assert legal_degrees(2, 4) == [2]
    assert legal_degrees(8, 4) == [4, 2]
    assert legal_degrees(8, 1) == []


def test_prefill_and_recheck_interval_do_not_pair():
    request = _request("prefill")
    request.num_computed_tokens = 3
    assert _publish(request) is None
    assert request.head_tp_plan is None
    assert request.head_tp_next_check_tokens == 0

    request.num_computed_tokens = request.num_prompt_tokens
    costly = HeadTPProfile(
        full_attn_s={0: 1.0},
        attn_s={(0, 2): 1.0},
        load_s={("gpu_copy", 0, 2): 100.0},
    )
    assert _publish(request, profile=costly) is None
    assert request.head_tp_next_check_tokens == (
        request.num_computed_tokens + RECHECK_INTERVAL
    )
    request.num_computed_tokens += 1
    assert _publish(request) is None
    assert request.head_tp_plan is None


def test_decode_pairs_at_degree_two_and_keeps_owner_on_shard_zero():
    request = _request("long")
    step = _publish(request)
    assert step is not None and step.active
    plan = request.head_tp_plan
    assert plan is not None
    assert plan.state == "pending_load"
    assert plan.kv_source == "gpu_copy"
    assert plan.degree == 2
    assert plan.visible_len == request.num_prompt_tokens
    assert [(shard.rank, shard.kv_range) for shard in plan.shards] == [
        (0, (0, 1)),
        (1, (1, 2)),
    ]
    assert request.predicted_remaining == request.max_tokens


def test_larger_legal_degree_wins_until_a_partner_is_short_on_blocks():
    request = _request("wide", prompt_len=32)
    step = _publish(
        request,
        dp_size=4,
        num_q_heads=32,
        num_kv_heads=8,
        peer_free=[8, 8, 8, 8],
        peer_running=[0, 0, 0, 0],
        peer_attn_us=[0, 0, 0, 0],
    )
    assert step is not None
    assert request.head_tp_plan is not None
    assert request.head_tp_plan.degree == 4

    request = _request("fallback", prompt_len=32)
    step = _publish(
        request,
        dp_size=4,
        num_q_heads=32,
        num_kv_heads=8,
        peer_free=[8, 8, 8, 0],
        peer_running=[0, 0, 0, 0],
        peer_attn_us=[0, 0, 0, 0],
    )
    assert step is not None
    assert request.head_tp_plan is not None
    assert request.head_tp_plan.degree == 2
    assert [shard.rank for shard in request.head_tp_plan.shards] == [0, 1]


def test_cpu_hit_beats_gpu_copy_and_a_partial_hit_does_not():
    assert select_kv_source(32, 32, True) == ("cpu_pool", 32)
    assert select_kv_source(31, 32, True) == ("gpu_copy", 0)
    assert select_kv_source(None, 32, True) == ("gpu_copy", 0)
    assert select_kv_source(None, 32, False) == ("none", 0)

    request = _request("cpu", prompt_len=32)
    step = _publish(request, cpu_hit_tokens={request.request_id: 32})
    assert step is not None
    assert request.head_tp_plan is not None
    assert request.head_tp_plan.kv_source == "cpu_pool"
    assert request.prefix_cpu_hit_tokens == 32

    missing = _request("nowhere")
    assert _publish(missing, has_gpu_blocks={missing.request_id: False}) is None
    assert missing.head_tp_plan is None


def test_pending_load_does_not_change_degree():
    request = _request("loading", prompt_len=16)
    request.head_tp_plan = make_plan(
        request.request_id,
        32,
        8,
        2,
        0,
        4,
        state="pending_load",
        kv_source="gpu_copy",
        visible_len=16,
        predicted_remaining=4,
    )
    _publish(
        request,
        dp_size=4,
        num_q_heads=32,
        num_kv_heads=8,
        peer_free=[8, 8, 8, 8],
        peer_running=[0, 0, 0, 0],
        peer_attn_us=[0, 0, 0, 0],
    )
    assert request.head_tp_plan.degree == 2
    assert request.head_tp_plan.state == "pending_load"


def test_active_plan_exits_only_after_64_tokens():
    request = _request("busy", max_tokens=8)
    request.append_output_token_ids([1] * 7)
    request.head_tp_collab_tokens = MIN_COLLAB_TOKENS - 1
    request.head_tp_plan = make_plan(
        request.request_id,
        16,
        2,
        2,
        0,
        2,
        state="active",
        kv_source="gpu_copy",
        visible_len=8,
        predicted_remaining=1,
    )
    _publish(request)
    assert request.head_tp_plan is not None
    assert request.head_tp_plan.degree == 2

    request.head_tp_collab_tokens = MIN_COLLAB_TOKENS
    _publish(request)
    assert request.head_tp_plan is None
    assert request.head_tp_next_check_tokens == (
        request.num_computed_tokens + RECHECK_INTERVAL
    )


def test_full_partner_queue_exits_when_the_remaining_work_does_not_pay():
    request = _request("queued")
    request.head_tp_collab_tokens = MIN_COLLAB_TOKENS
    request.head_tp_plan = make_plan(
        request.request_id,
        16,
        2,
        2,
        0,
        2,
        state="active",
        kv_source="gpu_copy",
        visible_len=8,
        predicted_remaining=10,
    )
    flat = HeadTPProfile()
    _publish(
        request,
        profile=flat,
        partner_queue_limit=1,
        peer_running=[0, 2],
    )
    assert request.head_tp_plan is None

    stayed = _request("worth-it")
    stayed.head_tp_collab_tokens = MIN_COLLAB_TOKENS
    stayed.head_tp_plan = make_plan(
        stayed.request_id,
        16,
        2,
        2,
        0,
        2,
        state="active",
        kv_source="gpu_copy",
        visible_len=8,
        predicted_remaining=10,
    )
    _publish(
        stayed,
        partner_queue_limit=1,
        peer_running=[0, 2],
    )
    assert stayed.head_tp_plan is not None
    assert stayed.head_tp_plan.degree == 2


def test_gathered_tables_merge_into_one_active_step():
    owner = _request("owner", prompt_len=16)
    step = _publish(owner, rank=1, peer_free=[8, 8])
    assert step is not None
    encoded, req_ids = encode_local_plans(
        step, ["owner"], rank=1, dp_size=2, max_plans=2
    )
    assert encoded[0] == 0
    assert req_ids[0] == "owner"
    assert req_ids[1] == ""
    empty = [-1] * len(encoded)
    blank = ["", ""]
    merged = decode_gathered_plans(
        [empty, encoded],
        [blank, req_ids],
        dp_size=2,
        num_q_heads=16,
        num_kv_heads=2,
    )
    assert merged is not None and merged.active
    plan = merged.plans["owner"]
    assert plan.owner_rank == 1
    assert [shard.rank for shard in plan.shards] == [1, 0]
    assert plan.shards[0].kv_range == (0, 1)
    assert plan.shards[1].kv_range == (1, 2)

    idle = decode_gathered_plans(
        [empty, empty],
        [blank, blank],
        dp_size=2,
        num_q_heads=16,
        num_kv_heads=2,
    )
    assert idle is None


def test_exchange_gathers_stats_before_plans():
    request = _request("synced")

    class _Wire:
        def __init__(self) -> None:
            self.order: list[str] = []

        def gather_stats(self, local: tuple[int, int, int]):
            self.order.append("stats")
            return [(4, 1, 0), (4, 0, 0)]

        def gather_plans(self, encoded: list[int], req_ids: list[str]):
            self.order.append("plans")
            empty = [-1] * len(encoded)
            blank = [""] * len(req_ids)
            return [encoded, empty], [req_ids, blank]

    wire = _Wire()
    step = exchange_head_tp_step(
        {request.request_id: request},
        {request.request_id: 1},
        rank=0,
        dp_size=2,
        num_q_heads=16,
        num_kv_heads=2,
        block_size=16,
        partner_queue_limit=8,
        profile=_profile(),
        lengths=LengthBucketEMA(),
        local_stats=(4, 1, 0),
        wire=wire,
        cpu_hit_tokens={request.request_id: None},
        has_gpu_blocks={request.request_id: True},
        max_plans=2,
    )
    assert wire.order == ["stats", "plans"]
    assert step is not None and step.active
    assert step.plans["synced"].state == "pending_load"


def test_length_bucket_starts_at_max_tokens_then_uses_the_mean():
    lengths = LengthBucketEMA(alpha=0.2)
    assert lengths.predict(256, 0, 40, 0) == 40
    done = _request("done", prompt_len=256, max_tokens=40)
    done.append_output_token_ids([1] * 10)
    lengths.observe(done)
    assert lengths.predict(256, 0, 40, 0) == 10
    done.append_output_token_ids([1] * 20)
    lengths.observe(done)
    assert lengths.means[(1, 0)] == 14
    assert lengths.predict(256, 0, 40, 9) == 5
    assert lengths.predict(256, 0, 12, 9) == 3
    assert lengths.predict(255, 0, 40, 0) == 40


def test_load_tracker_catches_up_then_releases_the_partial_cache():
    plan = make_plan(
        "move",
        16,
        2,
        2,
        0,
        2,
        state="pending_load",
        kv_source="gpu_copy",
        visible_len=10,
        predicted_remaining=4,
    )
    step = HeadTPStep({plan.req_id: plan}, active=True)
    tracker = HeadTPLoadTracker(0)
    caught, failed = tracker.observe(step, {"move": 10})
    assert caught == [] and failed == []
    assert tracker.pending_span("move") == (0, 10)

    tracker.mark_copy_done("move", 10)
    caught, failed = tracker.observe(step, {"move": 12})
    assert caught == [] and failed == []
    assert tracker.pending_span("move") == (10, 12)

    tracker.mark_copy_done("move", 12)
    caught, failed = tracker.observe(step, {"move": 12})
    assert caught == ["move"] and failed == []

    cache = HeadTPPartialCache(
        cache=torch.zeros(1),
        block_table=torch.zeros(1, dtype=torch.int32),
        seq_len=1,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
    )
    register_head_tp_partial_cache("move", cache)
    tracker.observe(step, {"move": 12})
    assert "move" in head_tp_partial_caches()
    tracker.observe(None, {})
    assert "move" not in head_tp_partial_caches()


def test_failed_copy_is_reported_and_the_cache_is_dropped():
    plan = make_plan(
        "fail",
        16,
        2,
        2,
        0,
        2,
        state="pending_load",
        kv_source="cpu_pool",
        visible_len=4,
        predicted_remaining=1,
    )
    step = HeadTPStep({plan.req_id: plan}, active=True)
    tracker = HeadTPLoadTracker(0)
    tracker.observe(step, {"fail": 4})
    register_head_tp_partial_cache(
        "fail",
        HeadTPPartialCache(
            cache=torch.zeros(1),
            block_table=torch.zeros(1, dtype=torch.int32),
            seq_len=1,
            slot_mapping=torch.zeros(1, dtype=torch.int64),
        ),
    )
    tracker.mark_copy_done("fail", 4, ok=False)
    caught, failed = tracker.observe(step, {"fail": 4})
    assert caught == [] and failed == ["fail"]
    assert "fail" not in head_tp_partial_caches()
