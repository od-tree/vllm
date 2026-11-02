# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading

import torch

from vllm import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.head_tp import (
    HeadTPLoadTracker,
    HeadTPStep,
    drop_head_tp_partial_cache,
    head_tp_partial_caches,
    make_plan,
)
from vllm.v1.head_tp_kv import (
    CPUPrefixLayer,
    exchange_prefix_loads,
    load_kv_range,
)
from vllm.v1.head_tp_sched import select_kv_source
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request


def _request(req_id: str) -> Request:
    request = Request.from_engine_core_request(
        EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=list(range(8)),
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        block_hasher=None,
    )
    request.num_computed_tokens = 8
    return request


def _cache(blocks: int, layout: str) -> torch.Tensor:
    # head-major: [blocks, heads, block, 2 * dim]
    # token-major: [blocks, block, heads, 2 * dim]
    if layout == "flash":
        cache = torch.zeros(blocks, 2, 4, 8)
        for block in range(blocks):
            for head in range(2):
                for token in range(4):
                    cache[block, head, token] = head * 100 + block * 10 + token
    else:
        cache = torch.zeros(blocks, 4, 2, 8)
        for block in range(blocks):
            for head in range(2):
                for token in range(4):
                    cache[block, token, head] = head * 100 + block * 10 + token
    return cache


def test_load_kv_range_copies_only_partner_heads_on_full_attention_layers():
    prefix_hash: dict[bytes, int] = {}
    byte_counts = []
    for backend in ("flash", "triton"):
        source_full = _cache(2, backend)
        source_gdn = source_full.clone()
        dest_full = torch.zeros_like(source_full)
        dest_before_gdn = source_gdn.clone()
        written = load_kv_range(
            {"full.0": source_full, "gdn.0": source_gdn},
            {"full.0": dest_full},
            [0, 1],
            (1, 2),
            ["full.0"],
            backend,
            num_kv_heads=2,
            head_dim=4,
            token_start=2,
            token_end=6,
            block_size=4,
        )
        byte_counts.append(written)
        assert torch.equal(source_gdn, dest_before_gdn)
        if backend == "flash":
            assert torch.equal(dest_full[:, 0], torch.zeros_like(dest_full[:, 0]))
            assert torch.equal(dest_full[0, 1, 2], source_full[0, 1, 2])
            assert torch.equal(dest_full[1, 1, 1], source_full[1, 1, 1])
            assert torch.equal(dest_full[0, 1, 0], torch.zeros(8))
        else:
            assert torch.equal(dest_full[:, :, 0], torch.zeros_like(dest_full[:, :, 0]))
            assert torch.equal(dest_full[0, 2, 1], source_full[0, 2, 1])
            assert torch.equal(dest_full[1, 1, 1], source_full[1, 1, 1])
            assert torch.equal(dest_full[0, 0, 1], torch.zeros(8))
        assert prefix_hash == {}
    assert byte_counts[0] == byte_counts[1] == 4 * 1 * 8 * 4


def test_cpu_prefix_hit_pins_only_a_replicated_ready_prefix():
    manager = CPUOffloadingManager(num_chunks=4)
    ready = CPUPrefixLayer(manager, block_size=16, replicated=True)
    closed = CPUPrefixLayer(manager, block_size=16, replicated=False)
    request = _request("cpu")
    request.block_hashes = [BlockHash(b"block-0"), BlockHash(b"block-1")]
    keys = [make_offload_key(bytes(item), 0) for item in request.block_hashes]
    ctx = ReqContext(req_id="seed")
    stored = manager.prepare_store(keys, ctx)
    assert stored is not None
    manager.complete_store(stored.keys_to_store[:1], ctx)

    assert ready.lookup_hit_tokens(request) == 16
    assert select_kv_source(16, 32, True) == ("gpu_copy", 0)
    manager.complete_store(stored.keys_to_store[1:], ctx)
    assert ready.lookup_hit_tokens(request) == 32
    assert select_kv_source(32, 32, True) == ("cpu_pool", 32)
    assert closed.lookup_hit_tokens(request) is None

    ready.pin(request)
    assert manager._policy.get(keys[0]).ref_cnt == 1
    ready.unpin(request.request_id)
    assert manager._policy.get(keys[0]).ref_cnt == 0
    assert request.request_id not in ready.pinned_request_ids()
    assert manager.lookup(keys[0], ctx).name == "HIT"


class _Link:
    def __init__(self, rank: int, queues: dict[tuple[int, int], object]) -> None:
        self.rank = rank
        self.queues = queues

    def send(self, dst: int, payload: dict) -> None:
        self.queues[(self.rank, dst)].put(payload)

    def recv(self, src: int) -> dict:
        return self.queues[(src, self.rank)].get(timeout=5)


def _exchange(kv_source: str) -> None:
    for key in list(head_tp_partial_caches()):
        drop_head_tp_partial_cache(key)
    plan = make_plan(
        "move",
        16,
        2,
        2,
        0,
        2,
        state="pending_load",
        kv_source=kv_source,  # type: ignore[arg-type]
        visible_len=6,
        predicted_remaining=4,
    )
    step = HeadTPStep({plan.req_id: plan}, active=True)
    source = {"full.0": _cache(2, "flash")}
    tracker = HeadTPLoadTracker(0)
    tracker.observe(step, {"move": 6})
    queues: dict[tuple[int, int], queue.Queue] = {
        (0, 1): queue.Queue(),
        (1, 0): queue.Queue(),
    }
    errors: list[BaseException] = []
    copied: dict[str, int] = {}

    def _run(rank: int, own_tracker: HeadTPLoadTracker) -> None:
        try:
            result = exchange_prefix_loads(
                step,
                rank,
                dp_size=2,
                tracker=own_tracker,
                transport=_Link(rank, queues),
                sources=(
                    {"gpu_copy": source}
                    if rank == 0 and kv_source == "gpu_copy"
                    else {"cpu_pool": source}
                    if rank == 1 and kv_source == "cpu_pool"
                    else {}
                ),
                block_ids={"move": [0, 1]} if rank == 0 else {},
                layers=["full.0"],
                backend="flash",
                num_kv_heads=2,
                head_dim=4,
                block_size=4,
            )
            if rank == 0:
                copied.update(result)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    helper = HeadTPLoadTracker(1)
    threads = [
        threading.Thread(target=_run, args=(0, tracker)),
        threading.Thread(target=_run, args=(1, helper)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert copied == {"move": 6 * 1 * 8 * 4}
    partial = head_tp_partial_caches()["move/full.0"]
    assert partial.seq_len == 6
    assert torch.equal(partial.cache[0, 1, 2], source["full.0"][0, 1, 2])
    assert torch.equal(partial.cache[:, 0], torch.zeros_like(partial.cache[:, 0]))
    caught, failed = tracker.observe(step, {"move": 6})
    assert caught == ["move"] and failed == []
    drop_head_tp_partial_cache("move")


def test_exchange_writes_the_helper_cache_and_reports_bytes():
    _exchange("gpu_copy")
    _exchange("cpu_pool")
