# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decide head collaboration and publish one step to every DP rank.

The scheduler calls this at the end of ``schedule``. Workers only execute
the step they are given. Pairing is refused when the degree does not divide
the KV heads, and a request that is still in prefill is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from vllm.utils.math_utils import cdiv
from vllm.v1.head_tp import (
    HeadTPPlan,
    HeadTPStep,
    make_plan,
    make_shards,
)
from vllm.v1.request import Request

RECHECK_INTERVAL = 32
MIN_COLLAB_TOKENS = 64
REQ_ID_BYTES = 256

_KV_CODE = {"none": 0, "cpu_pool": 1, "gpu_copy": 2}
_KV_NAME = {code: name for name, code in _KV_CODE.items()}
_STATE_CODE = {"pending_load": 1, "active": 2}
_STATE_NAME = {code: name for name, code in _STATE_CODE.items()}


class HeadTPWire(Protocol):
    """Two fixed collectives, in this order, on every DP rank."""

    def gather_stats(
        self, local: tuple[int, int, int]
    ) -> list[tuple[int, int, int]]:
        """Allgather ``(free_blocks, running_reqs, attn_us)``."""

    def gather_plans(
        self, encoded: list[int], req_ids: list[str]
    ) -> tuple[list[list[int]], list[list[str]]]:
        """Allgather the fixed plan table and its request ids."""


@dataclass
class HeadTPProfile:
    """Offline lookup tables. Keys are context lengths, not measured live.

    A missing table yields 0, so the benefit is 0 and the request stays local.
    """

    full_attn_s: dict[int, float] = field(default_factory=dict)
    attn_s: dict[tuple[int, int], float] = field(default_factory=dict)
    load_s: dict[tuple[str, int, int], float] = field(default_factory=dict)

    def full_attn(self, length: int) -> float:
        return _at_length(self.full_attn_s, length)

    def attn(self, length: int, degree: int) -> float:
        table = {
            length_key: value
            for (length_key, d), value in self.attn_s.items()
            if d == degree
        }
        return _at_length(table, length)

    def load(self, source: str, length: int, degree: int) -> float:
        table = {
            length_key: value
            for (src, length_key, d), value in self.load_s.items()
            if src == source and d == degree
        }
        return _at_length(table, length)


class LengthBucketEMA:
    """Default remaining-length estimate: exponential mean by prompt bucket.

    The bucket key is ``(prompt_len // 256, 1 if the CPU prefix hit else 0)``.
    An empty bucket falls back to the request's remaining ``max_tokens``.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.means: dict[tuple[int, int], float] = {}

    @staticmethod
    def bucket(prompt_len: int, prefix_hit_tokens: int) -> tuple[int, int]:
        hit = 1 if prefix_hit_tokens > 0 else 0
        return (prompt_len // 256, hit)

    def predict(
        self,
        prompt_len: int,
        prefix_hit_tokens: int,
        max_tokens: int,
        num_output_tokens: int,
    ) -> int:
        key = self.bucket(prompt_len, prefix_hit_tokens)
        if key not in self.means:
            return max(0, max_tokens - num_output_tokens)
        return max(0, int(self.means[key]))

    def observe(self, request: Request) -> None:
        key = self.bucket(request.num_prompt_tokens, request.prefix_cpu_hit_tokens)
        actual = float(request.num_output_tokens)
        old = self.means.get(key)
        if old is None:
            self.means[key] = actual
            return
        self.means[key] = self.alpha * actual + (1.0 - self.alpha) * old


def legal_degrees(num_kv_heads: int, dp_size: int) -> list[int]:
    """Legal collaboration degrees, largest first."""

    degrees = []
    for degree in range(num_kv_heads, 1, -1):
        if num_kv_heads % degree == 0 and degree <= dp_size:
            degrees.append(degree)
    return degrees


def select_kv_source(
    cpu_hit_tokens: int | None,
    visible_len: int,
    has_gpu_blocks: bool,
) -> tuple[str, int]:
    """Pick ``cpu_pool``, then ``gpu_copy``.

    A partial CPU hit, or a hit that is still pending, is not a hit.
    """

    if (
        cpu_hit_tokens is not None
        and visible_len > 0
        and cpu_hit_tokens >= visible_len
    ):
        return "cpu_pool", cpu_hit_tokens
    if has_gpu_blocks:
        return "gpu_copy", 0
    return "none", 0


def plan_slot_width(dp_size: int) -> int:
    """Ints per plan slot.

    The fields are batch index, degree, one rank per DP slot, KV source,
    visible length, state, and predicted remaining.
    """

    return 2 + dp_size + 4


def publish_head_tp_step(
    requests: dict[str, Request],
    scheduled_tokens: dict[str, int],
    *,
    rank: int,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    partner_queue_limit: int,
    profile: HeadTPProfile,
    lengths: LengthBucketEMA,
    peer_free: list[int],
    peer_running: list[int],
    peer_attn_us: list[int],
    cpu_hit_tokens: dict[str, int | None],
    has_gpu_blocks: dict[str, bool],
) -> HeadTPStep | None:
    """Update this rank's requests and return the plans it will publish.

    Only scheduled decode requests are judged. A plan that is already loading
    or active keeps its degree. ``None`` means this rank publishes nothing.
    """

    for request in requests.values():
        if request.request_id not in scheduled_tokens:
            continue
        _consider(
            request,
            rank=rank,
            dp_size=dp_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            partner_queue_limit=partner_queue_limit,
            profile=profile,
            lengths=lengths,
            peer_free=peer_free,
            peer_running=peer_running,
            peer_attn_us=peer_attn_us,
            cpu_hit_tokens=cpu_hit_tokens.get(request.request_id),
            has_gpu_blocks=has_gpu_blocks.get(request.request_id, False),
        )
    plans: dict[str, HeadTPPlan] = {}
    for request in requests.values():
        plan = request.head_tp_plan
        if plan is None or plan.state == "local":
            continue
        if plan.owner_rank != rank:
            continue
        plans[request.request_id] = plan
    if not plans:
        return None
    return HeadTPStep(plans=plans, active=True)


def exchange_head_tp_step(
    requests: dict[str, Request],
    scheduled_tokens: dict[str, int],
    *,
    rank: int,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    partner_queue_limit: int,
    profile: HeadTPProfile,
    lengths: LengthBucketEMA,
    local_stats: tuple[int, int, int],
    wire: HeadTPWire,
    cpu_hit_tokens: dict[str, int | None],
    has_gpu_blocks: dict[str, bool],
    max_plans: int,
) -> HeadTPStep | None:
    """Gather free-block stats, decide, then gather the plan table.

    Both collectives run on every rank, including a rank with no requests.
    """

    stats = wire.gather_stats(local_stats)
    if len(stats) != dp_size:
        raise RuntimeError(
            f"head-tp stats gathered {len(stats)} ranks, expected {dp_size}"
        )
    local = publish_head_tp_step(
        requests,
        scheduled_tokens,
        rank=rank,
        dp_size=dp_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        partner_queue_limit=partner_queue_limit,
        profile=profile,
        lengths=lengths,
        peer_free=[item[0] for item in stats],
        peer_running=[item[1] for item in stats],
        peer_attn_us=[item[2] for item in stats],
        cpu_hit_tokens=cpu_hit_tokens,
        has_gpu_blocks=has_gpu_blocks,
    )
    order = list(scheduled_tokens)
    encoded, req_ids = encode_local_plans(
        local, order, rank=rank, dp_size=dp_size, max_plans=max_plans
    )
    rows, id_rows = wire.gather_plans(encoded, req_ids)
    return decode_gathered_plans(
        rows,
        id_rows,
        dp_size=dp_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def encode_local_plans(
    step: HeadTPStep | None,
    scheduled_ids: list[str],
    *,
    rank: int,
    dp_size: int,
    max_plans: int,
) -> tuple[list[int], list[str]]:
    """Pack this rank's plans into a fixed table. Unused slots are -1."""

    width = plan_slot_width(dp_size)
    encoded = [-1] * (max_plans * width)
    req_ids = [""] * max_plans
    if step is None:
        return encoded, req_ids
    index = {req_id: pos for pos, req_id in enumerate(scheduled_ids)}
    slot = 0
    for req_id in sorted(step.plans):
        plan = step.plans[req_id]
        if plan.owner_rank != rank:
            continue
        if slot >= max_plans:
            raise RuntimeError(
                f"rank {rank} has more than {max_plans} head-collaboration plans"
            )
        _write_slot(encoded, slot, width, plan, index.get(req_id, -1), dp_size)
        req_ids[slot] = req_id
        slot += 1
    return encoded, req_ids


def decode_gathered_plans(
    rows: list[list[int]],
    id_rows: list[list[str]],
    *,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> HeadTPStep | None:
    """Rebuild one ``HeadTPStep`` from every rank's table."""

    if len(rows) != dp_size or len(id_rows) != dp_size:
        raise RuntimeError(
            f"head-tp plan gather has {len(rows)} int rows and {len(id_rows)} id rows, "
            f"expected {dp_size}"
        )
    width = plan_slot_width(dp_size)
    plans: dict[str, HeadTPPlan] = {}
    for encoded, req_ids in zip(rows, id_rows):
        if len(encoded) % width != 0 or len(req_ids) != len(encoded) // width:
            raise RuntimeError("head-tp plan table width does not match its ids")
        for slot, req_id in enumerate(req_ids):
            plan = _read_slot(
                encoded,
                slot,
                width,
                req_id,
                dp_size=dp_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            )
            if plan is None:
                continue
            if plan.req_id in plans:
                raise RuntimeError(f"duplicate head-tp plan for {plan.req_id}")
            plans[plan.req_id] = plan
    if not plans:
        return None
    return HeadTPStep(plans=plans, active=True)


class TorchDPHeadTPWire:
    """CPU allgather on the engine DP group. Ranks must call both methods."""

    def __init__(self, group: object, dp_size: int) -> None:
        self.group = group
        self.dp_size = dp_size

    def gather_stats(
        self, local: tuple[int, int, int]
    ) -> list[tuple[int, int, int]]:
        import torch
        import torch.distributed as dist

        tensor = torch.tensor(local, dtype=torch.int64)
        gathered = [torch.empty(3, dtype=torch.int64) for _ in range(self.dp_size)]
        dist.all_gather(gathered, tensor, group=self.group)
        return [tuple(int(value) for value in row.tolist()) for row in gathered]

    def gather_plans(
        self, encoded: list[int], req_ids: list[str]
    ) -> tuple[list[list[int]], list[list[str]]]:
        import torch
        import torch.distributed as dist

        ints = torch.tensor(encoded, dtype=torch.int64)
        int_out = [torch.empty_like(ints) for _ in range(self.dp_size)]
        dist.all_gather(int_out, ints, group=self.group)
        packed = pack_req_ids(req_ids)
        id_out = [torch.empty_like(packed) for _ in range(self.dp_size)]
        dist.all_gather(id_out, packed, group=self.group)
        return (
            [row.tolist() for row in int_out],
            [unpack_req_ids(row) for row in id_out],
        )


def pack_req_ids(req_ids: list[str]):
    import torch

    packed = torch.zeros((len(req_ids), REQ_ID_BYTES), dtype=torch.uint8)
    for row, req_id in enumerate(req_ids):
        raw = req_id.encode("utf-8")
        if len(raw) > REQ_ID_BYTES:
            raise ValueError(
                f"request id is {len(raw)} bytes; the head-tp table holds "
                f"{REQ_ID_BYTES}"
            )
        if raw:
            packed[row, : len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
    return packed


def unpack_req_ids(packed) -> list[str]:
    req_ids = []
    for row in packed.tolist():
        raw = bytes(int(value) for value in row).split(b"\0", 1)[0]
        req_ids.append(raw.decode("utf-8"))
    return req_ids


def _at_length(table: dict[int, float], length: int) -> float:
    if not table:
        return 0.0
    below = [key for key in table if key <= length]
    key = max(below) if below else min(table)
    return float(table[key])


def _consider(
    request: Request,
    *,
    rank: int,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    partner_queue_limit: int,
    profile: HeadTPProfile,
    lengths: LengthBucketEMA,
    peer_free: list[int],
    peer_running: list[int],
    peer_attn_us: list[int],
    cpu_hit_tokens: int | None,
    has_gpu_blocks: bool,
) -> None:
    if request.num_computed_tokens < request.num_prompt_tokens:
        return
    plan = request.head_tp_plan
    if plan is not None and plan.state == "pending_load":
        return
    if plan is not None and plan.state == "active":
        _maybe_exit(
            request,
            plan,
            partner_queue_limit=partner_queue_limit,
            profile=profile,
            lengths=lengths,
            peer_running=peer_running,
            peer_attn_us=peer_attn_us,
        )
        return
    if request.num_computed_tokens < request.head_tp_next_check_tokens:
        return
    _try_pair(
        request,
        rank=rank,
        dp_size=dp_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        profile=profile,
        lengths=lengths,
        peer_free=peer_free,
        peer_attn_us=peer_attn_us,
        cpu_hit_tokens=cpu_hit_tokens,
        has_gpu_blocks=has_gpu_blocks,
    )


def _try_pair(
    request: Request,
    *,
    rank: int,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    profile: HeadTPProfile,
    lengths: LengthBucketEMA,
    peer_free: list[int],
    peer_attn_us: list[int],
    cpu_hit_tokens: int | None,
    has_gpu_blocks: bool,
) -> None:
    visible = request.num_computed_tokens
    source, cpu_hit = select_kv_source(cpu_hit_tokens, visible, has_gpu_blocks)
    prefix_hit = cpu_hit if source == "cpu_pool" else 0
    remaining = lengths.predict(
        request.num_prompt_tokens,
        prefix_hit,
        request.max_tokens,
        request.num_output_tokens,
    )
    request.predicted_remaining = remaining
    if source == "none":
        _defer(request)
        return
    need = cdiv(visible, block_size) if visible > 0 else 0
    for degree in legal_degrees(num_kv_heads, dp_size):
        shards = make_shards(num_q_heads, num_kv_heads, degree, rank, dp_size)
        partners = [shard.rank for shard in shards[1:]]
        if any(peer_free[partner] < need for partner in partners):
            continue
        if _benefit(profile, visible, degree, remaining) <= _pair_cost(
            profile,
            visible,
            degree,
            source,
            partners,
            peer_attn_us,
        ):
            continue
        request.prefix_cpu_hit_tokens = prefix_hit
        request.head_tp_collab_tokens = 0
        request.head_tp_plan = make_plan(
            request.request_id,
            num_q_heads,
            num_kv_heads,
            degree,
            rank,
            dp_size,
            state="pending_load",
            kv_source=source,  # type: ignore[arg-type]
            visible_len=visible,
            predicted_remaining=remaining,
        )
        return
    _defer(request)


def _maybe_exit(
    request: Request,
    plan: HeadTPPlan,
    *,
    partner_queue_limit: int,
    profile: HeadTPProfile,
    lengths: LengthBucketEMA,
    peer_running: list[int],
    peer_attn_us: list[int],
) -> None:
    if request.head_tp_collab_tokens < MIN_COLLAB_TOKENS:
        return
    remaining = lengths.predict(
        request.num_prompt_tokens,
        request.prefix_cpu_hit_tokens,
        request.max_tokens,
        request.num_output_tokens,
    )
    request.predicted_remaining = remaining
    plan.predicted_remaining = remaining
    if request.max_tokens - request.num_output_tokens <= 1:
        _defer(request)
        return
    partners = [shard.rank for shard in plan.shards[1:]]
    if all(peer_running[partner] <= partner_queue_limit for partner in partners):
        return
    # The load is already paid. What remains is the slowest partner's attention.
    benefit = _benefit(profile, plan.visible_len, plan.degree, remaining)
    slowest = _slowest_partner(
        profile, plan.visible_len, plan.degree, partners, peer_attn_us
    )
    if benefit <= slowest:
        _defer(request)


def _defer(request: Request) -> None:
    request.head_tp_plan = None
    request.head_tp_collab_tokens = 0
    request.head_tp_next_check_tokens = request.num_computed_tokens + RECHECK_INTERVAL


def _benefit(
    profile: HeadTPProfile, length: int, degree: int, remaining: int
) -> float:
    saved = max(0.0, profile.full_attn(length) - profile.attn(length, degree))
    return remaining * saved


def _pair_cost(
    profile: HeadTPProfile,
    length: int,
    degree: int,
    source: str,
    partners: list[int],
    peer_attn_us: list[int],
) -> float:
    return profile.load(source, length, degree) + _slowest_partner(
        profile, length, degree, partners, peer_attn_us
    )


def _slowest_partner(
    profile: HeadTPProfile,
    length: int,
    degree: int,
    partners: list[int],
    peer_attn_us: list[int],
) -> float:
    fallback = profile.attn(length, degree)
    times = []
    for partner in partners:
        reported = peer_attn_us[partner]
        times.append(fallback if reported <= 0 else reported / 1_000_000)
    return max(times) if times else fallback


def _write_slot(
    buf: list[int],
    slot: int,
    width: int,
    plan: HeadTPPlan,
    req_index: int,
    dp_size: int,
) -> None:
    base = slot * width
    buf[base] = req_index
    buf[base + 1] = plan.degree
    for group in range(dp_size):
        if group < plan.degree:
            buf[base + 2 + group] = plan.shards[group].rank
        else:
            buf[base + 2 + group] = -1
    tail = base + 2 + dp_size
    buf[tail] = _KV_CODE[plan.kv_source]
    buf[tail + 1] = plan.visible_len
    buf[tail + 2] = _STATE_CODE[plan.state]
    buf[tail + 3] = plan.predicted_remaining


def _read_slot(
    buf: list[int],
    slot: int,
    width: int,
    req_id: str,
    *,
    dp_size: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> HeadTPPlan | None:
    base = slot * width
    degree = buf[base + 1]
    if degree < 2 or req_id == "":
        return None
    ranks = buf[base + 2 : base + 2 + degree]
    owner = ranks[0]
    shards = make_shards(num_q_heads, num_kv_heads, degree, owner, dp_size)
    if [shard.rank for shard in shards] != list(ranks):
        raise ValueError(
            f"{req_id}: gathered shard ranks {list(ranks)} do not match degree {degree}"
        )
    tail = base + 2 + dp_size
    source = _KV_NAME[buf[tail]]
    state = _STATE_NAME[buf[tail + 2]]
    return HeadTPPlan(
        req_id=req_id,
        owner_rank=owner,
        degree=degree,
        shards=shards,
        state=state,  # type: ignore[arg-type]
        kv_source=source,  # type: ignore[arg-type]
        visible_len=buf[tail + 1],
        predicted_remaining=buf[tail + 3],
    )
