# SPDX-License-Identifier: Apache-2.0
"""Coordinate physical head-sharded KV leases before a DP scheduling epoch."""

import json
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.math_utils import cdiv
from vllm.v1 import head_tp_profile as profile
from vllm.v1.head_tp import DecodePolicy, DecodeRow
from vllm.v1.head_tp_admission import AdmissionRow, LoadAdmission
from vllm.v1.kv_cache_interface import AttentionSpec, iter_layer_specs
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


@dataclass
class Lease:
    request: str | None = None
    blocks: list = field(default_factory=list)
    budget: int = 0


class PackedKVCoordinator:
    """Each rank holds one half of every lease. Block IDs are rank-local."""

    def __init__(self, engine, options):
        self.engine = engine
        self.rank = engine.dp_rank
        self.manager = engine.scheduler.kv_cache_manager
        self.pool = self.manager.block_pool
        self.group = engine.dp_group
        self.local = self.manager.coordinator.single_type_managers[0]
        self.block = self.local.block_size
        self.manager.head_tp_dense_limit = (self.pool.num_gpu_blocks - 1) * self.block
        self.limit = min(
            self.manager.max_model_len,
            int(options.get("max_context", self.manager.max_model_len)),
        )
        if self.limit < 1 or (
            self.limit < self.manager.max_model_len
            and self.manager.head_tp_dense_limit < self.manager.max_model_len
        ):
            raise ValueError(
                "max_context or dense KV capacity must cover max_model_len"
            )
        policy = options.get("policy_class")
        self.policy = (resolve_obj_by_qualname(policy) if policy else DecodePolicy)(
            options
        )
        self.admission = options.get("admission", "longest")
        if self.admission not in ("load", "local", "longest"):
            raise ValueError("head_tp.admission must be load, local or longest")
        if self.admission == "load":
            spec = next(iter(iter_layer_specs(self.local.kv_cache_spec)))
            assert isinstance(spec, AttentionSpec)
            shard_bytes = (
                (spec.num_kv_heads // 2)
                * (spec.head_size + spec.head_size_v)
                * spec.dtype.itemsize
            )
            self.cost_model = LoadAdmission(options, shard_bytes)
        self.leases = [Lease(), Lease()]
        self.step = 0

    def gather(self, value):
        # The engine's stateless Gloo group need not be in c10d's global registry.
        stamp = profile.start()
        raw = json.dumps(value).encode()
        n = torch.tensor([len(raw)], dtype=torch.int64)
        sizes = [torch.empty_like(n) for _ in self.leases]
        stamp = profile.finish("encode", self.rank, stamp)
        dist.all_gather(sizes, n, group=self.group)
        stamp = profile.finish("gather_sizes", self.rank, stamp)
        buf = torch.zeros(max(int(s.item()) for s in sizes), dtype=torch.uint8)
        buf[: len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
        parts = [torch.empty_like(buf) for _ in sizes]
        stamp = profile.finish("pack_payload", self.rank, stamp)
        dist.all_gather(parts, buf, group=self.group)
        stamp = profile.finish("gather_payload", self.rank, stamp)
        result = [
            json.loads(bytes(p[: int(s.item())].tolist())) for p, s in zip(parts, sizes)
        ]

        profile.finish("unpack_payload", self.rank, stamp)
        return result

    def advance(self):
        sched = self.engine.scheduler
        requests = sched.requests
        self.step += 1
        old = self.leases[self.rank].request
        prior = requests.get(old)
        if prior is not None and (
            prior.is_finished() or prior.status == RequestStatus.PREEMPTED
        ):
            prior = None
        # A waiting request that cannot be scheduled yet must not strand KV
        # on both workers while the head of the waiting queue needs memory.
        front = sched.waiting.peek_request() if sched.waiting else None
        if front is not None and front.status not in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            front = None
        if self.admission == "longest":
            # Custom selectors historically received rows in request insertion order.
            ready = [
                r
                for r in requests.values()
                if r.status == RequestStatus.RUNNING or r is front
            ]
        else:
            ready = [r for r in sched.running if r.status == RequestStatus.RUNNING]
            if front is not None:
                ready.append(front)
        required = None
        if (
            prior is not None
            and prior.status in (RequestStatus.RUNNING, RequestStatus.WAITING)
            and 0 < prior.num_computed_tokens < prior.num_prompt_tokens
        ):
            required = old
        mandatory = [
            r.request_id
            for r in ready
            if r.num_tokens <= self.limit
            and max(r.num_computed_tokens, r.num_prompt_tokens)
            > self.manager.head_tp_dense_limit
        ]
        if mandatory:
            required = old if old in mandatory else mandatory[0]

        def describe(r):
            if r is None:
                return None
            end = min(
                self.manager.max_model_len,
                r.num_tokens + r.num_output_placeholders,
                r.num_computed_tokens + sched.scheduler_config.max_num_batched_tokens,
            )
            bound = min(
                self.manager.max_model_len, max(end, r.num_prompt_tokens + r.max_tokens)
            )
            return (
                r.request_id,
                r.num_computed_tokens,
                end,
                cdiv(bound, 2 * self.block),
            )

        if self.admission == "longest":
            rows = [
                DecodeRow(r.request_id, max(r.num_computed_tokens, r.num_prompt_tokens))
                for r in ready
                if r.num_tokens <= self.limit
            ]
            loads = self.gather([r.position for r in rows])
            choice = self.policy.select(
                rows,
                peer_positions=loads[1 - self.rank],
                rank=self.rank,
                step=self.step,
            )
            if choice is not None and choice not in {r.req_id for r in rows}:
                raise ValueError("head_tp policy selected an ineligible request")
            if required is not None:
                choice = required
            states = self.gather(
                (
                    describe(prior),
                    describe(requests.get(choice)),
                    self.pool.get_num_free_blocks(),
                )
            )
        else:
            # Both ranks can derive the selected target's allocation from this
            # same pre-allocation snapshot, including when a forced lease wins.
            candidates = []
            if self.admission == "load":
                candidates = [
                    (
                        describe(r),
                        max(r.num_computed_tokens, r.num_prompt_tokens),
                        max(0, r.max_tokens - r.num_output_tokens),
                        r.num_tokens <= self.limit,
                    )
                    for r in ready
                ]
            gather_start = profile.start()
            snapshots = self.gather(
                (
                    describe(prior),
                    describe(requests.get(required)),
                    self.pool.get_num_free_blocks(),
                    candidates,
                )
            )
            choose_start = profile.finish("snapshot", self.rank, gather_start)
            states = self._joint_lease_states(snapshots)
            profile.finish("choose", self.rank, choose_start)
            if not any(target for _, target, _ in states) and not any(
                lease.request for lease in self.leases
            ):
                return
        # Promise space for accepted requests without allocating future pages.
        # Both schedulers must protect the same remaining growth; otherwise two
        # partial prefills can occupy the pool and repeatedly preempt each other.
        outstanding = sum(lease.budget - len(lease.blocks) for lease in self.leases)
        free = [s[2] - outstanding for s in states]
        for owner, (old_info, target_info, _) in enumerate(states):
            lease = self.leases[owner]
            if lease.request and (
                old_info is None
                or old_info[1] == 0
                and (target_info is None or target_info[0] != lease.request)
            ):
                self._publish("drop", owner, lease, [], 0)
                self.pool.free_blocks(lease.blocks)
                for i in (0, 1):
                    free[i] += lease.budget
                lease.blocks = []
                lease.budget = 0
                lease.request = None
                if owner == self.rank:
                    self.local.free(self.manager.packed_request)
                    self.manager.packed_request = None
                    self.manager.packed_capacity = 0
            if lease.request and (
                target_info is None or lease.request != target_info[0]
            ):
                pos = old_info[1]
                dense_count = cdiv(pos, self.block)
                used = cdiv(pos, 2 * self.block)
                extra = dense_count - used
                if free[owner] + lease.budget - len(lease.blocks) < extra:
                    target_info = old_info  # retain the authoritative shards
                else:
                    dense = []
                    if self.rank == owner:
                        added = iter(self.pool.get_new_blocks(extra))
                        dense = [
                            lease.blocks[i // 2] if i % 2 == 0 else next(added)
                            for i in range(dense_count)
                        ]
                    self._publish("unpack", owner, lease, dense, pos)
                    if self.rank == owner:
                        self._replace_local(lease.request, dense)
                        self.pool.free_blocks(lease.blocks[used:])
                        self.manager.packed_request = None
                    else:
                        self.pool.free_blocks(lease.blocks)
                    free[owner] += lease.budget - used - extra
                    free[1 - owner] += lease.budget
                    lease.blocks = []
                    lease.budget = 0
                    lease.request = None
            if target_info is None:
                continue
            req, pos, end, budget = target_info
            need = cdiv(end, 2 * self.block)
            delta = max(0, need - len(lease.blocks))
            if not delta and lease.request == req:
                continue
            commitment = max(0, budget - lease.budget)
            if min(free) < commitment:
                # An existing lease can only execute the tokens it has pages for.
                # Standard scheduler preemption frees the request if growth fails.
                continue
            fresh = lease.request is None
            lease.request = req
            lease.budget += commitment
            lease.blocks.extend(self.pool.get_new_blocks(delta))
            free = [v - commitment for v in free]
            dense = (
                list(self.local.req_to_blocks.get(req, ()))
                if self.rank == owner and fresh
                else []
            )
            self._publish("pack" if fresh else "grow", owner, lease, dense, pos)
            if self.rank == owner and fresh:
                self._replace_local(req, [self.pool.null_block] * cdiv(pos, self.block))
                self.pool.free_blocks(reversed(dense))
                self.manager.packed_request = req
            if self.rank == owner:
                self.manager.packed_capacity = len(lease.blocks) * 2 * self.block

        self.manager.packed_growth_reserve = sum(
            lease.budget - len(lease.blocks) for lease in self.leases
        )

    def _joint_lease_states(self, snapshots):
        states = [[prior, forced, free] for prior, forced, free, _ in snapshots]
        if self.admission == "local" or any(forced for _, forced, _ in states):
            return states
        batches = [
            [
                AdmissionRow(info[0], length, info[1], remaining, eligible)
                for info, length, remaining, eligible in candidates
            ]
            for _, _, _, candidates in snapshots
        ]
        residents = [lease.request for lease in self.leases]
        selected = self.cost_model.choose(batches, residents)
        if selected is not None:
            owner, request = selected
            # Unpack the opposite direction before admitting a new owner.
            if residents[1 - owner] is None:
                states[owner][1] = next(
                    info for info, *_ in snapshots[owner][3] if info[0] == request
                )
        return states

    def release_finished(self):
        if not any(lease.request for lease in self.leases):
            return
        ids = self.gather(
            self.leases[self.rank].request in self.engine.scheduler.requests
        )
        for owner, lease in enumerate(self.leases):
            if lease.request is not None and not ids[owner]:
                self._publish("drop", owner, lease, [], 0)
                self.pool.free_blocks(lease.blocks)
                self.leases[owner] = Lease()
                if owner == self.rank:
                    self.manager.packed_request = None
                    self.manager.packed_capacity = 0

        self.manager.packed_growth_reserve = sum(
            lease.budget - len(lease.blocks) for lease in self.leases
        )

    def _replace_local(self, req, blocks):
        self.local.req_to_blocks[req] = blocks
        self.local.num_cached_block[req] = 0

    def _publish(self, action, owner, lease, dense, pos):
        packed_ids = [b.block_id for b in lease.blocks]
        dense_ids = [b.block_id for b in dense]
        self.engine.collective_rpc(
            "change_head_tp_layout",
            args=(action, owner, lease.request, packed_ids, dense_ids, pos),
        )
        log = logger.debug if action == "grow" and len(packed_ids) < 2 else logger.info
        log(
            "head-TP %s owner=%d rank=%d position=%d physical_pages=%d",
            action,
            owner,
            self.rank,
            pos,
            len(packed_ids),
        )
