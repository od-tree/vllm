"""Paged head-TP on FA3 with one virtual row per local KV-head group."""
from __future__ import annotations

import os

import torch

from cuda_ipc_p2p import graph_streams_active, memcpy_peer
from tile_cp import _planner


def load_fa3():
    extension = os.environ.get("FA3_EXTENSION")
    if extension:
        torch.ops.load_library(extension)
    else:
        import vllm_flash_attn._vllm_fa3_C  # noqa: F401
    return torch.ops._vllm_fa3_C.fwd


def kv_block_tokens(head_dim):
    # BF16/FP16, one query, paged non-TMA, non-causal, on Hopper.
    # See the fork's hopper/tile_size.h. FP8 weights do not change KV dtype.
    return {128: 128, 256: 80}[head_dim]


def current_kv(batch):
    positions = torch.tensor(batch.lengths, dtype=torch.long, device=batch.table.device) - 1
    page_size = batch.k.shape[1]
    rows = torch.arange(len(batch.lengths), device=batch.table.device)
    pages = batch.table[rows, positions // page_size].long()
    offsets = positions % page_size
    return batch.k[pages, offsets], batch.v[pages, offsets]


class HeadShard:
    """One rank's cache with regular pages of one KV head and compact Q/O."""

    def __init__(self, ranges, device):
        # Each range is (logical batch, row slice, KV-head slice). K and V
        # are placed once. Only the current token is updated during a step.
        batch = ranges[0][0]
        _, page_size, kv_heads, dim = batch.k.shape
        group = batch.q.shape[1] // kv_heads
        rows = [
            (batch, row, heads)
            for batch, requests, heads in ranges
            for row in range(requests.start, requests.stop)
        ]
        self.lengths = [batch.lengths[row] for batch, row, _ in rows]
        self.heads = [heads.stop - heads.start for _, _, heads in rows]
        counts = [(length + page_size - 1) // page_size for length in self.lengths]
        n_groups = sum(self.heads)
        n_pages = sum(n * h for n, h in zip(counts, self.heads))
        self.k = torch.empty(n_pages, page_size, 1, dim, dtype=batch.k.dtype, device=device)
        self.v = torch.empty_like(self.k)
        table = torch.zeros(n_groups, max(counts), dtype=torch.int32)
        slots = []
        group_lengths = []
        cursor = group_start = 0
        for (source, row, heads), n, h, length in zip(rows, counts, self.heads, self.lengths):
            ids = source.table[row, :n].long()
            block = slice(cursor, cursor + h * n)
            for dest, cache in ((self.k, source.k), (self.v, source.v)):
                pages = cache.index_select(0, ids)[:, :, heads, :]
                dest[block].view(h, n, page_size, dim).copy_(pages.permute(2, 0, 1, 3))
            physical = torch.arange(cursor, cursor + h * n, dtype=torch.int32).view(h, n)
            table[group_start:group_start + h, :n] = physical
            last_token = (length - 1) % page_size
            slots.extend(
                (cursor + (head + 1) * n - 1) * page_size + last_token
                for head in range(h)
            )
            group_lengths.extend([length] * h)
            cursor += h * n
            group_start += h

        q, k_now, v_now = [], [], []
        for source, requests, heads in ranges:
            k, v = current_kv(source)
            q.append(source.q[requests, heads.start * group:heads.stop * group])
            k_now.append(k[requests, heads])
            v_now.append(v[requests, heads])
        self.q = torch.cat([x.reshape(-1, group, dim) for x in q]).to(device)
        self.k_now = torch.cat([x.reshape(-1, dim) for x in k_now]).to(device)
        self.v_now = torch.cat([x.reshape(-1, dim) for x in v_now]).to(device)
        self.out = torch.empty_like(self.q)
        self.table = table.to(device)
        self.used = torch.tensor(group_lengths, dtype=torch.int32, device=device)
        self.cu_q = torch.arange(n_groups + 1, dtype=torch.int32, device=device)
        self.slots = torch.tensor(slots, dtype=torch.long, device=device)
        self.maximum = max(self.lengths)
        block_size = kv_block_tokens(dim)
        self.kv_blocks = [(length + block_size - 1) // block_size for length in self.lengths]

    def write_kv(self):
        for cache, now in ((self.k, self.k_now), (self.v, self.v_now)):
            cache.view(-1, cache.shape[-1]).index_copy_(0, self.slots, now)

    def attention(self, fwd, plan):
        counts, bound = plan
        fwd(
            self.q, self.k, self.v, None, None, None, self.out,
            self.cu_q, None, None, None, self.used,
            1, self.maximum, self.table, None, None,
            None, None, None, None, None, None,
            self.q.shape[-1] ** -0.5, False, -1, -1, 0.0, True,
            None, bound, True, 0, None, 1, 0, None,
            **({"row_split_counts": counts} if counts is not None else {}),
        )

    def split_plan(self, counts):
        if counts is None:
            return None, 0
        per_group = [count for count, heads in zip(counts, self.heads) for _ in range(heads)]
        return torch.tensor(per_group, dtype=torch.int32, device=self.q.device), max(counts)

    def describe_plan(self, counts):
        if counts is None:
            return None
        chunks = [(n + s - 1) // s for n, s in zip(self.kv_blocks, counts)]
        effective = [(n + c - 1) // c for n, c in zip(self.kv_blocks, chunks)]
        return dict(
            work_tiles=sum(h * s for h, s in zip(self.heads, effective)),
            max_blocks_per_tile=max(chunks),
            # FA3 allocates storage up to the static upper bound even when
            # only a few rows use multiple splits.
            partial_buffer_bytes=(max(counts) * self.q.shape[0] * self.q.shape[1]
                                  * (self.q.shape[-1] + 1) * 4 if max(counts) > 1 else 0),
        )


class HeadTPFA3:
    def __init__(self, fwd, inputs, n_long):
        owner, partner = inputs
        _, q_heads, self.dim = owner.q.shape
        kv_heads = owner.k.shape[2]
        if not 1 <= n_long <= len(owner.lengths):
            raise ValueError("head-TP needs at least one cooperative request")
        if self.dim not in (128, 256) or kv_heads < 2 or kv_heads % 2 or q_heads % kv_heads:
            raise ValueError("FA3 head-TP requires head_dim 128 or 256 and even GQA KV heads")
        self.fwd = fwd
        self.n_long = n_long
        self.n_owner = len(owner.lengths)
        self.n_partner = len(partner.lengths)
        self.q_heads = q_heads
        self.kv_half = kv_heads // 2
        self.group = q_heads // kv_heads
        self.long_groups = n_long * self.kv_half
        self.streams = [torch.cuda.Stream(device=rank) for rank in range(2)]
        long_rows = slice(0, n_long)
        self.ranks = [
            HeadShard([(owner, long_rows, slice(0, self.kv_half)),
                       (owner, slice(n_long, self.n_owner), slice(0, kv_heads))], 0),
            HeadShard([(owner, long_rows, slice(self.kv_half, kv_heads)),
                       (partner, slice(0, self.n_partner), slice(0, kv_heads))], 1),
        ]
        # Keep the owner's next-token projections on the owner. Poison the
        # helper's source and both caches' current-token slots so correctness
        # depends on the captured exchange and updates.
        group_range = slice(0, self.long_groups)
        helper = self.ranks[1]
        owner_k, owner_v = current_kv(owner)
        remote_q = owner.q[:n_long, q_heads // 2:].reshape(-1, self.group, self.dim)
        self.remote = (
            remote_q.to(0),
            owner_k[:n_long, self.kv_half:].reshape(-1, self.dim).to(0),
            owner_v[:n_long, self.kv_half:].reshape(-1, self.dim).to(0),
        )
        for tensor in (helper.q, helper.k_now, helper.v_now):
            tensor[group_range].fill_(float("nan"))
        for rank in self.ranks:
            for cache in (rank.k, rank.v):
                cache.view(-1, self.dim).index_fill_(0, rank.slots, float("nan"))
        self.send = torch.empty(self.long_groups, self.group + 2, self.dim,
                                dtype=owner.q.dtype, device=0)
        self.recv = torch.empty_like(self.send, device=1)
        self.owner_out = torch.empty_like(owner.q, device=0)
        self.returned = torch.empty_like(helper.out[group_range], device=0)

    def step(self, counts, compute_only=False):
        plans = [rank.split_plan(plan) for rank, plan in zip(self.ranks, counts)]
        owner, helper = self.ranks
        s0, s1 = self.streams
        n = self.long_groups
        q_half = self.q_heads // 2

        def exchange(mark):
            with torch.cuda.stream(s0):
                self.send[:, :self.group].copy_(self.remote[0])
                self.send[:, -2].copy_(self.remote[1])
                self.send[:, -1].copy_(self.remote[2])
            mark("qkv_pack")
            if not graph_streams_active():
                s0.synchronize()
            memcpy_peer(self.recv, self.send)
            mark("qkv_transfer")
            with torch.cuda.stream(s1):
                helper.q[:n].copy_(self.recv[:, :self.group])
                helper.k_now[:n].copy_(self.recv[:, -2])
                helper.v_now[:n].copy_(self.recv[:, -1])

        def run(mark):
            if not compute_only:
                exchange(mark)
                for rank, stream in zip(self.ranks, self.streams):
                    with torch.cuda.stream(stream):
                        rank.write_kv()
                mark("kv_update")
            for rank, stream, plan in zip(self.ranks, self.streams, plans):
                with torch.cuda.stream(stream):
                    rank.attention(self.fwd, plan)
            mark("attention")
            partner_out = helper.out[n:].view(self.n_partner, self.q_heads, self.dim)
            if compute_only:
                return owner.out, partner_out
            if not graph_streams_active():
                s1.synchronize()
            memcpy_peer(self.returned, helper.out[:n])
            mark("output_transfer")
            with torch.cuda.stream(s0):
                self.owner_out[:self.n_long, :q_half].copy_(owner.out[:n].view(-1, q_half, self.dim))
                self.owner_out[:self.n_long, q_half:].copy_(self.returned.view(-1, q_half, self.dim))
                self.owner_out[self.n_long:].copy_(owner.out[n:].view(-1, self.q_heads, self.dim))
            mark("splice")
            return self.owner_out, partner_out

        if compute_only:
            exchange(lambda _: None)
            for rank, stream in zip(self.ranks, self.streams):
                with torch.cuda.stream(stream):
                    rank.write_kv()
            s0.synchronize()
            s1.synchronize()
        return run

    def plans(self, policy, limit):
        if policy == "auto":
            return {policy: (None, None)}
        sms = [torch.cuda.get_device_properties(rank.q.device).multi_processor_count
               for rank in self.ranks]
        block_size = kv_block_tokens(self.dim)
        legacy = []
        for rank, sm in zip(self.ranks, sms):
            _, counts, _ = _planner.plan_decode_tiles(
                rank.lengths, [(row, 0, self.kv_half) for row in range(self.n_long)],
                list(range(self.n_long, len(rank.lengths))), self.kv_half * 2, sm,
                kv_block_tokens=block_size,
            )
            legacy.append(tuple(min(value, limit) for value in counts))
        minimax = tuple(
            _planner.minimax_split_counts(rank.kv_blocks, rank.heads, max(sum(rank.heads), sm), limit)
            for rank, sm in zip(self.ranks, sms)
        )
        choices = {"legacy": tuple(legacy), "minimax": minimax}
        if policy != "sweep":
            return {policy: choices[policy]}

        # Full Cartesian scan over the small set of per-rank candidates.
        # The fastest measured candidate can come from different budgets on
        # the two ranks if their background loads differ.
        pools = []
        for rank, sm, legacy_counts in zip(self.ranks, sms, legacy):
            candidates = _planner.fa3_split_candidates(rank.kv_blocks, rank.heads, sm, limit)
            choices_for_rank = {"auto": None, **{str(b): c for b, c in candidates.items()}}
            if legacy_counts not in choices_for_rank.values():
                choices_for_rank["legacy"] = legacy_counts
            pools.append(choices_for_rank)
        return {f"{a}/{b}": (x, y) for a, x in pools[0].items() for b, y in pools[1].items()}
