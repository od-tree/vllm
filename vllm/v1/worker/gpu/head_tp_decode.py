# SPDX-License-Identifier: Apache-2.0
"""Two-worker head-sharded GQA for single-token Qwen3 MoE decode.

Cooperating ranks each store only their local KV heads in regular pool pages.
Virtual FA rows address one head each, including ordinary local requests.
"""

from numbers import Integral

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.head_tp import DecodePolicy as DecodePolicy
from vllm.v1.head_tp import DecodeRow as DecodeRow

logger = init_logger(__name__)


class HeadTPDecode:
    @classmethod
    def from_runner(cls, runner, options):
        from vllm.distributed.parallel_state import get_dp_group
        from vllm.utils.import_utils import resolve_obj_by_qualname
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        config = runner.vllm_config
        text = config.model_config.hf_text_config
        parallel = config.parallel_config
        if (
            text.model_type != "qwen3_moe"
            or runner.dp_size != 2
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or runner.speculator is not None
            or runner.ubatch_runner is not None
            or runner.dcp_size != 1
            or runner.pcp_manager is not None
            or len(runner.attn_groups) != 1
            or len(runner.attn_groups[0]) != 1
        ):
            raise ValueError(
                "head_tp requires two DP workers, Qwen3 MoE, single-token "
                "decode, one KV group, TP=PP=1 and no microbatching"
            )
        group = runner.attn_groups[0][0]
        layers = [
            runner.compilation_config.static_forward_context[name]
            for name in group.layer_names
        ]
        if any(not isinstance(layer.impl, FlashAttentionImpl) for layer in layers):
            raise ValueError("head_tp requires the FLASH_ATTN backend")
        policy = options.get("policy_class")
        policy = (resolve_obj_by_qualname(policy) if policy else DecodePolicy)(options)
        dp = get_dp_group()
        decode = cls(
            [layer.kv_cache for layer in layers],
            text.num_attention_heads,
            runner.max_model_len,
            runner.max_num_reqs,
            int(options.get("max_context", runner.max_model_len)),
            int(options.get("max_splits", 8)),
            policy,
            runner.dp_rank,
            dp.device_group,
            dp.cpu_group,
        )
        if decode.block != runner.block_tables.block_sizes[0]:
            raise ValueError(
                "head_tp requires matching allocator and kernel block sizes"
            )
        group.get_metadata_builder(0).head_tp_decode = decode
        for layer in layers:
            layer.impl.head_tp_decode = decode
        return decode

    def __init__(
        self,
        caches,
        q_heads,
        max_len,
        max_rows,
        capacity,
        split_limit,
        policy,
        rank,
        device_group,
        cpu_group,
    ):
        self.caches = caches
        self.h, self.block, width = caches[0].shape[1:]
        self.d = width // 2
        self.half = self.h // 2
        self.q_per_k = q_heads // self.h
        if self.h % 2 or q_heads % self.h or self.d != 128:
            raise ValueError(
                "head_tp requires grouped query heads, even KV heads and head_dim=128"
            )
        for cache in caches:
            if (
                cache.shape != caches[0].shape
                or not cache.is_contiguous()
                or cache.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "head_tp needs BF16 contiguous KV; set VLLM_KV_CACHE_LAYOUT=LBHNC"
                )
        self.device = caches[0].device
        self.rank = rank
        self.max_len = max_len
        if capacity < 1:
            raise ValueError("head_tp.max_context must be positive")
        self.fa_version = (
            3 if torch.cuda.get_device_capability(self.device)[0] == 9 else 2
        )
        self.limit = split_limit if self.fa_version == 3 else 1
        if not 1 <= self.limit <= 128:
            raise ValueError("head_tp.max_splits must be in [1,128]")
        self.policy = policy
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.enabled = False
        self.step = 0
        self.resident = [None, None]
        self.physical = [[], []]
        self.address = [None, None]
        self.mixed = False
        self.group = device_group
        self.cpu_group = cpu_group
        self.columns = cdiv(self.max_len, self.block)
        rows = max_rows * self.h + self.half
        self.table = torch.zeros(
            rows, self.columns, dtype=torch.int32, device=self.device
        )
        self.used = torch.ones(rows, dtype=torch.int32, device=self.device)
        self.splits = torch.ones_like(self.used)
        # Non-causal GQA metadata: Q lengths, optional dynamic splits, and
        # one mutable tile counter. Graph buckets share this stable allocation.
        rounded = cdiv(rows, 4) * 4
        self.metadata_buffer = (
            self.used.new_empty(rounded * (1 + int(self.limit > 1)) + 1)
            if self.fa_version == 3
            else None
        )
        self.scheduler_metadata = None
        self.cu_q = torch.arange(rows + 1, dtype=torch.int32, device=self.device)
        self.row = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_mask = torch.zeros(max_rows, dtype=torch.bool, device=self.device)
        self.remote_pages = torch.arange(
            self.half, dtype=torch.long, device=self.device
        )
        self.offset = torch.zeros(1, dtype=torch.long, device=self.device)
        self.owner_pages = self.remote_pages.clone()
        self.owner_offset = self.offset.clone()
        self.has_owner = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.write_mask = self.local_mask
        self.incoming = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.head_ids = torch.arange(self.h, device=self.device, dtype=torch.int32)

    def change_layout(self, action, owner, req_id, packed, dense, position):
        """Acknowledge migration before the allocator recycles source pages."""
        old = self.physical[owner]
        if action in ("pack", "unpack") and position:
            # A physical page owns 2 consecutive logical blocks per local head.
            # Read the whole chunk before writing because unpack can reuse the
            # same physical IDs for the even logical blocks.
            count = cdiv(position, self.block)
            ids = packed if action == "pack" else old
            chunk = max(
                1,
                (4 << 20)
                // (self.h * self.block * 2 * self.d * self.caches[0].element_size()),
            )
            for begin in range(0, cdiv(count, 2), chunk):
                end = min(begin + chunk, cdiv(count, 2))
                logical_end = min(2 * end, count)
                n = logical_end - 2 * begin
                page_ids = torch.tensor(
                    ids[begin:end], device=self.device, dtype=torch.long
                )
                if self.rank == owner:
                    dense_ids = torch.tensor(
                        dense[2 * begin : logical_end],
                        device=self.device,
                        dtype=torch.long,
                    )
                for cache in self.caches:
                    if action == "pack":
                        if self.rank == owner:
                            full = cache.index_select(0, dense_ids)
                            upper = full[:, self.half :].contiguous()
                            local = full[:, : self.half]
                        else:
                            upper = cache.new_empty(
                                n, self.half, self.block, 2 * self.d
                            )
                            local = upper
                        dist.broadcast(upper, group=self.group, group_src=owner)
                        chunks = cache.new_zeros(
                            2 * (end - begin), self.half, self.block, 2 * self.d
                        )
                        chunks[:n].copy_(local)
                        packed_kv = chunks.view(
                            end - begin, 2, self.half, self.block, 2 * self.d
                        ).transpose(1, 2)
                        cache.index_copy_(
                            0,
                            page_ids,
                            packed_kv.reshape(
                                end - begin, self.h, self.block, 2 * self.d
                            ),
                        )
                    else:
                        chunks = cache.index_select(0, page_ids).view(
                            end - begin, self.half, 2, self.block, 2 * self.d
                        )
                        local = chunks.transpose(1, 2).reshape(
                            -1, self.half, self.block, 2 * self.d
                        )[:n]
                        upper = (
                            torch.empty_like(
                                local, memory_format=torch.contiguous_format
                            )
                            if owner == self.rank
                            else local.contiguous()
                        )
                        dist.broadcast(upper, group=self.group, group_src=1 - owner)
                        if owner == self.rank:
                            cache.index_copy_(
                                0, dense_ids, torch.cat((local, upper), dim=1)
                            )
        leaving = action in ("drop", "unpack")
        self.resident[owner] = None if leaving else req_id
        self.physical[owner] = [] if leaving else packed
        self.address[owner] = self.packed_table(self.physical[owner])

    def packed_table(self, ids):
        physical = torch.tensor(ids, dtype=torch.int32, device=self.device)
        # slot = physical_page * H + 2 * local_head + logical_block % 2
        return (
            physical[:, None, None] * self.h
            + self.head_ids[:2][None, :, None]
            + 2 * self.head_ids[None, None, : self.half]
        ).reshape(-1, self.half)

    def stage(self, rows, seq_lens, b, table, *, counts=None, mixed=False):
        self.step += 1
        peer = 1 - self.rank
        idx = next(
            (i for i, r in enumerate(rows) if r.req_id == self.resident[self.rank]),
            None,
        )
        pos = rows[idx].position if idx is not None else -1
        counts = [1] * len(rows) if counts is None else counts
        nq = counts[idx] if idx is not None else 0
        routes = [None, None]
        dist.all_gather_object(routes, (pos, nq), group=self.cpu_group)
        peer_pos, peer_q = routes[peer]
        if peer_q and self.resident[peer] is None:
            raise RuntimeError("peer is using an unallocated head shard")
        self.mixed = mixed
        if mixed:
            self.mixed_args = rows, counts, idx, table, peer_pos, peer_q
            self.write_mask = torch.zeros(b, dtype=torch.bool, device=self.device)
            if idx is not None:
                begin = sum(counts[:idx])
                self.write_mask[begin : begin + nq] = True
            return
        self.row.fill_(idx if idx is not None else 0)
        self.local_mask.zero_()
        self.write_mask = self.local_mask
        local_groups = b * self.h
        mapping = self.table[:local_groups].view(b, self.h, self.columns)
        mapping.zero_()
        local_used = self.used[:local_groups].view(b, self.h)
        local_used.zero_()
        if rows:
            width = min(table.shape[1], self.columns)
            mapping[: len(rows), :, :width].copy_(
                table[: len(rows), None, :width] * self.h + self.head_ids[None, :, None]
            )
            local_used[: len(rows)].copy_(seq_lens[: len(rows), None])
        if idx is not None:
            n = cdiv(pos + 1, self.block)
            address = self.address[self.rank]
            mapping[idx, : self.half, :n].copy_(address[:n].T)
            local_used[idx, self.half :] = 0
            self.local_mask[idx] = True
            self.owner_pages.copy_(address[pos // self.block])
            self.owner_offset.fill_(pos % self.block)
        self.has_owner.fill_(idx is not None)
        self.used[local_groups : local_groups + self.half].fill_(
            peer_pos + 1 if peer_q else 0
        )
        self.incoming.fill_(bool(peer_q))
        if peer_q:
            n = cdiv(peer_pos + 1, self.block)
            address = self.address[peer]
            self.table[local_groups : local_groups + self.half, :n].copy_(address[:n].T)
            self.remote_pages.copy_(address[peer_pos // self.block])
            self.offset.fill_(peer_pos % self.block)
        lengths = [r.position + 1 for r in rows] + [max(1, peer_pos + 1)]
        heads = [self.half if i == idx else self.h for i in range(len(rows))] + [
            self.half
        ]
        counts = self.policy.splits(
            lengths,
            heads,
            block_n=128,
            sms=self.sms,
            limit=self.limit,
            rank=self.rank,
            step=self.step,
        )
        if len(counts) != len(lengths) or any(
            not isinstance(n, Integral) or not 1 <= n <= self.limit for n in counts
        ):
            raise ValueError("head_tp policy returned invalid split counts")
        splits = [n for n in counts[:-1] for _ in range(self.h)]
        splits += [1] * ((b - len(rows)) * self.h) + [counts[-1]] * self.half
        self.splits[: len(splits)].copy_(torch.tensor(splits, dtype=torch.int32))

        self.prepare_decode_metadata(b)

    def prepare_decode_metadata(self, batch_size):
        """Stage one rank-local FA3 launch plan before executing any layer."""
        if self.metadata_buffer is None:
            return
        from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

        n = batch_size * self.h + self.half
        meta = get_scheduler_metadata(
            batch_size=n,
            max_seqlen_q=1,
            max_seqlen_k=self.max_len,
            num_heads_q=self.q_per_k,
            num_heads_kv=1,
            headdim=self.d,
            cache_seqlens=self.used[:n],
            cu_seqlens_q=self.cu_q[: n + 1],
            page_size=self.block,
            num_splits=self.limit,
            row_split_counts=self.splits[:n],
        )
        self.scheduler_metadata = self.metadata_buffer[: meta.numel()]
        self.scheduler_metadata.copy_(meta)

    def mixed_forward(self, q, k, v, cache, output, scale):
        """Chunked causal prefill and mixed batches with bounded Q/K/V exchange."""
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        rows, counts, idx, table, peer_pos, peer_n = self.mixed_args
        n = counts[idx] if idx is not None else 0
        start = sum(counts[:idx]) if idx is not None else 0
        h, g, d = self.h, self.q_per_k, self.d
        width = max(n, peer_n, 1)
        grouped = q.view(-1, h, g, d)
        packet = q.new_zeros(width, self.half, g + 2, d)
        if n:
            sl = slice(start, start + n)
            packet[:n, :, :g] = grouped[sl, self.half :]
            packet[:n, :, g] = k[sl, self.half :]
            packet[:n, :, g + 1] = v[sl, self.half :]
        remote = self._exchange(packet)
        pages = cache.view(-1, self.block, 2 * d)
        key, value = pages.unsqueeze(2).split(d, dim=-1)

        def attend(query, address, pos, kv=None):
            nq, heads = query.shape[:2]
            if kv is not None:
                positions = torch.arange(pos, pos + nq, device=self.device)
                pages[
                    address[positions // self.block].long(),
                    (positions % self.block)[:, None],
                ] = kv
            queries = query.transpose(0, 1).reshape(heads * nq, g, d).contiguous()
            cu = torch.arange(heads + 1, device=self.device, dtype=torch.int32) * nq
            out = flash_attn_varlen_func(
                queries,
                key,
                value,
                max_seqlen_q=nq,
                cu_seqlens_q=cu,
                max_seqlen_k=pos + nq,
                seqused_k=torch.full(
                    (heads,), pos + nq, device=self.device, dtype=torch.int32
                ),
                block_table=address[: cdiv(pos + nq, self.block)].T.clone(
                    memory_format=torch.contiguous_format
                ),
                softmax_scale=scale,
                causal=True,
                fa_version=self.fa_version,
            )
            return out.view(heads, nq, g, d).transpose(0, 1)

        output.zero_()
        dest = output.view(-1, h, g, d)
        offset = 0
        for i, (r, num) in enumerate(zip(rows, counts)):
            sl = slice(offset, offset + num)
            if i == idx:
                dest[sl, : self.half] = attend(
                    grouped[sl, : self.half],
                    self.address[self.rank],
                    r.position,
                    torch.cat((k[sl, : self.half], v[sl, : self.half]), dim=-1),
                )
            else:
                addr = table[i, :, None] * h + self.head_ids
                dest[sl] = attend(grouped[sl], addr, r.position)
            offset += num
        returns = q.new_zeros(width, self.half, g, d)
        if peer_n:
            returns[:peer_n] = attend(
                remote[:peer_n, :, :g],
                self.address[1 - self.rank],
                peer_pos,
                remote[:peer_n, :, g:].reshape(peer_n, self.half, 2 * d),
            )
        recv = self._exchange(returns)
        if n:
            dest[start : start + n, self.half :] = recv[:n]
        return output

    def _exchange(self, data):
        """Both ranks execute the same two NCCL collectives in every layer."""
        gathered = data.new_empty((2 * data.shape[0], *data.shape[1:]))
        dist.all_gather_into_tensor(gathered, data.contiguous(), group=self.group)
        return gathered.chunk(2)[1 - self.rank]

    def paged_decode(self, queries, key, value, scale):
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        n = queries.shape[0]
        return flash_attn_varlen_func(
            queries,
            key,
            value,
            max_seqlen_q=1,
            cu_seqlens_q=self.cu_q[: n + 1],
            max_seqlen_k=self.max_len,
            seqused_k=self.used[:n],
            block_table=self.table[:n],
            softmax_scale=scale,
            causal=False,
            fa_version=self.fa_version,
            num_splits=self.limit,
            scheduler_metadata=self.scheduler_metadata,
            prepared_row_splits=self.fa_version == 3,
        )

    def forward(self, q, k, v, cache, output, scale):
        if self.mixed:
            return self.mixed_forward(q, k, v, cache, output, scale)
        b = q.shape[0]
        h, half, d, g = self.h, self.half, self.d, self.q_per_k
        grouped = q.view(b, h, g, d)
        packet = torch.cat(
            (
                grouped.index_select(0, self.row)[0, half:],
                k.index_select(0, self.row)[0, half:, None],
                v.index_select(0, self.row)[0, half:, None],
            ),
            dim=1,
        )
        remote = self._exchange(packet)
        pages = cache.view(-1, self.block, 2 * d)
        own = torch.cat(
            (
                k.index_select(0, self.row)[0, :half],
                v.index_select(0, self.row)[0, :half],
            ),
            dim=-1,
        )
        pages[self.owner_pages, self.owner_offset] = torch.where(
            self.has_owner, own, pages[self.owner_pages, self.owner_offset]
        )
        kv = remote[:, g:].reshape(half, 2 * d)
        previous = pages[self.remote_pages, self.offset]
        pages[self.remote_pages, self.offset] = torch.where(self.incoming, kv, previous)
        key, value = pages.unsqueeze(2).split(d, dim=-1)
        queries = torch.cat((grouped.flatten(0, 1), remote[:, :g]), dim=0)
        attn = self.paged_decode(queries, key, value, scale)
        returned = self._exchange(attn[-half:])
        local = attn[: b * h].view(b, h, g, d)
        upper = torch.where(
            self.local_mask[:b, None, None, None], returned, local[:, half:]
        )
        output.view(b, h, g, d).copy_(torch.cat((local[:, :half], upper), dim=1))
        return output
