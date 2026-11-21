# SPDX-License-Identifier: Apache-2.0
"""Two-worker head-sharded GQA for single-token Qwen3 MoE decode.

The owner always stores the full KV for its requests. A worker leases part of
the regular block pool for its peer's upper KV heads. Virtual FA rows address
one physical KV head each, including ordinary local requests.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.decode_tiles import minimax_split_counts

logger = init_logger(__name__)


@dataclass(frozen=True)
class DecodeRow:
    req_id: str
    position: int  # number of valid prior KV tokens


class DecodePolicy:
    """Called outside capture on every decode step. None keeps all rows local."""

    def __init__(self, options):
        self.threshold = int(options.get("min_context", 0))
        self.tile = options.get("tile", "minimax")
        if self.tile not in ("minimax", "one"):
            raise ValueError("head_tp.tile must be minimax or one")

    def select(
        self,
        rows: Sequence[DecodeRow],
        *,
        peer_positions: Sequence[int],
        rank: int,
        step: int,
    ) -> str | None:
        candidates = (r for r in rows if r.position >= self.threshold)
        row = max(candidates, key=lambda r: r.position, default=None)
        return row.req_id if row else None

    def splits(
        self,
        lengths: list[int],
        heads: list[int],
        *,
        block_n: int,
        sms: int,
        limit: int,
        rank: int,
        step: int,
    ) -> Sequence[int]:
        if self.tile == "one":
            return [1] * len(lengths)
        blocks = [max(1, cdiv(n, block_n)) for n in lengths]
        return minimax_split_counts(blocks, heads, max(sum(heads), sms), limit)


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
        self.capacity = min(max_len, capacity)
        if self.capacity < 1:
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
        self.previous = None
        self.reserved = None
        self.group = device_group
        self.cpu_group = cpu_group
        self.columns = cdiv(self.max_len, self.block)
        rows = max_rows * self.h + self.half
        self.table = torch.zeros(
            rows, self.columns, dtype=torch.int32, device=self.device
        )
        self.used = torch.ones(rows, dtype=torch.int32, device=self.device)
        self.splits = torch.ones_like(self.used)
        self.cu_q = torch.arange(rows + 1, dtype=torch.int32, device=self.device)
        self.row = torch.zeros(1, dtype=torch.long, device=self.device)
        self.local_mask = torch.zeros(max_rows, dtype=torch.bool, device=self.device)
        self.remote_pages = torch.arange(
            self.half, dtype=torch.long, device=self.device
        )
        self.offset = torch.zeros(1, dtype=torch.long, device=self.device)
        self.incoming = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.head_ids = torch.arange(self.h, device=self.device, dtype=torch.int32)

    def set_pages(self, block_ids):
        pages = torch.tensor(block_ids, dtype=torch.int32, device=self.device)
        # Pack two logical half-head pages into each allocator block.
        self.reserved = (pages[:, None] * self.h + self.head_ids).reshape(-1, self.half)

    def stage(self, rows, seq_lens, b, table, new_requests):
        """Agree on both routes before touching caches or replaying any graph."""
        self.step += 1
        eligible = [
            r for r in rows if self.reserved is not None and r.position < self.capacity
        ]
        positions = [None, None]
        dist.all_gather_object(
            positions, tuple(r.position for r in rows), group=self.cpu_group
        )
        selected = self.policy.select(
            eligible,
            peer_positions=positions[1 - self.rank],
            rank=self.rank,
            step=self.step,
        )
        idx = next((i for i, r in enumerate(rows) if r.req_id == selected), None)
        if selected is not None and (idx is None or rows[idx] not in eligible):
            raise ValueError("head_tp policy selected an ineligible row")
        pos = rows[idx].position if idx is not None else -1
        migrate = idx is not None and (
            self.previous != (selected, pos) or selected in new_requests
        )
        routes = [None, None]
        dist.all_gather_object(routes, (pos, migrate), group=self.cpu_group)
        peer_pos = routes[1 - self.rank][0]
        for owner, (position, copy_prefix) in enumerate(routes):
            if not copy_prefix or position <= 0:
                continue
            self._copy_prefix(owner, position, table, idx)
        self.previous = (selected, pos + 1) if selected is not None else None

        # The graph reads this stable storage even when the owner or row changes.
        self.row.fill_(idx if idx is not None else 0)
        self.local_mask.zero_()
        local_groups = b * self.h
        mapping = self.table[:local_groups].view(b, self.h, self.columns)
        mapping.zero_()
        local_used = self.used[:local_groups].view(b, self.h)
        local_used.zero_()
        if rows:
            n = len(rows)
            width = min(table.shape[1], self.columns)
            mapping[:n, :, :width].copy_(
                table[:n, None, :width] * self.h + self.head_ids[None, :, None]
            )
            local_used[:n].copy_(seq_lens[:n, None])
        if idx is not None:
            self.local_mask[idx] = True
            local_used[idx, self.half :] = 0
        self.used[local_groups : local_groups + self.half].fill_(peer_pos + 1)
        self.incoming.fill_(peer_pos >= 0)
        if peer_pos >= 0:
            pages = cdiv(peer_pos + 1, self.block)
            self.table[local_groups : local_groups + self.half, :pages].copy_(
                self.reserved[:pages].T
            )
            self.remote_pages.copy_(self.reserved[peer_pos // self.block])
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
            not isinstance(s, Integral) or not 1 <= s <= self.limit for s in counts
        ):
            raise ValueError("head_tp policy returned invalid split counts")
        splits = [s for s in counts[:-1] for _ in range(self.h)]
        splits += [1] * ((b - len(rows)) * self.h) + [counts[-1]] * self.half
        self.splits[: len(splits)].copy_(torch.tensor(splits, dtype=torch.int32))
        if migrate:
            logger.info(
                "head-TP rank %d joined request %s at position %d",
                self.rank,
                selected,
                pos,
            )

    def _copy_prefix(self, owner, position, table, idx):
        """Copy compact head pages with bounded temporary device memory."""
        assert self.reserved is not None
        n = cdiv(position, self.block)
        page_bytes = self.half * self.block * 2 * self.d * self.caches[0].element_size()
        chunk_pages = max(1, (4 << 20) // page_bytes)
        for start in range(0, n, chunk_pages):
            end = min(n, start + chunk_pages)
            if self.rank == owner:
                source = table[idx, start:end].long()
            else:
                destination = self.reserved[start:end].flatten().long()
            for cache in self.caches:
                if self.rank == owner:
                    payload = cache[source, self.half :].contiguous()
                else:
                    payload = cache.new_empty(
                        end - start, self.half, self.block, 2 * self.d
                    )
                dist.broadcast(payload, group=self.group, group_src=owner)
                if self.rank != owner:
                    pages = cache.view(-1, self.block, 2 * self.d)
                    pages.index_copy_(0, destination, payload.flatten(0, 1))

    def _exchange(self, data):
        """Both ranks execute the same two NCCL collectives in every layer."""
        gathered = data.new_empty((2 * data.shape[0], *data.shape[1:]))
        dist.all_gather_into_tensor(gathered, data.contiguous(), group=self.group)
        return gathered.chunk(2)[1 - self.rank]

    def forward(self, q, k, v, cache, output, scale):
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

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
        kv = remote[:, g:].reshape(half, 2 * d)
        previous = pages[self.remote_pages, self.offset]
        pages[self.remote_pages, self.offset] = torch.where(self.incoming, kv, previous)
        key, value = pages.unsqueeze(2).split(d, dim=-1)
        queries = torch.cat((grouped.flatten(0, 1), remote[:, :g]), dim=0)
        n = queries.shape[0]
        attn = flash_attn_varlen_func(
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
            row_split_counts=self.splits[:n] if self.fa_version == 3 else None,
        )
        returned = self._exchange(attn[-half:])
        local = attn[: b * h].view(b, h, g, d)
        upper = torch.where(
            self.local_mask[:b, None, None, None], returned, local[:, half:]
        )
        output.view(b, h, g, d).copy_(torch.cat((local[:, :half], upper), dim=1))
        return output
