# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load one partner's KV heads for Full Attention.

The destination blocks belong to that request on the helper. They are not
inserted into ``BlockPool`` and are not written back to the shared prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.head_tp import (
    HeadTPPartialCache,
    HeadTPStep,
    drop_head_tp_partial_cache,
    head_tp_partial_caches,
    register_head_tp_partial_cache,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.request import Request

_HEAD_MAJOR = "packed_head_major"
_TOKEN_MAJOR = "packed_token_major"
_BACKENDS = {
    "flash": _HEAD_MAJOR,
    "flash_attn": _HEAD_MAJOR,
    "packed_head_major": _HEAD_MAJOR,
    "triton": _TOKEN_MAJOR,
    "packed_token_major": _TOKEN_MAJOR,
}


def resolve_backend_layout(
    cache: torch.Tensor,
    backend: str,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[str, int]:
    """Head axis for one stored KV cache, checked against ``backend``."""

    if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
        raise ValueError(
            "unsupported kv cache shape "
            f"{tuple(cache.shape)} for head_dim {head_dim}"
        )
    if cache.shape[1] == num_kv_heads:
        layout, block = _HEAD_MAJOR, int(cache.shape[2])
    elif cache.shape[2] == num_kv_heads:
        layout, block = _TOKEN_MAJOR, int(cache.shape[1])
    else:
        raise ValueError(
            "kv cache shape "
            f"{tuple(cache.shape)} does not place {num_kv_heads} heads "
            "on axis 1 or axis 2"
        )
    if backend:
        expected = _BACKENDS.get(backend)
        if expected is None:
            raise ValueError(f"unknown kv backend {backend!r}")
        if expected != layout:
            raise ValueError(
                f"backend {backend} expects {expected}, cache is {layout} "
                f"{tuple(cache.shape)}"
            )
    return layout, block


def load_kv_range(
    source: Mapping[str, torch.Tensor],
    dest_cache: Mapping[str, torch.Tensor],
    block_ids: Sequence[int],
    kv_range: tuple[int, int],
    layers: Sequence[str],
    backend: str,
    *,
    num_kv_heads: int,
    head_dim: int,
    token_start: int = 0,
    token_end: int | None = None,
    block_size: int | None = None,
    src_block_ids: Sequence[int] | None = None,
) -> int:
    """Copy one KV-head interval of the listed Full Attention layers.

    ``block_ids`` is the destination blocks in token order. ``src_block_ids``
    defaults to the same ids. ``token_start`` and ``token_end`` are positions
    along that order. Returns the number of bytes written. Other heads, and
    layers that are not listed, are left untouched.
    """

    kv_start, kv_end = kv_range
    if kv_end < kv_start:
        raise ValueError(f"kv range {kv_range} is reversed")
    src_ids = list(block_ids if src_block_ids is None else src_block_ids)
    dst_ids = list(block_ids)
    if len(src_ids) != len(dst_ids):
        raise ValueError(
            f"{len(src_ids)} source blocks and {len(dst_ids)} dest blocks"
        )
    if not layers:
        return 0
    written = 0
    for layer in layers:
        if layer not in source or layer not in dest_cache:
            raise KeyError(f"full-attention layer {layer} is missing from the copy")
        src = source[layer]
        dst = dest_cache[layer]
        layout, inferred = resolve_backend_layout(
            src, backend, num_kv_heads, head_dim
        )
        dest_layout, _ = resolve_backend_layout(
            dst, backend, num_kv_heads, head_dim
        )
        if dest_layout != layout:
            raise ValueError(f"{layer}: source is {layout}, dest is {dest_layout}")
        block = inferred if block_size is None else block_size
        if block <= 0:
            raise ValueError(f"block size {block} is not positive")
        end = len(dst_ids) * block if token_end is None else token_end
        if end < token_start:
            raise ValueError(f"token range [{token_start}, {end}) is reversed")
        kv_per = kv_end - kv_start
        for logical, (src_block, dst_block) in enumerate(zip(src_ids, dst_ids)):
            block_lo = logical * block
            lo = max(token_start, block_lo)
            hi = min(end, block_lo + block)
            if lo >= hi:
                continue
            offset = lo - block_lo
            count = hi - lo
            if layout == _HEAD_MAJOR:
                dst[dst_block, kv_start:kv_end, offset : offset + count] = src[
                    src_block, kv_start:kv_end, offset : offset + count
                ]
            else:
                dst[dst_block, offset : offset + count, kv_start:kv_end] = src[
                    src_block, offset : offset + count, kv_start:kv_end
                ]
            written += count * kv_per * src.shape[-1] * src.element_size()
    return written


def ensure_partial_cache(
    req_id: str,
    example: torch.Tensor,
    num_tokens: int,
    block_size: int,
) -> HeadTPPartialCache:
    """Private blocks for one helper request. Not a prefix-hash entry."""

    need = max(1, cdiv(max(num_tokens, 1), block_size) + 1)
    current = head_tp_partial_caches().get(req_id)
    if current is not None and current.cache.shape[0] >= need:
        return current
    fresh = torch.zeros(
        (need, *example.shape[1:]),
        dtype=example.dtype,
        device=example.device,
    )
    if current is not None:
        fresh[: current.cache.shape[0]] = current.cache
        seq_len = current.seq_len
    else:
        seq_len = 0
    block_table = torch.arange(need, dtype=torch.int32, device=example.device)
    cache = HeadTPPartialCache(
        cache=fresh,
        block_table=block_table,
        seq_len=seq_len,
        slot_mapping=torch.empty(0, dtype=torch.int64),
    )
    register_head_tp_partial_cache(req_id, cache)
    return cache


class CPUPrefixLayer:
    """CPU-offload lookup and pin for one replicated shared region.

    ``replicated`` is false when workers do not map the same bytes. Lookup
    then returns ``None``, so pairing cannot pick ``cpu_pool``. ``HIT_PENDING``
    stops the scan and is not a hit. ``prepare_load`` pins; ``complete_load``
    runs when the plan goes away.
    """

    def __init__(
        self,
        manager: Any,
        *,
        block_size: int,
        replicated: bool,
        group_idx: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block size {block_size} is not positive")
        self.manager = manager
        self.block_size = block_size
        self.replicated = replicated
        self.group_idx = group_idx
        self._pinned: dict[str, list[OffloadKey]] = {}
        self._ctx: dict[str, ReqContext] = {}

    def pinned_request_ids(self) -> list[str]:
        return list(self._pinned)

    def lookup_hit_tokens(self, request: Request) -> int | None:
        if not self.replicated or not request.block_hashes:
            return None
        ctx = self._context(request)
        hits = 0
        with self.manager.lock():
            for block_hash in request.block_hashes:
                key = make_offload_key(bytes(block_hash), self.group_idx)
                if self.manager.lookup(key, ctx) is not LookupResult.HIT:
                    break
                hits += 1
        return hits * self.block_size

    def pin(self, request: Request) -> None:
        if not self.replicated or request.request_id in self._pinned:
            return
        ctx = self._context(request)
        keys: list[OffloadKey] = []
        with self.manager.lock():
            for block_hash in request.block_hashes:
                key = make_offload_key(bytes(block_hash), self.group_idx)
                if self.manager.lookup(key, ctx) is not LookupResult.HIT:
                    break
                keys.append(key)
            if keys:
                self.manager.prepare_load(keys, ctx)
        if keys:
            self._pinned[request.request_id] = keys

    def unpin(self, request_id: str) -> None:
        keys = self._pinned.pop(request_id, None)
        ctx = self._ctx.get(request_id)
        if not keys or ctx is None:
            return
        with self.manager.lock():
            self.manager.complete_load(keys, ctx)

    def _context(self, request: Request) -> ReqContext:
        ctx = self._ctx.get(request.request_id)
        if ctx is None:
            ctx = ReqContext(req_id=request.request_id)
            self._ctx[request.request_id] = ctx
        return ctx


def _prefix_pairs(step: HeadTPStep) -> list[tuple[int, int, str, int]]:
    pairs = []
    for req_id in sorted(step.plans):
        plan = step.plans[req_id]
        if plan.state != "pending_load":
            continue
        for shard_index, shard in enumerate(plan.shards):
            if shard_index == 0:
                continue
            pairs.append((plan.owner_rank, shard.rank, req_id, shard_index))
    pairs.sort()
    return pairs


def _layer_payload(cache: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"kv:{name}": tensor for name, tensor in cache.items()}


def _layers_from_payload(message: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key[3:]: value
        for key, value in message.items()
        if key.startswith("kv:") and isinstance(value, torch.Tensor)
    }


def exchange_prefix_loads(
    step: HeadTPStep | None,
    rank: int,
    *,
    dp_size: int,
    tracker: Any,
    transport: Any,
    sources: Mapping[str, Mapping[str, torch.Tensor]],
    block_ids: Mapping[str, Sequence[int]],
    layers: Sequence[str],
    backend: str,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> dict[str, int]:
    """Copy pending spans. The owner records the sum of helper byte counts.

    Every rank that appears in a ``pending_load`` shard must call this, in
    the same order. A rank with nothing to send still receives, so the owner
    is not left blocked. An empty span (nothing in flight) exchanges a header
    and does not mark the copy done.
    """

    if step is None or transport is None:
        return {}
    pairs = _prefix_pairs(step)
    if not pairs:
        return {}
    acks: dict[str, list[tuple[bool, int, int]]] = {}
    involved = {req_id for _, _, req_id, _ in pairs}
    for owner, helper, req_id, shard_index in pairs:
        plan = step.plans[req_id]
        shard = plan.shards[shard_index]
        if rank == owner:
            _owner_send(
                transport,
                helper,
                plan,
                shard.kv_range,
                tracker,
                sources,
                block_ids,
                layers,
                backend,
                num_kv_heads,
                head_dim,
                block_size,
            )
            ack = transport.recv(helper)
            acks.setdefault(req_id, []).append(
                (
                    bool(ack.get("ok")),
                    int(ack.get("up_to", 0)),
                    int(ack.get("bytes", 0)),
                )
            )
        elif rank == helper:
            message = transport.recv(owner)
            ack = _helper_recv(
                message,
                req_id,
                sources,
                layers,
                backend,
                num_kv_heads,
                head_dim,
                block_size,
            )
            transport.send(owner, ack)
    if tracker is None:
        return {}
    copied: dict[str, int] = {}
    for req_id in involved:
        plan = step.plans[req_id]
        if plan.owner_rank != rank:
            continue
        span = tracker.pending_span(req_id)
        if span is None:
            continue
        got = acks.get(req_id, [])
        expect = sum(1 for shard in plan.shards[1:])
        ok = len(got) == expect and all(item[0] for item in got)
        up_to = got[0][1] if got else span[0]
        tracker.mark_copy_done(req_id, up_to, ok=ok)
        if ok:
            copied[req_id] = sum(item[2] for item in got)
        else:
            drop_head_tp_partial_cache(req_id)
    return copied


def _owner_send(
    transport: Any,
    helper: int,
    plan: Any,
    kv_range: tuple[int, int],
    tracker: Any,
    sources: Mapping[str, Mapping[str, torch.Tensor]],
    block_ids: Mapping[str, Sequence[int]],
    layers: Sequence[str],
    backend: str,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    span = None if tracker is None else tracker.pending_span(plan.req_id)
    header: dict[str, Any] = {
        "req_id": plan.req_id,
        "ok": True,
        "empty": span is None,
        "token_start": 0 if span is None else span[0],
        "token_end": 0 if span is None else span[1],
        "kv_source": plan.kv_source,
        "kv_start": kv_range[0],
        "kv_end": kv_range[1],
        "src_block_ids": [],
        "layers": list(layers),
    }
    if span is None:
        transport.send(helper, header)
        return
    token_start, token_end = span
    ids = list(block_ids.get(plan.req_id, ()))
    need = cdiv(token_end, block_size) if token_end else 0
    if len(ids) < need:
        header["ok"] = False
        transport.send(helper, header)
        return
    header["src_block_ids"] = ids[:need]
    # cpu_pool is read on the helper from the replicated region. The owner
    # only sends the block ids. gpu_copy is staged here from the owner cache.
    if plan.kv_source == "cpu_pool":
        transport.send(helper, header)
        return
    source = sources.get(plan.kv_source)
    if source is None:
        header["ok"] = False
        header["src_block_ids"] = []
        transport.send(helper, header)
        return
    if plan.kv_source == "gpu_copy":
        try:
            staging = _stage_gpu_range(
                source,
                ids[:need],
                kv_range,
                layers,
                backend,
                num_kv_heads,
                head_dim,
                block_size,
                token_start,
                token_end,
            )
        except (KeyError, ValueError, RuntimeError):
            header["ok"] = False
            transport.send(helper, header)
            return
        header.update(_layer_payload(staging))
    transport.send(helper, header)


def _stage_gpu_range(
    source: Mapping[str, torch.Tensor],
    ids: Sequence[int],
    kv_range: tuple[int, int],
    layers: Sequence[str],
    backend: str,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    token_start: int,
    token_end: int,
) -> dict[str, torch.Tensor]:
    example = source[layers[0]]
    staging = {
        layer: torch.zeros(
            (len(ids), *example.shape[1:]),
            dtype=example.dtype,
            device=example.device,
        )
        for layer in layers
    }
    local_ids = list(range(len(ids)))
    load_kv_range(
        source,
        staging,
        local_ids,
        kv_range,
        layers,
        backend,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        token_start=token_start,
        token_end=token_end,
        block_size=block_size,
        src_block_ids=ids,
    )
    return staging


def _helper_recv(
    message: Mapping[str, Any],
    req_id: str,
    sources: Mapping[str, Mapping[str, torch.Tensor]],
    layers: Sequence[str],
    backend: str,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> dict[str, Any]:
    token_end = int(message.get("token_end", 0))
    ack = {"req_id": req_id, "ok": False, "up_to": token_end, "bytes": 0}
    if message.get("empty"):
        ack["ok"] = True
        ack["up_to"] = 0
        return ack
    if not message.get("ok", False):
        drop_head_tp_partial_cache(req_id)
        return ack
    kv_range = (int(message["kv_start"]), int(message["kv_end"]))
    src_ids = [int(block) for block in message.get("src_block_ids", ())]
    token_start = int(message["token_start"])
    local_ids = list(range(len(src_ids)))
    try:
        if message.get("kv_source") == "cpu_pool":
            source = sources.get("cpu_pool")
            if source is None:
                raise KeyError("cpu prefix tensor is not registered")
            src_block_ids: Sequence[int] = src_ids
        else:
            source = _layers_from_payload(message)
            if not source:
                raise KeyError("gpu copy arrived without kv tensors")
            src_block_ids = local_ids
        dest = _per_layer_dest(req_id, source, layers, token_end, block_size)
        nbytes = load_kv_range(
            source,
            dest,
            local_ids,
            kv_range,
            layers,
            backend,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            token_start=token_start,
            token_end=token_end,
            block_size=block_size,
            src_block_ids=src_block_ids,
        )
    except (KeyError, ValueError, RuntimeError):
        drop_head_tp_partial_cache(req_id)
        return ack
    for key, partial in head_tp_partial_caches().items():
        if key == req_id or key.startswith(f"{req_id}/"):
            partial.seq_len = token_end
    ack["ok"] = True
    ack["bytes"] = nbytes
    return ack


def _per_layer_dest(
    req_id: str,
    source: Mapping[str, torch.Tensor],
    layers: Sequence[str],
    num_tokens: int,
    block_size: int,
) -> dict[str, torch.Tensor]:
    """One private tensor per Full Attention layer, outside the prefix hash."""

    dest: dict[str, torch.Tensor] = {}
    for layer in layers:
        partial = ensure_partial_cache(
            f"{req_id}/{layer}",
            source[layer],
            num_tokens,
            block_size,
        )
        dest[layer] = partial.cache
    return dest
