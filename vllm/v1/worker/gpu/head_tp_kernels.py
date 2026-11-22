# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token head-TP staging for BF16 LBHNC cache and NCCL packets."""

from vllm.triton_utils import tl, triton


@triton.jit
def _write_kv(
    K,
    V,
    Cache,
    Slots,
    OwnerRow,
    OwnerPages,
    OwnerOffset,
    HasOwner,
    NUM_SLOTS,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if (row == tl.load(OwnerRow)) & tl.load(HasOwner):
        if head >= H // 2:
            return
        page = tl.load(OwnerPages + head).to(tl.int64)
        offset = tl.load(OwnerOffset)
    else:
        slot = tl.load(Slots + row, mask=row < NUM_SLOTS, other=-1).to(tl.int64)
        if slot < 0:
            return
        page = (slot // B) * H + head
        offset = slot % B
    dim = tl.arange(0, D)
    dest = Cache + (page * B + offset) * (2 * D) + dim
    tl.store(dest, tl.load(K + row * KS0 + head * KS1 + dim * KS2))
    tl.store(dest + D, tl.load(V + row * VS0 + head * VS1 + dim * VS2))


@triton.jit
def _prepare_queries_and_packet(
    Q,
    K,
    V,
    Queries,
    Packet,
    OwnerRow,
    HasOwner,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    TILE: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, TILE)
    q = tl.load(
        Q + row * QS0 + (head * G + i // D) * QS1 + (i % D) * QS2,
        mask=i < G * D,
        other=0,
    )
    tl.store(Queries + (row * H + head) * G * D + i, q, mask=i < G * D)
    if (row == tl.load(OwnerRow)) & (head >= H // 2):
        active = tl.load(HasOwner)
        dest = Packet + (head - H // 2) * (G + 2) * D
        tl.store(dest + i, tl.where(active, q, 0), mask=i < G * D)
        dim = tl.arange(0, D)
        k = tl.load(K + row * KS0 + head * KS1 + dim * KS2, mask=active, other=0)
        v = tl.load(V + row * VS0 + head * VS1 + dim * VS2, mask=active, other=0)
        tl.store(dest + G * D + dim, k)
        tl.store(dest + (G + 1) * D + dim, v)


@triton.jit
def _receive(
    Packet,
    RemoteQueries,
    Cache,
    Pages,
    Offset,
    Incoming,
    B: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0)
    src = Packet + head * (G + 2) * D
    active = tl.load(Incoming)
    i = tl.arange(0, TILE)
    q = tl.load(src + i, mask=active & (i < G * D), other=0)
    tl.store(RemoteQueries + head * G * D + i, q, mask=i < G * D)
    if active:
        page = tl.load(Pages + head).to(tl.int64)
        dim = tl.arange(0, 2 * D)
        dest = (page * B + tl.load(Offset)) * (2 * D)
        tl.store(Cache + dest + dim, tl.load(src + G * D + dim))


def write_kv(k, v, cache, slots, decode):
    _write_kv[(k.size(0), decode.h)](
        k,
        v,
        cache,
        slots,
        decode.row,
        decode.owner_pages,
        decode.owner_offset,
        decode.has_owner,
        slots.numel(),
        decode.h,
        decode.d,
        decode.block,
        *k.stride(),
        *v.stride(),
    )


def prepare_queries_and_packet(q, k, v, queries, packet, decode):
    _prepare_queries_and_packet[(q.size(0), decode.h)](
        q,
        k,
        v,
        queries,
        packet,
        decode.row,
        decode.has_owner,
        decode.h,
        decode.q_per_k,
        decode.d,
        triton.next_power_of_2(decode.q_per_k * decode.d),
        *q.stride(),
        *k.stride(),
        *v.stride(),
    )


def receive(packet, remote_queries, cache, decode):
    _receive[(decode.half,)](
        packet,
        remote_queries,
        cache,
        decode.remote_pages,
        decode.offset,
        decode.incoming,
        decode.block,
        decode.q_per_k,
        decode.d,
        triton.next_power_of_2(decode.q_per_k * decode.d),
    )
