"""Continuous-segment paged layout for decode CP.

Does not touch the head-TP operator. A long row is split on a page boundary.
Each side keeps whole original pages (``index_select``), compacted so token 0
of that cache is the first token of the segment. This is not vLLM interleaved
DCP, and it does not gather individual tokens.
"""

from __future__ import annotations

import torch

BLOCK = 16
KV = 2
Q = 16
DIM = 256
DTYPE = torch.bfloat16


def page_split(seq: int, owner_percent: int = 50) -> tuple[int, int]:
    """Round the requested owner share down to a page boundary."""
    if not 0 < owner_percent < 100:
        raise ValueError("a cooperative split requires an owner share between 0 and 100")
    left = (seq * owner_percent // (100 * BLOCK)) * BLOCK
    return left, seq - left


def segment_limits(seqs: list[int], n_long: int, owner_percent: int = 50) -> tuple[list[int], list[int]]:
    """CPU lengths for the master batch (long prefixes, then short full rows) and the partner suffixes."""
    master = []
    partner = []
    for row, seq in enumerate(seqs):
        if row < n_long:
            left, right = page_split(seq, owner_percent)
            master.append(left)
            partner.append(right)
        else:
            master.append(seq)
    return master, partner


def _slice_pages(cache: torch.Tensor, table_row: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Whole pages covering ``[start, start + length)``. ``start`` is a multiple of BLOCK."""
    if length <= 0:
        return cache.new_zeros(0, BLOCK, KV, DIM)
    if start % BLOCK != 0:
        raise ValueError(f"page slice start {start} is not a multiple of {BLOCK}")
    count = (length + BLOCK - 1) // BLOCK
    ids = table_row[start // BLOCK:start // BLOCK + count].long()
    return cache.index_select(0, ids)


def make_full_cache(seqs: list[int], device, seed: int = 11, *, shape=(Q, KV, DIM)):
    q_heads, kv_heads, head_dim = shape
    counts = [(seq + BLOCK - 1) // BLOCK for seq in seqs]
    total = sum(counts)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.randn(total, BLOCK, kv_heads, head_dim, dtype=DTYPE, device=device, generator=gen)
    v = torch.randn(total, BLOCK, kv_heads, head_dim, dtype=DTYPE, device=device, generator=gen)
    q = torch.randn(len(seqs), q_heads, head_dim, dtype=DTYPE, device=device, generator=gen)
    table = torch.zeros(len(seqs), max(counts) if counts else 1, dtype=torch.int32, device=device)
    cursor = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(cursor, cursor + count, device=device, dtype=torch.int32)
        cursor += count
    seqused = torch.tensor(seqs, dtype=torch.int32, device=device)
    return q, k, v, table, seqused


class SegmentBatch:
    """One rank's compacted view of a mixed batch.

    The partner holds routed suffixes followed by its own local rows.
    """

    def __init__(self, q, k, v, table, seqused, which: str, lengths=None, n_long: int = 0):
        self.q = q
        self.k = k
        self.v = v
        self.table = table
        self.seqused = seqused
        self.which = which  # "full" | "master" | "partner"
        # CPU lengths of the KV this rank holds, in row order. ``n_long`` is
        # how many of those rows are the long piece and may be tiled.
        self.lengths = list(lengths or [])
        self.n_long = n_long
        self.local_tail = 0
        self.offsets = {}


def relocate_kv(segment: SegmentBatch, device) -> SegmentBatch:
    """Move this rank's KV pages onto ``device``. Q is not part of the route.

    Decode communication does not carry K or V. Call this when the segment is
    first placed, then leave the pages there.
    """
    if isinstance(device, int):
        target = torch.device("cuda", device)
    else:
        target = torch.device(device)
    if segment.k.device == target:
        return segment
    from cuda_ipc_p2p import copy_tensor

    result = SegmentBatch(
        segment.q,
        copy_tensor(segment.k, target),
        copy_tensor(segment.v, target),
        copy_tensor(segment.table, target),
        copy_tensor(segment.seqused, target),
        segment.which,
        segment.lengths,
        segment.n_long,
    )
    result.local_tail = segment.local_tail
    return result


def full_batch(lengths, seed, device="cpu", *, shape=(Q, KV, DIM)):
    return SegmentBatch(*make_full_cache(lengths, device, seed, shape=shape), "full", lengths)


def _pack_rows(full, n_long, which, local=None, *, owner_percent=50):
    """Pack CP segments and optional partner-owned rows into one FA batch."""
    lengths, suffixes = segment_limits(full.lengths, n_long, owner_percent)
    master = which == "master"
    local_lengths = lengths if master else suffixes
    offsets = [0] * len(lengths) if master else lengths[:n_long]
    rows = [(full, i, offset, length)
            for i, (offset, length) in enumerate(zip(offsets, local_lengths))]
    if local is not None:
        rows.extend((local, i, 0, length) for i, length in enumerate(local.lengths))

    lengths = [length for _, _, _, length in rows]
    counts = [(length + BLOCK - 1) // BLOCK for length in lengths]
    total = sum(counts) + int(0 in counts)
    k = full.k.new_zeros((total, BLOCK, KV, DIM))
    v = torch.empty_like(k)
    table = full.table.new_zeros((len(rows), max(1, max(counts, default=0))))
    cursor = 0
    for i, ((source, row, offset, length), count) in enumerate(zip(rows, counts)):
        if count == 0:
            table[i, 0] = total - 1
            v[-1].zero_()
            continue
        k[cursor:cursor + count] = _slice_pages(source.k, source.table[row], offset, length)
        v[cursor:cursor + count] = _slice_pages(source.v, source.table[row], offset, length)
        table[i, :count] = torch.arange(cursor, cursor + count, device=table.device)
        cursor += count
    q = full.q if master else full.q[:n_long]
    if local is not None:
        q = torch.cat((q, local.q))
    segment = SegmentBatch(q, k, v, table, full.seqused.new_tensor(lengths),
                           which, lengths, n_long)
    segment.local_tail = len(local.lengths) if local is not None else 0
    return segment


def cp_segments(full, n_long, partner_local=None, *, owner_percent=50):
    if not 0 <= n_long <= len(full.lengths):
        raise ValueError("invalid number of cooperative rows")
    master = _pack_rows(full, n_long, "master", owner_percent=owner_percent)
    partner = _pack_rows(full, n_long, "partner", partner_local, owner_percent=owner_percent)
    mask = torch.zeros(2, len(full.lengths), dtype=torch.int32)
    mask[0] = 1
    mask[1, :n_long] = 1
    return master, partner, mask


def build_segments(seqs: list[int], n_long: int, device, seed: int = 11):
    """Return ``(full, master, partner, mask)``.

    ``mask`` is ``[2, batch]`` on CPU. Master (rank 0) owns every row. Partner
    (rank 1) owns only the long-row suffix, so short rows stay out of the route.
    """
    full = full_batch(seqs, seed, device)
    return full, *cp_segments(full, n_long)
