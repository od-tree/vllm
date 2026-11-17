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


def page_split(seq: int) -> tuple[int, int]:
    """Split on a page boundary. The prefix is the largest multiple of 16 not above ``seq // 2``."""
    left = (seq // 2 // BLOCK) * BLOCK
    return left, seq - left


def segment_limits(seqs: list[int], n_long: int) -> tuple[list[int], list[int]]:
    """CPU lengths for the master batch (long prefixes, then short full rows) and the partner suffixes."""
    master = []
    partner = []
    for row, seq in enumerate(seqs):
        left, right = page_split(seq)
        if row < n_long:
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


def make_full_cache(seqs: list[int], device, seed: int = 11):
    counts = [(seq + BLOCK - 1) // BLOCK for seq in seqs]
    total = sum(counts)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.randn(total, BLOCK, KV, DIM, dtype=DTYPE, device=device, generator=gen)
    v = torch.randn(total, BLOCK, KV, DIM, dtype=DTYPE, device=device, generator=gen)
    q = torch.randn(len(seqs), Q, DIM, dtype=DTYPE, device=device, generator=gen)
    table = torch.zeros(len(seqs), max(counts) if counts else 1, dtype=torch.int32, device=device)
    cursor = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(cursor, cursor + count, device=device, dtype=torch.int32)
        cursor += count
    seqused = torch.tensor(seqs, dtype=torch.int32, device=device)
    return q, k, v, table, seqused


class SegmentBatch:
    """One rank's compacted view of a mixed batch.

    The master holds every row. The partner holds long-row suffixes only.
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

    return SegmentBatch(
        segment.q,
        copy_tensor(segment.k, target),
        copy_tensor(segment.v, target),
        copy_tensor(segment.table, target),
        copy_tensor(segment.seqused, target),
        segment.which,
        segment.lengths,
        segment.n_long,
    )


def _pack_rows(q, k, v, table, seqs, n_long, which: str, device) -> SegmentBatch:
    """Master keeps every row. Partner keeps long-row suffixes only.

    Long rows are cut with ``page_split``. Pages are sliced from the original
    cache, not copied token by token. A zero-length piece points at one zero
    page and has ``seqused`` 0.
    """
    rows = range(len(seqs)) if which == "master" else range(n_long)
    pages_k = []
    pages_v = []
    counts = []
    lengths = []
    qs = []
    for row in rows:
        seq = seqs[row]
        left, right = page_split(seq)
        if which == "partner":
            start, length = left, right
        elif row < n_long:
            start, length = 0, left
        else:
            start, length = 0, seq
        pages_k.append(_slice_pages(k, table[row], start, length))
        pages_v.append(_slice_pages(v, table[row], start, length))
        counts.append(pages_k[-1].shape[0])
        lengths.append(length)
        qs.append(q[row])
    if not qs:
        empty_q = q[:0]
        empty_k = k.new_zeros(0, BLOCK, KV, DIM)
        empty_table = torch.zeros(0, 1, dtype=torch.int32, device=device)
        empty_used = torch.zeros(0, dtype=torch.int32, device=device)
        return SegmentBatch(empty_q, empty_k, empty_k, empty_table, empty_used, which, [], 0)
    nonempty_k = [pages for pages in pages_k if pages.shape[0] > 0]
    nonempty_v = [pages for pages in pages_v if pages.shape[0] > 0]
    if nonempty_k:
        cache_k = torch.cat(nonempty_k, dim=0)
        cache_v = torch.cat(nonempty_v, dim=0)
    else:
        cache_k = k.new_zeros(0, BLOCK, KV, DIM)
        cache_v = v.new_zeros(0, BLOCK, KV, DIM)
    zero_id = None
    if any(count == 0 for count in counts):
        zero_id = cache_k.shape[0]
        cache_k = torch.cat((cache_k, k.new_zeros(1, BLOCK, KV, DIM)), dim=0)
        cache_v = torch.cat((cache_v, v.new_zeros(1, BLOCK, KV, DIM)), dim=0)
    width = max(counts) if counts else 1
    if width == 0:
        width = 1
    seg_table = torch.zeros(len(counts), width, dtype=torch.int32, device=device)
    cursor = 0
    for row, count in enumerate(counts):
        if count == 0:
            seg_table[row, 0] = zero_id
            continue
        seg_table[row, :count] = torch.arange(cursor, cursor + count, device=device, dtype=torch.int32)
        cursor += count
    return SegmentBatch(
        torch.stack(qs, dim=0),
        cache_k,
        cache_v,
        seg_table,
        torch.tensor(lengths, dtype=torch.int32, device=device),
        which,
        lengths,
        n_long if which == "master" else len(lengths),
    )


def build_segments(seqs: list[int], n_long: int, device, seed: int = 11):
    """Return ``(full, master, partner, mask)``.

    ``mask`` is ``[2, batch]`` on CPU. Master (rank 0) owns every row. Partner
    (rank 1) owns only the long-row suffix, so short rows stay out of the route.
    """
    q, k, v, table, seqused = make_full_cache(seqs, device, seed)
    full = SegmentBatch(q, k, v, table, seqused, "full", list(seqs), 0)
    master = _pack_rows(q, k, v, table, seqs, n_long, "master", device)
    partner = _pack_rows(q, k, v, table, seqs, n_long, "partner", device)
    mask = torch.zeros(2, len(seqs), dtype=torch.int32)
    mask[0] = 1
    if n_long:
        mask[1, :n_long] = 1
    return full, master, partner, mask
