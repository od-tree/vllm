"""Per-row KV tiles for decode CP.

Same rule as head-TP's tile plan, without the head cut. Each rank still
holds both KV heads. While the long heads do not fill the SMs, those rows
share about one wave of CTAs. Once they already fill the SMs, each head is
split again so the partial last wave scans a shorter piece. The count is
capped at 16 and at the number of FA2 KV blocks (64 tokens for head dim
256). Short rows stay one CTA per KV head. The plan is host data; the
caller uploads it before the timed step.
"""

from __future__ import annotations

KV_HEADS = 2
KV_TILE = 64
SPLIT_CAP = 16


def pack_tile(row: int, head: int, split: int) -> int:
    """Bits 0-15 row, 16-23 KV head, 24-31 split. Same layout as head-TP."""

    if not (0 <= row <= 0xFFFF and 0 <= head <= 0xFF and 0 <= split <= 0xFF):
        raise ValueError(f"tile ({row}, {head}, {split}) does not fit in one int32")
    return row | (head << 16) | (split << 24)


def _splits_for_full_waves(long_heads: int, num_sms: int, cap: int) -> int:
    """Match ``vllm.v1.head_tp._splits_for_full_waves``."""

    if long_heads <= 0 or num_sms <= 0:
        return 1
    cap = max(1, cap)
    best_splits = 1
    best_waves = (long_heads + num_sms - 1) // num_sms
    for splits in range(2, cap + 1):
        waves = (long_heads * splits + num_sms - 1) // num_sms
        if waves * best_splits < best_waves * splits:
            best_splits = splits
            best_waves = waves
    return best_splits


def plan_tiles(lengths: list[int], n_long: int, num_sms: int, num_kv_heads: int = KV_HEADS):
    """Return ``(tiles, row_splits, max_splits)`` for one rank's batch.

    ``lengths`` is the KV this rank holds, long rows first. ``n_long`` of
    those rows may be split. The rest are short and get one split.
    """

    n = len(lengths)
    n_long = max(0, min(n_long, n))
    if num_sms <= 0 or num_kv_heads <= 0 or n == 0:
        return [pack_tile(0, 0, 0)], [1], 1
    row_splits = [1] * n
    long_heads = n_long * num_kv_heads
    if n_long and long_heads < num_sms:
        total = sum(max(lengths[index], 1) for index in range(n_long))
        for index in range(n_long):
            seq = max(lengths[index], 1)
            raw = (num_sms * seq) // (total * num_kv_heads)
            n_blocks = max(1, (seq + KV_TILE - 1) // KV_TILE)
            row_splits[index] = min(SPLIT_CAP, n_blocks, max(1, raw))
    elif n_long:
        max_seq = max(max(lengths[index], 1) for index in range(n_long))
        max_blocks = max(1, (max_seq + KV_TILE - 1) // KV_TILE)
        chosen = _splits_for_full_waves(long_heads, num_sms, min(SPLIT_CAP, max_blocks))
        for index in range(n_long):
            seq = max(lengths[index], 1)
            n_blocks = max(1, (seq + KV_TILE - 1) // KV_TILE)
            raw = max(1, chosen * seq // max_seq)
            row_splits[index] = min(SPLIT_CAP, n_blocks, raw)
    tiles: list[int] = []
    for row in range(n_long):
        for split in range(row_splits[row]):
            for head in range(num_kv_heads):
                tiles.append(pack_tile(row, head, split))
    for row in range(n_long, n):
        for head in range(num_kv_heads):
            tiles.append(pack_tile(row, head, 0))
    if not tiles:
        tiles.append(pack_tile(0, 0, 0))
        row_splits[0] = 1
    return tiles, row_splits, max(row_splits)


def segment_tiles(segment):
    """CUDA tile plan for ``segment``, built once per device.

    ``None`` means this batch should use FA2's own split heuristic.
    """

    import torch

    device = segment.k.device
    if device.type != "cuda" or segment.q.shape[0] == 0 or not segment.lengths:
        return None
    cached = getattr(segment, "_tiles", None)
    if cached is not None and cached[0].device == device:
        return cached
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    tiles, splits, max_splits = plan_tiles(segment.lengths, segment.n_long, num_sms)
    uploaded = (
        torch.tensor(tiles, dtype=torch.int32, device=device),
        torch.tensor(splits, dtype=torch.int8, device=device),
        max_splits,
    )
    segment._tiles = uploaded
    return uploaded
