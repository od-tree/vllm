"""CPU tile planner shared by paged GQA decode backends and benchmarks."""

HEAD_TP_SPLIT_CAP = 16
# row_splits is int8 and the FA2 combine is compiled up to 128 partials.
HEAD_TP_SPLIT_LIMIT = 127
FA3_EXPLICIT_SPLIT_LIMIT = 128


def flash_attn_split_kv_block_tokens(head_size: int) -> int:
    """FA2 split-KV tile width. Must match ``set_params_splitkv`` in flash_api.cpp."""

    if head_size <= 64:
        return 256
    if head_size <= 128:
        return 128
    return 64


def pack_head_tp_tile(row: int, head: int, split: int) -> int:
    """One compute CTA. Bits 0-15 are the row, 16-23 the KV head, 24-31 the split."""

    if not (0 <= row <= 0xFFFF and 0 <= head <= 0xFF and 0 <= split <= 0xFF):
        raise ValueError(f"tile ({row}, {head}, {split}) does not fit in one int32")
    return row | (head << 16) | (split << 24)


def _splits_for_full_waves(long_heads: int, num_sms: int, cap: int) -> int:
    """How many times to split each long head once the SMs are already full.

    One split leaves a partial last wave that still scans the whole segment.
    The result minimizes ``ceil(long_heads * splits / num_sms) / splits``,
    the scan time in units of one full segment. A tie keeps the smaller
    split so the combine loop stays short.
    """

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


def _fill_wave_splits(
    seqs: list[int],
    heads: list[int],
    num_sms: int,
    kv_block_tokens: int,
    short_work: int = 0,
) -> list[int]:
    """Split long rows so their share of one wave matches their share of KV.

    ``short_work`` is ``seq * kv_heads`` of the rows that stay one CTA per
    head. Those CTAs already occupy SMs. Giving the long rows a whole extra
    wave on top does not reduce bytes read; it only lengthens the combine.
    Each split launches ``heads[i]`` CTAs. Spare SMs inside the long-row
    budget are used only when every row on the tail can take one more split.
    """

    total = sum(seqs)
    long_work = sum(seq * head for seq, head in zip(seqs, heads))
    total_work = long_work + max(short_work, 0)
    if long_work <= 0 or total <= 0:
        return [1] * len(seqs)
    budget = num_sms if total_work == long_work else (num_sms * long_work) // total_work
    budget = max(budget, sum(heads))
    budget = min(budget, num_sms)
    limits: list[int] = []
    splits: list[int] = []
    for seq, head in zip(seqs, heads):
        if head <= 0:
            raise ValueError(f"a long row needs at least one KV head, got {head}")
        n_blocks = max(1, (seq + kv_block_tokens - 1) // kv_block_tokens)
        limit = min(HEAD_TP_SPLIT_LIMIT, n_blocks)
        limits.append(limit)
        raw = (budget * seq) // (total * head)
        splits.append(min(limit, max(1, raw)))

    def tail_cmp(left: int, right: int) -> int:
        return seqs[left] * splits[right] - seqs[right] * splits[left]

    while True:
        used = sum(splits[i] * heads[i] for i in range(len(splits)))
        spare = budget - used
        if spare <= 0:
            break
        worst = 0
        for i in range(1, len(splits)):
            if tail_cmp(i, worst) > 0:
                worst = i
        group: list[int] = []
        blocked = False
        for i in range(len(splits)):
            if tail_cmp(i, worst) != 0:
                continue
            if splits[i] >= limits[i]:
                blocked = True
                break
            group.append(i)
        if blocked or not group:
            break
        need = sum(heads[i] for i in group)
        if need > spare:
            break
        for i in group:
            splits[i] += 1
    return splits


def plan_decode_tiles(
    seq_lens: list[int], long_rows: list[tuple[int, int, int]], short_rows: list[int],
    num_kv_heads: int, num_sms: int, split_cap: int = HEAD_TP_SPLIT_CAP,
    kv_block_tokens: int = 64,
) -> tuple[list[int], list[int], int]:
    """Long rows specify (row, first KV head, exclusive last head)."""
    if not seq_lens or num_sms <= 0 or num_kv_heads <= 0:
        return [], [], 1
    if kv_block_tokens < 1 or not 1 <= split_cap <= HEAD_TP_SPLIT_LIMIT:
        raise ValueError("invalid KV tile width or split cap")
    long_heads = sum(end - start for _, start, end in long_rows)
    row_splits = [1] * len(seq_lens)
    if long_rows and long_heads < num_sms:
        assigned = _fill_wave_splits(
            [max(seq_lens[index], 1) for index, _, _ in long_rows],
            [end - start for _, start, end in long_rows],
            num_sms,
            kv_block_tokens,
            short_work=sum(max(seq_lens[index], 1) * num_kv_heads for index in short_rows),
        )
        for (index, _, _), count in zip(long_rows, assigned):
            row_splits[index] = count
    elif long_rows:
        max_seq = max(max(seq_lens[index], 1) for index, _, _ in long_rows)
        max_blocks = max(1, (max_seq + kv_block_tokens - 1) // kv_block_tokens)
        chosen = _splits_for_full_waves(long_heads, num_sms, min(split_cap, max_blocks))
        for index, _, _ in long_rows:
            seq = max(seq_lens[index], 1)
            n_blocks = max(1, (seq + kv_block_tokens - 1) // kv_block_tokens)
            raw = max(1, chosen * seq // max_seq)
            row_splits[index] = min(split_cap, n_blocks, raw)
    tiles: list[int] = []
    for index, start, end in long_rows:
        for split in range(row_splits[index]):
            for head in range(start, end):
                tiles.append(pack_head_tp_tile(index, head, split))
    for index in short_rows:
        for head in range(num_kv_heads):
            tiles.append(pack_head_tp_tile(index, head, 0))
    return tiles, row_splits, max(row_splits)


def minimax_split_counts(
    kv_blocks: list[int], kv_heads: list[int], cta_budget: int, split_limit: int,
) -> tuple[int, ...]:
    """Minimize the largest KV chunk under a budget of packed-GQA work tiles.

    Each split for a row costs one tile per local KV head. For a proposed
    maximum chunk of t blocks, ceil(blocks / t) is the least number of
    splits that can meet it. This gives a monotone feasibility test.
    """
    if len(kv_blocks) != len(kv_heads) or any(
        blocks < 1 or heads < 1 for blocks, heads in zip(kv_blocks, kv_heads)
    ):
        raise ValueError("each decode row needs a positive block and head count")
    if cta_budget < sum(kv_heads) or not 1 <= split_limit <= FA3_EXPLICIT_SPLIT_LIMIT:
        raise ValueError("invalid work tile budget or split limit")
    if not kv_blocks:
        return ()

    lo = (max(kv_blocks) + split_limit - 1) // split_limit
    hi = max(kv_blocks)
    while lo < hi:
        chunk = (lo + hi) // 2
        needed = sum(
            heads * ((blocks + chunk - 1) // chunk)
            for blocks, heads in zip(kv_blocks, kv_heads)
        )
        if needed <= cta_budget:
            hi = chunk
        else:
            lo = chunk + 1
    return tuple((blocks + lo - 1) // lo for blocks in kv_blocks)


def fa3_split_candidates(
    kv_blocks: list[int], kv_heads: list[int], num_sms: int, split_limit: int,
) -> dict[int, tuple[int, ...]]:
    """Generate distinct plans for a latency sweep, in increasing budget order."""
    if num_sms <= 0:
        raise ValueError("the SM count must be positive")
    # Sweep whole waves for sparse batches and extra work on top of the
    # unsplit baseline for batches that already occupy several waves.
    base = sum(kv_heads)
    budgets = {base, max(base, num_sms), max(base, 2 * num_sms)}
    budgets.update(base + extra for extra in (max(1, num_sms // 2), num_sms, 2 * num_sms))
    candidates = {}
    seen = set()
    for budget in sorted(budgets):
        counts = minimax_split_counts(kv_blocks, kv_heads, budget, split_limit)
        if counts not in seen:
            candidates[budget] = counts
            seen.add(counts)
    return candidates


def work_split_counts(
    kv_blocks: list[int],
    kv_heads: list[int],
    num_sms: int,
    split_limit: int,
    waves: float,
    min_chunk: int,
) -> tuple[int, ...]:
    """Size split chunks by total KV work per SM, including unsplit background work."""
    if (
        num_sms <= 0
        or waves <= 0
        or min_chunk < 1
        or not 1 <= split_limit <= FA3_EXPLICIT_SPLIT_LIMIT
    ):
        raise ValueError("invalid split planner parameters")
    if len(kv_blocks) != len(kv_heads) or any(
        blocks < 0 or heads < 1 for blocks, heads in zip(kv_blocks, kv_heads)
    ):
        raise ValueError("invalid KV block or head counts")
    total = sum(blocks * heads for blocks, heads in zip(kv_blocks, kv_heads))
    slots = max(1, round(num_sms * waves))
    target = max(min_chunk, (total + slots - 1) // slots)
    return tuple(
        max(1, min(split_limit, (n + target - 1) // target)) for n in kv_blocks
    )
