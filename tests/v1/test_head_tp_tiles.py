# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.head_tp import (
    HeadTPStep,
    build_head_tp_tile_plan,
    make_plan,
    pack_head_tp_tile,
)


def _unpack(tile: int) -> tuple[int, int, int]:
    return tile & 0xFFFF, (tile >> 16) & 0xFF, (tile >> 24) & 0xFF


def test_long_rows_are_split_ahead_of_short_rows():
    step = HeadTPStep(
        plans={
            "long-a": make_plan(
                "long-a", 16, 2, 2, 0, 2,
                state="active", kv_source="gpu_copy",
                visible_len=262144, predicted_remaining=0,
            ),
            "long-b": make_plan(
                "long-b", 16, 2, 2, 0, 2,
                state="active", kv_source="gpu_copy",
                visible_len=131072, predicted_remaining=0,
            ),
        },
        active=True,
    )
    seq_lens = [262144, 131072, 512, 512]
    tiles, splits, max_splits = build_head_tp_tile_plan(
        seq_lens, ["long-a", "long-b", "s0", "s1"], step, rank=0,
        num_kv_heads=2, num_sms=108,
    )

    # One local KV head. Short-row work reduces the long-row budget to 107 CTAs.
    # Together that is one wave. The old cap of 16 left only 32 CTAs.
    assert splits == [71, 36, 1, 1]
    assert max_splits == 71
    decoded = [_unpack(tile) for tile in tiles]
    long_tiles = decoded[: 71 + 36]
    short_tiles = decoded[71 + 36 :]
    assert long_tiles == [
        (0, 0, split) for split in range(71)
    ] + [
        (1, 0, split) for split in range(36)
    ]
    assert short_tiles == [(2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0)]
    assert pack_head_tp_tile(1, 0, 3) == tiles[71 + 3]


def test_partner_rank_uses_its_own_kv_head():
    plan = make_plan(
        "long", 16, 2, 2, 0, 2,
        state="active", kv_source="gpu_copy",
        visible_len=262144, predicted_remaining=0,
    )
    tiles, splits, _ = build_head_tp_tile_plan(
        [262144], ["long"], HeadTPStep(plans={"long": plan}, active=True),
        rank=1, num_kv_heads=2, num_sms=108,
    )
    assert splits == [108]
    assert [_unpack(tile)[1] for tile in tiles] == [1] * 108


def _saturated(n: int):
    plans = {
        f"r{i}": make_plan(
            f"r{i}", 16, 2, 2, 0, 2,
            state="active", kv_source="gpu_copy",
            visible_len=32768, predicted_remaining=0,
        )
        for i in range(n)
    }
    return build_head_tp_tile_plan(
        [32768] * n, list(plans), HeadTPStep(plans=plans, active=True),
        rank=0, num_kv_heads=2, num_sms=108,
    )


def test_saturated_long_batch_splits_to_shorten_the_tail():
    # 128 heads: ceil(128 * s / 108) / s is smallest at s=16 (1.1875 vs 2).
    # 256 heads: smallest at s=8 (2.375 vs 3). An exact multiple of 108 stays
    # at one split, because another split does not remove a partial wave.
    tiles, splits, max_splits = _saturated(128)
    assert splits == [16] * 128
    assert max_splits == 16
    assert len(tiles) == 128 * 16
    assert {_unpack(tile)[1] for tile in tiles} == {0}

    tiles, splits, max_splits = _saturated(256)
    assert splits == [8] * 256
    assert max_splits == 8
    assert len(tiles) == 256 * 8

    tiles, splits, max_splits = _saturated(108)
    assert splits == [1] * 108
    assert max_splits == 1
    assert len(tiles) == 108


def test_splits_stop_at_kv_blocks():
    # Formula alone would give 16 for every long row. With kBlockN=64,
    # 64 / 96 / 128 tokens only have 1 / 2 / 2 real KV tiles.
    step = HeadTPStep(
        plans={
            "a": make_plan(
                "a", 16, 2, 2, 0, 2,
                state="active", kv_source="gpu_copy",
                visible_len=64, predicted_remaining=0,
            ),
            "b": make_plan(
                "b", 16, 2, 2, 0, 2,
                state="active", kv_source="gpu_copy",
                visible_len=96, predicted_remaining=0,
            ),
            "c": make_plan(
                "c", 16, 2, 2, 0, 2,
                state="active", kv_source="gpu_copy",
                visible_len=128, predicted_remaining=0,
            ),
        },
        active=True,
    )
    tiles, splits, max_splits = build_head_tp_tile_plan(
        [64, 96, 128, 17], ["a", "b", "c", "short"], step, rank=0,
        num_kv_heads=2, num_sms=108, kv_block_tokens=64,
    )
    assert splits == [1, 2, 2, 1]
    assert max_splits == 2
    assert [_unpack(tile) for tile in tiles] == [
        (0, 0, 0),
        (1, 0, 0), (1, 0, 1),
        (2, 0, 0), (2, 0, 1),
        (3, 0, 0), (3, 1, 0),
    ]


def test_work_policy_splits_a_long_tail_even_when_short_rows_fill_the_wave():
    from vllm.v1.head_tp import DecodePolicy

    policy = DecodePolicy({"tile": "work"})
    lengths = [131073] + [128] * 256
    heads = [2] + [4] * 256
    counts = policy.splits(
        lengths, heads, block_n=128, sms=132, limit=64, rank=1, step=1
    )
    # 3074 head-blocks / 99 slots -> 32 blocks per split.
    assert counts == (33,) + (1,) * 256
    assert policy.splits(
        [0, 141698], [2, 2], block_n=128, sms=132, limit=8, rank=0, step=2
    ) == (1, 8)
