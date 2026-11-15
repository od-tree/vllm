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

    # 108 * 262144 / ((262144 + 131072) * 1) = 72, capped at 16.
    # 108 * 131072 / (393216 * 1) = 36, capped at 16.
    assert splits == [16, 16, 1, 1]
    assert max_splits == 16
    decoded = [_unpack(tile) for tile in tiles]
    long_tiles = decoded[: 16 + 16]
    short_tiles = decoded[16 + 16 :]
    assert long_tiles == [
        (0, 0, split) for split in range(16)
    ] + [
        (1, 0, split) for split in range(16)
    ]
    assert short_tiles == [(2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0)]
    assert pack_head_tp_tile(1, 0, 3) == tiles[16 + 3]


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
    assert splits == [16]
    assert [_unpack(tile)[1] for tile in tiles] == [1] * 16


def test_saturated_long_batch_stays_at_one_split():
    plans = {
        f"r{i}": make_plan(
            f"r{i}", 16, 2, 2, 0, 2,
            state="active", kv_source="gpu_copy",
            visible_len=32768, predicted_remaining=0,
        )
        for i in range(256)
    }
    tiles, splits, max_splits = build_head_tp_tile_plan(
        [32768] * 256, list(plans), HeadTPStep(plans=plans, active=True),
        rank=0, num_kv_heads=2, num_sms=108,
    )
    assert splits == [1] * 256
    assert max_splits == 1
    assert len(tiles) == 256
    assert {_unpack(tile)[1] for tile in tiles} == {0}


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
