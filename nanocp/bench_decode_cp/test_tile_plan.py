"""CPU checks for the CP tile counts. No FA2."""

from tile_cp import plan_tiles


def test_long_rows_share_the_sms_and_shorts_stay_one():
    lengths = [131072, 131072] + [512] * 32
    tiles, splits, max_splits = plan_tiles(lengths, n_long=2, num_sms=108)
    # Short rows already hold some of the wave. The two long rows get 25
    # splits each: 2 rows * 25 * 2 heads = 100 CTAs, their share of 108 SMs.
    assert splits[:2] == [25, 25]
    assert splits[2:] == [1] * 32
    assert max_splits == 25
    assert len(tiles) == 2 * 25 * 2 + 32 * 2


def test_saturated_batch_uses_the_engine_tail_wave_policy():
    lengths = [32768] * 128
    _tiles, splits, max_splits = plan_tiles(lengths, n_long=128, num_sms=108)
    assert splits == [8] * 128
    assert max_splits == 8


def test_split_stops_at_the_kv_blocks():
    _tiles, splits, _max_splits = plan_tiles([64, 64], n_long=2, num_sms=108)
    assert splits == [1, 1]


def main():
    test_long_rows_share_the_sms_and_shorts_stay_one()
    test_saturated_batch_uses_the_engine_tail_wave_policy()
    test_split_stops_at_the_kv_blocks()
    print("tile plan ok")


if __name__ == "__main__":
    main()
