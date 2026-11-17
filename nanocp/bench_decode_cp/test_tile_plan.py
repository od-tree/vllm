"""CPU checks for the CP tile counts. No FA2."""

from tile_cp import plan_tiles


def test_long_rows_share_the_sms_and_shorts_stay_one():
    lengths = [131072, 131072] + [512] * 32
    tiles, splits, max_splits = plan_tiles(lengths, n_long=2, num_sms=108)
    assert splits[:2] == [16, 16]
    assert splits[2:] == [1] * 32
    assert max_splits == 16
    assert len(tiles) == 2 * 16 * 2 + 32 * 2


def test_a_full_batch_of_long_rows_splits_to_shorten_the_tail():
    # 128 rows x 2 KV heads = 256. ceil(256 * s / 108) / s is smallest at s=8.
    # 256 rows x 2 heads = 512, smallest at s=4. 108 heads is an exact wave
    # and stays at one split.
    _tiles, splits, max_splits = plan_tiles([32768] * 128, n_long=128, num_sms=108)
    assert splits == [8] * 128
    assert max_splits == 8
    assert len(_tiles) == 128 * 8 * 2
    _tiles, splits, max_splits = plan_tiles([32768] * 256, n_long=256, num_sms=108)
    assert splits == [4] * 256
    assert max_splits == 4
    _tiles, splits, max_splits = plan_tiles([32768] * 54, n_long=54, num_sms=108)
    assert splits == [1] * 54
    assert max_splits == 1


def test_split_stops_at_the_kv_blocks():
    _tiles, splits, _max_splits = plan_tiles([64, 64], n_long=2, num_sms=108)
    assert splits == [1, 1]


def main():
    test_long_rows_share_the_sms_and_shorts_stay_one()
    test_a_full_batch_of_long_rows_splits_to_shorten_the_tail()
    test_split_stops_at_the_kv_blocks()
    print("tile plan ok")


if __name__ == "__main__":
    main()
