"""FA3 split planning, head-page packing, and Hopper graph correctness."""
from itertools import product
from random import Random
import unittest

import torch

from bench_perf import check_outputs, reference
from cuda_ipc_p2p import capture_graph, enable_peers
from head_tp_fa3 import HeadShard, HeadTPFA3, load_fa3
from layout_cp import full_batch
from tile_cp import _planner


class TestFA3HeadTP(unittest.TestCase):
    def test_minimax_matches_exhaustive_integer_splits(self):
        rng = Random(11)
        for _ in range(150):
            rows = rng.randrange(1, 5)
            blocks = [rng.randrange(1, 20) for _ in range(rows)]
            heads = [rng.randrange(1, 5) for _ in range(rows)]
            limit = rng.randrange(1, 5)
            budget = rng.randrange(sum(heads), sum(heads) * limit + 1)

            def tiles(counts):
                return sum(h * s for h, s in zip(heads, counts))

            def objective(counts):
                return max((n + s - 1) // s for n, s in zip(blocks, counts)), tiles(counts)

            oracle = min(objective(counts)
                         for counts in product(range(1, limit + 1), repeat=rows)
                         if tiles(counts) <= budget)
            actual = _planner.minimax_split_counts(blocks, heads, budget, limit)
            self.assertEqual(objective(actual), oracle)

    def test_packed_pages_and_current_token_slots(self):
        for shape in ((16, 2, 256), (32, 4, 128)):
            q_heads, kv_heads, dim = shape
            owner = full_batch((17, 65), 11, shape=shape)
            partner = full_batch((33,), 29, shape=shape)
            first = kv_heads // 2
            shard = HeadShard([(owner, slice(0, 2), slice(first, kv_heads)),
                               (partner, slice(0, 1), slice(0, kv_heads))], "cpu")
            expect = [(owner, row, h) for row in range(2) for h in range(first, kv_heads)]
            expect += [(partner, 0, h) for h in range(kv_heads)]
            for group, (source, row, head) in enumerate(expect):
                length = source.lengths[row]
                pages = (length + 15) // 16
                for got, original in ((shard.k, source.k), (shard.v, source.v)):
                    actual = got[shard.table[group, :pages].long()].reshape(-1, dim)
                    target = original[source.table[row, :pages].long(), :, head].reshape(-1, dim)
                    torch.testing.assert_close(actual[:length], target[:length], rtol=0, atol=0)
                group_width = q_heads // kv_heads
                torch.testing.assert_close(shard.q[group], source.q[row].view(kv_heads, group_width, dim)[head],
                                           rtol=0, atol=0)
            for cache in (shard.k, shard.v):
                cache.view(-1, dim).index_fill_(0, shard.slots, float("nan"))
            shard.write_kv()
            for cache, now in ((shard.k, shard.k_now), (shard.v, shard.v_now)):
                torch.testing.assert_close(cache.view(-1, dim)[shard.slots], now, rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2
        and all(torch.cuda.get_device_capability(i)[0] == 9 for i in (0, 1)),
        "requires two Hopper GPUs",
    )
    @torch.inference_mode()
    def test_graph_replays_mixed_shards_and_explicit_split_above_992(self):
        enable_peers([0, 1])
        fwd = load_fa3()
        shape = (32, 4, 128)
        owner = full_batch((8193,) + (17,) * 250, 11, shape=shape)
        partner = full_batch((33,) * 250, 29, shape=shape)
        case = HeadTPFA3(fwd, [owner, partner], 1)
        # 1002 virtual rows per rank. Ensure split and single-pass rows
        # coexist even when FA3's default path disables dynamic splitting.
        counts = [(5,) + (1,) * 250, (7,) + (1,) * 250]
        step = case.step(counts)
        graph = capture_graph(case.streams, lambda: step(lambda _: None))
        with torch.cuda.stream(case.streams[0]):
            graph.replay()
        case.streams[0].synchronize()
        actual = (case.owner_out, case.ranks[1].out[case.long_groups:].view(250, 32, 128))
        check_outputs(actual, (reference(owner), reference(partner)))

        case.remote[0].neg_()
        owner.q[0, 16:].neg_()
        torch.cuda.synchronize(0)
        with torch.cuda.stream(case.streams[0]):
            graph.replay()
        case.streams[0].synchronize()
        check_outputs(actual, (reference(owner), reference(partner)))


if __name__ == "__main__":
    unittest.main()
