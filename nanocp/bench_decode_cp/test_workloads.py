"""CPU regressions for the logical inputs shared by the three GPU runners."""
import unittest

import torch

from bench_perf import check_outputs, head_backend, reference
from layout_cp import BLOCK, DIM, KV, cp_segments, full_batch
from workloads import CASES, QUICK


def tokens(batch, tensor, row):
    length = batch.lengths[row]
    ids = batch.table[row, :(length + BLOCK - 1) // BLOCK].long()
    return tensor[ids].reshape(-1, KV, DIM)[:length]


class TestWorkload(unittest.TestCase):
    def test_asymmetric_cp_preserves_paged_kv_and_partner_batch(self):
        full = full_batch((101, 256, 17), 11)
        background = full_batch((33,), 29)
        for share, prefixes in ((75, [64, 192]), (90, [80, 224])):
            with self.subTest(owner_percent=share):
                master, partner, _ = cp_segments(full, 2, background, owner_percent=share)
                self.assertEqual(master.lengths, prefixes + [17])
                self.assertEqual(partner.lengths, [101 - prefixes[0], 256 - prefixes[1], 33])
                for field in ("k", "v"):
                    for row in range(2):
                        combined = torch.cat((tokens(master, getattr(master, field), row),
                                              tokens(partner, getattr(partner, field), row)))
                        self.assertTrue(torch.equal(combined, tokens(full, getattr(full, field), row)))
                    self.assertTrue(torch.equal(tokens(partner, getattr(partner, field), 2),
                                                tokens(background, getattr(background, field), 0)))

    def test_partner_background_and_long_prefix_survive_packing(self):
        case = QUICK[-1]
        full = full_batch(case.owner, 11)
        background = full_batch(case.partner, 29)
        master, partner, mask = cp_segments(full, case.n_long, background)
        self.assertEqual(partner.local_tail, len(case.partner))
        self.assertEqual(mask[1].tolist(), [1] * case.n_long + [0])
        torch.testing.assert_close(master.q, full.q, rtol=0, atol=0)
        torch.testing.assert_close(partner.q, torch.cat((full.q[:case.n_long], background.q)),
                                   rtol=0, atol=0)
        for field in ("k", "v"):
            for row in range(len(case.owner)):
                pieces = [tokens(master, getattr(master, field), row)]
                if row < case.n_long:
                    pieces.append(tokens(partner, getattr(partner, field), row))
                self.assertTrue(torch.equal(torch.cat(pieces), tokens(full, getattr(full, field), row)))
            for row in range(len(case.partner)):
                self.assertTrue(torch.equal(tokens(background, getattr(background, field), row),
                                            tokens(partner, getattr(partner, field), case.n_long + row)))

    def test_tp_packs_the_same_tokens_by_head(self):
        backend = head_backend()
        case = QUICK[-1]
        full = full_batch(case.owner, 11)
        background = full_batch(case.partner, 29)

        def spans(lengths):
            cursor = 0
            result = []
            for length in lengths:
                count = (length + BLOCK - 1) // BLOCK
                result.append((cursor, count))
                cursor += count
            return result

        owner = backend._owner_cache(full.k, full.v, full.table, spans(case.owner),
                                     case.owner, case.n_long)
        pk, pv, _ = backend._split_cache(full.k, full.v, full.table, case.owner,
                                         case.n_long, 1, "cpu")
        packed_counts = [backend._packed_pages(seq) for seq in case.owner[:case.n_long]]
        partner = backend._append_local(pk, pv, packed_counts, background.k, background.v,
                                        spans(case.partner))
        for head, (k, v, table) in enumerate((owner, partner)):
            for row, count in enumerate(packed_counts):
                ids = table[row, :count].long()
                for packed, field in ((k, "k"), (v, "v")):
                    actual = packed[ids].transpose(1, 2).reshape(-1, DIM)[:case.owner[row]]
                    original = tokens(full, getattr(full, field), row)
                    torch.testing.assert_close(actual, original[:, head], rtol=0, atol=0)
            for row in range(case.n_long, len(table)):
                source = background if head else full
                idx = row - case.n_long if head else row
                length = source.lengths[idx]
                ids = table[row, :(length + BLOCK - 1) // BLOCK].long()
                for packed, field in ((k, "k"), (v, "v")):
                    actual = packed[ids].reshape(-1, KV, DIM)[:length]
                    torch.testing.assert_close(actual, tokens(source, getattr(source, field), idx),
                                               rtol=0, atol=0)

    def test_large_paired_case_has_background_on_both_ranks(self):
        case = next(spec for spec in CASES if spec.name.startswith("mix_1024"))
        self.assertEqual(case.owner[case.n_long:], case.partner)
        self.assertEqual(len(case.partner), 1024)

    def test_reference_rejects_corrupt_partner_output(self):
        batch = full_batch((33,), 29)
        expected = reference(batch)
        got = expected.clone()
        got[0, 0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            check_outputs((expected, got), (expected, expected))


if __name__ == "__main__":
    unittest.main()
