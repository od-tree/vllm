"""One GPU, two logical ranks: page-sliced segments merged against full FA2.

Short rows sit in the master attention call. The partner mask for those rows
is 0. num_splits defaults to the FA2 heuristic.
"""

from __future__ import annotations

import torch

from decode_cp import run_routed
from layout_cp import build_segments, segment_limits
from reference import reference_full

MAX_ABS = 1e-2


def _check(seqs, n_long, num_splits=0):
    full, master, partner, mask = build_segments(seqs, n_long, "cuda", seed=11)
    ref, _ = reference_full(full, num_splits=num_splits)
    device = full.q.device
    master_lens, partner_lens = segment_limits(seqs, n_long)
    merged = run_routed(
        [master, partner], moe_rank=0, devices=[device, device], mask=mask,
        num_splits=num_splits,
        max_ks=[max(master_lens), max(partner_lens) if partner_lens else 0],
    )
    err = (merged.float() - ref.float()).abs()
    print(
        f"seqs {seqs} n_long {n_long} splits {num_splits} "
        f"max_abs {float(err.max()):.6e}",
        flush=True,
    )
    torch.testing.assert_close(merged.float(), ref.float(), atol=MAX_ABS, rtol=0)
    torch.cuda.synchronize()


def main():
    for seq in (64, 96, 100, 128, 1024, 8192):
        _check([seq], 1)
    _check([128], 1, num_splits=1)
    # Three long rows and one short row. 100 is covered above and splits 48 + 52.
    _check([64, 96, 128, 17], 3)
    print("halves ok", flush=True)


if __name__ == "__main__":
    main()
