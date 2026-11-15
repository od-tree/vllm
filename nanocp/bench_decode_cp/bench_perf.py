"""Wall time for decode CP, including the LSE merge.

KV for each half is resident before the timer. The timed window is the
Q / O / LSE route, both FA2 calls, and the merge. The route is the N-rank
table in ``comm_cp``; this script happens to use two devices. It does not
call the head-TP operator.

``python3 bench_perf.py`` runs a short matrix. ``python3 bench_perf.py --full``
uses the same long-row shapes as the head-TP perf cases (2 x 256k and friends).
"""

from __future__ import annotations

import sys

import torch

from comm_cp import CpComm, enable_peers
from decode_cp import run_routed
from layout_cp import build_segments, relocate_kv, segment_limits

QUICK = [
    ("n4_1024", [1024] * 4, 4),
    ("mix_4x128_1x1024", [1024, 128, 128, 128, 128], 1),
    ("n1_8192", [8192], 1),
]

# The quick mix includes short rows. They stay in the master's attention call
# and are inside the timer. The long cases below are all long rows.
FULL = QUICK + [
    ("n4_262144", [262144] * 4, 4),
    ("n16_262144", [262144] * 16, 16),
    ("mix_2x256k", [262144, 262144], 2),
    ("mix_4x128k", [131072] * 4, 4),
]


def _time_case(seqs, n_long, rounds=5):
    full, master, partner, mask = build_segments(seqs, n_long, "cuda:0", seed=11)
    del full
    # KV is resident before the timer. The route carries Q, O, and LSE only.
    partner = relocate_kv(partner, 1)
    devices = [0, 1]
    q_shape = tuple(master.q.shape[1:])
    comm = CpComm(
        devices, max_batch=master.q.shape[0],
        q_shape=q_shape, o_shape=q_shape, lse_shape=(q_shape[0], 1),
        q_dtype=master.q.dtype,
    )
    master_lens, partner_lens = segment_limits(seqs, n_long)
    max_ks = [max(master_lens), max(partner_lens)]
    stream0 = torch.cuda.current_stream(device=0)
    torch.cuda.synchronize()

    def once():
        wall0 = torch.cuda.Event(True)
        wall1 = torch.cuda.Event(True)
        marks = {}
        wall0.record(stream0)
        run_routed(
            [master, partner], moe_rank=0, devices=devices, mask=mask,
            max_ks=max_ks, comm=comm, sync=False, marks=marks,
        )
        wall1.record(stream0)
        wall1.synchronize()
        return wall0.elapsed_time(wall1), marks["merge_begin"].elapsed_time(marks["merge_end"])

    once()
    walls = []
    merges = []
    for _ in range(rounds):
        wall, merge = once()
        walls.append(wall)
        merges.append(merge)
    walls.sort()
    merges.sort()
    return walls[len(walls) // 2], merges[len(merges) // 2]


def main():
    enable_peers([0, 1])
    cases = FULL if "--full" in sys.argv else QUICK
    for name, seqs, n_long in cases:
        wall, merge = _time_case(seqs, n_long)
        share = 0.0 if wall == 0 else merge / wall
        print(
            f"{name} wall_ms {wall:.3f} merge_ms {merge:.3f} merge_share {share:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
