"""Wall time for decode CP, including the LSE merge.

KV for each half is resident before the timer. The timed window is the
Q / O / LSE route, both FA2 calls, and the merge. The route is the N-rank
table in ``comm_cp``; this script happens to use two devices. It does not
call the head-TP operator.

The default matrix is the head-TP perf list in
``test_head_tp_paged_gpu.py``: five pure-long batches, then a short batch
with a few 128k or 256k requests. Long rows are the prefix and are split.
Short rows stay on the master at full length. ``--quick`` is the small
smoke set.
"""

from __future__ import annotations

import sys

import torch

from comm_cp import CpComm, enable_peers
from cuda_ipc_p2p import capture_graph, replay_median
from decode_cp import run_routed
from layout_cp import SegmentBatch, build_segments, segment_limits

# Smoke only. Not the comparison matrix.
QUICK = [
    ("n4_1024", [1024] * 4, 4),
    ("mix_4x128_1x1024", [1024, 128, 128, 128, 128], 1),
    ("n1_8192", [8192], 1),
]

# Pure long. Every row is split. Same five shapes as the head-TP perf run.
PURE = [
    ("20x32768", [32768] * 20, 20),
    ("128x8192", [8192] * 128, 128),
    ("128x32768", [32768] * 128, 128),
    ("128x65536", [65536] * 128, 128),
    ("256x32768", [32768] * 256, 256),
]

# Many short requests plus a few long ones. Long rows come first, matching
# the head-TP batch order. Short rows are mask 0 on the partner.
MIXED = [
    ("mix_128x256_2x128k_1x256k", [131072, 131072, 262144] + [256] * 128, 3),
    ("mix_64x1024_4x128k", [131072] * 4 + [1024] * 64, 4),
    ("mix_32x512_2x256k", [262144, 262144] + [512] * 32, 2),
    ("mix_512x512_16x256k", [262144] * 16 + [512] * 512, 16),
    # Head-TP also gave the partner its own 1024 short rows. Those stay on
    # the master here: a short row is not routed.
    ("mix_1024x1024_4x256k", [262144] * 4 + [1024] * 1024, 4),
]

CASES = PURE + MIXED


def _place(segment, device) -> SegmentBatch:
    target = torch.device("cuda", device)
    return SegmentBatch(
        segment.q.to(target),
        segment.k.to(target),
        segment.v.to(target),
        segment.table.to(target),
        segment.seqused.to(target),
        segment.which,
        segment.lengths,
        segment.n_long,
    )


def _time_case(seqs, n_long, rounds=5):
    # One GPU cannot hold the full cache and both slices. Build on CPU, then
    # drop the full cache before either slice is copied. This is outside the timer.
    full, master, partner, mask = build_segments(seqs, n_long, "cpu", seed=11)
    del full
    master = _place(master, 0)
    partner = _place(partner, 1)
    devices = [0, 1]
    q_shape = tuple(master.q.shape[1:])
    comm = CpComm(
        devices, max_batch=master.q.shape[0],
        q_shape=q_shape, o_shape=q_shape, lse_shape=(q_shape[0], 1),
        q_dtype=master.q.dtype,
    )
    master_lens, partner_lens = segment_limits(seqs, n_long)
    max_ks = [max(master_lens), max(partner_lens)]
    stream0 = torch.cuda.Stream(device=0)
    stream1 = torch.cuda.Stream(device=1)
    marks = {}

    def once():
        run_routed(
            [master, partner], moe_rank=0, devices=devices, mask=mask,
            max_ks=max_ks, comm=comm, sync=False, marks=marks,
        )

    # The first eager step compiles Triton, so its event span is not the kernel.
    # A second eager step is. Capture then overwrites the events with ones
    # recorded inside the graph, which this driver will not time.
    once()
    once()
    torch.cuda.synchronize()
    merge = marks["merge_begin"].elapsed_time(marks["merge_end"])
    graph = capture_graph([stream0, stream1], once)
    return replay_median(graph, stream0, rounds), merge


def main():
    enable_peers([0, 1])
    cases = QUICK if "--quick" in sys.argv else CASES
    for name, seqs, n_long in cases:
        wall, merge = _time_case(seqs, n_long)
        share = 0.0 if wall == 0 else merge / wall
        print(
            f"{name} wall_ms {wall:.3f} merge_ms {merge:.3f} merge_share {share:.3f} graph",
            flush=True,
        )


if __name__ == "__main__":
    main()
