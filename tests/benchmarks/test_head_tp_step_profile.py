# SPDX-License-Identifier: Apache-2.0

from benchmarks.kernels.head_tp.summarize_step_profile import summarize


def test_profile_excludes_a_discontinuous_decode_and_keeps_idle_peer():
    # One FULL graph is missing between steps 2 and 4. Only the later,
    # longer consecutive run should contribute to the statistics.
    events = []
    for rank in (0, 1):
        for step in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
            start = step * 0.01
            events.append(
                dict(
                    stage="replay",
                    rank=rank,
                    start=start,
                    ms=7.0,
                    packed=True,
                    bucket=1,
                    lengths=[100 + step] if rank == 0 else [],
                )
            )
            events.append(dict(stage="replay_gap", rank=rank, start=start, ms=step))
    client = dict(rep=1, records=[dict(rank=0, n=100, generated=10, start=0, finish=1)])
    summary = summarize(events, client, packed=True, min_steps=3)
    for r in summary["ranks"].values():
        assert r["intervals"] == 3
        # The selected intervals start on 5, 6, 7 and end on 6, 7, 8.
        assert r["stages"]["replay_gap"]["mean_ms"] == 7
