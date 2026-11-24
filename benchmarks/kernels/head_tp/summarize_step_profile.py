# SPDX-License-Identifier: Apache-2.0
"""Summarize same-host worker and engine traces for bench_serving_batch.py."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_records(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def matches(replay, requests, packed):
    # Match computed context lengths to the client requests, allowing for
    # decode progress and scheduler reordering. Earliest deadline first is
    # sufficient for assigning sorted points to inclusive intervals.
    bounds = [(r["n"] - 1, r["n"] + r["generated"]) for r in requests]
    if len(bounds) != len(replay["lengths"]) or replay["packed"] != packed:
        return False
    for length in sorted(replay["lengths"]):
        eligible = [x for x in bounds if x[0] <= length <= x[1]]
        if not eligible:
            return False
        bounds.remove(min(eligible, key=lambda x: x[1]))
    return True


def stable_runs(replays, requests, packed, bucket, start, end):
    runs = []
    active = []
    previous_lengths = []
    for r in replays:
        lengths = sorted(r["lengths"])
        if active and lengths != [n + 1 for n in previous_lengths]:
            runs.append(active)
            active = []
        previous_lengths = lengths
        if (
            start <= r["start"] <= end
            and r["bucket"] == bucket
            and matches(r, requests, packed)
        ):
            active.append(r["start"])
        else:
            if active:
                runs.append(active)
            active = []
    if active:
        runs.append(active)
    return runs


def summarize(events, client, packed, min_steps):
    requests = client["records"]
    start = max(r["start"] for r in requests)
    end = min(r["finish"] for r in requests)
    bucket = max(sum(r["rank"] == rank for r in requests) for rank in (0, 1))
    by_rank = {}
    for rank in (0, 1):
        rank_replays = [
            r for r in events if r["stage"] == "replay" and r["rank"] == rank
        ]
        local = [r for r in requests if r["rank"] == rank]
        by_rank[rank] = stable_runs(rank_replays, local, packed, bucket, start, end)
    overlaps = [
        (max(a[1], b[1]), min(a[-2], b[-2]))
        for a in by_rank[0]
        for b in by_rank[1]
        if min(len(a), len(b)) >= 4
    ]
    if not overlaps:
        raise ValueError(f"No matching two-rank batch in repetition {client['rep']}")
    lo, hi = max(overlaps, key=lambda pair: pair[1] - pair[0])
    result = {}
    for rank in (0, 1):
        steps = [
            r["start"]
            for r in events
            if r["rank"] == rank and r["stage"] == "replay" and lo <= r["start"] <= hi
        ]
        n = len(steps) - 1
        if n < min_steps:
            raise ValueError(f"Only {n} stable intervals on rank {rank}")
        groups = defaultdict(list)
        for event in events:
            t = event["start"]
            # A start-to-start interval includes the first graph and the gap
            # before the next graph. The gap is stamped with that next start.
            inside = (
                (steps[0] < t <= steps[-1])
                if event["stage"] == "replay_gap"
                else (steps[0] <= t < steps[-1])
            )
            if event["rank"] == rank and inside:
                groups[event["stage"]].append(event["ms"])
        if len(groups["replay_gap"]) != n:
            raise ValueError(f"Incomplete GPU gap timings on rank {rank}")
        result[rank] = dict(
            intervals=n,
            wall_ms=1000 * (steps[-1] - steps[0]) / n,
            stages={
                k: dict(n=len(v), mean_ms=mean(v), median_ms=median(v))
                for k, v in groups.items()
            },
        )
    return dict(rep=client["rep"], ranks=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("client_jsonl", type=Path)
    parser.add_argument("--case", default="mixed4x2")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--min-steps", type=int, default=20)
    args = parser.parse_args()
    if args.min_steps < 1:
        parser.error("--min-steps must be positive")
    events = sorted(
        (
            r
            for path in args.profile_dir.glob("steps-*.jsonl")
            for r in read_records(path)
        ),
        key=lambda r: r["start"],
    )
    rounds = [
        r
        for r in read_records(args.client_jsonl)
        if r["case"] == args.case and r["rep"] > 0
    ]
    if not events or not rounds:
        parser.error("expected profile records and measured client rounds")
    print(
        json.dumps(
            [summarize(events, r, args.packed, args.min_steps) for r in rounds],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
