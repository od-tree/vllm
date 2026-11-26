# SPDX-License-Identifier: Apache-2.0
"""Two-rank decode admission. Costs are per full-attention layer."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AdmissionRow:
    req_id: str
    length: int
    history: int
    remaining: int
    eligible: bool


class LoadAdmission:
    def __init__(self, options, shard_bytes_per_token: int):
        # Initial H100/BF16 estimates. Calibrate the full-graph overhead for
        # the selected tile policy before using this model on another platform.
        self.tokens_per_us = float(options.get("kv_tokens_per_us", 1400))
        self.overhead = float(options.get("split_overhead_us", 32))
        bandwidth = float(options.get("migration_gbps", 8))
        self.return_fraction = float(options.get("return_fraction", 0.0))
        self.horizon = int(options.get("horizon", 64))
        self.threshold = int(options.get("min_context", 0))
        if not (
            isfinite(self.tokens_per_us)
            and self.tokens_per_us > 0
            and isfinite(bandwidth)
            and bandwidth > 0
            and isfinite(self.overhead)
            and self.overhead >= 0
            and 0 <= self.return_fraction <= 1
            and self.horizon >= 1
            and self.threshold >= 0
            and shard_bytes_per_token > 0
        ):
            raise ValueError("Invalid head_tp load admission cost parameters")
        self.copy_us_per_token = shard_bytes_per_token / (1000 * bandwidth)

    def suggest(self, contexts, owner, length, remaining):
        # Only aggregate background work is needed for an arriving prompt.
        batches = [[AdmissionRow("", total, 0, 0, False)] for total in contexts]
        batches[owner].append(AdmissionRow("new", length, 0, remaining, True))
        return self.choose(batches, [None, None]) is not None

    def choose(
        self, batches: Sequence[Sequence[AdmissionRow]], residents: Sequence[str | None]
    ) -> tuple[int, str] | None:
        """Choose local, keep, or split one row using identical rank snapshots."""
        loads = [sum(row.length for row in rows) for rows in batches]
        dense = max(loads)
        exit_cost = sum(
            row.history * self.copy_us_per_token
            for rank, rows in enumerate(batches)
            for row in rows
            if row.req_id == residents[rank]
        )
        best = None
        best_score = -exit_cost  # immediately restore all resident requests
        for rank, rows in enumerate(batches):
            for row in rows:
                if (
                    not row.eligible
                    or row.length < self.threshold
                    or row.remaining <= 0
                ):
                    continue
                shard = row.length / 2
                critical = max(loads[rank] - shard, loads[1 - rank] + shard)
                saving = (dense - critical) / self.tokens_per_us - self.overhead
                steps = min(self.horizon, row.remaining)
                copy = row.history * self.copy_us_per_token
                keep = row.req_id == residents[rank]
                transition = exit_cost + (-copy if keep else copy)
                future_return = (
                    self.return_fraction * (row.length + steps) * self.copy_us_per_token
                )
                score = steps * saving - transition - future_return
                if score > best_score:
                    best_score, best = score, (rank, row.req_id)
        return best
