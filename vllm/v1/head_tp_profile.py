# SPDX-License-Identifier: Apache-2.0
"""Opt-in, per-process timing for a native-DP versus Head-TP serving run."""

import json
import os
from collections import deque
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch

from vllm import envs

DIRECTORY = envs.VLLM_HEAD_TP_PROFILE_DIR


@lru_cache
def _output(pid: int):
    assert DIRECTORY is not None
    path = Path(DIRECTORY)
    path.mkdir(parents=True, exist_ok=True)
    return (path / f"steps-{pid}.jsonl").open("a", buffering=1)


def _write(rank, stage, start, ms, **fields):
    record = dict(rank=rank, stage=stage, start=start, ms=ms, **fields)
    _output(os.getpid()).write(json.dumps(record, separators=(",", ":")) + "\n")


def start():
    return perf_counter() if DIRECTORY else None


def finish(stage, rank, stamp):
    """Return the start of the next section, excluding this record's write."""
    if stamp is not None:
        elapsed = (perf_counter() - stamp) * 1000
        _write(rank, stage, stamp, elapsed)
        return perf_counter()
    return None


class ReplayProfile:
    """Resolve completed event pairs on later steps without a device sync."""

    def __init__(self):
        self.pending = deque()
        self.previous_end = None

    def drain(self, rank):
        while self.pending and self.pending[0][2].query():
            stamp, start_event, end_event, previous, fields = self.pending.popleft()
            _write(rank, "replay", stamp, start_event.elapsed_time(end_event), **fields)
            if previous is not None:
                _write(rank, "replay_gap", stamp, previous.elapsed_time(start_event))

    def replay(self, manager, desc, batch, rank, dummy):
        self.drain(rank)
        fields = dict(
            bucket=desc.num_tokens,
            packed=desc.packed_heads,
            lengths=[]
            if dummy
            else batch.num_computed_tokens_np[: batch.num_reqs].tolist(),
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stamp = perf_counter()
        start_event.record()
        output = manager.run_fullgraph(desc)
        end_event.record()
        self.pending.append((stamp, start_event, end_event, self.previous_end, fields))
        self.previous_end = end_event
        return output
