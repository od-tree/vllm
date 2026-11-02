"""Controllable short/long mix used by the head-collaboration eval."""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from ..arrivals import arrival_names, make_timestamps
from ..trace import DEFAULT_CHUNK_SIZE, TraceRow, make_hash_ids, require_chunk_multiple
from .base import TraceBuild
from .registry import register_dataset

DEFAULT_SHORT_INPUT = 1024
DEFAULT_SHORT_OUTPUT = 256
DEFAULT_LONG_INPUT = 8192
DEFAULT_LONG_OUTPUT = 1024
DEFAULT_PREFIX_RATIO = 0.75
DEFAULT_REQUEST_RATE = 1.0


def proportion_count(total: int, fraction: float) -> int:
    """Return ``fraction`` of ``total``, rounding halves away from zero.

    ``round`` uses bankers' rounding, so ``round(0.5)`` is ``0``. One long
    request at prefix fraction 0.5 must still count as shared.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if fraction == 0 or total == 0:
        return 0
    if fraction == 1:
        return total
    return int(math.floor(total * fraction + 0.5))


@dataclass(frozen=True)
class MixedParams:
    """Knobs for one mixed trace. Lengths are token counts."""

    num_requests: int
    long_ratio: float
    short_input: int = DEFAULT_SHORT_INPUT
    short_output: int = DEFAULT_SHORT_OUTPUT
    long_input: int = DEFAULT_LONG_INPUT
    long_output: int = DEFAULT_LONG_OUTPUT
    prefix_fraction: float = 0.0
    prefix_ratio: float = DEFAULT_PREFIX_RATIO
    arrival: str = "poisson"
    request_rate: float = DEFAULT_REQUEST_RATE
    burstgpt_csv: str | None = None
    seed: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> MixedParams:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown mixed dataset params: {unknown}")
        values = {name: params[name] for name in known if name in params}
        return cls(**values)

    def check(self) -> None:
        if self.num_requests < 1:
            raise ValueError(f"num_requests must be positive, got {self.num_requests}")
        if self.long_ratio < 0 or self.long_ratio > 1:
            raise ValueError(f"long_ratio must be in [0, 1], got {self.long_ratio}")
        if self.prefix_fraction < 0 or self.prefix_fraction > 1:
            raise ValueError(
                "prefix_fraction must be in [0, 1], got "
                f"{self.prefix_fraction}"
            )
        if self.prefix_ratio < 0 or self.prefix_ratio > 1:
            raise ValueError(f"prefix_ratio must be in [0, 1], got {self.prefix_ratio}")
        require_chunk_multiple(self.short_input, self.chunk_size, "short_input")
        require_chunk_multiple(self.long_input, self.chunk_size, "long_input")
        if self.short_output < 1 or self.long_output < 1:
            raise ValueError("output lengths must be positive")
        if self.prefix_fraction > 0:
            shared_chunks(self.long_input, self.prefix_ratio, self.chunk_size)


def shared_chunks(long_input: int, prefix_ratio: float, chunk_size: int) -> int:
    """Chunks of a long request that repeat a shared hash-id prefix."""
    if prefix_ratio == 0:
        return 0
    tokens = prefix_ratio * long_input
    if abs(tokens - round(tokens)) > 1e-6:
        raise ValueError(
            f"prefix_ratio {prefix_ratio} of long_input {long_input} "
            "is not a whole number of tokens"
        )
    token_count = int(round(tokens))
    require_chunk_multiple(token_count, chunk_size, "shared prefix")
    return token_count // chunk_size


class MixedDataset:
    """Length mix from the eval plan, including an optional shared prefix.

    Long versus short is chosen by request count, not by token count. Arrival
    times use their own seed stream, so changing the arrival source does not
    reshuffle which requests are long.
    """

    name = "mixed"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num-requests", type=int, required=True)
        parser.add_argument("--long-ratio", type=float, required=True)
        parser.add_argument("--short-input", type=int, default=DEFAULT_SHORT_INPUT)
        parser.add_argument("--short-output", type=int, default=DEFAULT_SHORT_OUTPUT)
        parser.add_argument("--long-input", type=int, default=DEFAULT_LONG_INPUT)
        parser.add_argument("--long-output", type=int, default=DEFAULT_LONG_OUTPUT)
        parser.add_argument("--prefix-fraction", type=float, default=0.0)
        parser.add_argument("--prefix-ratio", type=float, default=DEFAULT_PREFIX_RATIO)
        parser.add_argument(
            "--arrival",
            choices=tuple(arrival_names()),
            default="poisson",
        )
        parser.add_argument("--request-rate", type=float, default=DEFAULT_REQUEST_RATE)
        parser.add_argument("--burstgpt-csv", default=None)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    def build(self, params: Mapping[str, Any]) -> TraceBuild:
        spec = MixedParams.from_mapping(params)
        spec.check()
        n_long = proportion_count(spec.num_requests, spec.long_ratio)
        rng = random.Random(spec.seed)
        long_indexes = set(rng.sample(range(spec.num_requests), n_long))
        n_shared = proportion_count(len(long_indexes), spec.prefix_fraction)
        shared_indexes = set(rng.sample(sorted(long_indexes), n_shared))
        prefix: tuple[str, ...] = ()
        if shared_indexes:
            n_prefix = shared_chunks(
                spec.long_input, spec.prefix_ratio, spec.chunk_size
            )
            prefix = tuple(f"shared-c{chunk}" for chunk in range(n_prefix))
        timestamps, cycled = make_timestamps(
            spec.arrival,
            num_requests=spec.num_requests,
            seed=spec.seed,
            request_rate=spec.request_rate,
            burstgpt_csv=spec.burstgpt_csv,
        )
        if len(timestamps) != spec.num_requests:
            raise ValueError(
                f"arrival {spec.arrival} returned {len(timestamps)} timestamps "
                f"for {spec.num_requests} requests"
            )
        rows: list[TraceRow] = []
        for index, timestamp in enumerate(timestamps):
            is_long = index in long_indexes
            input_length = spec.long_input if is_long else spec.short_input
            output_length = spec.long_output if is_long else spec.short_output
            row_prefix = prefix if index in shared_indexes else ()
            rows.append(
                TraceRow(
                    timestamp=timestamp,
                    input_length=input_length,
                    output_length=output_length,
                    hash_ids=make_hash_ids(
                        index, input_length, spec.chunk_size, row_prefix
                    ),
                )
            )
        return TraceBuild(rows=rows, cycled=cycled)


register_dataset(MixedDataset())
