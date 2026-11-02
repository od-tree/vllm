"""Service matrix from the head-collaboration eval plan.

The server configuration is chosen by the person who started the process.
This scheme only names that configuration and loops the workloads. It does
not start the server or change its flags.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence

from ..datasets.mixed import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LONG_OUTPUT,
    DEFAULT_PREFIX_RATIO,
    DEFAULT_SHORT_INPUT,
    DEFAULT_SHORT_OUTPUT,
    MixedParams,
)
from .base import RunSpec
from .registry import register_scheme

CONFIGS = ("A", "B", "C", "D")
DEFAULT_LONG_RATIOS = (0.01, 0.05, 0.2)
DEFAULT_LONG_INPUTS = (8192, 32768)
DEFAULT_RATES = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SEEDS = (0, 1)
DEFAULT_NUM_REQUESTS = 200
DEFAULT_SHARED_RATIOS = (0.01, 0.05)
DEFAULT_SHARED_FRACTION = 0.5


def format_num(value: float) -> str:
    """Render a ratio or rate without trailing zeros."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def result_name(
    config: str,
    long_ratio: float,
    long_input: int,
    rate: float,
    seed: int,
    prefix_fraction: float,
) -> str:
    name = (
        f"{config}_{format_num(long_ratio)}_{long_input}_"
        f"{format_num(rate)}_{seed}"
    )
    if prefix_fraction > 0:
        name += f"_prefix{format_num(prefix_fraction)}"
    return name


def csv_floats(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return tuple(float(part) for part in parts)


def csv_ints(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return tuple(int(part) for part in parts)


class HeadTPMatrix:
    """Configs A–D crossed with the mixed, control, and shared-prefix loads."""

    name = "head_tp_matrix"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", required=True, choices=CONFIGS)
        parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
        parser.add_argument(
            "--long-ratios",
            type=csv_floats,
            default=DEFAULT_LONG_RATIOS,
        )
        parser.add_argument(
            "--long-inputs",
            type=csv_ints,
            default=DEFAULT_LONG_INPUTS,
        )
        parser.add_argument("--rates", type=csv_floats, default=DEFAULT_RATES)
        parser.add_argument("--seeds", type=csv_ints, default=DEFAULT_SEEDS)
        parser.add_argument("--short-input", type=int, default=DEFAULT_SHORT_INPUT)
        parser.add_argument("--short-output", type=int, default=DEFAULT_SHORT_OUTPUT)
        parser.add_argument("--long-output", type=int, default=DEFAULT_LONG_OUTPUT)
        parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        parser.add_argument(
            "--shared-prefix-fraction",
            type=float,
            default=DEFAULT_SHARED_FRACTION,
        )
        parser.add_argument(
            "--prefix-ratio",
            type=float,
            default=DEFAULT_PREFIX_RATIO,
        )
        parser.add_argument(
            "--controls",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--shared-prefix",
            action=argparse.BooleanOptionalAction,
            default=True,
        )

    def expand(self, args: argparse.Namespace) -> list[RunSpec]:
        if args.num_requests < 1:
            raise ValueError(f"num_requests must be positive, got {args.num_requests}")
        if args.shared_prefix and args.shared_prefix_fraction <= 0:
            raise ValueError("shared-prefix-fraction must be positive")
        _require_positive(args.rates, "rates")
        cells: list[tuple[float, int, float, int, float]] = []
        cells.extend(
            _product(
                args.long_ratios,
                args.long_inputs,
                args.rates,
                args.seeds,
                prefix_fraction=0.0,
            )
        )
        if args.controls:
            cells.extend(
                _product(
                    (0.0,),
                    (args.long_inputs[0],),
                    args.rates,
                    args.seeds,
                    prefix_fraction=0.0,
                )
            )
            cells.extend(
                _product(
                    (1.0,),
                    args.long_inputs,
                    args.rates,
                    args.seeds,
                    prefix_fraction=0.0,
                )
            )
        if args.shared_prefix:
            cells.extend(
                _product(
                    DEFAULT_SHARED_RATIOS,
                    args.long_inputs,
                    args.rates,
                    args.seeds,
                    prefix_fraction=args.shared_prefix_fraction,
                )
            )
        specs: list[RunSpec] = []
        seen: set[str] = set()
        for long_ratio, long_input, rate, seed, prefix_fraction in cells:
            name = result_name(
                args.config, long_ratio, long_input, rate, seed, prefix_fraction
            )
            if name in seen:
                continue
            seen.add(name)
            params = MixedParams(
                num_requests=args.num_requests,
                long_ratio=long_ratio,
                short_input=args.short_input,
                short_output=args.short_output,
                long_input=long_input,
                long_output=args.long_output,
                prefix_fraction=prefix_fraction,
                prefix_ratio=args.prefix_ratio if prefix_fraction > 0 else 0.0,
                arrival="poisson",
                request_rate=rate,
                seed=seed,
                chunk_size=args.chunk_size,
            )
            params.check()
            specs.append(
                RunSpec(
                    config=args.config,
                    dataset="mixed",
                    dataset_args=_params_dict(params),
                    result_name=name,
                )
            )
        return specs


def _product(
    ratios: Sequence[float],
    long_inputs: Sequence[int],
    rates: Sequence[float],
    seeds: Sequence[int],
    *,
    prefix_fraction: float,
) -> Iterable[tuple[float, int, float, int, float]]:
    for long_ratio in ratios:
        for long_input in long_inputs:
            for rate in rates:
                for seed in seeds:
                    yield (long_ratio, long_input, rate, seed, prefix_fraction)


def _require_positive(values: Sequence[float], name: str) -> None:
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _params_dict(params: MixedParams) -> dict[str, object]:
    return {
        "num_requests": params.num_requests,
        "long_ratio": params.long_ratio,
        "short_input": params.short_input,
        "short_output": params.short_output,
        "long_input": params.long_input,
        "long_output": params.long_output,
        "prefix_fraction": params.prefix_fraction,
        "prefix_ratio": params.prefix_ratio,
        "arrival": params.arrival,
        "request_rate": params.request_rate,
        "burstgpt_csv": params.burstgpt_csv,
        "seed": params.seed,
        "chunk_size": params.chunk_size,
    }


register_scheme(HeadTPMatrix())
