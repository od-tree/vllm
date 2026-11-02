"""Arrival-time sources for timed traces.

A new source is a function registered with ``register_arrival``. It receives
``num_requests``, ``seed``, ``request_rate``, and ``burstgpt_csv``, and returns
``(timestamps, cycled)``. Timestamps are seconds from zero. ``cycled`` is true
when a finite trace had to reuse intervals to fill the request count.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

Arrival = Callable[..., tuple[list[float], bool]]

_ARRIVALS: dict[str, Arrival] = {}


def register_arrival(name: str, arrival: Arrival) -> None:
    existing = _ARRIVALS.get(name)
    if existing is not None and existing is not arrival:
        raise ValueError(f"arrival {name!r} is already registered")
    _ARRIVALS[name] = arrival


def unregister_arrival(name: str) -> None:
    _ARRIVALS.pop(name)


def arrival_names() -> list[str]:
    return sorted(_ARRIVALS)


def make_timestamps(
    arrival: str,
    *,
    num_requests: int,
    seed: int,
    request_rate: float,
    burstgpt_csv: str | None,
) -> tuple[list[float], bool]:
    try:
        function = _ARRIVALS[arrival]
    except KeyError as exc:
        known = ", ".join(arrival_names()) or "(none)"
        raise ValueError(f"unknown arrival {arrival!r}; known: {known}") from exc
    return function(
        num_requests=num_requests,
        seed=seed,
        request_rate=request_rate,
        burstgpt_csv=burstgpt_csv,
    )


def poisson_timestamps(
    num_requests: int, request_rate: float, seed: int
) -> list[float]:
    """Exponential gaps accumulated from zero."""
    if num_requests < 1:
        raise ValueError(f"num_requests must be positive, got {num_requests}")
    if request_rate <= 0:
        raise ValueError(
            f"request_rate must be positive for a Poisson arrival, got {request_rate}"
        )
    rng = random.Random(seed)
    times = [0.0]
    for _ in range(num_requests - 1):
        times.append(times[-1] + rng.expovariate(request_rate))
    return times


def burstgpt_timestamps(
    csv_path: Path, num_requests: int, seed: int
) -> tuple[list[float], bool]:
    """Replay gaps from a BurstGPT CSV, ignoring that file's token lengths.

    Only GPT-4 rows with a positive response-token count contribute. Numeric
    timestamps are already seconds; ISO datetimes are converted to seconds.
    Gaps are rebased so the first request is at zero. When the CSV does not
    hold enough gaps, or the seed offset wraps past the end, intervals cycle
    and the second return value is true.
    """
    if num_requests < 1:
        raise ValueError(f"num_requests must be positive, got {num_requests}")
    timestamps = read_burstgpt_timestamps(csv_path)
    if num_requests == 1:
        return [0.0], False
    if len(timestamps) < 2:
        raise ValueError(
            f"{csv_path} has {len(timestamps)} usable BurstGPT timestamps; need 2"
        )
    ordered = sorted(timestamps)
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    start = seed % len(gaps)
    cycled = start + (num_requests - 1) > len(gaps)
    times = [0.0]
    for step in range(num_requests - 1):
        times.append(times[-1] + gaps[(start + step) % len(gaps)])
    return times, cycled


def read_burstgpt_timestamps(csv_path: Path) -> list[float]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header")
        fields = {name.strip(): name for name in reader.fieldnames}
        timestamp_column = _require_column(fields, "Timestamp")
        model_column = _require_column(fields, "Model")
        response_column = _require_column(fields, "Response tokens")
        values: list[float] = []
        for row in reader:
            if (row.get(model_column) or "").strip() != "GPT-4":
                continue
            if _response_tokens(row.get(response_column)) <= 0:
                continue
            values.append(parse_timestamp(row.get(timestamp_column) or ""))
    return values


def parse_timestamp(raw: str) -> float:
    text = raw.strip()
    if not text:
        raise ValueError("empty BurstGPT timestamp")
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise ValueError(f"cannot parse BurstGPT timestamp {raw!r}") from exc


def _require_column(fields: Mapping[str, str], name: str) -> str:
    try:
        return fields[name]
    except KeyError as exc:
        raise ValueError(
            f"BurstGPT CSV is missing {name!r}. Columns: {list(fields)}"
        ) from exc


def _response_tokens(raw: str | None) -> float:
    try:
        return float(raw or 0)
    except ValueError as exc:
        raise ValueError(f"bad BurstGPT response token count {raw!r}") from exc


def _poisson_arrival(**kwargs: Any) -> tuple[list[float], bool]:
    times = poisson_timestamps(
        kwargs["num_requests"], kwargs["request_rate"], kwargs["seed"]
    )
    return times, False


def _burstgpt_arrival(**kwargs: Any) -> tuple[list[float], bool]:
    csv_path = kwargs.get("burstgpt_csv")
    if not csv_path:
        raise ValueError("burstgpt_csv is required when arrival is 'burstgpt'")
    return burstgpt_timestamps(Path(csv_path), kwargs["num_requests"], kwargs["seed"])


register_arrival("poisson", _poisson_arrival)
register_arrival("burstgpt", _burstgpt_arrival)
