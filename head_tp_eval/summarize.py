"""Split a ``vllm bench serve --save-result`` JSON into short and long requests.

``T0`` is passed in. It is the short-request mean TPOT from the pure-short,
collaboration-off, lowest-rate run. This module does not call the server.
A service log is optional. Lines that contain ``HEAD_TP`` and a JSON object
add pairing counts to the same report. Latency rows are still produced when
the log is absent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HEAD_TP_PREFIX = "HEAD_TP"


def summarize_bench(
    bench: Mapping[str, Any],
    *,
    short_input: int,
    t0_s: float | None = None,
) -> dict[str, Any]:
    """Build the short/long latency report.

    A request is long when its input length is greater than ``short_input``.
    TPOT for one request is the mean of that request's ``itls``. Failed
    requests, marked by a non-empty ``errors`` entry, stay out of the means.
    """
    if short_input < 1:
        raise ValueError(f"short_input must be positive, got {short_input}")
    if t0_s is not None and t0_s < 0:
        raise ValueError(f"t0_s must be non-negative, got {t0_s}")
    input_lens = _column(bench, "input_lens")
    ttfts = _column(bench, "ttfts")
    itls = _column(bench, "itls")
    count = len(input_lens)
    if len(ttfts) != count or len(itls) != count:
        raise ValueError("input_lens, ttfts, and itls must have the same length")
    errors = bench.get("errors")
    if errors is not None and len(errors) != count:
        raise ValueError("errors must have the same length as input_lens")

    grouped: dict[str, list[dict[str, Any]]] = {"short": [], "long": []}
    for index, input_length in enumerate(input_lens):
        kind = "long" if int(input_length) > short_input else "short"
        failed = bool(errors[index]) if errors is not None else False
        request_itls = itls[index]
        if not isinstance(request_itls, list):
            raise ValueError(f"itls[{index}] must be a list")
        tpot = sum(request_itls) / len(request_itls) if request_itls else None
        grouped[kind].append(
            {
                "failed": failed,
                "ttft": None if failed else float(ttfts[index]),
                "tpot": None if failed else tpot,
            }
        )

    groups: dict[str, dict[str, Any]] = {}
    short_tpots: list[float] = []
    for kind, rows in grouped.items():
        stats = _group_stats(rows)
        if kind == "short":
            short_tpots = stats.pop("tpots_s")
        else:
            stats.pop("tpots_s", None)
        groups[kind] = stats
    report: dict[str, Any] = {
        "short_input": short_input,
        "t0_s": t0_s,
        "goodput_limit_s": None if t0_s is None else 3 * t0_s,
        "output_throughput": _output_throughput(bench),
        "groups": groups,
        "short_within_3x_t0": _goodput(short_tpots, t0_s),
        "head_tp": None,
    }
    return report


def summarize_log(path: Path) -> dict[str, Any]:
    """Count ``HEAD_TP`` JSON records in a service log."""
    paired = 0
    rejected = 0
    invalid = 0
    skipped = 0
    by_d: dict[int, int] = {}
    bytes_by_source: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = _head_tp_record(line)
            if record is None:
                if HEAD_TP_PREFIX in line:
                    skipped += 1
                continue
            event = str(record.get("event", "")).lower()
            if event in {"pair", "paired"}:
                paired += 1
                if "d" in record and record["d"] is not None:
                    degree = int(record["d"])
                    by_d[degree] = by_d.get(degree, 0) + 1
                source = str(record.get("kv_source") or "unknown")
                copied = int(record.get("bytes_copied") or 0)
                bytes_by_source[source] = bytes_by_source.get(source, 0) + copied
                if record.get("invalid"):
                    invalid += 1
            elif event in {"reject", "rejected"}:
                rejected += 1
            else:
                skipped += 1
    return {
        "paired": paired,
        "rejected": rejected,
        "by_d": by_d,
        "bytes_by_source": bytes_by_source,
        "invalid": invalid,
        "skipped": skipped,
    }


def attach_log(report: dict[str, Any], path: Path | None) -> dict[str, Any]:
    if path is not None:
        report["head_tp"] = summarize_log(path)
    return report


def render_report(report: Mapping[str, Any]) -> str:
    """Plain-text table. Times are seconds, matching the bench JSON."""
    lines = [
        (
            f"{'group':<8} {'ok':>6} {'fail':>6} "
            f"{'ttft_mean_s':>12} {'ttft_p99_s':>12} "
            f"{'tpot_mean_s':>12} {'tpot_p99_s':>12}"
        )
    ]
    groups = report["groups"]
    for name in ("short", "long"):
        group = groups[name]
        lines.append(
            f"{name:<8} {group['completed']:>6} {group['failed']:>6} "
            f"{_fmt(group['ttft_mean_s']):>12} {_fmt(group['ttft_p99_s']):>12} "
            f"{_fmt(group['tpot_mean_s']):>12} {_fmt(group['tpot_p99_s']):>12}"
        )
    throughput = report["output_throughput"]
    lines.append(f"output_throughput_tok_s {_fmt(throughput)}")
    if report["t0_s"] is not None:
        lines.append(
            f"t0_s {_fmt(report['t0_s'])} "
            f"limit_3x_t0_s {_fmt(report['goodput_limit_s'])} "
            f"short_within_3x_t0 {_fmt(report['short_within_3x_t0'])}"
        )
    head_tp = report.get("head_tp")
    if head_tp:
        lines.append(
            "head_tp "
            f"paired {head_tp['paired']} "
            f"rejected {head_tp['rejected']} "
            f"invalid {head_tp['invalid']} "
            f"by_d {head_tp['by_d']} "
            f"bytes_by_source {head_tp['bytes_by_source']}"
        )
    return "\n".join(lines)


def load_bench(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear percentile, matching NumPy's default method on a sorted sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if pct < 0 or pct > 100:
        raise ValueError(f"percentile must be in [0, 100], got {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _column(bench: Mapping[str, Any], name: str) -> list[Any]:
    try:
        value = bench[name]
    except KeyError as exc:
        raise ValueError(f"bench JSON is missing {name!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if not row["failed"]]
    ttfts = [float(row["ttft"]) for row in completed if row["ttft"] is not None]
    tpots = [float(row["tpot"]) for row in completed if row["tpot"] is not None]
    return {
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "ttft_mean_s": _mean(ttfts),
        "ttft_p99_s": percentile(ttfts, 99) if ttfts else None,
        "tpot_mean_s": _mean(tpots),
        "tpot_p99_s": percentile(tpots, 99) if tpots else None,
        "tpots_s": tpots,
    }


def _goodput(tpots: Sequence[float], t0_s: float | None) -> float | None:
    if t0_s is None or not tpots:
        return None
    limit = 3 * t0_s
    return sum(tpot <= limit for tpot in tpots) / len(tpots)


def _output_throughput(bench: Mapping[str, Any]) -> float | None:
    if "output_throughput" in bench and bench["output_throughput"] is not None:
        return float(bench["output_throughput"])
    duration = bench.get("duration")
    output_lens = bench.get("output_lens")
    if duration and isinstance(output_lens, list) and output_lens:
        return float(sum(output_lens) / duration)
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _head_tp_record(line: str) -> dict[str, Any] | None:
    marker = line.find(HEAD_TP_PREFIX)
    if marker < 0:
        return None
    payload = line[marker + len(HEAD_TP_PREFIX) :].strip()
    start = payload.find("{")
    if start < 0:
        return None
    try:
        record, _ = json.JSONDecoder().raw_decode(payload[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    return record
