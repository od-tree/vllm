"""Timed-trace rows written for ``vllm bench serve --dataset-name timed_trace``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``TimedTrace`` expands each hash id into this many tokens unless the bench
# command overrides ``--timed-trace-chunk-hash-size``.
DEFAULT_CHUNK_SIZE = 16

TRACE_FIELDS = ("timestamp", "input_length", "output_length", "hash_ids")


@dataclass(frozen=True)
class TraceRow:
    """One JSONL record. Field names match ``TimedTrace``'s default labels."""

    timestamp: float
    input_length: int
    output_length: int
    hash_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "hash_ids": list(self.hash_ids),
        }


def require_chunk_multiple(length: int, chunk_size: int, name: str) -> None:
    """Raise if ``length`` is not a positive multiple of ``chunk_size``."""
    if chunk_size <= 0:
        raise ValueError(f"chunk size must be positive, got {chunk_size}")
    if length <= 0 or length % chunk_size != 0:
        raise ValueError(
            f"{name}={length} must be a positive multiple of chunk size {chunk_size}"
        )


def make_hash_ids(
    request_index: int,
    input_length: int,
    chunk_size: int,
    shared_prefix: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build hash ids that expand back to ``input_length`` tokens.

    ``shared_prefix`` occupies the leading chunks. The tail is unique to this
    request so unshared tokens do not collide in the prefix cache.
    """
    require_chunk_multiple(input_length, chunk_size, "input_length")
    n_chunks = input_length // chunk_size
    if len(shared_prefix) > n_chunks:
        raise ValueError(
            f"shared prefix has {len(shared_prefix)} chunks but "
            f"input_length={input_length} only holds {n_chunks}"
        )
    unique = tuple(
        f"r{request_index}-c{chunk}" for chunk in range(len(shared_prefix), n_chunks)
    )
    return tuple(shared_prefix) + unique


def cycled_output_path(path: Path) -> Path:
    """Mark a trace whose BurstGPT intervals wrapped around the CSV."""
    if path.stem.endswith("_cycled"):
        return path
    return path.with_name(f"{path.stem}_cycled{path.suffix}")


def write_jsonl(path: Path, rows: Sequence[TraceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def assert_trace_contract(rows: Sequence[Mapping[str, Any]], chunk_size: int) -> None:
    """Check the field names and the chunk alignment ``TimedTrace`` requires."""
    if chunk_size <= 0:
        raise ValueError(f"chunk size must be positive, got {chunk_size}")
    for index, row in enumerate(rows):
        missing = [field for field in TRACE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} is missing {missing}")
        extra = [field for field in row if field not in TRACE_FIELDS]
        if extra:
            raise ValueError(f"row {index} has unexpected fields {extra}")
        input_length = int(row["input_length"])
        require_chunk_multiple(input_length, chunk_size, f"row {index} input_length")
        hash_ids = row["hash_ids"]
        expected = input_length // chunk_size
        if not isinstance(hash_ids, list) or len(hash_ids) != expected:
            raise ValueError(
                f"row {index} hash_ids must contain {input_length // chunk_size} ids"
            )
