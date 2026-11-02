"""Trace generation for the mixed length dataset."""

from pathlib import Path

from head_tp_eval.__main__ import main
from head_tp_eval.datasets.mixed import MixedDataset
from head_tp_eval.trace import (
    TRACE_FIELDS,
    assert_trace_contract,
    read_jsonl,
)


def test_seed0_mixed_trace_contract(tmp_path: Path) -> None:
    output = tmp_path / "smoke.jsonl"
    code = main(
        [
            "trace",
            "--dataset",
            "mixed",
            "--num-requests",
            "20",
            "--long-ratio",
            "0.05",
            "--prefix-fraction",
            "0",
            "--seed",
            "0",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    rows = read_jsonl(output)
    assert len(rows) == 20
    assert_trace_contract(rows, chunk_size=16)
    shorts = [row for row in rows if row["input_length"] == 1024]
    longs = [row for row in rows if row["input_length"] == 8192]
    assert len(shorts) == 19
    assert len(longs) == 1
    assert all(row["output_length"] == 256 for row in shorts)
    assert longs[0]["output_length"] == 1024
    assert rows[0]["timestamp"] == 0
    assert all(
        rows[index]["timestamp"] < rows[index + 1]["timestamp"] for index in range(19)
    )
    hash_ids = [item for row in rows for item in row["hash_ids"]]
    assert len(hash_ids) == len(set(hash_ids))
    assert list(rows[0]) == list(TRACE_FIELDS)


def test_trace_is_deterministic() -> None:
    params = {
        "num_requests": 20,
        "long_ratio": 0.05,
        "prefix_fraction": 0,
        "seed": 0,
    }
    dataset = MixedDataset()
    first = dataset.build(params)
    second = dataset.build(params)
    first_rows = [row.to_json() for row in first.rows]
    second_rows = [row.to_json() for row in second.rows]
    assert first_rows == second_rows
    assert first.cycled is False


def test_shared_prefix_is_only_on_selected_long_requests() -> None:
    built = MixedDataset().build(
        {
            "num_requests": 20,
            "long_ratio": 0.5,
            "prefix_fraction": 0.5,
            "prefix_ratio": 0.75,
            "long_input": 8192,
            "seed": 0,
        }
    )
    longs = [row for row in built.rows if row.input_length == 8192]
    assert len(longs) == 10
    shared = [row for row in longs if row.hash_ids[0] == "shared-c0"]
    assert len(shared) == 5
    prefix = shared[0].hash_ids[: 6144 // 16]
    assert all(row.hash_ids[: len(prefix)] == prefix for row in shared)
    shared_tail = shared[0].hash_ids[len(prefix)]
    assert all(row.hash_ids[len(prefix)] != shared_tail for row in shared[1:])
    unshared = [row for row in built.rows if row not in shared]
    for row in unshared:
        assert all(not item.startswith("shared-") for item in row.hash_ids)


def test_input_length_must_match_chunk_size() -> None:
    try:
        MixedDataset().build(
            {"num_requests": 1, "long_ratio": 0, "short_input": 1000, "seed": 0}
        )
    except ValueError as exc:
        assert "chunk size" in str(exc)
    else:
        raise AssertionError("misaligned input length should fail")


def test_burstgpt_cycles_intervals_and_renames_output(tmp_path: Path) -> None:
    csv_path = tmp_path / "burst.csv"
    csv_path.write_text(
        "Timestamp,Model,Request tokens,Response tokens,Total tokens\n"
        "10,GPT-4,100,20,120\n"
        "12,GPT-4,80,10,90\n"
        "30,GPT-3.5,100,20,120\n"
        "40,GPT-4,100,0,100\n"
        "45,GPT-4,50,5,55\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace.jsonl"
    code = main(
        [
            "trace",
            "--dataset",
            "mixed",
            "--num-requests",
            "5",
            "--long-ratio",
            "0",
            "--arrival",
            "burstgpt",
            "--burstgpt-csv",
            str(csv_path),
            "--seed",
            "0",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    cycled = tmp_path / "trace_cycled.jsonl"
    assert cycled.exists()
    assert not output.exists()
    rows = read_jsonl(cycled)
    assert [row["timestamp"] for row in rows] == [0.0, 2.0, 35.0, 37.0, 70.0]
    assert all(row["input_length"] == 1024 for row in rows)
