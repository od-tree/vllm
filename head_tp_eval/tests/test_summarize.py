"""Latency summary over a bench JSON, with an optional HEAD_TP log."""

import json
from pathlib import Path

import pytest

from head_tp_eval.__main__ import main
from head_tp_eval.summarize import load_bench, summarize_bench


def _bench() -> dict:
    return {
        "input_lens": [1024, 1024, 8192, 8192],
        "output_lens": [256, 256, 1024, 0],
        "ttfts": [0.1, 0.3, 0.4, 0.0],
        "itls": [[0.01, 0.03], [0.05], [0.2, 0.4], []],
        "latencies": [1.0, 1.2, 2.0, 0.1],
        "start_times": [0.0, 0.1, 0.2, 0.3],
        "errors": ["", "", "", "boom"],
        "output_throughput": 12.5,
    }


def test_short_and_long_ttft_and_tpot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(_bench()), encoding="utf-8")
    report = summarize_bench(load_bench(path), short_input=1024, t0_s=0.01)
    short = report["groups"]["short"]
    long = report["groups"]["long"]
    assert short["completed"] == 2
    assert short["failed"] == 0
    assert short["ttft_mean_s"] == pytest.approx(0.2)
    assert short["ttft_p99_s"] == pytest.approx(0.298)
    assert short["tpot_mean_s"] == pytest.approx(0.035)
    assert short["tpot_p99_s"] == pytest.approx(0.0497)
    assert long["completed"] == 1
    assert long["failed"] == 1
    assert long["ttft_mean_s"] == pytest.approx(0.4)
    assert long["tpot_mean_s"] == pytest.approx(0.3)
    assert report["short_within_3x_t0"] == pytest.approx(0.5)
    assert report["output_throughput"] == pytest.approx(12.5)
    assert report["head_tp"] is None

    log_path = tmp_path / "server.log"
    log_path.write_text(
        'INFO HEAD_TP {"event": "pair", "d": 2, "kv_source": "cpu_pool", '
        '"bytes_copied": 100, "invalid": true}\n'
        'HEAD_TP {"event": "reject", "d": 2}\n'
        "HEAD_TP not-json\n",
        encoding="utf-8",
    )
    code = main(
        [
            "summarize",
            "--bench",
            str(path),
            "--short-input",
            "1024",
            "--t0",
            "0.01",
            "--log",
            str(log_path),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "short" in text
    assert "long" in text
    assert "0.200000" in text
    assert "0.300000" in text
    assert "paired 1" in text
    assert "rejected 1" in text


def test_boundary_uses_short_input_length() -> None:
    bench = {
        "input_lens": [1024, 1025],
        "ttfts": [0.2, 0.8],
        "itls": [[0.1], [0.4]],
    }
    report = summarize_bench(bench, short_input=1024)
    assert report["groups"]["short"]["completed"] == 1
    assert report["groups"]["long"]["ttft_mean_s"] == 0.8
    assert report["short_within_3x_t0"] is None
