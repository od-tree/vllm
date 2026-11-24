# SPDX-License-Identifier: Apache-2.0
"""Replay matched, rank-pinned decode batches from qwen3-original-v1/issue05."""

import argparse
import csv
import gzip
import hashlib
import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

LONG = 35009
TRACE_SHA256 = "ffb1c265bfa69fdc554ca5bc9d4f3b53653920e142158349d642e6813960ca02"
SHORT = (64, 70, 81, 97, 132, 164, 192, 222)


def request_one(url, model, row, rank, first_token):
    row_id = int(row["row"])
    n = int(row["input_length"])
    output_len = int(row["output_length"])
    prompt_bytes = hashlib.shake_256(
        f"headtp-qwen3-v1/20261122/{row_id}".encode()
    ).digest(2 * n)
    body = {
        "model": model,
        "prompt": struct.unpack(f"<{n}H", prompt_bytes),
        "temperature": 0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_tokens_as_token_ids": True,
        "logprobs": 1,
    }
    req = Request(
        url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    start = perf_counter()
    tokens, times = [], []
    usage = None
    with urlopen(req, timeout=600) as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for ch in event["choices"]:
                chunk = (ch.get("logprobs") or {}).get("tokens", [])
                if chunk:
                    tokens.extend(chunk)
                    times.extend([perf_counter()] * len(chunk))
                    if first_token is not None:
                        first_token.set()
    finish = perf_counter()
    assert len(tokens) == output_len and usage["completion_tokens"] == output_len
    assert usage["prompt_tokens"] == n
    return dict(
        row=row_id,
        rank=rank,
        n=n,
        generated=output_len,
        start=start,
        finish=finish,
        first=times[0],
        last=times[-1],
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True, help="Frozen issue05.csv.gz")
    ap.add_argument("--url", default="http://127.0.0.1:18548")
    ap.add_argument("--model", default="qwen-2507")
    ap.add_argument(
        "--repeats", type=int, default=3, help="Measured rounds after warmup"
    )
    cases = ("short4x2", "mixed4_idle", "mixed4x2")
    ap.add_argument("--case", choices=cases, help="Run a single batch shape")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    if hashlib.sha256(args.trace.read_bytes()).hexdigest() != TRACE_SHA256:
        ap.error("--trace must match qwen3-original-v1/issue05.csv.gz")
    with gzip.open(args.trace, "rt") as f:
        rows = {
            int(r["row"]): r
            for r in csv.DictReader(f)
            if int(r["row"]) in (LONG, *SHORT)
        }
    for rep in range(args.repeats + 1):
        for case in (args.case,) if args.case else cases:
            jobs = []
            with ThreadPoolExecutor(8) as pool:
                if case.startswith("mixed"):
                    ready = threading.Event()
                    anchor = pool.submit(
                        request_one, args.url, args.model, rows[LONG], 0, ready
                    )
                    anchor.add_done_callback(lambda _, event=ready: event.set())
                    jobs.append(anchor)
                    # Submit the background at the same decode boundary in both
                    # modes, so it cannot finish during the anchor's long prefill.
                    if not ready.wait(timeout=180):
                        raise TimeoutError("Long request did not reach decode")
                    if anchor.done():
                        anchor.result()
                    background = [(i, 0) for i in SHORT[:3]]
                    if case == "mixed4x2":
                        background.extend((i, 1) for i in SHORT[3:7])
                else:
                    background = [(i, j // 4) for j, i in enumerate(SHORT)]
                jobs.extend(
                    pool.submit(request_one, args.url, args.model, rows[i], rank, None)
                    for i, rank in background
                )
                records = [f.result() for f in jobs]
            print(json.dumps(dict(rep=rep, case=case, records=records)), flush=True)


if __name__ == "__main__":
    main()
