"""Call ``vllm bench serve`` for runs a scheme has already expanded.

The server is not started here. A closed port ends the command before any
request is sent. Arrival rate lives in the trace timestamps; the bench command
does not pass ``--request-rate``, because ``timed_trace`` replays those
timestamps itself.
"""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .datasets import get_dataset
from .schemes.base import RunSpec
from .trace import cycled_output_path, write_jsonl


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def serve_args(
    trace_path: Path,
    num_prompts: int,
    model: str,
    host: str,
    port: int,
    result_dir: Path,
) -> list[str]:
    """Arguments after ``vllm bench serve``."""
    return [
        "--backend",
        "openai",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dataset-name",
        "timed_trace",
        "--dataset-path",
        str(trace_path),
        "--num-prompts",
        str(num_prompts),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "bench.json",
    ]


def serve_command(
    trace_path: Path,
    num_prompts: int,
    model: str,
    host: str,
    port: int,
    result_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        *serve_args(trace_path, num_prompts, model, host, port, result_dir),
    ]


def execute(
    specs: list[RunSpec],
    *,
    model: str,
    host: str,
    port: int,
    result_root: Path,
    dry_run: bool = False,
    server_command: str | None = None,
) -> int:
    """Write each trace and, unless ``dry_run``, invoke bench serve."""
    if not specs:
        print("scheme expanded to zero runs", file=sys.stderr)
        return 1
    if not dry_run and not port_open(host, port):
        print(
            f"server is not accepting connections on {host}:{port}; "
            "start it before running the matrix",
            file=sys.stderr,
        )
        return 1
    for spec in specs:
        code = _execute_one(
            spec,
            model=model,
            host=host,
            port=port,
            result_root=result_root,
            dry_run=dry_run,
            server_command=server_command,
        )
        if code != 0:
            return code
    return 0


def _execute_one(
    spec: RunSpec,
    *,
    model: str,
    host: str,
    port: int,
    result_root: Path,
    dry_run: bool,
    server_command: str | None,
) -> int:
    run_dir = result_root / spec.result_name
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(spec.dataset)
    built = dataset.build(spec.dataset_args)
    trace_path = run_dir / "trace.jsonl"
    if built.cycled:
        trace_path = cycled_output_path(trace_path)
    write_jsonl(trace_path, built.rows)
    command = serve_command(
        trace_path, len(built.rows), model, host, port, run_dir
    )
    command_path = run_dir / "client_command.txt"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    params: dict[str, Any] = {
        "config": spec.config,
        "dataset": spec.dataset,
        "dataset_args": dict(spec.dataset_args),
        "result_name": spec.result_name,
        "trace": trace_path.name,
        "cycled": built.cycled,
    }
    (run_dir / "trace_params.json").write_text(
        json.dumps(params, indent=2) + "\n", encoding="utf-8"
    )
    if server_command is not None:
        (run_dir / "server_command.txt").write_text(
            server_command.rstrip() + "\n", encoding="utf-8"
        )
    if dry_run:
        print(f"dry-run {spec.result_name} -> {trace_path}")
        return 0
    print(f"run {spec.result_name}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"vllm bench serve failed for {spec.result_name} "
            f"with exit code {completed.returncode}",
            file=sys.stderr,
        )
    return completed.returncode
