"""CLI for traces, summaries, and scheme runs.

Dataset and scheme flags are installed only after ``--dataset`` or ``--scheme``
selects the plugin, so a new plugin can add flags without colliding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import datasets as datasets_pkg
from . import schemes as schemes_pkg
from .runner import execute, port_open
from .summarize import attach_log, load_bench, render_report, summarize_bench
from .trace import cycled_output_path, write_jsonl

_USAGE = """\
usage: python -m head_tp_eval <command>

commands:
  trace        write a timed-trace JSONL
  summarize    split a bench JSON into short and long latency
  run          expand a scheme and call vllm bench serve
  check-serve  exit 0 when the server port accepts connections

datasets: {datasets}
schemes: {schemes}
"""


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        _print_root_help()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "trace":
        return _cmd_trace(rest)
    if command == "summarize":
        return _cmd_summarize(rest)
    if command == "run":
        return _cmd_run(rest)
    if command == "check-serve":
        return _cmd_check_serve(rest)
    print(f"unknown command {command!r}", file=sys.stderr)
    _print_root_help()
    return 2


def _cmd_trace(argv: list[str]) -> int:
    if not argv or argv in (["-h"], ["--help"]):
        print("datasets: " + ", ".join(datasets_pkg.dataset_names()))
        print("trace requires --dataset NAME --output PATH plus that dataset's flags")
        return 0
    name, remaining = _extract_option(argv, "--dataset")
    if name is None:
        print("trace requires --dataset NAME", file=sys.stderr)
        return 2
    dataset = datasets_pkg.get_dataset(name)
    parser = argparse.ArgumentParser(
        prog=f"python -m head_tp_eval trace --dataset {name}"
    )
    parser.add_argument("--dataset", default=name)
    parser.add_argument("--output", type=Path, required=True)
    dataset.add_arguments(parser)
    args = parser.parse_args(remaining)
    params = _dataset_params(args)
    built = dataset.build(params)
    output = args.output
    if built.cycled:
        output = cycled_output_path(output)
    write_jsonl(output, built.rows)
    print(output)
    return 0


def _cmd_summarize(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m head_tp_eval summarize"
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--short-input", type=int, required=True)
    parser.add_argument(
        "--t0",
        type=float,
        default=None,
        help="Short-request mean TPOT in seconds from the baseline run.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional service log. HEAD_TP JSON lines are included when present.",
    )
    args = parser.parse_args(argv)
    report = summarize_bench(
        load_bench(args.bench), short_input=args.short_input, t0_s=args.t0
    )
    attach_log(report, args.log)
    print(render_report(report))
    return 0


def _cmd_run(argv: list[str]) -> int:
    if not argv or argv in (["-h"], ["--help"]):
        print("schemes: " + ", ".join(schemes_pkg.scheme_names()))
        print("run requires --scheme NAME --model NAME and that scheme's flags")
        return 0
    name, remaining = _extract_option(argv, "--scheme")
    if name is None:
        print("run requires --scheme NAME", file=sys.stderr)
        return 2
    scheme = schemes_pkg.get_scheme(name)
    parser = argparse.ArgumentParser(
        prog=f"python -m head_tp_eval run --scheme {name}"
    )
    parser.add_argument("--scheme", default=name)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("local_research/eval_runs"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-command",
        default=None,
        help="Text stored next to the trace. The server itself is not started.",
    )
    scheme.add_arguments(parser)
    args = parser.parse_args(remaining)
    specs = scheme.expand(args)
    return execute(
        specs,
        model=args.model,
        host=args.host,
        port=args.port,
        result_root=args.result_root,
        dry_run=args.dry_run,
        server_command=args.server_command,
    )


def _cmd_check_serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m head_tp_eval check-serve"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args(argv)
    if port_open(args.host, args.port, args.timeout):
        print(f"{args.host}:{args.port} is accepting connections")
        return 0
    print(f"{args.host}:{args.port} is not accepting connections", file=sys.stderr)
    return 1


def _dataset_params(args: argparse.Namespace) -> dict[str, object]:
    skipped = {"dataset", "output"}
    return {
        key: value
        for key, value in vars(args).items()
        if key not in skipped and not key.startswith("_")
    }


def _extract_option(argv: list[str], option: str) -> tuple[str | None, list[str]]:
    value: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            if value is not None:
                raise SystemExit(f"{option} was given more than once")
            value = argv[index + 1]
            index += 2
            continue
        prefix = option + "="
        if arg.startswith(prefix):
            if value is not None:
                raise SystemExit(f"{option} was given more than once")
            value = arg[len(prefix) :]
            index += 1
            continue
        rest.append(arg)
        index += 1
    return value, rest


def _print_root_help() -> None:
    print(
        _USAGE.format(
            datasets=", ".join(datasets_pkg.dataset_names()) or "(none)",
            schemes=", ".join(schemes_pkg.scheme_names()) or "(none)",
        ),
        end="",
    )


if __name__ == "__main__":
    raise SystemExit(main())
