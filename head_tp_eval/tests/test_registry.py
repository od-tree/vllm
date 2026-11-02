"""Plugin registration, the head-collaboration matrix, and bench argument parsing."""

import argparse
import ast
import json
import socket
import uuid
from pathlib import Path

from head_tp_eval.__main__ import main
from head_tp_eval.arrivals import (
    register_arrival,
    unregister_arrival,
)
from head_tp_eval.datasets import (
    get_dataset,
    register_dataset,
    unregister_dataset,
)
from head_tp_eval.datasets.base import TraceBuild
from head_tp_eval.runner import serve_args
from head_tp_eval.schemes import (
    get_scheme,
    register_scheme,
    unregister_scheme,
)
from head_tp_eval.schemes.base import RunSpec
from head_tp_eval.schemes.head_tp_matrix import HeadTPMatrix
from head_tp_eval.trace import TraceRow, read_jsonl


class _WidthDataset:
    name = "width_probe"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--width", type=int, default=1)

    def build(self, params):
        rows = [
            TraceRow(float(index), 16, 1, (f"w{index}",))
            for index in range(int(params["width"]))
        ]
        return TraceBuild(rows=rows)


class _OnceScheme:
    name = "once_probe"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--width", type=int, default=1)

    def expand(self, args: argparse.Namespace) -> list[RunSpec]:
        return [
            RunSpec(
                config="Z",
                dataset="width_probe",
                dataset_args={"width": args.width},
                result_name="probe",
            )
        ]


def test_new_dataset_and_scheme_run_without_touching_mixed(tmp_path: Path) -> None:
    register_dataset(_WidthDataset())
    register_scheme(_OnceScheme())
    try:
        assert get_dataset("width_probe").name == "width_probe"
        assert get_scheme("once_probe").name == "once_probe"
        code = main(
            [
                "run",
                "--scheme",
                "once_probe",
                "--model",
                "dummy",
                "--dry-run",
                "--width",
                "2",
                "--result-root",
                str(tmp_path),
            ]
        )
        assert code == 0
        rows = read_jsonl(tmp_path / "probe" / "trace.jsonl")
        assert len(rows) == 2
        command_path = tmp_path / "probe" / "client_command.txt"
        command = command_path.read_text(encoding="utf-8")
        assert "--dataset-name timed_trace" in command
        assert "--request-rate" not in command
    finally:
        unregister_dataset("width_probe")
        unregister_scheme("once_probe")


def test_new_arrival_can_be_registered() -> None:
    def zeros(**kwargs):
        return [0.0] * kwargs["num_requests"], False

    register_arrival("zeros_probe", zeros)
    try:
        built = get_dataset("mixed").build(
            {
                "num_requests": 3,
                "long_ratio": 0,
                "arrival": "zeros_probe",
                "seed": 0,
            }
        )
        assert [row.timestamp for row in built.rows] == [0.0, 0.0, 0.0]
    finally:
        unregister_arrival("zeros_probe")


def test_matrix_names_and_narrowed_cell() -> None:
    scheme = HeadTPMatrix()
    parser = argparse.ArgumentParser()
    scheme.add_arguments(parser)
    full = scheme.expand(parser.parse_args(["--config", "A"]))
    assert len(full) == 104
    names = {spec.result_name for spec in full}
    assert "A_0.05_8192_0.5_0" in names
    assert "A_0_8192_0.5_0" in names
    assert "A_1_32768_4_1" in names
    assert "A_0.01_8192_0.5_0_prefix0.5" in names

    narrowed = scheme.expand(
        parser.parse_args(
            [
                "--config",
                "C",
                "--no-controls",
                "--no-shared-prefix",
                "--long-ratios",
                "0.05",
                "--long-inputs",
                "8192",
                "--rates",
                "1",
                "--seeds",
                "0",
                "--num-requests",
                "20",
            ]
        )
    )
    assert len(narrowed) == 1
    spec = narrowed[0]
    assert spec.result_name == "C_0.05_8192_1_0"
    assert spec.dataset == "mixed"
    assert spec.dataset_args["num_requests"] == 20
    assert spec.dataset_args["request_rate"] == 1
    assert spec.dataset_args["prefix_fraction"] == 0


def test_dry_run_writes_the_smoke_trace(tmp_path: Path) -> None:
    code = main(
        [
            "run",
            "--scheme",
            "head_tp_matrix",
            "--model",
            "dummy",
            "--dry-run",
            "--result-root",
            str(tmp_path),
            "--config",
            "A",
            "--no-controls",
            "--no-shared-prefix",
            "--long-ratios",
            "0.05",
            "--long-inputs",
            "8192",
            "--rates",
            "1",
            "--seeds",
            "0",
            "--num-requests",
            "20",
        ]
    )
    assert code == 0
    rows = read_jsonl(tmp_path / "A_0.05_8192_1_0" / "trace.jsonl")
    assert len(rows) == 20
    assert sum(row["input_length"] == 8192 for row in rows) == 1


def test_closed_port_stops_a_real_run(tmp_path: Path) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    code = main(
        [
            "run",
            "--scheme",
            "head_tp_matrix",
            "--model",
            "dummy",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--result-root",
            str(tmp_path),
            "--config",
            "A",
            "--no-controls",
            "--no-shared-prefix",
            "--long-ratios",
            "0.05",
            "--long-inputs",
            "8192",
            "--rates",
            "1",
            "--seeds",
            "0",
            "--num-requests",
            "1",
        ]
    )
    assert code == 1
    assert not any(tmp_path.iterdir())


def test_check_serve_sees_an_open_port() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert main(["check-serve", "--port", str(port)]) == 0
    finally:
        sock.close()


def test_timed_trace_serve_args_parse(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    result_dir = tmp_path / "out"
    argv = serve_args(trace, 20, "dummy", "127.0.0.1", 8000, result_dir)
    assert "--request-rate" not in argv
    parser = argparse.ArgumentParser(prog="vllm bench serve")
    _load_add_cli_args()(parser)
    parsed = parser.parse_args(argv)
    assert parsed.dataset_name == "timed_trace"
    assert parsed.dataset_path == str(trace)
    assert parsed.num_prompts == 20
    assert parsed.save_result
    assert parsed.result_dir == str(result_dir)
    assert parsed.backend == "openai"


def _load_add_cli_args():
    """Load the current ``vllm bench serve`` parser without importing vLLM.

    ``vllm.benchmarks.serve`` imports torch. The argument definitions themselves
    only need the parser, so the test executes those functions against the real
    backend names.
    """
    repo = Path(__file__).resolve().parents[2]
    dataset_path = repo / "vllm/benchmarks/datasets/datasets.py"
    serve_path = repo / "vllm/benchmarks/serve.py"
    request_path = repo / "vllm/benchmarks/lib/endpoint_request_func.py"
    namespace = {
        "argparse": argparse,
        "json": json,
        "uuid": uuid,
        "FlexibleArgumentParser": argparse.ArgumentParser,
        "DEFAULT_NUM_PROMPTS": 1000,
        "ASYNC_REQUEST_FUNCS": {
            name: None for name in _backend_names(request_path)
        },
        "BFCLDataset": _BFCLDataset,
        "SpeedBench": _SpeedBench,
        "RandomMultiModalDataset": _RandomMultiModalDataset,
    }
    source = "\n\n".join(
        [
            _function_source(dataset_path, "add_random_dataset_base_args"),
            _function_source(
                dataset_path, "add_random_multimodal_dataset_args"
            ),
            _function_source(dataset_path, "add_dataset_parser"),
            _function_source(serve_path, "add_cli_args"),
        ]
    )
    exec(source, namespace)
    return namespace["add_cli_args"]


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(text, node)
            if segment is not None:
                return segment
    raise AssertionError(f"{name} was not found in {path}")


def _backend_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        value = None
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target == "ASYNC_REQUEST_FUNCS" and isinstance(value, ast.Dict):
            return [
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
    raise AssertionError(f"ASYNC_REQUEST_FUNCS was not found in {path}")


class _BFCLDataset:
    """Stub docstring for the upstream dataset parser."""

    DEFAULT_CATEGORIES = ("simple",)


class _SpeedBench:
    """Stub docstring for the upstream dataset parser."""


class _RandomMultiModalDataset:
    DEFAULT_BASE_ITEMS_PER_REQUEST = 1
    DEFAULT_NUM_MM_ITEMS_RANGE_RATIO = 0.0
    DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 255, "video": 1}
    DEFAULT_MM_ITEM_BUCKET_CONFIG = {(256, 256, 1): 1.0}
