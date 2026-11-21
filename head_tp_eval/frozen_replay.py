"""Freeze the NanoEngine length traces and export exact token-ID windows."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import struct
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "replays" / "qwen3-v1"
ORIGINAL = HERE / "replays" / "qwen3-original-v1"
SOURCES = {
    "short": "sharegpt4o-mixed-random-60k.csv",
    "issue01": "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
    "issue05": "sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
    "issue_pool": "Gemini_Issues_Stats_rename-shuffle.csv",
}
SOURCE_REV = "8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5"
CONTEXT = 262144
SEED = 20261122


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_bytes(row, length):
    # UInt16 values 0..65535 are ordinary tokens in the target Qwen tokenizer.
    key = f"headtp-qwen3-v1/{SEED}/{row}".encode("ascii")
    return hashlib.shake_256(key).digest(2 * length)


@contextmanager
def compressed_writer(path):
    with (
        path.open("wb") as f,
        gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="ascii", newline="") as stream,
    ):
        yield csv.writer(stream, lineterminator="\n")


def freeze(source, dest, *, original_lengths=False):
    context_limit = None if original_lengths else CONTEXT
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir()
        manifest = {
            "format": "headtp-qwen3-v1",
            "seed": SEED,
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "model_revision": "61c2ee12e2e018cd6cdcaa3713e6ffd19e7b8086",
            "source_revision": SOURCE_REV,
            "max_context": context_limit,
            "poisson_requests_per_second": 1,
            "prompts": (
                "SHAKE256(ascii(headtp-qwen3-v1/seed/zero_based_row)), "
                "little-endian uint16"
            ),
            "workloads": {},
        }
        if original_lengths:
            manifest["profile"] = "original"
            manifest["required_context"] = 0
        for name, filename in SOURCES.items():
            original = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(source),
                    "show",
                    f"{SOURCE_REV}:ae_scripts/dataset/{filename}",
                ]
            )
            if (source / filename).read_bytes() != original:
                raise ValueError(
                    f"Source differs from pinned NanoEngine revision: {filename}"
                )
            path = root / (name + ".csv.gz")
            digest = hashlib.sha256()
            rng = random.Random(SEED)
            seconds = 0.0
            count = clipped = longs = tokens = required_context = 0
            with (
                (source / filename).open(newline="") as f,
                compressed_writer(path) as writer,
            ):
                writer.writerow(
                    [
                        "row",
                        "arrival_us",
                        "source_input",
                        "input_length",
                        "output_length",
                        "class",
                    ]
                )
                for i, row in enumerate(csv.DictReader(f)):
                    raw, out = int(row["prompt_len"]), int(row["output_len"])
                    if (
                        raw <= 0
                        or out <= 0
                        or (context_limit is not None and out >= context_limit)
                    ):
                        raise ValueError((filename, i, raw, out))
                    length = (
                        raw if context_limit is None else min(raw, context_limit - out)
                    )
                    required_context = max(required_context, length + out)
                    kind = "long" if row.get("type") in ("issue", "long") else "short"
                    if i:
                        seconds += rng.expovariate(1)
                    writer.writerow([i, round(seconds * 1e6), raw, length, out, kind])
                    digest.update(token_bytes(i, length))
                    clipped += length != raw
                    longs += kind == "long"
                    tokens += length
                    count += 1
            manifest["workloads"][name] = dict(
                source=filename,
                source_sha256=sha(source / filename),
                sha256=sha(path),
                prompt_bytes_sha256=digest.hexdigest(),
                requests=count,
                long=longs,
                clipped=clipped,
                input_tokens=tokens,
            )
            if original_lengths:
                manifest["workloads"][name]["required_context"] = required_context
                manifest["required_context"] = max(
                    manifest["required_context"], required_context
                )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if dest.exists():
            if any(
                not (dest / p.name).is_file() or sha(dest / p.name) != sha(p)
                for p in root.iterdir()
            ):
                raise ValueError(
                    "Different freeze at destination; use a new version directory"
                )
        else:
            root.rename(dest)


def export(bundle, name, start, count, output, route_output):
    meta = json.loads((bundle / "manifest.json").read_text())
    path = bundle / (name + ".csv.gz")
    if (
        meta["format"] != "headtp-qwen3-v1"
        or meta["seed"] != SEED
        or sha(path) != meta["workloads"][name]["sha256"]
    ):
        raise ValueError("Unexpected freeze version or checksum")
    if start < 0 or count < 1 or start + count > meta["workloads"][name]["requests"]:
        raise ValueError("Window outside frozen trace")
    if route_output and route_output.resolve() == output.resolve():
        raise ValueError("Route overlay and trace need separate paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    routes = []
    with gzip.open(path, "rt") as f, output.open("w") as dst:
        first_us = None
        short_count = 0
        for row in csv.DictReader(f):
            i = int(row["row"])
            short = row["class"] == "short"
            rank = short_count % 2 if short else 0
            short_count += short
            if i < start:
                continue
            if i >= start + count:
                break
            length = int(row["input_length"])
            us = int(row["arrival_us"])
            if first_us is None:
                first_us = us
            ids = [v for (v,) in struct.iter_unpack("<H", token_bytes(i, length))]
            key = f"{name}/{i}"
            data = dict(
                request_key=key,
                timestamp=(us - first_us) / 1e6,
                input_length=length,
                output_length=int(row["output_length"]),
                prompt_token_ids=ids,
            )
            dst.write(json.dumps(data, separators=(",", ":")) + "\n")
            if route_output:
                routes.append(dict(request_key=key, rank=rank))
    if route_output:
        route_output.parent.mkdir(parents=True, exist_ok=True)
        route_output.write_text("".join(json.dumps(row) + "\n" for row in routes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    build = subs.add_parser("freeze")
    build.add_argument(
        "--source", type=Path, default=HERE.parents[1] / "NanoEngine/ae_scripts/dataset"
    )
    build.add_argument("--dest", type=Path)
    build.add_argument(
        "--original-lengths",
        action="store_true",
        help="Keep every source prompt and output length.",
    )
    cut = subs.add_parser("export")
    cut.add_argument("workload", choices=SOURCES)
    cut.add_argument("--bundle", type=Path, default=DEFAULT)
    cut.add_argument("--start", type=int, default=0)
    cut.add_argument("--count", type=int, required=True)
    cut.add_argument("--output", type=Path, required=True)
    cut.add_argument("--two-rank-overlay", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        dest = args.dest or (ORIGINAL if args.original_lengths else DEFAULT)
        freeze(args.source, dest, original_lengths=args.original_lengths)
    else:
        export(
            args.bundle,
            args.workload,
            args.start,
            args.count,
            args.output,
            args.two_rank_overlay,
        )


if __name__ == "__main__":
    main()
