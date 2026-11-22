# SPDX-License-Identifier: Apache-2.0
"""Compare native DP and head-TP rank-local FA3 calls on one Hopper."""

import argparse
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from bench_fa3_kv_layout import profile_graph, time_graph
from bench_two_rank_attention import (
    CASES,
    B,
    D,
    H,
    Pages,
    Q,
    logical_inputs,
    native,
    reference,
)

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.head_tp_decode import DecodePolicy, DecodeRow, HeadTPDecode


def local_attention(data, rank, cap):
    long_q, long_kv = data[0][0]
    length = len(long_kv)
    group = Q // H
    half = H // 2
    segments = [kv for _, kv in data[rank]]
    batch = max(1, len(segments))
    n_pages = cdiv(length, 2 * B)
    pages = Pages(segments, length, n_pages, long_q.device)
    packed = list(range(pages.first_spare, len(pages.cache)))
    # Each physical page stores 512 consecutive tokens for each local KV head.
    dest = pages.cache[pages.first_spare :].view(n_pages, half, 2 * B, 2 * D)
    shard = long_kv[:, rank * half : (rank + 1) * half]
    full, tail = divmod(length, 2 * B)
    dest[:full].transpose(1, 2).copy_(
        shard[: full * 2 * B].reshape(full, 2 * B, half, 2 * D)
    )
    if tail:
        dest[full, :, :tail].copy_(shard[full * 2 * B :].transpose(0, 1))

    dec = HeadTPDecode(
        [pages.cache],
        Q,
        length,
        batch,
        length,
        cap,
        DecodePolicy({"tile": "work"}),
        rank,
        None,
        None,
    )
    # KV is already placed, so only register the packed addresses.
    dec.change_layout("pack", 0, "long", packed, [], 0)
    rows = [
        DecodeRow("long" if rank == 0 and i == 0 else f"local{i}", n - 1)
        for i, n in enumerate(pages.lengths)
    ]

    # Simulate only the host routing collective, before capture. Production
    # stage() constructs both the virtual rows (including padding) and splits.
    def stage_routes(routes, local, *, group):
        routes[rank] = local
        routes[1 - rank] = (length - 1, 1) if rank else (-1, 0)

    with patch("torch.distributed.all_gather_object", stage_routes):
        dec.stage(rows, pages.lens, batch, pages.table)

    local_q = (
        torch.stack([q for q, _ in data[rank]])
        if rows
        else torch.zeros_like(long_q[None])
    )
    remote_q = long_q[Q // 2 :] if rank else torch.zeros_like(long_q[: Q // 2])
    queries = torch.cat((local_q.reshape(-1, group, D), remote_q.view(half, group, D)))
    key, value = pages.cache.view(-1, B, 1, 2 * D).split(D, dim=-1)
    return (
        lambda: dec.paged_decode(queries, key, value, D**-0.5),
        dict(row_lengths=dec.used.tolist(), requested_splits=dec.splits.tolist()),
    )


def reassemble(owner, helper, lengths):
    """Return all real Q heads in the original owner-then-helper order."""
    left = owner[: len(lengths[0]) * H].reshape(-1, Q, D).clone()
    left[0, Q // 2 :] = helper[-H // 2 :].reshape(Q // 2, D)
    right = helper[: len(lengths[1]) * H].reshape(len(lengths[1]), Q, D)
    return torch.cat((left, right))


@torch.inference_mode()
def measure(args, case):
    lengths = CASES[case]
    data, digest = logical_inputs(lengths, args.seed, "cuda")
    paths = {}
    expected = {}
    for rank, name in enumerate(("owner", "helper")):
        rows = data[rank]
        if rows:
            q = torch.stack([query for query, _ in rows])
            kv = [history for _, history in rows]
            pages = Pages(kv, max(map(len, kv)), 0, q.device)
            forward, _ = native(pages, q)
            native_name = f"dp_{name}"
            paths[native_name] = forward, dict(row_lengths=pages.lengths)
            expected[native_name] = reference(rows, q, torch.stack([x[-1] for x in kv]))
        paths[name] = local_attention(data, rank, args.cap)

    graphs, outputs, metrics = {}, {}, {}
    for name, (forward, metadata) in paths.items():
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = forward()
        for _ in range(5):
            graph.replay()
        graphs[name] = graph
        metrics[name] = dict(**metadata, gpu_samples_us=[], host_samples_us=[])

    errors = {}
    split = reassemble(outputs["owner"], outputs["helper"], lengths)
    expected["reassembled"] = torch.cat(list(expected.values()))
    outputs["reassembled"] = split
    for name, ref in expected.items():
        out = outputs[name].float()
        torch.testing.assert_close(out, ref, atol=0.005, rtol=0.01)
        errors[name] = (out - ref).abs().max().item()

    names = list(paths)
    for i in range(args.rounds):
        offset = i % len(names)
        for name in names[offset:] + names[:offset]:
            gpu, host = time_graph(graphs[name], args.replays)
            metrics[name]["gpu_samples_us"].append(gpu)
            metrics[name]["host_samples_us"].append(host)
    for name, result in metrics.items():
        result["median_gpu_us"] = statistics.median(result["gpu_samples_us"])
        if args.profile_rounds:
            result["kernels"], result["stage_us"] = profile_graph(
                graphs[name], args.profile_rounds
            )

    def critical_samples(owner, helper):
        owner_times = metrics[owner]["gpu_samples_us"]
        if helper not in metrics:
            return owner_times
        return list(map(max, zip(owner_times, metrics[helper]["gpu_samples_us"])))

    dp_times = critical_samples("dp_owner", "dp_helper")
    sharded_times = critical_samples("owner", "helper")
    if "dp_helper" not in metrics:
        # Idle DP has no FA3 launch and contributes zero to the critical path.
        metrics["dp_helper"] = dict(
            row_lengths=[], median_gpu_us=0.0, gpu_samples_us=[], host_samples_us=[]
        )
    return dict(
        case=case,
        lengths=lengths,
        seed=args.seed,
        input_sha256=digest,
        gpu=torch.cuda.get_device_name(),
        torch_version=str(torch.__version__),
        tile="work",
        cap=args.cap,
        rounds=args.rounds,
        replays=args.replays,
        paths=metrics,
        max_abs_error=errors,
        estimated_dp_us=statistics.median(dp_times),
        estimated_sharded_us=statistics.median(sharded_times),
        dp_minus_sharded_us=statistics.median(
            d - h for d, h in zip(dp_times, sharded_times)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=7251)
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--profile-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.cap <= 128
        or min(args.rounds, args.replays) < 1
        or args.profile_rounds < 0
    ):
        parser.error("cap: 1..128, rounds and replays > 0, profile-rounds >= 0")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error("requires one Hopper GPU and the patched FA3 extension")
    torch.backends.cuda.matmul.allow_tf32 = False
    results = [
        measure(args, case) for case in (CASES if args.case == "all" else [args.case])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
