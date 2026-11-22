# SPDX-License-Identifier: Apache-2.0
"""Measure per-row split plans for mixed full-head and head-TP decode on one H100."""

import argparse
import json
import math
import random
import statistics

import torch
from bench_fa3_kv_layout import (
    BLOCK,
    DIM,
    GROUP,
    View,
    capture,
    prepare_metadata,
    time_graph,
)

from vllm.v1.decode_tiles import minimax_split_counts, work_split_counts

# (number of shared requests, long length, number of background rows, short length)
TRAIN = [(1, n, 0, 0) for n in (256, 2048, 8192, 32768, 141698)] + [
    (1, 8192, 4, 512),
    (1, 32768, 8, 2048),
    (1, 141698, 8, 16384),
    (1, 141698, 32, 512),
    (4, 32768, 32, 1024),
    (1, 65536, 96, 128),
]
HOLDOUT = [(1, n, 0, 0) for n in (513, 12289, 65537, 196609)] + [
    (1, 12289, 6, 768),
    (1, 98305, 12, 4096),
    (1, 141698, 48, 1024),
    (2, 65537, 16, 8192),
    (1, 131073, 256, 128),
]


def fixture(shape, seed):
    nl, long_len, ns, short_len = shape
    lengths = [long_len] * nl + [short_len] * ns
    heads = [2] * nl + [4] * ns
    pages = [
        math.ceil(n / (2 * BLOCK if h == 2 else BLOCK)) for n, h in zip(lengths, heads)
    ]
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randperm(sum(pages), generator=gen, dtype=torch.int32) + 1
    cache = torch.full((sum(pages) + 1, 4, BLOCK, 2 * DIM), 9.0, dtype=torch.bfloat16)
    query = torch.randn(sum(heads), GROUP, DIM, dtype=torch.bfloat16, generator=gen)
    table = torch.zeros(sum(heads), math.ceil(max(lengths) / BLOCK), dtype=torch.int32)
    used = []
    references = []
    offset = 0
    for length, h, physical in zip(lengths, heads, ids.split(pages)):
        span = 2 * BLOCK if h == 2 else BLOCK
        raw = torch.randn(
            len(physical), h, span, 2 * DIM, dtype=torch.bfloat16, generator=gen
        )
        raw[:, :, :, DIM:] += torch.arange(h)[None, :, None, None] * 0.25
        cache[physical.long()] = raw.reshape(-1, 4, BLOCK, 2 * DIM)
        page_ids = (
            physical[:, None, None] * 4
            + torch.arange(h)[None, :, None] * (span // BLOCK)
            + torch.arange(span // BLOCK)
        )
        slots = page_ids.transpose(0, 1).reshape(h, -1)[:, : math.ceil(length / BLOCK)]
        table[offset : offset + h, : slots.shape[1]] = slots
        logical = raw.transpose(1, 2).reshape(-1, h, 2 * DIM)[:length].float()
        k, v = logical.split(DIM, -1)
        scores = (
            torch.einsum("hgd,thd->hgt", query[offset : offset + h].float(), k)
            * DIM**-0.5
        )
        references.append(torch.einsum("hgt,thd->hgd", scores.softmax(-1), v))
        offset += h
        used.extend([length] * h)
    k, v = cache.cuda().view(-1, BLOCK, 1, 2 * DIM).split(DIM, -1)
    data = View(
        query.cuda(),
        k,
        v,
        table.cuda(),
        torch.tensor(used, device="cuda", dtype=torch.int32),
        torch.arange(len(used) + 1, device="cuda", dtype=torch.int32),
    )
    return lengths, heads, data, torch.cat(references)


def candidates(lengths, heads, sms, work=False, block_n=128):
    blocks = [math.ceil(n / block_n) for n in lengths]
    base = sum(heads)
    choices = {}

    def add(name, counts, bound):
        key = (tuple(counts), bound)
        choices.setdefault(key, []).append(name)

    # The static graph bound also applies to an all-ones replay.
    for cap in (8, 64, 128):
        add(
            f"minimax{cap}",
            minimax_split_counts(blocks, heads, max(base, sms), cap),
            cap,
        )
    add("auto", [], 0)
    for s in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        add(f"uniform{s}", [min(b, s) for b in blocks], s)
    for extra in (0, 0.25, 0.5, 1, 1.5, 2, 3, 4):
        budget = base + round(sms * extra)
        cs = minimax_split_counts(blocks, heads, budget, 128)
        cap = 1 << (max(cs) - 1).bit_length()
        add(f"budget{extra}", cs, cap)
        if cap <= 64:
            add(f"fixed64_budget{extra}", cs, 64)
    # One shared graph with cap=64 may also replay the all-ones plan.
    add("fixed64_one", [1] * len(heads), 64)
    if work:
        for waves in (0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            for chunk in (1, 2, 4, 8, 16):
                add(
                    f"work_{waves}_{chunk}",
                    work_split_counts(blocks, heads, sms, 64, waves, chunk),
                    64,
                )
    return choices


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument(
        "--suite", choices=("grid", "work", "confirm", "cta"), default="grid"
    )
    parser.add_argument(
        "--block-n",
        type=int,
        default=128,
        help="must match the separately built SM90a kernel",
    )
    parser.add_argument("--prepare-per-step", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.replays, args.block_n) < 1:
        parser.error("rounds, replays and block-n must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error(
            "this experiment requires one Hopper GPU and the research FA3 extension"
        )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    sets = [("train", TRAIN), ("holdout", HOLDOUT)]
    if args.suite == "confirm":
        sets = [
            (
                "confirm",
                [
                    (1, 12345, 64, 256),
                    (1, 200001, 18, 3072),
                    (2, 90001, 7, 8192),
                    (3, 28673, 96, 512),
                    (1, 8001, 310, 128),
                    (0, 0, 24, 1024),
                    (1, 98001, 1, 65536),
                ],
            )
        ]
    if args.suite == "cta":
        sets = [
            (
                "cta",
                [
                    (1, 8193, 0, 0),
                    (1, 141698, 0, 0),
                    (1, 141698, 8, 16384),
                    (1, 141698, 32, 512),
                    (1, 131073, 256, 128),
                    (2, 90001, 7, 8192),
                ],
            )
        ]
    for subset, shapes in sets:
        for shape in shapes:
            lengths, heads, data, ref = fixture(shape, 11 + args.repeat)
            options = candidates(
                lengths, heads, sms, args.suite != "grid", args.block_n
            )
            if args.suite == "cta":
                keep = {
                    "auto",
                    "uniform8",
                    "uniform32",
                    "uniform64",
                    "uniform128",
                    "minimax64",
                    "work_0.75_1",
                }
                options = {
                    k: [n for n in v if n in keep]
                    for k, v in options.items()
                    if keep.intersection(v)
                }
            keys = list(options)
            records = []
            # Bound the memory of captured FP32 partial-result buffers. Randomize
            # candidate order independently in each repeat, then rotate within groups.
            random.Random(871 + args.repeat).shuffle(keys)
            for start in range(0, len(keys), 4):
                group = keys[start : start + 4]
                graphs = []
                for cs, cap in group:
                    vals = [s for s, h in zip(cs, heads) for _ in range(h)]
                    splits = (
                        torch.tensor(vals, dtype=torch.int32, device="cuda")
                        if cs
                        else None
                    )
                    metadata = (
                        prepare_metadata(data, cap, splits, max(lengths))
                        if args.prepare_per_step
                        else None
                    )
                    graph, out = capture(data, cap, splits, max(lengths), metadata)
                    for _ in range(5):
                        graph.replay()
                    torch.testing.assert_close(
                        out.cpu().float(), ref, rtol=0.01, atol=0.005
                    )
                    graphs.append((graph, out, splits, metadata))
                samples = [[] for _ in group]
                for r in range(args.rounds):
                    for j in range(len(group)):
                        i = (j + r) % len(group)
                        gpu, _ = time_graph(graphs[i][0], args.replays)
                        samples[i].append(gpu)
                for (cs, cap), (graph, _, _, _), times in zip(group, graphs, samples):
                    records.append(
                        dict(
                            names=options[(cs, cap)],
                            counts=cs,
                            cap=cap,
                            graph_us=statistics.median(times),
                            samples=times,
                        )
                    )
                    graph.reset()
            print(
                json.dumps(
                    dict(
                        subset=subset,
                        shape=shape,
                        repeat=args.repeat,
                        sms=sms,
                        block_n=args.block_n,
                        prepare_per_step=args.prepare_per_step,
                        candidates=records,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
