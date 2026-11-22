# SPDX-License-Identifier: Apache-2.0
"""Compare FA3 interpretations of the same rank-local, packed BF16 KV bytes."""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

BLOCK = 256
HEADS = 2  # Local half of Qwen3's four KV heads.
GROUP = 8  # Q heads per KV head.
DIM = 128
LAYOUTS = ("virtual256", "virtual512", "physical512")


@dataclass
class Inputs:
    cache: torch.Tensor
    q: torch.Tensor
    lengths: list[int]
    physical: torch.Tensor
    logical: list[torch.Tensor]


def make_inputs(lengths, device, seed, ordered):
    rng = torch.Generator().manual_seed(seed)
    counts = [math.ceil(n / (2 * BLOCK)) for n in lengths]
    total = sum(counts)
    # Physical page order must not change the logical Q/K/V samples.
    page_rng = torch.Generator().manual_seed(seed)
    ids = torch.arange(total) if ordered else torch.randperm(total, generator=page_rng)
    # Page zero is padding. Store finite sentinels in every unused position.
    cache = torch.full(
        (total + 1, HEADS, 2 * BLOCK, 2 * DIM), 9.0, dtype=torch.bfloat16
    )
    table = torch.zeros(len(lengths), max(counts), dtype=torch.int32)
    q = torch.randn(
        len(lengths), HEADS * GROUP, DIM, generator=rng, dtype=torch.bfloat16
    )
    logical = []
    for row, (n, pages) in enumerate(zip(lengths, ids.split(counts))):
        pages = pages + 1
        table[row, : len(pages)] = pages
        kv = torch.randn(n, HEADS, 2 * DIM, dtype=torch.bfloat16, generator=rng)
        # A wrong head/row or an all-zero result must also fail on long rows.
        kv[..., DIM:] += torch.tensor([-0.5, 0.5])[:, None] + (row % 5) * 0.125
        logical.append(kv)
        packed = torch.full(
            (len(pages) * 2 * BLOCK, HEADS, 2 * DIM), 9.0, dtype=torch.bfloat16
        )
        packed[:n] = kv
        cache[pages] = packed.view(-1, 2 * BLOCK, HEADS, 2 * DIM).transpose(1, 2)
    return Inputs(cache.to(device), q.to(device), lengths, table.to(device), logical)


@dataclass
class View:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    table: torch.Tensor
    used: torch.Tensor
    cu: torch.Tensor


def view(inputs, layout):
    cache, q, table = inputs.cache, inputs.q, inputs.physical
    head = torch.arange(HEADS, dtype=torch.int32, device=q.device)
    used = torch.tensor(inputs.lengths, dtype=torch.int32, device=q.device)
    if layout == "physical512":
        kv = cache.transpose(1, 2)
    else:
        q = q.view(-1, GROUP, DIM)
        used = used.repeat_interleave(HEADS)
        # Each row now describes one head of one request.
        ids = table[:, None, :] * HEADS + head[None, :, None]
        if layout == "virtual256":
            halves = torch.arange(2, dtype=torch.int32, device=q.device)
            ids = (2 * ids[..., None] + halves).flatten(-2)
            ids = ids[..., : math.ceil(max(inputs.lengths) / BLOCK)]
            page_size = BLOCK
        elif layout == "virtual512":
            page_size = 2 * BLOCK
        else:
            raise ValueError(layout)
        table = ids.reshape(q.shape[0], -1).clone(memory_format=torch.contiguous_format)
        kv = cache.view(-1, page_size, 1, 2 * DIM)
    k, v = kv.split(DIM, dim=-1)
    cu = torch.arange(len(used) + 1, dtype=torch.int32, device=q.device)
    assert k.data_ptr() == cache.data_ptr()
    assert v.data_ptr() == cache.data_ptr() + DIM * cache.element_size()
    assert q.data_ptr() == inputs.q.data_ptr()
    return View(q, k, v, table, used, cu)


def check_layout(inputs, data):
    """Decode the exposed page table and check every valid KV token exactly."""
    virtual = data.k.shape[2] == 1
    page_size = data.k.shape[1]
    for i, original in enumerate(inputs.logical):
        n = len(original)
        for h in range(HEADS):
            row, head = (i * HEADS + h, 0) if virtual else (i, h)
            assert data.used[row].item() == n
            pages = data.table[row, : math.ceil(n / page_size)].long()
            for cache, ref in zip((data.k, data.v), original.split(DIM, dim=-1)):
                actual = cache[pages, :, head].reshape(-1, DIM)[:n]
                torch.testing.assert_close(actual.cpu(), ref[:, h], atol=0, rtol=0)


def reference(inputs):
    """FP32 decode from the original chronological KV, independent of paging."""
    outputs = []
    for i, kv in enumerate(inputs.logical):
        q = inputs.q[i].cpu().float().view(HEADS, GROUP, DIM)
        k, v = kv.float().split(DIM, dim=-1)
        scores = torch.einsum("hgd,thd->hgt", q, k) * DIM**-0.5
        outputs.append(torch.einsum("hgt,thd->hgd", scores.softmax(-1), v))
    return torch.stack(outputs).reshape_as(inputs.q)


def prepare_metadata(data, split, split_counts, maximum):
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

    return get_scheduler_metadata(
        len(data.used),
        1,
        maximum,
        data.q.size(1),
        data.k.size(2),
        DIM,
        data.used,
        cu_seqlens_q=data.cu,
        page_size=data.k.size(1),
        num_splits=split,
        row_split_counts=split_counts,
    )


def capture(data, split, split_counts, maximum, metadata=None):
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    output = torch.empty_like(data.q)

    def run():
        flash_attn_varlen_func(
            data.q,
            data.k,
            data.v,
            max_seqlen_q=1,
            cu_seqlens_q=data.cu,
            max_seqlen_k=maximum,
            seqused_k=data.used,
            block_table=data.table,
            softmax_scale=DIM**-0.5,
            causal=False,
            fa_version=3,
            out=output,
            num_splits=split,
            row_split_counts=split_counts if metadata is None else None,
            scheduler_metadata=metadata,
            prepared_row_splits=metadata is not None and split_counts is not None,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph, output


def time_graph(graph, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    host_us = (time.perf_counter() - wall) * 1e6 / replays
    return start.elapsed_time(end) * 1000 / replays, host_us


def kernel_stage(name):
    if "prepare_varlen" in name:
        return "prepare"
    if "combine" in name.lower():
        return "combine"
    if "FlashAttnFwd" in name:
        return "attention"
    return "other"


def profile_graph(graph, rounds):
    from torch.profiler import ProfilerActivity, profile

    samples = []
    for _ in range(rounds):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        kernels = defaultdict(list)
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels[event.name].append(event.device_time_total)
        if not kernels:
            raise RuntimeError("No CUDA activities returned by CUPTI")
        samples.append(kernels)
    names = sorted(set().union(*samples))
    kernels = [
        dict(
            name=name,
            stage=kernel_stage(name),
            launches=[len(s.get(name, [])) for s in samples],
            median_us=statistics.median(sum(s.get(name, [])) for s in samples),
        )
        for name in names
    ]

    stage_times = {
        stage: statistics.median(
            sum(
                sum(values)
                for name, values in sample.items()
                if kernel_stage(name) == stage
            )
            for sample in samples
        )
        for stage in ("prepare", "attention", "combine", "other")
    }
    return kernels, stage_times


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[141698])
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ordered-pages", action="store_true")
    parser.add_argument(
        "--prepare-per-step",
        action="store_true",
        help="time layer replay after staging its FA3 schedule",
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument(
        "--profile-rounds", type=int, default=3, help="0 disables CUPTI"
    )
    parser.add_argument(
        "--layout-check-only",
        action="store_true",
        help="verify page addresses and bytes on CPU",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="emit one NVTX-marked replay per case for Nsight Compute",
    )
    args = parser.parse_args()
    if (
        min(args.lengths) < 1
        or not all(1 <= n <= 128 for n in args.splits)
        or min(args.rounds, args.replays) < 1
        or args.profile_rounds < 0
        or len(set(args.layouts)) != len(args.layouts)
    ):
        parser.error("invalid lengths, splits, layout list, or measurement counts")
    device = "cpu" if args.layout_check_only else "cuda"
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9
    ):
        parser.error(
            "GPU timing requires one Hopper GPU and the research FA3 extension"
        )
    inputs = make_inputs(args.lengths, device, args.seed, args.ordered_pages)
    layouts = {name: view(inputs, name) for name in args.layouts}
    for data in layouts.values():
        check_layout(inputs, data)
    print(
        json.dumps(
            dict(
                event="fixture",
                lengths=args.lengths,
                seed=args.seed,
                ordered_pages=args.ordered_pages,
                torch=torch.__version__,
                device="cpu" if device == "cpu" else torch.cuda.get_device_name(),
                views={
                    n: dict(
                        k_shape=list(v.k.shape),
                        k_stride=list(v.k.stride()),
                        table_shape=list(v.table.shape),
                        q_shape=list(v.q.shape),
                    )
                    for n, v in layouts.items()
                },
            )
        ),
        flush=True,
    )
    if args.layout_check_only:
        return
    ref = reference(inputs)
    names = list(layouts)
    for split in args.splits:
        # CUDA graphs keep input addresses; retain these external buffers through
        # timing, CUPTI profiling and the final replay of every layout.
        split_counts = {n: torch.full_like(layouts[n].used, split) for n in names}
        metadata = {
            n: prepare_metadata(layouts[n], split, split_counts[n], max(args.lengths))
            if args.prepare_per_step
            else None
            for n in names
        }
        cases = {
            n: capture(
                layouts[n], split, split_counts[n], max(args.lengths), metadata[n]
            )
            for n in names
        }
        errors = {}
        for name, (graph, output) in cases.items():
            for _ in range(5):
                graph.replay()
            actual = output.reshape_as(inputs.q).cpu().float()
            torch.testing.assert_close(actual, ref, atol=0.005, rtol=0.01)
            errors[name] = (actual - ref).abs().max().item()
        times = defaultdict(list)
        if not args.ncu:
            for i in range(args.rounds):
                # Rotate the first case to balance clock/order effects.
                order = names[i % len(names) :] + names[: i % len(names)]
                for name in order:
                    times[name].append(time_graph(cases[name][0], args.replays))
        for name, (graph, _) in cases.items():
            result = dict(
                layout=name,
                splits=split,
                max_abs_error=errors[name],
                prepare_per_step=args.prepare_per_step,
            )
            if args.ncu:
                with torch.cuda.nvtx.range(f"headtp_{name}_s{split}"):
                    graph.replay()
                    torch.cuda.synchronize()
            else:
                gpu, host = zip(*times[name])
                result.update(
                    gpu_median_us=statistics.median(gpu),
                    gpu_samples_us=gpu,
                    host_median_us=statistics.median(host),
                    host_samples_us=host,
                    replays_per_sample=args.replays,
                )
                if args.profile_rounds:
                    result["kernels"], result["stage_median_us"] = profile_graph(
                        graph, args.profile_rounds
                    )
            print(json.dumps(result), flush=True)
            graph.reset()


if __name__ == "__main__":
    main()
