# SPDX-License-Identifier: Apache-2.0
"""Matched two-rank, one-layer Qwen3 GQA decode. See docs/features/head_tp_decode.md."""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch
import torch.distributed as dist

from vllm.utils.math_utils import cdiv

Q, H, D, B = 32, 4, 128, 256
CASES = {
    "smoke": ([513, 17], [129]),
    "idle": ([141698], []),
    "light": ([141698], [512] * 4),
    "busy": ([141698], [16384] * 8),
}


def logical_inputs(lengths, seed, rank):
    """Identical logical inputs independently generated for every implementation."""
    digest = hashlib.sha256()
    data = []
    for owner, batch in enumerate(lengths):
        rows = []
        for row, length in enumerate(batch):
            generator = torch.Generator().manual_seed(seed + owner * 1000 + row)
            q = torch.randn(Q, D, generator=generator, dtype=torch.bfloat16)
            kv = torch.randn(
                length, H, 2 * D, generator=generator, dtype=torch.bfloat16
            )
            for tensor in (q, kv):
                digest.update(memoryview(tensor.view(torch.uint8).numpy()))
            rows.append((q.to(rank), kv.to(rank)))
        data.append(rows)
    return data, digest.hexdigest()


def reference(rows, q, kv):
    """Unpaged FP32 grouped dot/softmax/value product; includes the current token."""
    outputs = []
    for i, (_, history) in enumerate(rows):
        complete = torch.cat((history[:-1], kv[i : i + 1]))
        k, v = complete.transpose(0, 1).float().split(D, dim=-1)
        scores = q[i].view(H, Q // H, D).float() @ k.transpose(1, 2)
        outputs.append(((scores * D**-0.5).softmax(-1) @ v).reshape(Q, D))
    return torch.stack(outputs) if outputs else q[:0].float()


class Pages:
    def __init__(self, segments, maximum, spare, rank):
        self.lengths = [len(s) for s in segments]
        counts = [cdiv(s, B) for s in self.lengths]
        self.first_spare = 1 + sum(counts)  # page zero is safe scratch for padding
        self.cache = torch.zeros(
            self.first_spare + spare, H, B, 2 * D, device=rank, dtype=torch.bfloat16
        )
        self.table = torch.zeros(
            max(1, len(counts)), cdiv(maximum, B), device=rank, dtype=torch.int32
        )
        self.lens = self.table.new_tensor(self.lengths or [0])
        self.k, self.v = self.cache.transpose(1, 2).split(D, dim=-1)
        self.dense = []
        first = 1
        for row, (segment, count) in enumerate(zip(segments, counts)):
            ids = list(range(first, first + count))
            self.dense.append(ids)
            self.table[row, :count] = self.table.new_tensor(ids)
            dest = self.cache[first : first + count].transpose(1, 2)
            full, tail = divmod(len(segment), B)
            dest[:full].copy_(segment[: full * B].view(full, B, H, 2 * D))
            if tail:
                dest[full, :tail].copy_(segment[full * B :])
            first += count

    def writer(self, mapping, k, v, packed_mask=None):
        import vllm._C_stable_libtorch  # noqa: F401 -- registers CUDA cache ops

        slots = [-1] * len(k)
        for row, segment in mapping:
            slots[row] = self.dense[segment][-1] * B + (self.lengths[segment] - 1) % B
        indices = torch.tensor(slots, device=k.device)
        scale = torch.ones((), device=k.device)

        def write():
            slots = (
                indices if packed_mask is None else indices.masked_fill(packed_mask, -1)
            )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                k, v, self.k, self.v, slots, "auto", scale, scale
            )

        return write


def native(pages, q):
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    n = len(pages.lengths)
    if not n:
        return lambda: q[:0], []
    maximum = max(pages.lengths)
    cu = torch.arange(n + 1, device=q.device, dtype=torch.int32)
    meta = fa.get_scheduler_metadata(
        n, 1, maximum, Q, H, D, pages.lens, cu_seqlens_q=cu, page_size=B, causal=True
    )
    out = torch.empty_like(q)

    def forward():
        return fa.flash_attn_varlen_func(
            q,
            pages.k,
            pages.v,
            1,
            cu,
            maximum,
            seqused_k=pages.lens,
            block_table=pages.table,
            causal=True,
            fa_version=3,
            out=out,
            scheduler_metadata=meta,
        )

    return forward, [(fa, "flash_attn_varlen_func", "fa3")]


def sharded(pages, q, k, v, maximum, rank, cpu, cap):
    from vllm.v1.worker.gpu.head_tp_decode import DecodePolicy, DecodeRow, HeadTPDecode
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    n = len(q)
    dec = HeadTPDecode(
        [pages.cache],
        Q,
        maximum,
        n,
        maximum,
        cap,
        DecodePolicy({"tile": "work"}),
        rank,
        dist.group.WORLD,
        cpu,
    )
    packed = list(range(pages.first_spare, len(pages.cache)))
    dec.change_layout(
        "pack", 0, "long", packed, pages.dense[0] if rank == 0 else [], maximum - 1
    )
    rows = [
        DecodeRow("long" if rank == 0 and i == 0 else f"local{i}", s - 1)
        for i, s in enumerate(pages.lengths)
    ]
    dec.stage(rows, pages.lens, n, pages.table)
    out = torch.empty_like(q)
    return (
        lambda: dec.forward(q, k, v, pages.cache, out, D**-0.5),
        [(dec, "_exchange", "exchange"), (fa, "flash_attn_varlen_func", "fa3")],
        dec.write_mask[:n],
    )


def nanocp(root, pages, q, count, prefix, slots, rank):
    root = root.resolve()
    sys.path.insert(0, str(root))
    path = root / "tests/test_sp_attention_cudagraph.py"
    spec = importlib.util.spec_from_file_location("nano_upstream_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.setup_distributed_context(attention_sp=2, rank=rank, world_size=2)
    module.setup_sp_context(slots, D, Q)
    ctx = module.get_sp_context()

    zero = q.new_zeros(1, dtype=torch.int32)
    rows = torch.arange(count, device=q.device, dtype=torch.int32)
    local = rows + rank  # helper computes the incoming prefix in row 0
    mask = torch.zeros(2, slots, device=q.device, dtype=torch.int32)
    result_mask = torch.zeros_like(mask)
    lens = torch.zeros_like(mask)
    lens[rank, :count] = pages.lens[rank : rank + count]
    if rank == 0:
        mask[1, 0] = 1
        lens[1, 0] = prefix
    else:
        result_mask[0, 0] = 1
    args = dict(
        q=q,
        k_cache=pages.k,
        v_cache=pages.v,
        context_lens=pages.lens,
        block_tables=pages.table,
        global_context_lens=lens,
        q_mask_adjusted=mask,
        res_lse_mask_adjusted=result_mask,
        q_slice_get=rows if count else zero,
        q_slice_fill=local if count else zero,
        q_copy_mask=torch.ones_like(rows) if count else zero,
        res_slice_get_to_buffer_output=local if count else zero,
        res_slice_fill_to_buffer_output=rank * slots + rows if count else zero,
        res_to_buffer_output_mask=torch.ones_like(rows) if count else zero,
        res_slice_get_to_buffer_input=zero,
        res_slice_fill_to_buffer_input=zero,
        res_to_buffer_input_mask=torch.full_like(zero, rank),
        q_offsets=zero.new_tensor([0, 1, 1]),
        attention_compute_bs=len(pages.lengths),
        num_heads=Q,
        head_dim=D,
        scale=D**-0.5,
        sp_size=2,
    )
    return (
        lambda: module.sp_attention_forward_gqa(**args),
        [
            (ctx.q_buffer, "all_to_all_ll", "q_exchange"),
            (module, "flash_attn_with_kvcache", "fa3"),
            (ctx.res_buffer, "all_to_all_ll", "o_exchange"),
            (ctx.lse_buffer, "all_to_all_ll", "lse_exchange"),
            (module, "inter_rank_gqa_fwd_batch_decode_combine_kv", "rank_combine"),
        ],
    )


def capture(write, forward, spans, fence):
    graph = torch.cuda.CUDAGraph()
    events = []

    def mark(label):
        event = torch.cuda.Event(enable_timing=True, external=True)
        event.record()
        events.append((label, event))

    def instrument(fn, label):
        def call(*args, **kwargs):
            mark(label + "_begin")
            result = fn(*args, **kwargs)
            mark(label + "_end")
            return result

        return call

    with ExitStack() as stack:
        for obj, attr, label in spans:
            stack.enter_context(
                patch.object(obj, attr, instrument(getattr(obj, attr), label))
            )
        with torch.cuda.graph(graph):
            # Upstream NanoCP uses this rank fence; apply it equally and time after it.
            dist.all_reduce(fence)
            mark("begin")
            write()
            if spans:
                mark("kv_written")
            out = forward()
            mark("end")
    return graph, out, events


def run(args, rank, cpu):
    lengths = CASES[args.case]
    maximum = lengths[0][0]
    data, digest = logical_inputs(lengths, args.seed, rank)
    local = data[rank]
    count = len(local)
    q = torch.stack([x[0] for x in local]) if count else data[0][0][0][None].clone()
    current = (
        torch.stack([kv[-1] for _, kv in local])
        if count
        else data[0][0][1][-1:].clone()
    )
    segments = [kv for _, kv in local]
    mapping = [(i, i) for i in range(count)]
    prefix = max(B, (maximum // 2) // B * B)
    spare = 0
    if args.method == "nanocp":
        if rank == 0:
            segments[0] = segments[0][prefix:]
        else:
            segments.insert(0, data[0][0][1][:prefix])
            mapping = [(i, i + 1) for i in range(count)]
    if args.method == "headtp":
        spare = cdiv(maximum, 2 * B)
    pages = Pages(segments, maximum, spare, rank)
    k, v = (t.contiguous() for t in current.split(D, dim=-1))
    if args.method == "nanocp":
        # Match the two separate NHD paged caches in the published GQA fixture.
        pages.k, pages.v = pages.k.contiguous(), pages.v.contiguous()
    packed_mask = None
    if args.method == "dp":
        forward, spans = native(pages, q)
    elif args.method == "headtp":
        forward, spans, packed_mask = sharded(
            pages, q, k, v, maximum, rank, cpu, args.cap
        )
    else:
        slots = max(len(lengths[0]), 1 + len(lengths[1]))
        forward, spans = nanocp(args.nano_root, pages, q, count, prefix, slots, rank)
    write = pages.writer(mapping, k, v, packed_mask) if count else lambda: None
    fence = q.new_zeros(1)
    for _ in range(3):
        dist.all_reduce(fence)
        write()
        forward()
    torch.cuda.synchronize()
    graph, out, events = capture(write, forward, [], fence)
    diagnostic, diag_out, detail = capture(write, forward, spans, fence)
    errors = []
    for changed in (False, True):
        if changed:
            # Make the current token visible even at 141k. Every KV head has
            # one Q head that strongly attends to the newly written K/V.
            q.neg_()
            k.copy_(q[:, :: Q // H] * 4)
            v.fill_(0.75)
        expected = reference(local, q, torch.cat((k, v), dim=-1))
        for g, output in ((graph, out), (diagnostic, diag_out)):
            g.replay()
            torch.cuda.synchronize()
            observed = output[:count].float()
            errors.append((observed - expected).abs().max().item() if count else 0.0)
            # NanoCP rounds partial LSE to BF16 before its rank combine.
            ok = torch.allclose(observed, expected, atol=0.008, rtol=0.02)
            failures = [None, None]
            dist.all_gather_object(failures, None if ok else errors[-1], group=cpu)
            if any(x is not None for x in failures):
                raise AssertionError(f"{args.method} FP32 check failed: {failures}")
    q.neg_()
    k.copy_(current[..., :D])
    v.copy_(current[..., D:])

    def sample(g, markers, samples):
        measurements = {}
        for _ in range(args.warmup):
            g.replay()
            torch.cuda.synchronize()
        for _ in range(samples):
            dist.barrier(group=cpu)
            start = perf_counter()
            g.replay()
            torch.cuda.synchronize()
            measurements.setdefault("host_us", []).append(
                (perf_counter() - start) * 1e6
            )
            for i, ((first, a), (last, b)) in enumerate(zip(markers, markers[1:])):
                measurements.setdefault(f"{i}.{first}..{last}", []).append(
                    a.elapsed_time(b) * 1000
                )
        return measurements

    timing = sample(graph, events, args.samples)
    detail_times = sample(diagnostic, detail, args.samples)
    records = [None, None]
    dist.all_gather_object(
        records,
        dict(
            timing=timing,
            stages_us={k: statistics.median(v) for k, v in detail_times.items()},
            max_abs=errors,
        ),
        group=cpu,
    )
    if rank == 0:
        critical = list(map(max, zip(*(r["timing"]["0.begin..end"] for r in records))))
        result = dict(
            method=args.method,
            case=args.case,
            lengths=lengths,
            seed=args.seed,
            input_sha256=digest,
            cap=args.cap if args.method == "headtp" else None,
            hardware=torch.cuda.get_device_name(),
            torch_version=str(torch.__version__),
            page_size=B,
            q_heads=Q,
            kv_heads=H,
            head_dim=D,
            median_slower_rank_us=statistics.median(critical),
            ranks=records,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{args.method} {args.case}: {statistics.median(critical):.2f} us",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["dp", "nanocp", "headtp"], required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--nano-root", type=Path, default=Path("../NanoEngine"))
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7251)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or args.samples < 1 or args.warmup < 0:
        parser.error("requires torchrun with two local ranks and positive sample count")
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("this comparison requires Hopper FA3 on both ranks")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    try:
        run(args, rank, cpu)
    finally:
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()
