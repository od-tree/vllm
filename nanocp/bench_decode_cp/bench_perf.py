"""Compare DP, head-TP and NanoCP-style GQA on identical two-rank inputs."""
import argparse
import importlib.util
import json
import os
from itertools import groupby
from pathlib import Path

import torch

from attn_local import decode_offsets, paged_fwd
from comm_cp import CpComm, enable_peers
from cuda_ipc_p2p import assign_tensor, p2p_backend
from decode_cp import run_routed
from layout_cp import BLOCK, DIM, KV, Q, SegmentBatch, cp_segments, full_batch
from measurement import measure
from tile_cp import device_tiles, segment_tiles
from workloads import CASES, QUICK


def tensors(batch):
    return batch.q, batch.k, batch.v, batch.table, batch.seqused


def place(batch, rank):
    result = SegmentBatch(*(t.to(rank) for t in tensors(batch)), batch.which,
                          batch.lengths, batch.n_long)
    result.local_tail = batch.local_tail
    return result


def reference(batch):
    """FP32 grouped attention on the original CPU pages, one query per row."""
    outputs = []
    for row, length in enumerate(batch.lengths):
        ids = batch.table[row, :(length + BLOCK - 1) // BLOCK].long()
        k = batch.k[ids].reshape(-1, KV, DIM)[:length].float()
        v = batch.v[ids].reshape(-1, KV, DIM)[:length].float()
        q = batch.q[row].reshape(KV, Q // KV, DIM).float()
        scores = torch.einsum("hgd,thd->hgt", q, k) * (DIM ** -0.5)
        out = torch.einsum("hgt,thd->hgd", scores.softmax(-1), v)
        outputs.append(out.reshape(Q, DIM))
    return torch.stack(outputs) if outputs else batch.q.float()


def local_operator(batch, rank):
    if not batch.lengths:
        return lambda: batch.q
    rows = len(batch.lengths)
    offsets, zeros = decode_offsets(rows, rank)
    out = torch.empty_like(batch.q)
    tiles = segment_tiles(batch)
    maximum = max(batch.lengths)
    if maximum == 0:
        return lambda: out.zero_()

    def run():
        paged_fwd(*tensors(batch), offsets, zeros, out, maximum, tile_plan=tiles)
        return out
    return run


def local_step(batches):
    ops = [local_operator(batch, rank) for rank, batch in enumerate(batches)]

    def step(mark):
        outputs = [op() for op in ops]
        mark("attention")
        return outputs
    return step


def cp_step(spec, inputs, compute_only):
    master, partner, mask = cp_segments(inputs[0], spec.n_long, inputs[1])
    segments = [place(master, 0), place(partner, 1)]
    if compute_only:
        return local_step(segments)
    shape = (Q, DIM)
    comm = CpComm([0, 1], len(master.lengths), shape, shape, (Q, 1))
    max_ks = [max(batch.lengths) for batch in segments]
    # Precompute route metadata outside all replays.
    lens = torch.zeros(2, len(master.lengths), dtype=torch.int32)
    lens[0] = master.seqused
    lens[1, :spec.n_long] = partner.seqused[:spec.n_long]
    lens = lens.to(0)
    for batch in segments:
        segment_tiles(batch)

    def step(mark):
        local = {}
        owner = run_routed(segments, 0, [0, 1], mask, comm=comm, max_ks=max_ks,
                           sync=False, local_outputs=local, context_lens=lens, mark=mark)
        return owner, local.get(1, segments[1].q[:0])
    return step


def head_backend():
    root = Path(os.environ.get("VLLM_ROOT", Path(__file__).resolve().parents[3] / "vllm"))
    default = root / "local_research/vendor/flash-attention"
    path = Path(os.environ.get("FLASH_ATTN_ROOT", default)) / "tests/head_tp_decode.py"
    module_spec = importlib.util.spec_from_file_location("head_tp_decode", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def head_step(spec, inputs, compute_only):
    module = head_backend()
    module._enable_peers()
    case = module.HeadTPDecode(module._ops(), spec.owner, spec.n_long, spec.partner,
                               inputs=[tensors(batch) for batch in inputs], tile_plan=device_tiles)
    case.pack()
    if not compute_only:
        case.prepare_exchange()
    streams = [case.s0, case.s1]
    if compute_only:
        assign_tensor(case.partner[0][:spec.n_long], case.owner[0][:spec.n_long])

    return (case.compute_step if compute_only else case.decode_step), streams


def check_outputs(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("missing owner outputs")
    for rank, (got, ref) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got.cpu().float(), ref, atol=1e-2, rtol=0,
                                   msg=lambda msg: f"owner rank {rank}: {msg}")


def run_case(spec, methods, args, compute_only):
    inputs = [full_batch(spec.owner, 11), full_batch(spec.partner, 29)]
    inputs[0].n_long = spec.n_long
    expected = [reference(batch) for batch in inputs] if args.check or args.check_only else None
    for method in methods:
        if method == "head":
            work, streams = head_step(spec, inputs, compute_only)
        else:
            streams = [torch.cuda.Stream(device=i) for i in range(2)]
            if method == "cp":
                work = cp_step(spec, inputs, compute_only)
            else:
                work = local_step([place(batch, i) for i, batch in enumerate(inputs)])
        check = None if expected is None else lambda out: check_outputs(out, expected)
        result = measure(work, streams, args.rounds, args.profile_rounds, check, args.check_only)
        print(json.dumps(dict(case=spec.name, method=method, compute_only=compute_only,
                              checked=expected is not None, rank_requests=[len(batch.lengths) for batch in inputs],
                              rank_tokens=[sum(batch.lengths) for batch in inputs], n_long=spec.n_long,
                              lengths_rle=[[(length, sum(1 for _ in rows))
                                            for length, rows in groupby(batch.lengths)]
                                           for batch in inputs],
                              seeds=[11, 29], backend=p2p_backend(), torch_version=torch.__version__,
                              devices=[torch.cuda.get_device_name(i) for i in range(2)], **result)),
              flush=True)
        del work, streams


def main(compute_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[s.name for s in QUICK + CASES])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--method", choices=("all", "dp", "head", "cp"), default="all")
    parser.add_argument("--check", action="store_true", help="compare both owners with FP32 attention")
    parser.add_argument("--check-only", action="store_true", help="validate graph outputs and skip profiling")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--profile-rounds", type=int, default=3)
    args = parser.parse_args()
    if min(args.rounds, args.profile_rounds) < 1:
        parser.error("timing and profiling rounds must be positive")
    if (args.check or args.check_only) and compute_only:
        parser.error("use bench_perf.py --check to validate assembled outputs")
    if torch.cuda.device_count() < 2:
        parser.exit(2, "two CUDA GPUs required for graph measurements\n")
    enable_peers([0, 1])
    cases = QUICK if args.quick else CASES
    if args.case:
        cases = [spec for spec in QUICK + CASES if spec.name == args.case]
    methods = ("dp", "head", "cp") if args.method == "all" else (args.method,)
    with torch.inference_mode():
        for spec in cases:
            run_case(spec, methods, args, compute_only)


if __name__ == "__main__":
    main()
