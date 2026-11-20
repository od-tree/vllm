"""Run NanoEngine's pinned single-request GQA forward over the 5090 memcpy transport."""
import argparse
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from greenlet import getcurrent, greenlet
import torch

from attn_local import attn_local, decode_offsets
from bench_perf import place, reference
from cuda_ipc_p2p import enable_peers, memcpy_peer, p2p_backend
from layout_cp import BLOCK, DIM, KV, Q, cp_segments, full_batch
from measurement import measure
from workloads import CASES, QUICK


def load_upstream():
    root = Path(os.environ.get("NANOENGINE_ROOT", Path(__file__).resolve().parents[3] / "NanoEngine"))
    path = root / "tests/test_sp_attention_cudagraph.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "aed556d326712ccacc14bd3740fe04115539a6202f344a1bc4006e6c538c6544":
        raise ValueError("expected the NanoEngine 8abce29 GQA benchmark")
    import sys
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("nanoengine_sp_benchmark", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class MemcpyBuffer:
    def __init__(self, channel, width, rank, slots):
        self.channel = channel
        self.local_buffer = torch.zeros(2 * slots, width, device=rank, dtype=torch.bfloat16)

    def all_to_all_ll(self, x, mask, offsets=None, is_transpose=False):
        # The parent launches the transfer after both upstream rank functions
        # have staged their data. Graph replay contains no Python coordination.
        return getcurrent().parent.switch((self.channel, x))


def single_fixture(module, length, published_metadata):
    slots = 2  # The upstream microbenchmark reserves two slots for one owner row.
    half = length // 2
    full = full_batch((length,), 11)
    batches = []
    for rank in range(2):
        data = module.create_test_data(Q, DIM, KV, length, slots, 2, device=f"cuda:{rank}")
        pages = half // BLOCK
        for field, original in (("k_cache", full.k), ("v_cache", full.v)):
            data[field][:pages].copy_(original[rank * pages:(rank + 1) * pages])
        data["q"].copy_(full.q if rank == 0 else -full.q)
        qmask = data.pop("q_mask").bool().int()
        rmask = data.pop("res_lse_mask").bool().int()
        qmask[rank].zero_()
        rmask[rank].zero_()
        data.update(q_mask_adjusted=qmask, res_lse_mask_adjusted=rmask)
        # The published fixture masks every scatter to the return send input.
        # The partner must place its local partial in destination 0, slot 0.
        if rank == 1 and not published_metadata:
            for field in ("res_slice_get_to_buffer_input", "res_slice_fill_to_buffer_input"):
                data[field][0] = 0
            data["res_to_buffer_input_mask"][0] = 1
        data["attention_compute_bs"] = 1
        batches.append(data)
    return slots, batches, (reference(full), full.q[:0].float()), 1


def matrix_fixture(spec):
    """Supply the existing two-owner workloads to the upstream forward."""
    full = [full_batch(spec.owner, 11), full_batch(spec.partner, 29)]
    expected = tuple(reference(batch) for batch in full)
    segments = cp_segments(full[0], spec.n_long, full[1])[:2]
    slots = max(len(segment.lengths) for segment in segments)
    batches = []
    for rank in range(2):
        seg = place(segments[rank], rank)
        count = len(full[rank].lengths)
        rows = torch.arange(count, device=rank, dtype=torch.int32)
        longs = torch.arange(spec.n_long if rank else 0, device=rank, dtype=torch.int32)
        local = rows + (spec.n_long if rank else 0)
        mask = torch.ones_like(rows)
        q = full[rank].q.to(rank) if count else torch.zeros_like(full[0].q[:1], device=rank)
        qmask = torch.zeros(2, slots, device=rank, dtype=torch.int32)
        rmask = torch.zeros_like(qmask)
        lens = torch.zeros_like(qmask)
        lens[rank, :count] = seg.seqused[spec.n_long:] if rank else seg.seqused
        if rank == 0:
            lens[1, :spec.n_long] = segments[1].seqused[:spec.n_long].to(rank)
            qmask[1, :spec.n_long] = 1
        else:
            rmask[0, :spec.n_long] = 1
        # Empty index batches cannot launch every upstream Triton variant.
        # One masked slot represents an empty scatter, as in the single-row fixture.
        masked = torch.zeros(1, dtype=torch.int32, device=rank)
        batches.append(dict(q=q, k_cache=seg.k, v_cache=seg.v, block_tables=seg.table,
                            context_lens=seg.seqused, global_context_lens=lens,
                            q_mask_adjusted=qmask, res_lse_mask_adjusted=rmask,
                            q_offsets=torch.zeros(3, dtype=torch.int32, device=rank),
                            q_slice_get=rows if count else masked,
                            q_slice_fill=local if count else masked,
                            q_copy_mask=mask if count else masked,
                            res_slice_get_to_buffer_output=local if count else masked,
                            res_slice_fill_to_buffer_output=rank * slots + rows if count else masked,
                            res_to_buffer_output_mask=mask if count else masked,
                            res_slice_get_to_buffer_input=longs if rank else masked,
                            res_slice_fill_to_buffer_input=longs if rank else masked,
                            res_to_buffer_input_mask=torch.ones_like(longs) if rank else masked,
                            attention_compute_bs=len(seg.lengths)))
    return slots, batches, expected, spec.n_long


def case(module, fixture, sync):
    slots, batches, expected, n_long = fixture
    streams = [torch.cuda.Stream(device=r) for r in range(2)]
    contexts, calls, offsets, max_lengths, empty_rows = [], [], [], [], []
    transfers = {"q": [], "o": [], "lse": []}
    buffers = {c: [MemcpyBuffer(c, width, r, slots) for r in range(2)]
               for c, width in (("q", Q * DIM), ("o", Q * DIM), ("lse", Q))}
    for rank, data in enumerate(batches):
        contexts.append(SimpleNamespace(max_num_seqs=slots, dtype=torch.bfloat16,
                                        q_buffer=buffers["q"][rank], res_buffer=buffers["o"][rank],
                                        lse_buffer=buffers["lse"][rank]))
        qoffset = data["q_offsets"].tolist()[rank]
        for channel in transfers:
            mask = data["q_mask_adjusted" if channel == "q" else "res_lse_mask_adjusted"]
            for dst, flags in enumerate(mask.tolist()):
                rows = [i for i, active in enumerate(flags) if active]
                if not rows:
                    continue
                if rows != list(range(len(rows))):
                    raise ValueError("this workload adapter expects a contiguous routed prefix")
                first = 0 if channel == "q" else dst * slots
                target = qoffset if channel == "q" else rank * slots
                transfers[channel].append((rank, slice(first, first + len(rows)),
                                           dst, slice(target, target + len(rows))))
        data.update(num_heads=Q, head_dim=DIM, scale=DIM ** -0.5, sp_size=2)
        params = inspect.signature(module.sp_attention_forward_gqa).parameters
        calls.append(partial(module.sp_attention_forward_gqa, **{k: data[k] for k in params}))
        n = data["attention_compute_bs"]
        offsets.append(decode_offsets(n, rank))
        lengths = data["context_lens"][:n].tolist()
        max_lengths.append(max(lengths))
        empty_rows.append(0 in lengths)

    partials = {}

    def fa2(q, k, v, *, cache_seqlens, page_table, softmax_scale, causal, return_softmax_lse):
        if not (causal and return_softmax_lse and softmax_scale == DIM ** -0.5):
            raise ValueError("unsupported GQA call")
        rank = q.device.index
        o, lse = attn_local(q.squeeze(1), k, v, page_table, cache_seqlens, max_seqlen_k=max_lengths[rank],
                            offsets=offsets[rank], may_have_empty_rows=empty_rows[rank])
        partials[rank] = o, lse
        return o.unsqueeze(1), lse.T.unsqueeze(-1)

    module.get_sp_context = lambda: contexts[torch.cuda.current_device()]
    module.flash_attn_with_kvcache = fa2
    tokens = [torch.zeros(1, device=r) for r in range(2)]

    def step(mark):
        if sync == "nccl":
            torch.cuda.nccl.all_reduce(tokens, streams=streams)
            mark("entry_all_reduce")
        ranks = [greenlet(call) for call in calls]
        received = [(), ()]
        for channel in ("q", "o", "lse"):
            submissions = []
            for rank, worker in enumerate(ranks):
                torch.cuda.set_device(rank)
                name, payload = worker.switch(*received[rank])
                if name != channel:
                    raise RuntimeError(f"upstream collective order changed: {name}")
                submissions.append(payload)
            mark({"q": "q_stage", "o": "attention_and_return_stage", "lse": "lse_stage"}[channel])
            ready = [s.record_event() for s in streams]
            for src, row, dst, slot in transfers[channel]:
                # Upstream may have written the destination slot locally first.
                streams[src].wait_event(ready[dst])
                memcpy_peer(buffers[channel][dst].local_buffer[slot], submissions[src][row])
            received = [(b.local_buffer,) for b in buffers[channel]]
            mark(channel + "_route")
        out = []
        for rank, worker in enumerate(ranks):
            torch.cuda.set_device(rank)
            out.append(worker.switch(*received[rank]))
            if not worker.dead:
                raise RuntimeError("upstream spawned an unexpected collective")
        torch.cuda.set_device(0)
        mark("upstream_combine")
        return out

    errors = {}
    def check(out):
        partner_o, partner_lse = partials[1]
        # The upstream return path sends bf16 O and bf16 LSE. Both must be
        # copied bit-for-bit before assessing the numerical merge tolerance.
        for channel, value in (("o", partner_o), ("lse", partner_lse.T)):
            actual = buffers[channel][0].local_buffer[slots:slots + n_long].cpu()
            want = value[:n_long].to(torch.bfloat16).reshape(n_long, -1).cpu()
            torch.testing.assert_close(actual, want, rtol=0, atol=0,
                                       msg=lambda msg: f"partner {channel} return: {msg}")
        errors["max_abs_fp32"] = []
        for result, ref in zip(out, expected):
            got = result[:len(ref)].cpu().float()
            errors["max_abs_fp32"].append((got - ref).abs().max().item() if len(ref) else None)
            torch.testing.assert_close(got, ref, rtol=0, atol=1e-2)
    return step, streams, check, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, nargs="+", default=[32768])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--profile-rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--entry-sync", choices=("nccl", "graph"), default="nccl")
    parser.add_argument("--matrix", action="store_true", help="Supply the matched two-rank workloads")
    parser.add_argument("--published-metadata", action="store_true",
                        help="Use the published return scatter metadata and check its output")
    args = parser.parse_args()
    if args.matrix and args.published_metadata:
        parser.error("the published metadata is specific to the upstream single-request workload")
    if min(args.length) <= 0 or any(n % (2 * BLOCK) for n in args.length):
        parser.error("each length must split evenly into two whole-page segments")
    if min(args.rounds, args.profile_rounds, args.repeats) < 1:
        parser.error("measurement counts must be positive")
    if enable_peers([0, 1]) != "memcpy":
        parser.error("the 5090 adapter requires P2P_BACKEND=memcpy")
    module = load_upstream()
    with torch.inference_mode():
        for repeat in range(args.repeats):
            for spec in ((QUICK[-1], *CASES) if args.matrix else args.length):
                fixture = (matrix_fixture(spec) if args.matrix else
                           single_fixture(module, spec, args.published_metadata))
                work, streams, check, errors = case(module, fixture, args.entry_sync)
                name = spec.name if args.matrix else f"single_{spec}"
                timing = measure(work, streams, args.rounds, args.profile_rounds, check)
                print(json.dumps(dict(method="upstream-gqa-5090", upstream="8abce29",
                                      case=name, repeat=repeat+1, entry_sync=args.entry_sync,
                                      backend=p2p_backend(), seeds=[11,29] if args.matrix else [11],
                                      rank_requests=[len(ref) for ref in fixture[2]],
                                      return_metadata=("matched_workload" if args.matrix else
                                                       "published" if args.published_metadata else "partner_slot_fixed"),
                                      **errors, **timing)), flush=True)
                del fixture, work, streams, check


if __name__ == "__main__":
    main()
