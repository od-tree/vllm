"""Compose one decode-CP step: local FA2 on each rank, then the NanoCP combine.

The master segment contains every row, including short rows at full length.
The partner segment contains long-row suffixes only. Short rows are mask 0
on the partner, so they stay inside the master's single attention call.
Nothing here writes the head-TP exchange buffers.
"""

from __future__ import annotations

import torch

from attn_local import attn_local
from comm_cp import CpComm, as_device
from layout_cp import SegmentBatch
from merge_cp import combine_kv
from tile_cp import segment_tiles


def merge_halves(prefix, suffix, num_splits: int = 0,
                 max_k_prefix: int | None = None, max_k_suffix: int | None = None):
    """Two segments on one device. No route: both KV caches are already local."""
    if prefix.q.shape[0] != suffix.q.shape[0]:
        raise ValueError(
            f"prefix batch {prefix.q.shape[0]} != suffix batch {suffix.q.shape[0]}"
        )
    plan0 = segment_tiles(prefix) if num_splits < 1 else None
    plan1 = segment_tiles(suffix) if num_splits < 1 else None
    o0, l0 = attn_local(
        prefix.q, prefix.k, prefix.v, prefix.table, prefix.seqused,
        num_splits=num_splits, max_seqlen_k=max_k_prefix, tile_plan=plan0,
    )
    o1, l1 = attn_local(
        suffix.q, suffix.k, suffix.v, suffix.table, suffix.seqused,
        num_splits=num_splits, max_seqlen_k=max_k_suffix, tile_plan=plan1,
    )
    if o0.is_cuda:
        torch.cuda.set_device(o0.device)
    o_stack = torch.stack((o0, o1), dim=0)
    lse_stack = torch.stack((l0.transpose(0, 1), l1.transpose(0, 1)), dim=0).unsqueeze(-1)
    lens = torch.stack((prefix.seqused, suffix.seqused), dim=0)
    merged = torch.empty_like(o0)
    combine_kv(merged, o_stack, lse_stack, lens)
    return merged


def _stage_return(comm, rank: int, slots: torch.Tensor, out: torch.Tensor,
                  lse: torch.Tensor, moe_rank: int):
    """Indexed copies into the local buffer and a fresh input buffer.

    Does not launch. Every rank has to enter the O kernel before any rank
    launches LSE: the LSE launch touches this GPU, and that waits out an O
    kernel which is still spinning until the partner's O kernel is queued.
    """
    from copy_idx import copy_batch_indexed_triton

    device = out.device
    torch.cuda.set_device(device)
    world = comm.world
    comm_bs = comm.max_batch
    n = out.shape[0]
    slot_key = tuple(int(value) for value in slots.tolist())
    key = (rank, slot_key, moe_rank, device.index, n)
    cached = comm._return_cache.get(key)
    if cached is None:
        src_idx = torch.arange(n, device=device, dtype=torch.int32)
        slot_gpu = slots.to(device=device, dtype=torch.int32)
        ones = torch.ones(n, device=device, dtype=torch.int32)
        mask = torch.zeros(world, comm_bs, dtype=torch.int32)
        mask[moe_rank, slots.to(dtype=torch.long)] = 1
        cached = (src_idx, rank * comm_bs + slot_gpu, moe_rank * comm_bs + slot_gpu, ones, mask)
        comm._return_cache[key] = cached
    src_idx, dst_out, dst_in, ones, mask = cached
    copy_batch_indexed_triton(out, comm.staging["o"][rank], src_idx, dst_out, ones)
    copy_batch_indexed_triton(lse, comm.staging["lse"][rank], src_idx, dst_out, ones)
    o_in = torch.empty(world * comm_bs, out.shape[1], out.shape[2], dtype=out.dtype, device=device)
    lse_in = torch.empty(world * comm_bs, lse.shape[1], lse.shape[2], dtype=lse.dtype, device=device)
    copy_batch_indexed_triton(out, o_in, src_idx, dst_in, ones)
    copy_batch_indexed_triton(lse, lse_in, src_idx, dst_in, ones)
    return o_in, lse_in, mask


def _select_segment(segment: SegmentBatch, slots: torch.Tensor, full_batch: int) -> tuple:
    """KV rows are either already in route-slot order, or the full MoE batch."""
    rows = segment.seqused.shape[0]
    if rows == slots.numel():
        return segment.k, segment.v, segment.table, segment.seqused
    if rows == full_batch:
        index = slots.to(segment.k.device)
        return (
            segment.k,
            segment.v,
            segment.table.index_select(0, index),
            segment.seqused.index_select(0, index),
        )
    raise ValueError(
        f"segment has {rows} KV rows, route selected {slots.numel()}, MoE batch is {full_batch}"
    )


def run_routed(segments: list[SegmentBatch], moe_rank: int, devices, mask: torch.Tensor,
               num_splits: int = 0, max_ks: list[int] | None = None,
               comm: CpComm | None = None, sync: bool = True, marks: dict | None = None):
    """One decode step. KV already lives on each rank. Q, O, and LSE are routed.

    ``segments[r]`` is rank ``r``'s compacted pages, on ``devices[r]``. A
    segment is either the full MoE batch or already packed in route-slot
    order. ``mask[r, row] == 1`` means rank ``r`` holds a KV segment of that
    row. Q is read back from the hao_basic buffer, including on the source rank.
    The combine reads that buffer directly and returns O only.
    """
    world = len(devices)
    if len(segments) != world:
        raise ValueError(f"{len(segments)} segments for world {world}")
    q = segments[moe_rank].q
    if mask.shape != (world, q.shape[0]):
        raise ValueError(f"mask {tuple(mask.shape)} != {(world, q.shape[0])}")
    for rank, segment in enumerate(segments):
        expected = as_device(devices[rank])
        if segment.k.device != expected:
            raise RuntimeError(
                f"rank {rank} KV is on {segment.k.device}, expected {expected}. "
                "Place KV before the step; the route does not move it."
            )
    if comm is None:
        comm = CpComm(
            devices, max_batch=max(q.shape[0], 1),
            q_shape=tuple(q.shape[1:]), o_shape=tuple(q.shape[1:]),
            lse_shape=(q.shape[1], 1), q_dtype=q.dtype, o_dtype=q.dtype,
        )
    q_launch = comm.launch_q(moe_rank, q, mask=mask, sync=False)
    order = [moe_rank] + [rank for rank in range(world) if rank != moe_rank]
    produced = []
    for rank in order:
        slots = q_launch.slots_at(rank)
        if slots is None or slots.numel() == 0:
            continue
        comm.wait(q_launch, rank)
        q_rank = comm.buffer_rows("q", rank, moe_rank, slots)
        k, v, table, used = _select_segment(segments[rank], slots, q.shape[0])
        max_k = None if max_ks is None else max_ks[rank]
        out, lse = attn_local(
            q_rank, k, v, table, used, num_splits=num_splits, max_seqlen_k=max_k,
            tile_plan=segment_tiles(segments[rank]) if num_splits < 1 else None,
        )
        produced.append((rank, slots, out, lse))
    staged = []
    produced_ranks = set()
    for rank, slots, out, lse in produced:
        produced_ranks.add(rank)
        lse_rows = lse.transpose(0, 1).contiguous().unsqueeze(-1)
        staged.append((rank, _stage_return(comm, rank, slots, out, lse_rows, moe_rank)))
    # O from every rank, then LSE. One process cannot enter the next channel
    # while this GPU's O kernel is still waiting for a partner that has not
    # launched. A rank with no partial still has to enter the kernel.
    partial_launches = []
    for rank, (o_in, lse_in, mask) in staged:
        partial_launches.append(comm.launch_return("o", rank, o_in, mask, sync=False))
    for rank in range(world):
        if rank not in produced_ranks:
            comm.participate("o", rank)
    for rank, (o_in, lse_in, mask) in staged:
        partial_launches.append(comm.launch_return("lse", rank, lse_in, mask, sync=False))
    for rank in range(world):
        if rank not in produced_ranks:
            comm.participate("lse", rank)
    for launch in partial_launches:
        comm.wait(launch, moe_rank)
    if sync:
        for dst in q_launch.written:
            comm.wait(q_launch, dst)
            if comm.devices[dst].type == "cuda":
                torch.cuda.current_stream(comm.devices[dst]).synchronize()
        for launch in partial_launches:
            for dst in launch.written:
                comm.wait(launch, dst)
                if comm.devices[dst].type == "cuda":
                    torch.cuda.current_stream(comm.devices[dst]).synchronize()
    moe_device = q.device
    if moe_device.type == "cuda":
        torch.cuda.set_device(moe_device)
    batch, heads, dim = q.shape
    lens = torch.zeros(world, batch, dtype=torch.int32, device=moe_device)
    for rank, slots, _out, _lse in produced:
        from cuda_ipc_p2p import copy_tensor

        lens[rank].index_copy_(
            0, copy_tensor(slots, moe_device),
            copy_tensor(segments[rank].seqused, moe_device, dtype=torch.int32),
        )
    out = torch.empty(batch, heads, dim, dtype=q.dtype, device=moe_device)
    if marks is not None:
        marks["merge_begin"] = torch.cuda.Event(True)
        marks["merge_begin"].record()
    combine_kv(
        out,
        comm.buffers["o"][moe_rank][:, :batch],
        comm.buffers["lse"][moe_rank][:, :batch],
        lens,
    )
    if marks is not None:
        marks["merge_end"] = torch.cuda.Event(True)
        marks["merge_end"].record()
    return out


def short_rows(full, n_long: int, num_splits: int = 0, max_seqlen_k: int | None = None):
    """Full-length FA2 for rows after ``n_long``. The routed path does not use this."""
    if n_long >= full.q.shape[0]:
        empty = full.q[:0]
        return empty, torch.empty(16, 0, dtype=torch.float32, device=full.q.device)
    return attn_local(
        full.q[n_long:], full.k, full.v, full.table[n_long:], full.seqused[n_long:],
        num_splits=num_splits, max_seqlen_k=max_seqlen_k,
    )


def assemble(long_out: torch.Tensor, short_out: torch.Tensor) -> torch.Tensor:
    if short_out.numel() == 0:
        return long_out
    return torch.cat((long_out, short_out), dim=0)
