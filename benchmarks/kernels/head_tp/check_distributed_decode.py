"""torchrun --standalone --nproc-per-node=2 check_distributed_decode.py"""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from vllm.v1.worker.gpu.head_tp_decode import DecodePolicy, DecodeRow, HeadTPDecode


class SwitchingPolicy(DecodePolicy):
    def select(self, rows, *, peer_positions, rank, step):
        phase = (step - 1) % 7 + 1
        if not rows or phase == 4 or (phase <= 3 and rank != 0):
            return None
        return rows[-1].req_id if phase >= 5 else rows[0].req_id

    def splits(self, lengths, heads, *, block_n, sms, limit, rank, step):
        return [1 if step % 2 else limit] * len(lengths)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    device = torch.device("cuda", rank)
    h, g, d, rows, block = 4, 8, 128, 2, 256
    pages_per_request = 18
    table = torch.arange(
        1, 1 + rows * pages_per_request, device=device, dtype=torch.int32
    ).view(rows, pages_per_request)
    helper_begin = 1 + rows * pages_per_request
    helper_end = helper_begin + pages_per_request
    torch.manual_seed(2026 + rank)
    cache = torch.randn(
        helper_end, h, block, 2 * d, device=device, dtype=torch.bfloat16
    )
    decode = HeadTPDecode(
        [cache],
        h * g,
        pages_per_request * block,
        rows,
        pages_per_request * block,
        4,
        SwitchingPolicy({}),
        rank,
        dist.group.WORLD,
        cpu,
    )
    oracle = cache.clone()
    # Qwen's projections can have row strides larger than their logical width.
    qkv = cache.new_empty(rows, h * g + 2 * h, d).normal_()
    q, k, v = qkv.split((h * g, h, h), dim=1)
    slots = torch.full((rows,), -1, device=device, dtype=torch.long)
    out = q.new_empty(rows + 1, h * g, d)
    scale = d**-0.5

    def run():
        decode.write_decode_kv(k, v, cache, slots)
        visible = decode.forward(q, k, v, cache, out, scale, local_kv_written=True)
        assert visible.shape == q.shape and visible.data_ptr() == out.data_ptr()

    decode.prepare_decode_metadata(rows)
    with patch.object(decode, "paged_decode", wraps=decode.paged_decode) as paged:
        run()  # initialize communicator and check FA's actual output destination
        fa_out = paged.call_args.kwargs["out"]
        assert fa_out.data_ptr() == out.data_ptr()
        assert fa_out.shape == (rows * h + h // 2, g, d)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for step in range(1, 8):
        # Rank 1 starts without a local batch. Both later help each other.
        n = 0 if rank == 1 and step == 1 else rows
        positions = [block - 3 + step, 17 * block - 4 + step][:n]
        scheduled = [DecodeRow(f"r{rank}-{i}", p) for i, p in enumerate(positions)]
        lens = torch.tensor(
            [p + 1 for p in positions], device=device, dtype=torch.int32
        )
        selected = [
            0 if step <= 3 and owner == 0 else 1 if step >= 5 else None
            for owner in range(2)
        ]
        for owner, selected_row in enumerate(selected):
            old_id = decode.resident[owner]
            target = None if selected_row is None else f"r{owner}-{selected_row}"
            if old_id and target != old_id:
                old_row = int(old_id[-1])
                dense = table[old_row].tolist()
                position = (
                    positions[old_row]
                    if rank == owner
                    else (block - 3 + step if old_row == 0 else 17 * block - 4 + step)
                )
                decode.change_layout("unpack", owner, old_id, [], dense, position)
                if rank == owner:
                    actual = (
                        cache[dense].transpose(1, 2).reshape(-1, h, 2 * d)[:position]
                    )
                    expected = (
                        oracle[dense].transpose(1, 2).reshape(-1, h, 2 * d)[:position]
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if target is not None:
                at = block - 3 + step if selected_row == 0 else 17 * block - 4 + step
                needed = (at + 1 + 2 * block - 1) // (2 * block)
                first = helper_begin + owner * (pages_per_request // 2)
                packed = list(range(first, first + needed))
                if old_id != target:
                    decode.change_layout(
                        "pack", owner, target, packed, table[selected_row].tolist(), at
                    )
                    if rank == owner:
                        cache.index_fill_(0, table[selected_row].long(), float("nan"))
                elif decode.physical[owner] != packed:
                    decode.change_layout("grow", owner, target, packed, [], at)
                assert len(decode.physical[owner]) == needed
        decode.stage(scheduled, lens, rows, table, table_revision=0)
        q.normal_()
        k.normal_()
        v.normal_()
        ref = []
        slots.fill_(-1)
        for i, pos in enumerate(positions):
            page = table[i, pos // block]
            oracle[page, :, pos % block] = torch.cat((k[i], v[i]), dim=-1)
            if selected[rank] != i:
                slots[i] = page * block + pos % block
            keys, vals = (
                oracle[table[i].long()]
                .transpose(1, 2)
                .reshape(-1, h, 2 * d)[: pos + 1]
                .float()
                .split(d, -1)
            )
            scores = torch.einsum("hgd,thd->hgt", q[i].float().view(h, g, d), keys)
            ref.append(torch.einsum("hgt,thd->hgd", (scores * scale).softmax(-1), vals))
        graph.replay()
        # Inspect the token itself: a wrong write at a long position can have
        # too little softmax weight to trip the attention output tolerance.
        current_kv = torch.cat((k, v), dim=-1)
        both_kv = current_kv.new_empty(2 * rows, h, 2 * d)
        dist.all_gather_into_tensor(both_kv, current_kv)
        virtual = cache.view(-1, block, 2 * d)
        for i, pos in enumerate(positions):
            if i == selected[rank]:
                actual = virtual[decode.owner_pages, decode.owner_offset]
                expected = current_kv[i, : h // 2]
            else:
                actual = cache[table[i, pos // block], :, pos % block]
                expected = current_kv[i]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        peer = 1 - rank
        if selected[peer] is not None:
            expected = both_kv[peer * rows + selected[peer], h // 2 :]
            torch.testing.assert_close(
                virtual[decode.remote_pages, decode.offset], expected, rtol=0, atol=0
            )
        torch.testing.assert_close(cache[0], oracle[0], rtol=0, atol=0)
        if n:
            torch.testing.assert_close(
                out[:n].float().view(n, h, g, d),
                torch.stack(ref),
                rtol=0.01,
                atol=0.005,
            )
    print(f"rank {rank}: seven replays matched FP32, FA{decode.fa_version}", flush=True)
    torch.cuda.synchronize()
    graph.reset()
    del graph
    # Fill four logical blocks directly from chunked prefill. Neither worker
    # has a dense copy in the tested cache. Check every stored K and V element.
    for owner in (0, 1):
        decode.change_layout("drop", owner, "", [], [], 0)
    packed = [helper_begin, helper_begin + 1]
    decode.change_layout("pack", 0, "prefill", packed, [], 0)
    gen = torch.Generator().manual_seed(83)
    history = torch.randn(4 * block, h, 2 * d, generator=gen, dtype=torch.bfloat16).to(
        device
    )
    pos = 0
    for size in (130, 160, 250, 228, 256):
        scheduled = [DecodeRow("prefill", pos)] if rank == 0 else []
        count = size if rank == 0 else 1
        decode.stage(
            scheduled,
            None,
            count,
            table,
            counts=[size] if rank == 0 else [],
            mixed=True,
        )
        q = cache.new_empty(count, h * g, d).normal_()
        kv = history[pos : pos + count]
        out = decode.forward(
            q, kv[:, :, :d], kv[:, :, d:], cache, torch.empty_like(q), scale
        )
        if rank == 0:
            keys, vals = history[: pos + size].float().split(d, -1)
            scores = (
                torch.einsum("nhgd,thd->nhgt", q.float().view(size, h, g, d), keys)
                * scale
            )
            future = (
                torch.arange(pos + size, device=device)[None, :]
                > (pos + torch.arange(size, device=device))[:, None]
            )
            probs = scores.masked_fill(future[:, None, None, :], float("-inf")).softmax(
                -1
            )
            ref = torch.einsum("nhgt,thd->nhgd", probs, vals)
            torch.testing.assert_close(
                out.view_as(ref).float(), ref, rtol=0.01, atol=0.006
            )
        pos += size
    assert pos == 4 * block and len(decode.physical[0]) == 2
    shard = history[:, rank * (h // 2) : (rank + 1) * (h // 2)]
    expected = shard.reshape(2, 2 * block, h // 2, 2 * d).transpose(1, 2)
    torch.testing.assert_close(
        cache[packed].view_as(expected), expected, rtol=0, atol=0
    )
    print(
        f"rank {rank}: 1024 prefill tokens, 2 physical pages, head bytes exact",
        flush=True,
    )
    # Reorder dense rows without changing the bucket, then remap one head
    # page at the same position. Neither change may reuse stale addresses.
    decode.change_layout("drop", 0, "", [], [], 0)
    ids = [DecodeRow("A", 10), DecodeRow("B", 10)]
    lens = table.new_full((2,), 11)
    heads = torch.arange(h, device=device, dtype=table.dtype)
    for order, dense, revision in (
        (ids, table, 1),
        (ids[::-1], table.flip(0), 1),
        (ids[::-1], table.flip(0).roll(1, dims=1), 2),
    ):
        decode.stage(order, lens, rows, dense, table_revision=revision)
        torch.testing.assert_close(
            decode.table[: rows * h, 0].reshape(rows, h),
            dense[:, :1] * h + heads,
            atol=0,
            rtol=0,
        )
    print(f"rank {rank}: reordered and replaced page maps match", flush=True)
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
