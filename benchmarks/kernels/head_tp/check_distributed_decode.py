"""torchrun --standalone --nproc-per-node=2 check_distributed_decode.py"""

import os

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
    helper_end = helper_begin + pages_per_request // 2
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
    decode.set_pages(list(range(helper_begin, helper_end)))
    q = cache.new_empty(rows, h * g, d).normal_()
    k = cache.new_empty(rows, h, d).normal_()
    v = torch.empty_like(k).normal_()
    out = torch.empty_like(q)
    scale = d**-0.5

    def run():
        decode.forward(q, k, v, cache, out, scale)

    run()  # initialize communicator before capture
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
        decode.stage(scheduled, lens, rows, table, set())
        q.normal_()
        k.normal_()
        v.normal_()
        ref = []
        for i, pos in enumerate(positions):
            page = table[i, pos // block]
            cache[page, :, pos % block] = torch.cat((k[i], v[i]), dim=-1)
            keys, vals = (
                cache[table[i].long()]
                .transpose(1, 2)
                .reshape(-1, h, 2 * d)[: pos + 1]
                .float()
                .split(d, -1)
            )
            scores = torch.einsum("hgd,thd->hgt", q[i].float().view(h, g, d), keys)
            ref.append(torch.einsum("hgt,thd->hgd", (scores * scale).softmax(-1), vals))
        graph.replay()
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
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
