# SPDX-License-Identifier: Apache-2.0
"""One Hopper: verify the serving FA3 call across layers, steps and graph buckets."""

import argparse
import json

import torch

from vllm.v1.worker.gpu.head_tp_decode import DecodePolicy, HeadTPDecode
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


@torch.inference_mode()
def check(limit, layers):
    device = torch.device("cuda")
    # Both buckets share the worker's fixed metadata buffer. 994 virtual rows
    # exercise the multi-CTA prepare path above the automatic 992-row cutoff.
    batches = (1, 248)
    maximum = 4097
    torch.manual_seed(11)
    cache = torch.randn(17, 4, 256, 256, device=device, dtype=torch.bfloat16)
    decode = HeadTPDecode(
        [cache],
        32,
        maximum,
        max(batches),
        maximum,
        limit,
        DecodePolicy({}),
        0,
        None,
        None,
    )
    key, value = cache.view(-1, 256, 1, 256).split(128, dim=-1)
    # Nontrivial page ordering, reused for all virtual rows.
    decode.table.copy_(torch.randperm(key.size(0), device=device)[: decode.columns])
    graphs = {}

    def stage(b, step):
        n = b * decode.h + decode.half
        idx = torch.arange(n, device=device)
        lengths = (idx * 13 + step * 3) % 67
        lengths[0] = maximum - step
        splits = 1 + (idx + step) % limit
        decode.used[:n].copy_(lengths)
        decode.splits[:n].copy_(splits)
        decode.prepare_decode_metadata(b)

    for b in batches:
        n = b * decode.h + decode.half
        q = cache.new_empty(n, 8, 128).normal_()
        stage(b, 0)
        decode.paged_decode(q, key, value, 128**-0.5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [
                decode.paged_decode(q, key, value, 128**-0.5) for _ in range(layers)
            ]
        graphs[b] = (q, graph, outputs)
    for step, b in enumerate((1, 248, 1, 248), 1):
        q, graph, outputs = graphs[b]
        n = q.size(0)
        q.neg_()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            stage(b, step)
            graph.replay()
            torch.cuda.synchronize()
        meta = decode.scheduler_metadata
        assert (
            meta.data_ptr() == decode.metadata_buffer.data_ptr()
            and meta[-1].item() == 0
        )
        if limit > 1:
            blocks = (decode.used[:n] + 127) // 128
            width = ((blocks + decode.splits[:n] - 1) // decode.splits[:n]).clamp(min=1)
            effective = ((blocks + width - 1) // width).clamp(min=1)
            off = ((n + 3) // 4) * 4
            torch.testing.assert_close(meta[off : off + n], effective, rtol=0, atol=0)
        ref = flash_attn_varlen_func(
            q,
            key,
            value,
            max_seqlen_q=1,
            cu_seqlens_q=decode.cu_q[: n + 1],
            max_seqlen_k=maximum,
            seqused_k=decode.used[:n],
            block_table=decode.table[:n],
            softmax_scale=128**-0.5,
            fa_version=3,
            num_splits=limit,
            row_split_counts=decode.splits[:n],
        )
        for out in outputs:
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        prep = sum("prepare_varlen_num_blocks" in n for n in names)
        main = sum("FlashAttnFwdSm90" in n for n in names)
        combine = sum("combine" in n.lower() for n in names)
        assert (prep, main, combine) == (1, layers, layers if limit > 1 else 0), (
            prep,
            main,
            combine,
        )
        print(
            json.dumps(
                dict(
                    step=step,
                    rows=n,
                    bound=limit,
                    prepare=prep,
                    main=main,
                    combine=combine,
                    semaphore=0,
                )
            ),
            flush=True,
        )
    for _, graph, _ in graphs.values():
        graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=48)
    args = parser.parse_args()
    if args.layers < 2:
        parser.error("use at least two layers")
    for cap in (1, 8):
        check(cap, args.layers)
