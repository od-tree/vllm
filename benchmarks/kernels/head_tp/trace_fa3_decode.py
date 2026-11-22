# SPDX-License-Identifier: Apache-2.0
"""One marked eager FA3 call for external SASS/trace tools; no timed instrumentation."""

import argparse

import torch
from bench_fa3_kv_layout import DIM, LAYOUTS, make_inputs, reference, view

from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=141698)
    parser.add_argument("--split", type=int, default=64)
    parser.add_argument("--layout", choices=LAYOUTS, default="virtual256")
    args = parser.parse_args()
    if args.length < 1 or not 1 <= args.split <= 128:
        parser.error("positive length and split between 1 and 128 required")
    # Construct all metadata on CPU to keep diagnostic kernel launches identifiable.
    inputs = make_inputs([args.length], "cpu", 11, False)
    expected = reference(inputs)
    host = view(inputs, args.layout)
    cache = inputs.cache.cuda()

    # Retain the exact strides and K/V alias, including interleaving.
    def gpu_view(t):
        return cache.as_strided(t.shape, t.stride(), t.storage_offset())

    q, k, v = host.q.cuda(), gpu_view(host.k), gpu_view(host.v)
    splits = torch.full_like(host.used, args.split).cuda()
    table, used, cu = host.table.cuda(), host.used.cuda(), host.cu.cuda()
    out = torch.empty_like(q)
    with torch.cuda.nvtx.range("fa3_decode"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cu,
            max_seqlen_k=args.length,
            seqused_k=used,
            block_table=table,
            softmax_scale=DIM**-0.5,
            causal=False,
            fa_version=3,
            out=out,
            num_splits=args.split,
            row_split_counts=splits,
        )
        torch.cuda.synchronize()
    torch.testing.assert_close(
        out.cpu().float().reshape_as(expected), expected, atol=0.005, rtol=0.01
    )
    print("FP32 reference checked", args.layout, args.length, args.split, flush=True)


if __name__ == "__main__":
    main()
