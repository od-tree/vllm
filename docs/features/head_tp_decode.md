# Two-rank head-TP decode for Qwen3 MoE

The V2 runner can split a Qwen3 MoE request's GQA heads across two DP workers
during prefill and decode. The request owner runs QKV projection, rotary
embedding, output projection and MoE. Each rank can lend half of its attention
heads for one request and simultaneously compute a peer's half-head shard.
Other requests use full-head local KV. A packed request can start on both
workers from its first prefill chunk. Mixed and chunked prefill are eager;
single-token decode supports full model graph replay.

## Run

Build the Hopper FA3 extension with `VLLM_HEAD_TP_FA3=1` in the build
environment. Alternatively point `VLLM_FLASH_ATTN_SRC_DIR` at the research
flash-attention checkout at `8e52178ea7471b6f368032882c018b1d7c9b18bc`.
The operator needs the FA3 `row_split_counts` argument. On the 5090 development
machine the same distributed path uses FA2 with a single split.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_KV_CACHE_LAYOUT=LBHNC \
vllm serve /path/to/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tensor-parallel-size 1 --data-parallel-size 2 --enable-expert-parallel \
  --attention-backend FLASH_ATTN --block-size 256 --max-model-len 4096 \
  --additional-config '{"head_tp":{"enabled":true,"max_context":4096,"min_context":0,"max_splits":8}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Use BF16 KV, one full-attention KV group, and equal allocator and kernel page
sizes. The initial implementation supports head dimension 128, an even number
of KV heads, PP=1 and regular decode. Omit the head_tp configuration for the
native DPA baseline. For a bounded comparison, set the same graph capture sizes,
KV page size, EP options and request stream in both runs.

With H KV heads and a B-token page, dense local KV occupies
`ceil(length / B)` pool blocks on the owner. A sharded request occupies
`ceil(length / (2*B))` blocks **on each participating GPU**. For local head
`h` and absolute position `t`, the physical page index is `t // (2*B)`,
the head slot within that page is `2*h + (t // B) % 2`, and the offset is
`t % B`. Each local head has 2B consecutive tokens. The scheduler accounts
for allocated pages on both workers, protects space for admitted requests'
growth, and returns pages to the regular pool on completion. Native logical
slot entries for packed requests refer to the null block. Native KV writes
are masked for those tokens; partial-head pages do not enter the prefix cache.

## Change the decision during decode

`head_tp.policy_class` accepts the fully qualified Python class name of a
policy available on both workers. Its constructor receives the head_tp options.
It can subclass `DecodePolicy` in `vllm.v1.worker.gpu.head_tp_decode`.
The DP engine invokes selection before scheduling. The worker chooses splits
outside graph capture for each decode. Policy instances in the engine and
worker have separate state.

- `select(rows, *, peer_positions, rank, step)` receives local eligible
  `DecodeRow(req_id, position)` objects and the peer's batch history lengths.
  During admission the candidate length includes the prompt. Return a local
  request ID or `None`. The default selects the longest eligible
  local request at or above `min_context`. This is a forced-collaboration
  policy for experiments.
- `splits(lengths, heads, *, block_n, sms, limit, rank, step)` receives local
  logical rows in batch order followed by one inbound slot. Return one integer
  in `[1, limit]` per entry. Return all ones to disable splitting for the step.
  Each value applies to the row's live KV heads. On Hopper the maximum split
  count is fixed at capture and the per-row values are device inputs to replay.
  Built-in choices for `tile` are `minimax` and `one`.

A request admitted as packed stores its two halves directly, including during
chunked prefill. Entering from dense layout migrates the prefix in bounded
chunks before releasing the old dense blocks. Leaving copies back the remote
heads and reuses the packed owner pages for even logical blocks. When expansion
would exceed the available pool budget, the request keeps its packed layout.
The engine holds each layout through an in-progress chunked prefill. Requests
too large for full-head storage require packed admission. Transitions run at
completed scheduling boundaries with one model step in flight. The MoE batch
continues to contain only the rank's own tokens.

## Verify

`torchrun --standalone --nproc-per-node=2 benchmarks/kernels/head_tp/check_distributed_decode.py`
captures one graph per process. Seven replays change request ownership,
cooperation, page offsets, input data and split counts, and check outputs
against FP32 attention.

For an end-to-end comparison, run
`python benchmarks/kernels/head_tp/check_serving.py --url http://localhost:8000 --model NAME --save baseline.json`
on the native server, then repeat on the enabled server with
`--save headtp.json --reference baseline.json`. This exercises two
concurrent requests, idle partners and a 256-token page crossing.

For a constrained-pool test use B=256, model length 4096 and
`--num-gpu-blocks-override 9` on the head-TP server (eight usable blocks
per worker). Run the comparison with `--prompt-length 3067` against an
ordinary DPA reference with sufficient cache. Dense KV needs 12 blocks for
the prompt; packed KV needs six per GPU.

## Single-GPU FA3 layout experiment

Run `benchmarks/kernels/head_tp/bench_fa3_kv_layout.py` on one Hopper GPU
using the research FA3 extension described above. It benchmarks the attention
call for pure cooperative decode rows after the local shard is resident. Q
and BF16 KV are synthetic; the three cases share their underlying storage.
Each request has two local KV heads, eight Q heads per KV head and dimension
128. Pages are shuffled with a fixed seed; `--ordered-pages` selects a
contiguous physical allocation. Compare layouts within the same invocation.

| Layout | Rows per request | KV heads / tokens per FA page | Question |
| --- | ---: | --- | --- |
| virtual256 | 2 | 1 / 256 | Current per-head page-table addressing of the active shard |
| virtual512 | 2 | 1 / 512 | What changes when FA sees the full contiguous span of one head? |
| physical512 | 1 | 2 / 512 | What changes when both local heads share a row and one physical page table? |

The middle case allows the page-length effect and the row/head-grouping
effect to be measured separately. Per-head splits and their static bound
are identical across layouts at each sweep point. `--lengths` accepts a
ragged batch of cooperative requests; ordinary full-head background rows,
empty serving slots, KV update, NCCL and output splice are outside the timing
window. The page-table and Q views are built once before capture.

```bash
script=benchmarks/kernels/head_tp/bench_fa3_kv_layout.py
python "$script" --layout-check-only --lengths 1 255 256 257 511 512 513 8193
python "$script" --lengths 141698 --splits 1 8 32 64 128 > layout.jsonl
python "$script" --lengths 8193 141698 --splits 8 64 --ordered-pages > ordered.jsonl
```

The CPU check gathers every valid K/V token through each page table and
compares it bitwise to the chronological input. GPU runs also check every
graph output against independent FP32 attention. JSONL contains the shapes
and strides, complete uninstrumented graph GPU and host times in microseconds,
and kernel names, launch counts and stage timings from separate CUPTI replays.
Layout order rotates between timing rounds. `--profile-rounds 0` disables CUPTI.
Kernel names that do not match prepare, attention or combine remain visible
under other. Short cases can fit in L2; use the long case for bandwidth studies.

For DRAM traffic use Nsight Compute on separate runs. For example, run the
following command for each layout, changing `layout` and keeping the split,
length and seed fixed. The NVTX range contains one checked graph replay.

```bash
layout=virtual256
ncu --nvtx --nvtx-include "headtp_${layout}_s64/" --graph-profiling node \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum --csv \
  --log-file "${layout}-s64.csv" \
  python "$script" --lengths 141698 --splits 64 --layouts "$layout" --ncu
```

Use the main kernel's counters and duration when investigating KV loads;
the complete graph also contains scheduler preparation and, for split
attention, reduction. Scaling with page length or grouped heads indicates
sensitivity to that representation. Timing alone cannot establish a TMA
transition or a pipeline stall. The fork's single-query decode TMA guard
still sees an effective query height of eight in all three cases. Inspect
the selected kernel and SASS/counters before attributing a difference to TMA.
