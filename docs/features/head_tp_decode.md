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
flash-attention checkout at `8e52178ea7471b6f368032882c018b1d7c9b18bc`
with `cmake/external_projects/patches/head_tp_step_metadata.patch` applied.
The vendor checkout at `57db5bb602f200b864acc600c3a50df4a50595c0` already
includes the same change. The pinned FetchContent build applies the patch automatically. Rebuild the
FA3 extension when upgrading from the earlier per-layer split interface.
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
  Built-in choices for `tile` are `minimax`, `one` and experimental `work`.
  For `work`, set `max_splits=64` to reproduce the calibrated H100 profile.
  It sums local KV blocks across active heads, divides by 0.75 times the SM
  count, and splits each row just enough to meet that target chunk size,
  up to the captured bound. The default remains `minimax`.

A request admitted as packed stores its two halves directly, including during
chunked prefill. Entering from dense layout migrates the prefix in bounded
chunks before releasing the old dense blocks. Leaving copies back the remote
heads and reuses the packed owner pages for even logical blocks. When expansion
would exceed the available pool budget, the request keeps its packed layout.
The engine holds each layout through an in-progress chunked prefill. Requests
too large for full-head storage require packed admission. Transitions run at
completed scheduling boundaries with one model step in flight. The MoE batch
continues to contain only the rank's own tokens.

### Sharing the FA3 decode schedule across layers

On each rank the runner stages lengths, page tables and requested row splits
before the model forward. One FA3 metadata build converts those requests to
effective splits in the backend's tile units. The runner copies the result into
fixed-address storage shared by all full-attention layers of that decode step.
Full graph buckets bind the matching slice during capture. Each forward uses
precomputed metadata and its own KV pointers and outputs. The FA3 combine
kernel resets the tile semaphore; for a single split FA3 clears the counter
after its main kernel. Reuse assumes sequential layers on the worker stream.
Mixed/chunked prefill uses its own causal launch shapes.

Run `python benchmarks/kernels/head_tp/check_fa3_step_metadata.py` on one
Hopper to exercise a rank's 48 FA3 layer calls, alternating 6 and 994 virtual
rows in two graph buckets. The test checks graph outputs against inline
preparation, effective counts, an empty row, semaphore reset and CUPTI launch
counts. The expected trace is one prepare per staged step, 48 main kernels
and 48 combines when the static split bound exceeds one.

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

## Calibrating the split planner and observing Hopper instructions

Use `benchmarks/kernels/head_tp/bench_fa3_splits.py --suite work --repeat 0`
on a single Hopper GPU. Run with repeat IDs 0, 1, 2 in separate processes.
Each shape specifies counts and lengths of two-KV-head foreground shards
and full four-head background requests. The virtual page table preserves
the two physical layouts. All candidates for a shape see the same Q/K/V
and are checked against FP32. The measured interval contains FA3 metadata,
forward and combine in a graph. Worker staging, empty serving slots, KV
writes and inter-rank communication are outside this experiment. The JSONL
records include the static bound, per-row requests, alias names for identical
plans and the uninstrumented replay samples. Use `--suite confirm` for the
additional evaluation shapes. `--block-n` describes the experimental binary's
compiled BlockN; it does not change kernel dispatch.

The H100 calibration used 11 training shapes to select a target chunk size
from seven wave factors and five minimum chunk lengths at fixed cap 64. For
`tile=work`, let b[i] be the row's KV length rounded up to blocks and h[i]
its local KV head count. The target is ceil(sum(b[i]*h[i]) /
round(0.75*SMs)), at least one block. Request ceil(b[i]/target) splits per
row, clamped to [1, max_splits]. The kernel may coalesce empty tail splits.
The same graph can change counts on replay. Calibrate separately for a
different device, physical tile or head size.

`trace_fa3_decode.py` invokes one marked eager FA3 operation using the same
packed BF16 bytes and verifies its output. This separates intrusive SASS
instrumentation from timing. For CUDA 13.0 on H100, the following worked with
the official NVBit v1.8.2 x86_64 release:

```bash
cmake -S "$NVBIT/tools/opcode_hist" -B /tmp/opcode-hist -DNVBIT_TOOL_CUDA_ARCHITECTURES=90a
cmake --build /tmp/opcode-hist -j 3
LD_PRELOAD=/tmp/opcode-hist/opcode_hist.so COUNT_WARP_LEVEL=1 EXCLUDE_PRED_OFF=1 \
  python benchmarks/kernels/head_tp/trace_fa3_decode.py --split 64
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  -o layout-graph python benchmarks/kernels/head_tp/bench_fa3_kv_layout.py \
  --lengths 141698 --splits 64 --layouts virtual256 --ncu
```

For per-PC counts, the research tool keys selected main-kernel instructions
by the function-relative offset from NVBit's `Instr::getOffset()` and opcode.
Match these offsets with `cuobjdump --dump-sass --function MANGLED_SYMBOL LIB`
on the exact loaded library. Nsight Systems' CUDA kernel table identifies the
symbol and graph node. Dynamic warp instruction counts and static PCs describe
execution, not stall cycles, achieved bandwidth or DRAM transactions.

For the layout and split microbenchmarks, pass `--prepare-per-step` to time
layer replay using a prebuilt schedule. The default includes inline preparation
to reproduce the earlier sweeps. Charge the one metadata build and buffer copy
separately when estimating complete step latency.

## Two-Hopper single-layer comparison

Run one process per GPU on the same NVLink-connected Hopper pair. Use this
fork's cache ops and patched FA3 extension. For NanoCP, install the FA3 wheel
specified in NanoEngine's AE Dockerfile and its native DLSlime `hao_basic`
backend. Use the NanoEngine checkout at
`8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5`. Each command starts fresh
processes so the upstream communication buffers belong to one graph shape.

```bash
script=benchmarks/kernels/head_tp/bench_two_rank_attention.py
out=benchmarks/kernels/head_tp/results/two-rank
for load in smoke idle light busy; do
  for method in dp nanocp headtp; do
    torchrun --standalone --nproc-per-node=2 "$script" \
      --method "$method" --case "$load" --nano-root ../NanoEngine \
      --output "$out/$load-$method.json" || exit 1
  done
done
```

All methods use the same BF16 Q/K/V, request owners, page size 256, and
32 Q / 4 KV heads of dimension 128. Compare `input_sha256` across each
triplet. Smoke has owner lengths [513, 17] and partner length [129].
The other cases put a 141,698-token request on rank 0 and respectively
zero, four 512-token, or eight 16,384-token requests on rank 1.
DP keeps each owner's original batch and uses native FA3 auto. NanoCP runs
the upstream GQA operator with its published FA3 and Q/O/LSE IPC exchanges,
splitting the long prefix at a page boundary. It uses separate NHD caches.
Head-TP uses the serving worker with packed LBHNC pages, `tile=work`, and
`max_splits=64`. Regular rows use vLLM's KV write kernel in all three runs.

The GPU interval starts after a common NCCL start fence and includes the
current token KV write, attention, required exchanges, output assembly and
rank combine. The JSON reports the median of the paired per-sample maximum
GPU times over both ranks. It also retains individual rank samples, host
replay-and-sync latency including the fence, and per-stage CUDA events on
a separate instrumented graph. Each layer consumes precomputed DP/head-TP
step metadata. Initial prefix placement, CPU step staging, projections and
MoE are outside this interval. FP32 output checks cover every owner's rows
before timing. A second replay changes Q and makes the newest KV token
dominant for one Q head per KV group, including across the rank boundary.
