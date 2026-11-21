# Two-rank head-TP decode for Qwen3 MoE

The V2 runner can split a Qwen3 MoE request's GQA heads across two DP workers
for single-token decode. The request owner runs QKV projection, rotary
embedding, output projection and MoE. Each rank can lend half of its attention
heads for one request and simultaneously compute a peer's half-head shard.
Other requests use their local KV. Prefill and mixed prefill/decode steps use
the regular attention backend.

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
KV page size, EP options and request stream in both runs. The enabled
configuration permanently budgets `ceil(max_context / (2 * block_size))`
physical KV blocks per rank for incoming shards. Capacity is capped at the
model length. The local scheduler sees the remaining capacity.

## Change the decision during decode

`head_tp.policy_class` accepts the fully qualified Python class name of a
policy available on both workers. Its constructor receives the head_tp options.
It can subclass `DecodePolicy` in `vllm.v1.worker.gpu.head_tp_decode`.
The worker calls the following methods outside graph capture on every eligible
decode step.

- `select(rows, *, peer_positions, rank, step)` receives local eligible
  `DecodeRow(req_id, position)` objects and the peer's batch history lengths.
  Return a local request ID or `None`. The default selects the longest eligible
  local request at or above `min_context`. This is a forced-collaboration
  policy for experiments.
- `splits(lengths, heads, *, block_n, sms, limit, rank, step)` receives local
  logical rows in batch order followed by one inbound slot. Return one integer
  in `[1, limit]` per entry. Return all ones to disable splitting for the step.
  Each value applies to the row's live KV heads. On Hopper the maximum split
  count is fixed at capture and the per-row values are device inputs to replay.
  Built-in choices for `tile` are `minimax` and `one`.

The owner retains full-width KV throughout decode. On first use, switch or resume
the workers agree on an absolute token position and synchronously copy the
peer's prefix heads before attention. Consecutive tokens of the same request
only send current Q/K/V. The helper writes the absolute slot, returns the output
of its Q heads, and reuses its private pages when the next request takes over.
The rank's regular MoE batch contains only its own tokens. An empty local rank
can still help a peer. A mixed step ends the current collaboration and the next
eligible step re-evaluates it. Both directions use fixed-size NCCL exchanges
even when the current policy selects no request.

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
