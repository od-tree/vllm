# Qwen3 decode replays

## Original NanoCP lengths

`qwen3-original-v1/` retains the source prompt and output lengths for every
request. The manifest gives the minimum total context needed by each workload.
Across all four files the longest prompt plus output is 996,596 tokens.
Provision the model's context extension and sufficient KV capacity on each
tested server before running those windows.

From the vLLM checkout:

```sh
python head_tp_eval/frozen_replay.py freeze --original-lengths
python head_tp_eval/frozen_replay.py export issue05 \
  --bundle head_tp_eval/replays/qwen3-original-v1 --start 0 --count 16 \
  --output head_tp_eval/results/qwen3-original-v1/issue05-first16.jsonl
```

The two profiles share the row keys, arrival times, output budgets, routing
overlay and synthetic prompt stream. A previously clipped row expands by
extending its original token prefix. GPU count and DP rank assignment are
deployment parameters. The model ID in the manifest identifies the token
vocabulary used to freeze the input.

## 256K profile

The four compact traces in `qwen3-v1/` preserve every source row in order.
`short`, `issue01`, and `issue05` are the three 60,000-request evaluation
workloads. `issue_pool` is the 2,776-row Issue source population, also frozen
for long-only diagnostics. Do not concatenate it with the three main traces.
The ShareGPT conversation example and the synthetic kernel correctness cases
are separate fixtures.

The CSV files contain length statistics. Their source labels define short and
long. Prompts in this replay are synthetic Qwen token IDs, with independent
row streams and no intentional shared prefixes across requests within a trace.
The three main traces share the same seed and arrival schedule to support paired
comparisons. The row ID selects a SHAKE256 stream of little-endian uint16 IDs.
All 65,536 values are ordinary tokens of the pinned Qwen tokenizer. Using IDs
directly avoids retokenization and Python's per-process hash salt. Each
workload has both a compressed-file SHA-256 and a SHA-256 over all expanded
prompt bytes in order. The compact files are the inputs; expanded windows
can be recreated on demand.

Arrival times are synthetic Poisson arrivals at a nominal 1 request/s,
starting at zero, stored as integer microseconds. Output budgets are the
source values. The Qwen profile limits input to 262144 minus output budget;
the original input length is retained in a separate column. No rows are
dropped. The profile is model-specific and independent of DP size and hardware.
For a shorter model limit use a separately named workload profile across all
compared methods. For a different offered load scale the same gaps identically
and record the multiplier.

From the vLLM checkout, verify the source and reproduce the freeze with
`python head_tp_eval/frozen_replay.py freeze`. An existing version must match
byte for byte. A different freeze needs a new directory and version.

```sh
python head_tp_eval/frozen_replay.py export issue05 --start 0 --count 16 \
  --output head_tp_eval/results/qwen3-v1/issue05-first16.jsonl
vllm bench serve --backend openai --dataset-name timed_trace \
  --dataset-path head_tp_eval/results/qwen3-v1/issue05-first16.jsonl \
  --num-prompts 16 --model MODEL --host HOST --port PORT
```

Use the benchmark **client from this checkout** for all servers. It accepts
explicit prompt_token_ids and timed_trace defaults to self-timed arrivals
and ignore_eos. Windows rebase the first arrival to zero while retaining
all gaps; they preserve source order and need enough server context for the
longest prompt plus output. Export bounded windows to limit client RAM.

For default load balancing, use the canonical requests directly. For a
two-DP-rank skew experiment, export with `--two-rank-overlay PATH` to emit an
independent map keyed by `request_key`. It sends long requests to rank 0 and
round-robins short requests across 0 and 1. The example in `routing/` covers
the first 16 rows of issue05. The standard bench command above uses native
routing; a client applying the overlay must set the per-request
`X-data-parallel-rank` header. Use a mapping appropriate to the number of
DP ranks, which can differ from the number of GPUs when TP or PP is used.
