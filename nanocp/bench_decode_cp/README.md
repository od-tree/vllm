# Decode CP microbench (NanoCP-style)

Independent of the head-TP operator. `vendor/flash-attention` is only called
through the already installed `varlen_fwd`, with head-TP arguments left off
(`src_rank=-1`, no tile plan). None of these files edit that operator.

| File | Role |
| --- | --- |
| `layout_cp.py` | Page-aligned split. Master batch holds short rows at full length; partner gets long suffixes only |
| `attn_local.py` | FA2 paged decode. `num_splits=0` uses the occupancy heuristic. LSE is `[16, B]` |
| `merge_cp.py` | Online combine from `[W, B, H, *]`. A rank with `context_lens <= 0` is skipped. Writes O only |
| `copy_idx.py` | Indexed copies that stage partial O and LSE before the return all-to-all |
| `hao_basic.py` | DLSlime Basic all-to-all: one kernel, int4 warp copy, flag handshake inside it |
| `comm_cp.py` | N-rank Q / O / LSE route. Mask or `dst_row_indices`, preallocated `[src, slot]` buffers. KV is not moved |
| `decode_cp.py` | `run_routed` for any rank count. `merge_halves` is the one-GPU check |
| `reference.py` | Single-card full-length FA2 |
| `test_comm_route.py` | Route table on CPU, world size 4, no FA2 |
| `test_merge_unit.py` | Merge formula, no FA2 |
| `test_merge_fa2_halves.py` | One GPU, two segments vs full FA2 |
| `test_e2e_correctness.py` | Two GPUs. Master attention includes short rows |
| `bench_perf.py` | Q copy + both FA2 + O/LSE copy + merge. `--full` adds the long head-TP shapes |

Run from this directory:

```text
python3 test_comm_route.py
python3 test_merge_unit.py
python3 test_merge_fa2_halves.py
python3 test_e2e_correctness.py
python3 bench_perf.py
```

Output gate versus a single-card full FA2 is max abs `1e-2` on bf16, including
short rows. The combine kernel matches NanoCP and does not return a merged
LSE. Both sides use `num_splits=0` (FA2's occupancy heuristic), so a short
row is not bitwise identical to the reference.
