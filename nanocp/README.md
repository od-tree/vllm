# NanoCP local research

Dedicated tree inside `vllm_nanocp` for request-level decode CP
(NanoCP-style) reproduce and comparison against head-TP.

| Path | Role |
| --- | --- |
| `bench_decode_cp/` | Independent operator bench (layout, hao_basic, merge, e2e, perf) |
| `docs/nanocp_comparison_plan.md` | Comparison experiment plan |
| `docs/nanocp_operator_impl_plan.md` | Operator impl notes |
| `docs/nanocp_paper_and_implementation.md` | Paper / upstream notes |
| `images/` | Request-level CP / TP figures |

## Run (after GPU env is ready)

```bash
cd nanocp/bench_decode_cp
python3 test_comm_route.py
python3 test_merge_unit.py
python3 test_merge_fa2_halves.py
python3 test_e2e_correctness.py
python3 bench_perf.py
```

FA2 is imported as `vllm_flash_attn` / `torch.ops._vllm_fa2_C.varlen_fwd`.
The flash-attention tree used to build that extension stays in the sibling
`vllm` checkout, under `local_research/vendor/flash-attention`.

`.nanocp_tree.json` is the file list of the upstream snippets used while
aligning the operator. The bench does not import it.

## Peer backend (5090)

`bench_decode_cp/cuda_ipc_p2p.py` keeps the historical
`EnablePeerAccess` + foreign `data_ptr` path. `P2P_BACKEND=auto` (default)
uses `cudaMemcpyPeer` on sm_120 / RTX 5090. Force with `memcpy` or `peer`.
The head-TP GPU test uses the same switch: the kernel writes a local
payload, then the host copies it.
