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

## 上游 NanoCP GQA 基线

作者的 [NanoEngine 源码](https://github.com/nano-infra/NanoEngine/tree/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5) 位于与 `vllm_nanocp` 同级的 `NanoEngine/`。基线固定为 `decentral-aug` 的提交 `8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5`。换机器时可在两个 vLLM 仓库的父目录复原这个 checkout。

```bash
git clone --branch decentral-aug --single-branch https://github.com/nano-infra/NanoEngine.git NanoEngine
git -C NanoEngine checkout --detach 8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5
```

在 `vllm_nanocp` 根目录运行。若源码放在其他目录，可设置 `NANOENGINE_ROOT`。

```bash
bash nanocp/bench_decode_cp/run_upstream_gqa.sh --check
# 在配好 NanoDeploy、FlashAttention 和 DLSlime 的双卡 CUDA 环境中运行
bash nanocp/bench_decode_cp/run_upstream_gqa.sh 32768
```

`--check` 校验源码版本和入口。第二条命令调用作者的 `tests/test_sp_attention_cudagraph.py`，用两个进程测单请求、SP=2、16 Q/2 KV、head_dim=256、bf16、page=16 的 graph replay。传入的长度须能均分成两个整页段。脚本检查两个 rank 都输出计时。上游输出是各 rank 的平均延迟，当前本地 `bench_perf.py` 输出双卡整步中位数。论文对比时应统一请求清单与计时口径，并补上相对单卡的数值对拍。

上游服务的 GQA 实现在 `nanodeploy/layers/attention.py` 的 `FlashAttentionImpl`，模型入口包括 `Qwen3MoeForCausalLM`。上面的 microbench 有自己的 GQA forward，覆盖 Q 路由、本地 FA 和 O/LSE 合并。它只评估这条数据通路。完整 Qwen3.5 服务还涉及混合层、动态调度和 EP。原生脚本使用 `flash_attn_interface` 和默认的 `hao_basic` 通信后端。目标 GPU 上的依赖和 P2P 能力需实测，现有 5090 memcpy 路径由本地 `bench_perf.py` 提供。依赖版本见 [上游 README](https://github.com/nano-infra/NanoEngine/blob/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5/README.md)。

## 本地 DP、head-TP 与 CP 的统一测试

输入生成、两卡输出校验、CUDA Graph 和分阶段计时口径见[双卡 GQA decode 对比](bench_decode_cp/README.md)。

```bash
cd nanocp/bench_decode_cp
python3 test_comm_route.py
python3 test_merge_unit.py
python3 test_merge_fa2_halves.py
python3 test_e2e_correctness.py
python3 bench_perf.py --quick --check
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
