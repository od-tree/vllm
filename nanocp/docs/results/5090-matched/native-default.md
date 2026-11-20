# 双 RTX 5090 上的 NanoCP 默认 GQA 补测

## 测了哪些代码

| 路径 | Attention 和通信 | 工作负载 |
| --- | --- | --- |
| 作者原版 | 固定提交的原生 FA3 wheel、DLSlime 的 `hao_basic`，双进程 | 作者的单请求 SP=2 测试 |
| 作者 forward 的 5090 移植 | 直接导入同一提交的 `sp_attention_forward_gqa`，使用 FA2 默认 split 和项目的跨卡 memcpy，单进程双设备 | 作者的单请求规格，以及调用方提供的同输入双卡 batch |
| 本地 CP 对照 | 项目自己的 CP 路由和 FP32 LSE，分别启用 FA2 默认 split 与自定义 per-row tile | 与前次相同的十组 batch，另加单请求 32768 |

**原版结果为 N/A。** 已通过 torchrun 在两张 RTX 5090 上运行未经修改的作者入口，SP=2、16 Q / 2 KV、head_dim 256、bf16、长度 32768。两个 rank 均在建立 SP 缓冲时报告 `mapping of buffer object failed`。独立的两进程测试在两个方向调用 DLSlime 使用的 `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)`，均返回 CUDA 205。原脚本捕获异常后仍会打印完成提示，本次日志中有效的 CUDA Graph 时间为 0 条。

FA3 还有独立的设备兼容问题。已核对作者 Dockerfile 指定 wheel 的 SHA-256（`186595796ffab2449f57c429ca9da94ebf081a5c84bad9f4ca9c49c36d3ebe48`）。cuobjdump 列出了 sm80、sm90a SASS，没有列出 PTX。在 5090 上单独执行相同规格的 paged GQA 得到 `no kernel image is available for execution on the device`。DLSlime 使用 `4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e` 的原版 buffer、CUDA 和绑定源码，在隔离目录针对当前 PyTorch 和 GPU 编译。服务器原有环境继续使用 Torch 2.14.1+cu130。[作者依赖版本](https://github.com/nano-infra/NanoEngine/blob/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5/ae_scripts/docker/Dockerfile#L56-L89) · [DLSlime 打开 IPC 句柄](https://github.com/nano-infra/DLSlime/blob/4f1fce6a45e3310cbc80f474f0264e4fa88b1f5e/csrc/dlslime/ops/intra_ll/all_to_all/alltoall_buffer.cpp#L114-L142)

## 对 5090 的适配与校验

- 上游 `8abce29` 的 GQA forward、Triton 索引拷贝、局部 LSE 转 bf16、O/LSE 分别回传及最终 Triton 归约均直接执行作者源码。FA3 调用替换为同规格 paged FA2，使用其内置 split 启发式。三次 collective 按相应 mask、offset 和返回槽调用项目的 `cuda_ipc_p2p.memcpy_peer`。capture 内使用源卡流上的 `cudaMemcpyAsync` 和跨卡事件，连续行合并传输。Python 协程只在预热和建图时协调两个 rank。[作者 GQA forward](https://github.com/nano-infra/NanoEngine/blob/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5/tests/test_sp_attention_cudagraph.py#L224-L398)
- 作者单请求用例中，伙伴的返回发送区是 `torch.empty`，写入它的索引为 −1，mask 为 0。原样传给 5090 通信适配器时，长度 32768 的伙伴 O 回传有 4096/4096 个元素与其实际计算值不符。修正用例中只把伙伴第 0 行 O/LSE 写入发给 owner 的第 0 槽。没有修改上游 forward。[原始测试元数据](https://github.com/nano-infra/NanoEngine/blob/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5/tests/test_sp_attention_cudagraph.py#L797-L815)
- 单请求输入采用相同 CPU 种子的逻辑 Q/K/V，均分页后分别放置，沿用作者预留两个槽的设定。矩阵测试使用上轮 workload 的两个原始 batch 和请求归属，为上游 forward 构造相应的输入元数据。每个有效 owner 都与 CPU FP32 GQA 对拍（绝对容差 0.01）；伙伴的 O 与 bf16 LSE 到达 owner 后还须逐元素精确相等。
- `nccl` 模式保留作者每步入口的单元素 NCCL all-reduce；`graph` 模式沿用上一轮的双流 fork/join。均使用两卡完成的 GPU makespan、30 次 replay 中位数。单请求每种长度独立建图三次，再取三个中位数的中位数。矩阵每格建图一次。每张图另有三轮 CUPTI 和阶段采样。矩阵内的请求选择与页放置由测试输入给定，未测 NanoCP 的服务调度、端到端吞吐或 MoE 尾延迟。

## 运行入口

服务器隔离目录为 `/root/bench-runs/native-default-3g3Ff6`，其中 `bench/` 存放本次执行的测试代码。`NANOENGINE_ROOT` 指向固定提交，Python 环境需能导入上游代码及其导入依赖；协程使用 greenlet 3.2.4。指定 `P2P_BACKEND=memcpy`，并加载前次修正的 FA2 扩展。

```bash
source /root/dev-env.sh
run=/root/bench-runs/native-default-3g3Ff6
matched=/root/bench-runs/matched-WTVd1i
cd "$run/bench"
export NANOENGINE_ROOT="$matched/upstream-ref" VLLM_ROOT="$matched/vllm" FLASH_ATTN_ROOT="$matched/fa"
export PYTHONPATH="$run/packages:$NANOENGINE_ROOT:$matched/fixed-fa:$PWD"
export LD_LIBRARY_PATH="$run/packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export P2P_BACKEND=memcpy OMP_NUM_THREADS=6

python3 bench_upstream_5090.py --length 8192 32768 131072 262144
python3 bench_upstream_5090.py --entry-sync graph --length 8192 32768 131072 262144
python3 bench_upstream_5090.py --matrix --repeats 1
python3 bench_upstream_5090.py --matrix --entry-sync graph --repeats 1
python3 bench_upstream_5090.py --published-metadata --length 32768 --repeats 1
python3 bench_perf.py --method cp-fa2-default --check --rounds 30 --profile-rounds 3
```

其中 `--published-metadata` 用于重现返回槽校验失败。作者原版用同目录的 `run_upstream_gqa.sh 32768` 在固定的 Git checkout 下执行。

## 单请求结果

单位为毫秒。每格取三张独立图的 replay 中位数的中位数。修正返回槽后，四种长度均通过伙伴回传逐位校验和 owner 最终输出 FP32 校验；最大的绝对误差为 0.00179。

| 请求长度 | 上游 5090 移植，保留入口 NCCL | 上游 5090 移植，使用 graph fork/join |
| ---: | ---: | ---: |
| 8,192 | 0.124 | 0.105 |
| 32,768 | 0.141 | 0.133 |
| 131,072 | 0.225 | 0.212 |
| 262,144 | 0.312 | 0.303 |

## 双卡固定 batch

以下均为整张 decode attention 图的毫秒数。比较本地 CP 时优先读后三列，其计时窗口一致。每个用例的原始请求归属、Q/K/V、页切分和随机种子在四列间一致。最后一行两张卡各有 1024 条长度 1024 的短请求，owner 另有 4 条长度 262144 的长请求。其余用例由 owner 持有原始请求，再将指定的长行均分到两卡。

| 输入 | 上游移植，入口 NCCL | 上游移植，graph | 本地 CP，默认 split | 本地 CP，自定义 tile |
| --- | ---: | ---: | ---: | ---: |
| 20×32768 | 0.534 | 0.520 | 0.491 | 0.501 |
| 128×8192 | 0.829 | 0.820 | 0.814 | 0.833 |
| 128×32768 | 2.737 | 2.734 | 2.727 | 2.752 |
| 128×65536 | 5.274 | 5.262 | 5.255 | 5.290 |
| 256×32768 | 6.071 | 6.050 | 6.043 | 5.366 |
| mix_128×256_2×128k_1×256k | 1.575 | 1.561 | 1.524 | 0.474 |
| mix_64×1024_4×128k | 0.881 | 0.869 | 0.826 | 0.480 |
| mix_32×512_2×256k | 1.565 | 1.546 | 1.507 | 0.419 |
| mix_512×512_16×256k | 7.523 | 7.512 | 7.466 | 3.004 |
| mix_1024×1024_4×256k | 8.229 | 8.239 | 8.189 | 2.144 |

末行在本地 CP 中只切换 tile 就从 8.189 降至 2.144 ms，约 3.82 倍。伙伴卡上，CUPTI 测到的 FA2 主 attention kernel 从 8.071 降至 1.959 ms。相同窗口的上游移植版该 kernel 为 8.101 ms。差距主要来自长短混合输入上的 tile 分配。对于 128×32768，自定义 tile 和默认 split 的整图时间则相差约 1%。单格仅重建一次图，不能据此区分接近的结果。

[本次原始数据及诊断日志](native-default.tar.gz) 包含 68 条性能记录，均保留 30 次 replay 的 GPU 与主机用时、两卡 kernel 和阶段明细。归档还包括双向 IPC 探针、FA3 独立执行结果、上游原版启动日志，以及使用作者原始返回槽元数据时的逐元素校验失败记录。
