# head-TP 与 NanoCP：工作记录

整理日期：2026-11-18。覆盖 vLLM 仓库（`/Users/ly/Documents/code/vllm`）和 NanoCP 仓库（`/Users/ly/Documents/code/vllm_nanocp`），以及 flash-attention fork 里的算子。来源是仓库里的设计文档、提交，和历次对话里实际跑出来的表。同一组数字只保留同机、同协议下能互相比较的那一版；被后来的测量取代的表会标明。

单位都是毫秒。模型规格始终是 Qwen3.5 Full Attention：16 Q / 2 KV，`head_dim=256`，bf16，page 16。协作度 `d=2`。这不是 `tensor_parallel_size`。权重完整，每张卡仍是一个 DP rank。

## 两条路线

要解决的是 DP+EP decode 里的同一件事：Attention 耗时随本卡 KV 变长，MoE 的 dispatch/combine 要等最慢的卡。一条长请求会把所在卡拖成慢卡。

| | head-TP | 这里复现的 NanoCP |
| --- | --- | --- |
| 切什么 | 长请求按 KV 头。owner 持 KV0 和 Q[0:8)，partner 持 KV1 和 Q[8:16)。短请求留在原卡，全头 | 长请求按 token，在 page 边界对半。短请求留在 master，全长 |
| 每张卡看到的 KV | 一个 KV head、整段 token | 两个 KV head、大约一半 token |
| 跨卡合并 | 按头拼回，没有 LSE | 两段 partial 用 LSE 合成 |
| 每步交换 | 当前 token 的 Q、这一 token 的 K/V，以及伙伴算完的 Q head | 长行的 Q 下去，O 和 LSE 回来 |

论文是 ASPLOS 2027 的 [NanoCP](https://arxiv.org/abs/2605.21100)。开源全栈在 NanoEngine 的 `decentral-aug`，包名 NanoDeploy。本仓库没有搬那套调度、MoE 和 Ray，只复现了 decode 的数据面：连续段 paged FA2、Q 路由、O/LSE 回传、`combine_kv`。MLA 按头切会把共享 KV 复制一遍，所以论文走 CP；Qwen3.5 是 GQA，按头切能少读 KV。

## 代码库

三棵 git 树互不相通。vLLM 和 NanoCP 的远端都是 `od-tree/vllm` 的 fork，flash-attention 是 `od-tree/flash-attention`。算子没有接进 vLLM 的 forward。框架里 head-TP 的 attention 仍走 SDPA，真正的 paged kernel 只在独立测试里调用。`local_research/` 被 vLLM 的 gitignore，设计文档留在那里；算子源码在它下面的 vendor 目录，那是另一个 git 仓库。

| 树 | 路径 | 分支 | 已提交尖端 |
| --- | --- | --- | --- |
| vLLM | `/Users/ly/Documents/code/vllm` | `dtp_test` | `92cd045f30` |
| NanoCP | `/Users/ly/Documents/code/vllm_nanocp` | `nanocp_5090` | `cf481f9b49` |
| FA2 | `vllm/local_research/vendor/flash-attention` | `5090` | `39ac30e` |

NanoCP 的 `nanocp/` 是从 vLLM 的 `local_research` 挪出去的，见 `local_research/NANOCP_MOVED.md`。FA2 扩展装成 `vllm_flash_attn`，两边 bench 都 `import vllm_flash_attn._vllm_fa2_C` 再调 `torch.ops._vllm_fa2_C.varlen_fwd`。

### vLLM：框架，不跑这个 kernel

`dtp_test` 相对 `b8c04dd3b4` 的 7 个提交，从旧到新：

- `ff11006024` Qwen3.5 decode 的 eager 按头协作
- `7b791d418c` 评测 harness `head_tp_eval/`
- `0252b22f89` 调度协作方案，并拷贝伙伴 KV
- `d1de380fd5` 把共享的 CPU prefix 当成 attention block 读
- `72196bcf0d` 随 batch 生成 CTA 列表
- `223b5f48a4` split 不超过 FA2 真实的 KV tile 数
- `92cd045f30` 长行已经铺满 SM 时，再压尾波

| 文件 | 作用 |
| --- | --- |
| `vllm/v1/head_tp.py` | 请求方案、分片区间、tile 计划。`make_plan` / `make_shards` 按 `d` 切 Q 和 KV。`build_head_tp_tile_plan` 给一整个 step 的 CTA 列表 |
| `vllm/v1/head_tp_sched.py` | scheduler 末尾决定哪些请求协作。`d` 不整除 KV 头、或请求还在 prefill，就保持本地。每 32 步复查。短于 64 token 不协作 |
| `vllm/v1/head_tp_kv.py` | 把伙伴卡需要的 KV head 装进该请求自己的 partial cache。不进 `BlockPool`，也不写回共享 prefix |
| `vllm/v1/worker/gpu/head_tp.py` | GPU model runner 的胶水：组 batch、触发 prefix 搬运、把 tile 放进 forward state |
| `vllm/model_executor/layers/head_tp_attn.py` | eager 层。远端行不进原来的 `Attention`。第一版计算是 PyTorch SDPA，读的是后端实际存的 KV 布局。owner 在 gate 和 `o_proj` 之前把各头拼上。传输走现有 DP group 的 `send_tensor_dict` |
| `vllm/model_executor/models/qwen3_next.py` | `head_tp_is_active` 时改走 `run_mixed_head_attention`。GDN 层不动 |
| `head_tp_eval/` | 独立评测包。数据集和方案都是注册表，方便加 ShareGPT / GitHub / AgentX。还没有在 GPU 上跑过端到端 |
| `tests/v1/test_head_tp_{eager,prefix,schedule,tiles}.py` | 对应上面四块的 CPU 测试。本机没有 torch，这些 pytest 跑不了 |

未提交，只在 `vllm/v1/head_tp.py` 和 `tests/v1/test_head_tp_tiles.py`：

- 新增 `_fill_wave_splits`。长行还没铺满 SM 时，按它们占这一步 KV 字节的比例分一轮 SM。短行的 `seq * kv_heads` 先从预算里扣掉。多出来的 SM 只有在所有最慢的长行都能再拿一份时才用。
- `HEAD_TP_SPLIT_LIMIT = 127`。`row_splits` 是 int8，FA2 combine 只编到 128。原来的 `HEAD_TP_SPLIT_CAP = 16` 只留给已经铺满 SM 的那条路径。
- 测试期望从「cap 16、两个长行一共 32 个 CTA」改成一轮 SM 的 split，例如 108 个 SM 上 `[72, 36]` 再加短行。

### flash-attention：paged head-TP kernel

相对上游 `af5422c`，`5090` 分支上和 head-TP 有关的是这些文件。`csrc/composable_kernel` 和 `third_party/aiter` 是子模块脏状态，不属于这次改动。

| 文件 | 作用 |
| --- | --- |
| `csrc/flash_attn/src/head_tp_exchange.h` | 设备侧交换。发送、伙伴把当前 token 的 K/V 写入自己的页、伙伴把 Q head 写进 payload、owner 拼回。`head_tp_memcpy` 为真时写本地平面，不写对端地址 |
| `csrc/flash_attn/src/head_tp_page.h` | 长页寻址。一个物理页是本卡 `kv_per` 个头、每头连续 `d * block_size` 个 token |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | split-KV kernel 里调用上面的发送、等待和拼回。combine 之后才发布返回 |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | `HEAD_TP_PHASE=1` 只发 send CTA，不扫 KV。phase ≥ 2 不再发 send。`num_splits > 1` 且不是 phase 1 时才启动 `flash_fwd_splitkv_combine_kernel` |
| `csrc/flash_attn/flash_api.cpp` | `mha_varlen_fwd` 增加 row kind、slot、Q/KV 区间、payload 指针和 tile。GQA decode 仍会把 Q 换成 `[batch * group, kv_heads, dim]` |
| `flash_attn/head_tp_page.py` | 和设备侧一致的页偏移，给 CPU 测试对拍 |
| `tests/test_head_tp_page_d2.py`、`test_head_tp_page_d4.py` | 页字节和拼回下标，不需要 GPU 通信 |
| `tests/test_head_tp_paged_gpu.py` | 双卡正确性和性能。case、tile、三阶段交换都在这里 |
| `tests/cuda_ipc_p2p.py` | 和 NanoCP 那份同源。5090 上 `P2P_BACKEND=memcpy` |

分支上的提交，从接到 paged 路径开始：

- `31b2217` paged 交换，以及只编 bf16 head-dim 256 的测试构建
- `c863de7` 修长页 load 走进下一块
- `79a4e83` 发送排到计算 tile 之前，并允许 split-KV
- `6858613` 每行自己的 split 列表
- `d085c7b` 只有 split 0 写当前 token 的 K/V
- `dfae77d`、`3041cde` 正确性修复
- `35ff4e0` 5090 测试改走 CUDA IPC
- `67ea8aa` 5090 改走 `cudaMemcpyPeer`，A100 的 kernel 直写留下
- `39ac30e` memcpy 分成 send / compute / splice，并用 CUDA graph 计时

未提交：

- `csrc/flash_attn/src/flash_fwd_head_tp_splice.cu`（新文件）。一个 block 一个 token，128 线程。只在 owner、且 `row_kind == 1` 时，把伙伴的 Q head 从 payload 拷进用户 out。不读 KV。
- `flash_api.cpp` 在 `head_tp_memcpy && head_tp_phase == 3` 时调用它然后返回，不再 `run_mha_fwd`，也不做随后的 GQA transpose。那个 transpose 会用一块新的临时输出盖掉 phase 2 已经写好的 owner head。GQA 交换过的话，splice 写的是用户最终的 out，并把 `seqlenq_ngroups_swapped` 关掉。
- `CMakeLists.txt` 把这个 `.cu` 加进 `FA2_MINIMAL_HDIM256_BF16` 的源文件列表。
- `tests/test_head_tp_paged_gpu.py` 的 `_copy_rank_plane` 只拷长行前缀。`_std_timed` 不再两个方向都拷：compute 前只把 owner 的 Q/K/V/arrived 拷给 partner，splice 前只把 partner 的 out/arrived 拷回 owner。`_fill_wave_splits` 与 vLLM 那份是同一条规则，head 数固定为 1。

phase 编码：`HEAD_TP_PHASE` 为 `send` / `compute` / `splice`，对应 1 / 2 / 3。不设则是 0，A100 上发送、计算、拼回仍在同一次 attention 里。

### NanoCP：独立 bench，不进 vLLM

`nanocp_5090` 相对 vLLM 的 `b8c04dd3b4` 有 6 个提交：

- `3abb4dce99` decode CP 对照的第一版
- `2be62c17e9` 长行 tile，并修两卡 handshake 死锁
- `dff0dfd16e` 5090 上用 CUDA IPC 映射 peer store
- `058c03b74e` 先把 O/LSE 分段 stage，再启动回传
- `653cc21262` 5090 传输改成 `cudaMemcpyPeer`
- `cf481f9b49` 用 CUDA graph 给这次 memcpy 计时

`nanocp/bench_decode_cp/` 里各文件：

| 文件 | 作用 |
| --- | --- |
| `layout_cp.py` | `page_split`：前缀是不超过 `seq // 2` 的最大 16 的倍数。`segment_limits` 把长行前缀和全部短行留给 master，长行后缀给 partner |
| `attn_local.py` | 一段 paged FA2，返回 O 和 LSE。有 tile 时把 `tile_cp` 的计划传进去 |
| `tile_cp.py` | 与 head-TP 相同的一轮 SM 规则，但每个长行有 2 个 KV head，所以同样 CTA 数下 split 大约是一半 |
| `hao_basic.py` | 从 DLSlime `feat/hao-basic-alltoall-offsets` 的 `4f1fce6` 移植的节点内 all-to-all。`int4` 拷贝，双向 flag 在同一个 kernel 里。A100 上使用。0 行 dummy 不能 `reshape(0, -1)` |
| `comm_cp.py` | `CpComm`。A100 走 hao_basic。5090 上 `use_memcpy()` 为真时改走 `_memcpy_rows` |
| `merge_cp.py` | `combine_kv`。在线 softmax 把 `[W, B, H, D]` 收成一份 O，不写合并后的 LSE。`context_lens <= 0` 的那一侧跳过。CUDA 上是 Triton |
| `decode_cp.py` | 一步：路由 Q、两侧 FA2、回传 O/LSE、`combine_kv`。短行不进合并 |
| `cuda_ipc_p2p.py` | peer、IPC、memcpy 三种后端，以及 `capture_graph` / `replay_median`。head-TP 测试把这份文件放到自己的 `tests/` 下使用 |
| `bench_perf.py` | 和 head-TP 同一组 10 个 case 的墙钟，外加每 kernel 的 CUPTI 中位数 |
| `bench_compute.py` | 未提交。只比两侧 FA2，不把路由和合并算进墙钟 |
| `reference.py` | 单卡参照 |
| `copy_idx.py` | 回传时的行下标 |
| `test_merge_unit.py`、`test_merge_fa2_halves.py` | 合并公式，以及两半 FA2 拼起来对单卡 |
| `test_comm_route.py` | 路由拷贝。未提交的用例覆盖打包后的一段拷贝 |
| `test_e2e_correctness.py` | 两卡整步对单卡，门槛 `1e-2` |
| `test_tile_plan.py` | CPU 上对 split 数。不需要 torch |

文档：`docs/nanocp_paper_and_implementation.md` 是论文和 NanoEngine 的对照。`nanocp_comparison_plan.md`、`nanocp_operator_impl_plan.md` 是动手前的计划。`5090_compare.md` 是当前 5090 对照表。`images/` 是请求级 CP 和 TP 的示意图。`.nanocp_tree.json` 是对照上游时用过的文件清单，bench 不 import 它。

未提交的行为，除了上面的 `bench_compute.py` 和本文件：

- `tile_cp.py` 去掉固定的 16 split 上限，改成和 vLLM `_fill_wave_splits` 一样的一轮 SM，并把短行工作量扣出预算。上限 127。
- `comm_cp.py` 的 memcpy 路径不再按 token 各发一次 `cudaMemcpyPeer`。连续的目标行是一次切片拷贝；不连续则先收成一块再拷一次，然后 scatter。注释写明没有设备侧握手。
- `bench_perf.py` 把同一次 graph replay 的 CUDA 活动按设备和 kernel 短名归并，打印中位数。跳过 `cudaGraphLaunch`、`cudaDeviceSynchronize` 和 CUPTI 自己的 buffer 请求。
- `cuda_ipc_p2p.py` 增加图捕获时把拷贝钉在各卡自己的流上，并用 `wait_event` 让目标流看见字节。
- `5090_compare.md` 改成只拷长行、tile 扣过短行之后的表。

GPU 上的入口：

```bash
cd nanocp/bench_decode_cp
python3 test_comm_route.py
python3 test_merge_unit.py
python3 test_tile_plan.py
python3 test_e2e_correctness.py
P2P_BACKEND=memcpy python3 bench_perf.py
```

head-TP 的 GPU 入口在 FA2 树里：`P2P_BACKEND=memcpy python3 tests/test_head_tp_paged_gpu.py`。`HEAD_TP_ONLY=<case>` 只跑一组。5090 上非交互 shell 要先 `. /root/dev-env.sh`。

调度可以生成请求粒度的方案，并异步搬 KV、切换并行方式。剩余长度预测没做。端到端 TPOT 和 ShareGPT / GitHub / AgentX 负载没跑。

## 计时约定

两边比的都是单层 decode attention，不含 MoE。KV 在计时前已经放在各自的卡上。

- head-TP 单卡：整段 KV、全部头，在原卡上的 FA2。
- head-TP 无通信：`src_rank=-1`，只切 KV，不交换。早期大 batch 上这条路径有非法访存，那些时间作废。后来修过的表才使用这一列。
- head-TP 带通信：send、compute、splice。A100 上 splice 融合在 attention 的 epilogue 里。5090 上 phase 3 是单独的拷贝 kernel。
- NanoCP：Q 路由、两侧 FA2、O/LSE 回传、`combine_kv`。`merge_ms` 只是最后这次 LSE 合成，不是整步。

A100 上 head-TP 是预热 3 次、测量 8 次的 CUDA event 和墙钟。NanoCP 是丢掉 1 次预热后再取 5 次中位数。5090 两边都是 CUDA graph：图在 GPU0 的流结束前会等 GPU1，所以墙钟是两张卡的 makespan。head-TP 8 次中位数，NanoCP 5 次。kernel 时间是另一次 CUPTI replay，两张卡重叠，不能加起来当成墙钟。

正确性：head-TP 拼回后与单卡全头的最大绝对误差门槛是 `4e-3`（`check_mix`，输入 `[64, 96, 128, 17]`）。早期 dense 路径用过 1 个 fp16 ulp。NanoCP 两卡 e2e 在 5090 上是 `2.0e-3` 到 `3.9e-3`，门槛 `1e-2`。不要为了消掉这点差把两张卡的 batch 凑成一样长。`d=4` 只对过页字节，没有 4 卡性能数。

## 环境

### 2×A100-SXM4-40GB

Vast.ai，镜像 `pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel`。真实 NVLink：`nvidia-smi topo -m` 是 `NV12`，每向 12×25 GB/s，`can_device_access_peer` 双向为真。不要信广告里的 `bw_nvlink`。PCIe 那台（实例 `60172964`）peer 是 false，通信测试作废。

用过的实例：`60177457`、`60235220`、`60393954`，以及同机对比用的 `60811672`（标签 `nanocp-decode-cp`）。停机不销毁。SSH 端口每次重启会变。FA2 编的是 `FA2_MINIMAL_HDIM256_BF16=1`，装上的是 `vllm-flash-attn 2.7.2.post1+cu130`。108 个 SM。kernel 可以直接写对端地址。

早期 dense 基线（`operator_refactor.md` 第 2 节）也是这种卡，走的是 `fwd_kvcache`，KV 被 pad 成 `[batch, max_seq, 2, 256]`。那张表不能拿来和后来的 paged 数逐格对比。

### 2×RTX 5090

客人机，不是 Vast。SSH `ssh -p 42971 root@114.214.240.30`。Ubuntu 26.04，KVM 直通，GPU 是 GB202 `10de:2b85`，各 32 GB，170 个 SM，sm_120。拓扑是 PHB，没有 NVLink。FA2 只编了 bf16 head-dim 256，二进制是 sm_80 cubin 加 compute_80 PTX，驱动在 5090 上 JIT。Torch 保持 `2.14.1+cu130`，不要让 pip 把 torch 拉回 2.4。

P2P：打过社区驱动之后 `cudaDeviceCanAccessPeer` 为真，但 `cudaDeviceEnablePeerAccess` 返回 205 `cudaErrorMapBufferObjectFailed`。kernel 写对端 `data_ptr` 是非法地址。`cudaMemcpyPeer` / 源卡流上的 `cudaMemcpyAsync` 可用，大块大约 39 GB/s。`cudaMemcpyPeerAsync` 在 CUDA graph 捕获期间会失败，图里用的是对端指针上的 `cudaMemcpyAsync`，目标流 `wait_event`。PyTorch 的 `.to()` / `.copy_()` 跨这两张卡也会 205。

非交互 SSH 不读 bashrc，用 `. /root/dev-env.sh`。不要设 `CUDA_LAUNCH_BLOCKING`，不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，计时区里不要 `seqused.max().item()`。

## 1. dense 算子，2×A100

分支 `head-tp-kernels`，尖端曾经是 `b9c0fe6`。发送放在同一次 launch 里空闲的 head tile 上。预热 3，测量 8。`half` 是「一个 head 的时间 /（单卡两个 head 的一半）」，1.0 表示刚好一半。

| 用例 | base0 | split 墙钟比 | std 墙钟比 | half_split | half_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20×32768 | 1.071 | 0.637 | 0.899 | 1.042 | 1.520 |
| 128×8192 | 2.142 | 0.723 | 1.145 | 1.346 | 2.185 |
| 128×32768 | 7.252 | 0.665 | 0.986 | 1.300 | 1.935 |
| 128×65536 | 14.374 | 0.659 | 0.964 | 1.301 | 1.909 |
| 256×32768 | 13.006 | 0.593 | 0.944 | 1.166 | 1.870 |

不通信已经比单卡快，256×32768 的墙钟比 0.593。一个 head 到不了两个 head 的一半，因为两个 head 在一张卡上会重叠。通信放进算子之后优势几乎没了，墙钟比 0.94–1.15。20 条填不满 108 个 SM。

## 2. paged 第一版，2×A100，实例 60235220

改到 `block_table + seqused_k`。长请求的物理页只放本卡的 KV head，一个物理页收 `d` 个逻辑页。短请求的页仍是全部头。`d=2` 和 `d=4` 的页布局通过。带一条本地行的拼回，长行不超过 1 个 bf16 ulp，本地行逐位相同。

这一版 harness 把 `num_splits` 写成 1，发送 CTA 排在本地 head 后面。纯长请求：

| 用例 | base0 | split0 | std0 | w_spl | w_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20×32768 | 1.683 | 1.751 | 2.072 | 1.131 | 1.288 |
| 128×8192 | 1.756 | 1.201 | 1.860 | 0.744 | 1.119 |
| 128×32768 | 6.879 | 4.513 | 6.912 | 0.658 | 1.021 |
| 128×65536 | 14.426 | 8.942 | 13.656 | 0.621 | 0.954 |
| 256×32768 | 13.058 | 7.379 | 12.219 | 0.567 | 0.944 |

不通信的 256×32768 墙钟比 0.567，和 dense 的 0.593 同一档。带通信仍是 0.944。

短 batch 里夹超长请求，paged 放得下（128×256 + 2×128k + 1×256k 实际页 1.06 GB，若 pad 到 256k 要 65.5 GB），但墙钟比是 1.09–1.12。只有两三条长请求，拆开之后也填不满 SM。

接着做了 CTA 扫描，长度固定 32768，单卡强制一个 head 一个 CTA。单卡每个正在发 load 的 CTA 大约 16–21 GB/s，整卡在大约 64 个 CTA 时到 1.2–1.3 TB/s。

| 条数 | 单卡 | 墙钟比 | 单卡 CTA | 每卡 CTA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.058 | 1.250 | 2 | 1 |
| 32 | 1.762 | 1.337 | 64 | 32 |
| 64 | 4.408 | 0.851 | 128 | 64 |
| 80 | 4.458 | 0.918 | 160 | 80 |
| 108 | 5.591 | 1.011 | 216 | 108 |
| 256 | 13.052 | 0.945 | 512 | 256 |

条数少时，字节少一半，发 load 的 CTA 也少一半，时间不变。48–80 条是唯一稳定低于 1 的一段。每张卡的 CTA 铺满 108 个 SM 之后，发送排在本地 head 后面，墙钟比回到 1 附近。

把到达标志提前置上、返回标志也置上，可以把「等发送」和「等返回」拆开。64 条起，置上标志之后 kernel 从 3.6–12.2 ms 掉到大约一半。256×32768：现在 12.21，标志已在 7.38，两边都不等 7.31。返回只占 0.1–0.2 ms。少条数的 256k 置标志几乎不变，仍是 13 ms 量级，因为一张卡上只有几个 CTA。单卡若把 4×256k 沿 KV 切开，可以到大约 1.8 ms，两卡不切开是 13.6 ms。

## 3. 发送提前，并允许按 KV 拆 CTA

提交 `79a4e83`。发送改到本轮计算 tile 之前。`check_mix` 在 1 split 和启发式 split 下都过。无通信路径在较大 batch 上仍有非法访存，下表只有单卡和带通信。

| 输入 | 单卡 | 带通信 | 墙钟比 | 拆分 |
| --- | ---: | ---: | ---: | --- |
| 4×256k | 1.761 | 1.276 | 0.802 | 启发式，约 23 |
| 16×256k | 8.073 | 4.150 | 0.507 | 6 |
| 32×256k | 26.098 | 6.841 | 0.246 | 3 |
| 4×256k，强制 1 split | 27.303 | 13.665 | 0.465 | 1 |
| 2×256k + 32×512 | 5.467 | 4.991 | 0.906 | 约 3 |
| 同上，强制 1 split | 16.004 | 17.385 | 1.083 | 1 |
| 4×128k + 64×1024 | 2.821 | 2.982 | 1.097 | 启发式，接近 1 |
| 1×256k + 2×128k + 128×256 | 15.896 | 13.393 | 0.840 | 1 |
| 16×256k + 512×512 | 16.467 | 13.619 | 0.825 | 1 |
| 4×256k + 1024×1024 | 16.304 | 15.345 | 0.939 | 1 |
| 128×32768 | 7.220 | 5.441 | 0.749 | 1 |
| 256×32768 | 12.208 | 7.710 | 0.629 | 1 |

4×256k 强制 1 split 时带通信仍是 13.7 ms。收益来自按 KV 拆 CTA，不是发送提前本身。256×32768 从 12.2 ms 降到 7.7 ms，这是发送不再排在整波计算后面。短请求已经很多时，启发式对整批返回 1，超长行不会被单独拆开，`4×256k+1024×1024` 仍是 0.94。32×256k 的单卡从大约 13 ms 变成 26 ms，墙钟比 0.246 偏乐观，不要单独拿这个比当结论。

## 4. 每行自己的 split 列表，上限 16

`6858613` 之后，长行可以有自己的 split，不再整批一个数。`dfae77d` 上、实例 `60393954`，`check_mix` 通过。先是单卡仍强制少拆、tile 把长尾切到 16，混合看起来非常快（mix_32 墙钟比 0.111）。把单卡改成 FA2 启发式之后，对比才公平：

| case | base0 | std 墙钟 | 墙钟比 | 单卡 split | tile split |
| --- | ---: | ---: | ---: | ---: | ---: |
| mix_32x512_2x256k | 5.511 | 1.779 | 0.320 | 3 | 16 |
| mix_64x1024_4x128k | 2.867 | 1.235 | 0.424 | 3 | 16 |
| 4×262144 | 1.844 | 1.779 | 0.941 | 23 | 16 |
| 16×262144 | 6.694 | 3.957 | 0.587 | 6 | 6 |
| 32×262144 | 13.339 | 7.042 | 0.526 | 3 | 3 |
| mix_512x512_16x256k | 13.072 | 4.107 | 0.313 | 1 | 6 |
| mix_1024x1024_4x256k | 13.827 | 3.268 | 0.235 | 1 | 16 |
| 128×32768 | 7.138 | 5.009 | 0.697 | 1 | 1 |
| 256×32768 | 13.092 | 7.791 | 0.593 | 1 | 1 |

短请求多的时候，启发式给单卡的 split 很少，tile 仍切长尾，加速还在。纯 4×256k 上启发式开到 23，单卡已经很快，两卡比接近 1。

## 5. 同机 NanoCP，2×A100 实例 60811672

FA2 是 `head-tp-paged` 的 `3041cde`。NV12，peer 双向可用。head-TP 只跑了单卡和带通信（无通信路径仍跳过）。NanoCP 这一版还没有 per-row tile，传输是节点内 all-to-all 形状的 handshake kernel，不是 5090 后来的 memcpy。`combine_kv` 稳定在 0.08 ms 左右。

| case | head-TP 单卡墙钟 | head-TP 带通信 | NanoCP 墙钟 | NanoCP merge |
| --- | ---: | ---: | ---: | ---: |
| 20×32768 | 1.164 | 1.124 | 1.753 | 0.081 |
| 128×8192 | 1.998 | 1.828 | 1.892 | 0.083 |
| 128×32768 | 7.018 | 5.466 | 4.306 | 0.081 |
| 128×65536 | 14.399 | 9.251 | 7.815 | 0.083 |
| 256×32768 | 13.118 | 7.732 | 7.071 | 0.083 |
| mix_128 | 12.677 | 1.443 | 8.652 | 0.084 |
| mix_64 | 2.323 | 1.031 | 2.097 | 0.084 |
| mix_32 | 4.367 | 1.426 | 3.407 | 0.084 |
| mix_512 | 13.029 | 4.085 | 7.305 | 0.082 |
| mix_1024 | 13.795 | 3.179 | 8.078 | 0.084 |

短的纯 batch 上 head-TP 更快或打平。更长的纯 batch 上 NanoCP 更快：两边每卡 KV 字节相近，但 head-TP 一条请求只有 1 个 CTA 扫完整段，NanoCP 一条请求有 2 个 CTA、各扫一半。128 条时 head-TP 的第二波只占 20/108 个 SM，每个 CTA 仍扫完 32k。merge 的 0.08 ms 不是这个差距。混合长尾上 head-TP 更快，因为 tile 把那几条超长请求拆开了，而这一版 NanoCP 还没有同样的 tile。

40 GB 上不能把完整 KV 和两半同时放在 GPU0。`128×65536` 的 K+V 大约 16 GB，三份拷贝会 OOM。bench 改为在 CPU 上切段，删掉完整 cache，再分别放到两张卡，这一步在计时外。

这次同机对比之后，NanoCP 也加了和 head-TP 相同公式的 tile（提交 `2be62c17e9`）。A100 上 tile 之后的整表没有留在同一次记录里。下面的 5090 表是两边都有 tile 之后的数。

## 6. 5090 上传输走不通，改成 memcpy

先试了 kernel 直写和 CUDA IPC。`cudaIpcOpenMemHandle` 失败。kernel store 非法。改成 `cudaMemcpyPeer`，A100 上原来的 kernel 直写留在 phase 0，不改。

5090 的 head-TP 被拆成三次 launch，环境变量 `HEAD_TP_PHASE`：

| phase | 做什么 |
| --- | --- |
| 1 send | 只把本卡平面写好，不扫 KV |
| 2 compute | 等平面，算 attention，把伙伴的 Q head 写进本地 payload |
| 3 splice | 写进用户的 out |

最初 phase 3 又跑了一遍 attention，带通信大约是无通信的两倍。改成专用 kernel `head_tp_splice_only` 之后，splice 本身是 0.003–0.006 ms。

NanoCP 最初按 token 调同步的 `cudaMemcpyPeer`。`mix_1024` 从 64.561 ms 降到把活跃行打包成一次拷贝之后的 6.482 ms，再收进 CUDA graph、并改掉逐行拷贝之后，才到下面表里的 2 ms 量级。

## 7. 5090 上 head-TP 带通信的几次墙钟

机器和 case 不变。NanoCP 列是最终数，中间没有每次重跑。head-TP 四列都是带通信墙钟。

| case | phase3 仍是 attention | 专用 splice | 一轮 SM 的 tile | 只拷长行 | NanoCP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20×32768 | 1.085 | 0.642 | 0.640 | 0.566 | 0.527 |
| 128×8192 | 1.644 | 0.967 | 0.963 | 0.841 | 0.841 |
| 128×32768 | 5.451 | 2.880 | 2.876 | 2.761 | 2.800 |
| 128×65536 | 10.522 | 5.415 | 5.407 | 5.278 | 5.415 |
| 256×32768 | 10.890 | 5.620 | 5.620 | 5.478 | 6.128 |
| mix_128 | 2.250 | 1.279 | 0.783 | 0.584 | 0.502 |
| mix_64 | 1.333 | 0.789 | 0.727 | 0.565 | 0.493 |
| mix_32 | 2.134 | 1.173 | 0.643 | 0.505 | 0.438 |
| mix_512 | 6.393 | 3.441 | 3.481 | 3.082 | 3.031 |
| mix_1024 | 5.039 | 2.895 | 2.874 | 2.283 | 2.141 |

tile 那一列曾经先忽略短行已经占住的 SM。`mix_1024` 因此退到 3.141，combine 从 0.173 升到 0.430。扣掉短行之后回到 2.874。表里用的是扣过之后的数。

只拷长行：交换缓冲是 `[degree, tokens, ...]`，以前把整个 batch 的平面按两个方向都拷。短行对端不读。`mix_1024` 把 4 条长行放大成 1028 个 token，Q 面约 4.2 MB，两轮两个方向合计约 19 MB，走 PCIe 是零点几毫秒。CUPTI 把这 12 次 memcpy 记成 0.05 ms，那不是流被堵住的时间。现在只拷长行前缀，而且只走有数据的方向：compute 前 owner 到 partner，splice 前 partner 的输出回到 owner。`mix_1024` 这 6 次是 0.006 ms。`check_mix` 仍通过。

最终比：

| case | head-TP 单卡 | 无通信 | 带通信 | NanoCP | 带通信/NanoCP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20×32768 | 0.841 | 0.445 | 0.566 | 0.527 | 1.074 |
| 128×8192 | 1.315 | 0.680 | 0.841 | 0.841 | 1.000 |
| 128×32768 | 5.114 | 2.596 | 2.761 | 2.800 | 0.986 |
| 128×65536 | 10.171 | 5.131 | 5.278 | 5.415 | 0.975 |
| 256×32768 | 11.724 | 5.268 | 5.478 | 6.128 | 0.894 |
| mix_128 | 2.938 | 0.491 | 0.584 | 0.502 | 1.163 |
| mix_64 | 1.522 | 0.479 | 0.565 | 0.493 | 1.146 |
| mix_32 | 2.926 | 0.410 | 0.505 | 0.438 | 1.153 |
| mix_512 | 14.629 | 2.974 | 3.082 | 3.031 | 1.017 |
| mix_1024 | 15.082 | 2.141 | 2.283 | 2.141 | 1.066 |

纯长 batch 打平或 head-TP 更快。`256×32768` 是 5.478 对 6.128，因为 head-TP 每卡只有 1 个 KV head，条数已经够填 170 个 SM，不必再拆。小混合 NanoCP 仍快约 15%。`mix_512` 和 `mix_1024` 是 1.02 和 1.07。

`mix_1024` 的伙伴卡不一样。head-TP 伙伴拿长行另一个 KV head 的整段，再加上自己的 1024 条短请求，FA2 是 1.960 ms，和 owner 的 1.954 重叠。NanoCP 伙伴只有长行后半段、两个 KV head，短请求留在 master，FA2 是 0.669 ms。长行 KV 字节两边相同，多出来的是这 1024 条短请求，不单独加到墙钟上。

## 8. 剩下的 combine 差

卡间 head-TP 不合并，只把伙伴的 Q head 拷到对应位置。`fa2_combine` 是同一次 FA 调用里的第二个 kernel：一条 KV 被拆成多段 CTA 之后，把各段 partial 收成这一行的输出。128 和 256 条纯长请求 split 是 1，这个 kernel 不启动。

170 个 SM 上，长行 CTA 数两边一样，所以 attention 时间对齐。head-TP 每个长行只有 1 个 KV head，同样的 CTA 数就是两倍的 split。combine 按最大 split 数循环。

| case | head-TP split | NanoCP split | CTA |
| --- | ---: | ---: | ---: |
| mix_32 | 80 | 40 | 160 |
| mix_64 | 34 | 17 | 136 |
| mix_128 | 75 | 37 | 151 / 150 |
| mix_1024 | 14 | 7 | 56 |

测得的 `fa2_combine`（owner）：`mix_32` 0.040 对 0.021，`mix_128` 0.102 对 0.051，`mix_1024` 0.151 对 0.076。

NanoCP 跨卡还要做一次 `combine_kv`，只收两份已经收完的结果。5090 上是 0.001–0.009 ms。总份数接近：head-TP 一张卡收 80 份；NanoCP 两张卡各收 40 份，再跨卡收 2 份。两张卡的 40 份是同时收的，墙钟只看到 40，再加那 2 份。head-TP 的 80 份在每张卡自己的关键路径上。`mix_128` 墙钟 0.584 对 0.502，差 0.082，其中大约 0.05 是这次卡内 combine。

## 还没做

- 算子接进 `Qwen3NextAttention.forward`，并让 CUDA graph 走这条路径。现在框架里仍是 SDPA。
- 剩余长度预测，以及 prefill / CPU prefix 命中时按 head 直接把 KV 送到目标卡。调度和异步搬运的骨架已经在 `dtp_test` 里。
- 端到端：1% / 5% / 20% 长请求，短的用 ShareGPT，长的用 GitHub 轨迹，再加 AgentX。指标是 TPOT 和端到端延迟。机器计划过同架构、更长上下文的模型，没有实际跑。
- `d=4` 的带通信性能。页布局可以在一两张卡上对字节。
- A100 上 NanoCP 加 tile 之后，没有和 head-TP 再同机重跑一整表。5090 的表是当前对照。
