# NanoCP 论文与开源实现

NanoCP 要解决的是 MoE 在 DP–EP decode 里的一种结构性失衡：每个请求的 Attention、MoE 通信和 KV cache 被绑在同一个 DP 实例上，而 Attention 延迟跟 KV 体量走、dispatch/combine 延迟跟 batch 走，这两件事无法同时均衡。它的做法是按请求动态选择上下文并行度，把「谁做 MoE」和「KV 放在哪」拆开。长请求把 KV 分到多个实例，短请求留在本地。

论文是 ASPLOS 2027 的 *NanoCP: Taming Load Imbalance in Data-Expert Parallel Decoding under Variable-Length Workloads*（[arXiv:2605.21100](https://arxiv.org/abs/2605.21100)）。开源实现在 [NanoEngine 的 `decentral-aug` 分支](https://github.com/nano-infra/NanoEngine/tree/decentral-aug)，Python 包名是 NanoDeploy。代码里不叫 DCP，叫 sequence parallelism（SP）。

下面按论文章节讲解。每一节先讲论文在说什么，再对到代码。

## 术语对照

| 论文 | 代码 | 含义 |
| --- | --- | --- |
| DP instance | 一个 SP rank（`attention_sp` 里的一员） | 一份完整 Attention 副本加一份专家分片 |
| 一个 node 上的实例集合 \(\mathcal{S}_n\) | 一个 engine，内部有 `attention_sp` 个 rank | KV 绑定不出这个 engine |
| MoE binding \(m_r\) | `master_sp_idx` | 这个请求的 Q 投影、MoE dispatch/combine、采样都在这一张卡 |
| KV binding \(\mathcal{P}_r\) | `num_dispatched_tokens[sp] > 0` 的 rank 集合 | 这些卡存该请求的 KV 分片并算 partial attention |
| CP degree \(k_r\) | 上述集合的大小 | 1 表示纯本地 |
| \(\textit{Split}_r\) | `num_dispatched_tokens` | 每个参与 rank 分到多少 token |
| Q-Route / Res-Route | `sp_q_matrix`、`sp_res_matrix`，再落到 `q_mask`、`res_lse_mask` | 稀疏路由表 |
| 执行形状 \((M, N)\) | `master_bs`、`attention_compute_bs` | \(M\) 是本卡作为 master 的请求数，\(N\) 是本卡要算 attention 的请求数 |
| DCP | SP，backend 为 `hao_basic` 或 `nccl` | 同一套数据通路 |

论文正文写的是一个全局 C++ 调度器，用 RDMA 把元数据送给数据面。公开的 `decentral-aug` 已经是第 7 节讨论的分层控制面：跨 engine 的 `RequestRouter`，加上每个 engine 里的 C++ `Scheduler` / `SPStateManager`。`DecodeCoordinator` 只负责多 engine 的 wave 同步，不拥有请求，也不拥有 KV。

## 1. Introduction

生产里的 MoE 服务通常是 prefill–decode 分离，再加上 DP–EP。非专家层（Attention、Norm、QKV/O 投影）在每个 DP 实例上各有一份完整副本；专家按 EP 切开。每一层 MoE 都有一次 dispatch all-to-all 和一次 combine all-to-all，集体通信的延迟由最慢的 rank 决定。

论文给请求放置定义了两个绑定：

- **MoE binding**：做这个请求的 dispatch 和 combine 的实例。全文里它始终是单例。
- **KV binding**：存放这个请求的 KV、并执行它的 Attention 的实例集合。

现有请求级系统（vLLM、以及同类 DP 调度）把这两个绑定焊在同一个实例上。变长流量下这会造成两个独立的问题。

**负载问题。** Attention 延迟随该实例上驻留的 KV token 数增长；DeepEP 的 dispatch/combine 延迟随进入 all-to-all 的 token batch 增长。为了 KV 均衡而路由，batch 就会歪；为了 batch 均衡而路由，KV 就会歪。EP 是锁步的，任一维度的慢实例都会把整组拖住。这和专家内部不均衡（EPLB、冗余专家）是正交的：专家即使完全均衡，实例之间的 Attention 时间和 batch 时间仍会对不齐。

**容量问题。** 1T 参数的 MoE 在 32×H200、32DP–32EP 上，单实例 KV 容量大约 100 万 token，已经和一条长上下文请求同一个量级。KV 不能跨实例切的时候，集群里加起来够用的空闲块，也会因为没有任何一个实例单独够用而让队头请求进不去。这就是 head-of-line blocking。

已有的上下文并行补不上这个洞。LoongServe 在请求组粒度上弹性调整 CP：长请求组的 CP 度变大之后，Attention 更均衡，但各 DP 实例上的 batch 又歪了，EP straggler 还在。Helix 给每个请求同一个固定 CP 度：MoE 仍在一个实例上，KV 被扩到固定大小的组。短请求也要跨实例传 Q、回 partial、做合并，而且每个 CP 实例看到的有效 Attention batch 变大，FlashMLA 的 kernel 效率下降。两者都没有在同一步 decode 里、按请求独立选择 MoE binding 和 KV binding。

NanoCP 的 DCP 就是这件事：MoE binding 用来均衡 batch；KV binding 的大小按该请求的 KV 体量来选，并用通信代价约束它。长请求 CP 度大，短请求 CP 度为 1。

开源仓库约 1 万行，叠在 FlashMLA、Triton-distributed、DeepEP、DeepGEMM、nano-vllm 和 Ray 上。通信库是他们自己的 DLSlime（`hao_basic`，带 offsets 的节点内 all-to-all），不是把 DeepEP 拿来传 Q。

## 2. Background and Motivation

### 2.1 变长负载

OpenRouter 上 Gemini 3 Flash Preview 的调用统计（2026-02-23）里，大约 1.67% 的请求超过 100k token，多数请求仍然很短。评测按 Medha 的做法混合两条轨迹：ShareGPT-4o 当短请求，GitHub Issues 派生的软件工程轨迹当长请求。长请求占比取 1% 和 5%，分别代表常见生产和更重的长上下文压力。另外还有纯短、纯长两组对照。

长度分布（论文 Table 1）：

| 数据集 | <1k | 1k–10k | 10k–100k | 100k–500k | 500k–1M |
| --- | --- | --- | --- | --- | --- |
| OpenRouter | 31.82% | 50.08% | 16.42% | 1.67% | |
| ShareGPT-4o | 85.7% | 10.7% | 3.5% | — | — |
| GitHub Issues | — | — | — | 65.06% | 34.94% |

### 2.2 为什么是 DP–EP，而不是把 Attention 做 TP

对 MLA，按头切 TP 会把共享的 latent KV 在 TP rank 上复制，MLA 省 KV 的好处被吃掉。专家本身很小，再按 TP 切开会得到喂不饱 GPU 的碎 GEMM。所以 Attention 整份复制，专家用 EP 切开，靠请求级并行和聚合显存带宽来扩。

Figure 3 的微基准把两个缩放律分开了：

- FlashMLA：固定每卡缓存 token 数时，延迟随总 token 数上升；同样的总 token 数下，batch=1024 比 batch=1 或 128 更慢。序列更多、每条更短，persistent tile scheduler 的序列开销和尾 tile 摊不掉。
- DeepEP：dispatch 和 combine 都随每卡 batch 变大，combine 更贵。

### 2.3 现有均衡策略为什么失败

#### 请求级均衡

**LeastCache。** 新请求去当前 KV 最少的实例。Attention 齐了，batch 会极歪。论文 Figure 4(a) 的例子是一边 2 个请求、一边 98 个。batch 小的那一侧 dispatch receive 很久，另一侧 MLP 和 combine send 早就结束，卡在 combine receive 前空等。32 个 DeepSeek-V3 实例、Issue 1%、30 req/s 时，多数实例的 MoE 通信在 100–500 µs，straggler 到 1020.6 µs，均值 354.7 µs。若能拉平，最大延迟能降 65.2%。

**LeastBatch。** 新请求去当前 batch 最小的实例，这是 vLLM 的默认策略。batch 齐了，KV 会极歪，例如一边 100K token、一边 500K。轻的一侧 Attention 和 dispatch send 结束后，要等重的一侧算完才能继续。同设置下 Attention 多数在 40–55 µs，straggler 539.5 µs，均值 184.3 µs，拉平后最大延迟可降 65.8%。

**队头阻塞。** 一条长请求必须整段放进某一个实例。Figure 5(c) 里，集群空闲块一直很高，队头请求的需求却高于任一实例的本地空闲，于是后面的请求全被堵住。

#### 均匀 CP

Helix 把每个请求的 KV binding 扩到同一个固定 CP 组，MoE binding 仍是单例。Figure 6 在每卡约 100 万 token 的前提下扫 CP2/4/8：通信在每档 CP 上都在，短序列加大 batch（例如 8K×128）时，通信占 Attention 时间的大头；序列变长、batch 变小后，通信占比才降下来。同时，均匀 CP 把组内全部请求的 Attention 都摊到每个成员上，有效 batch 变大，正好撞上 Figure 3(a) 里「同样 KV token、更大 batch 更慢」的 kernel 效率损失。

### 2.4 DCP 想同时拿到的两件事

Figure 7 把三种绑定画在一起：

- DP：Q、KV、MoE 在同一实例。
- 均匀 CP：每个请求的 KV 组一样大，短请求也付钱。
- DCP：长请求的 KV 组大，短请求的 KV 组小；MoE 绑定独立，而且是单例。

Figure 4(d) 的示意结果是两侧各 300K KV、各 50 的 batch，通信只发生在真正被拆开的请求上，这一层的时间最短。长请求的 KV 可以落在多个实例的碎片空闲里，所以不再要求「某一个实例单独装得下」。

## 3. 系统总览

Figure 8 把系统拆成控制面和数据面。一次 decode 请求的生命周期是：

1. 外部 prefill 结束后进入全局等待队列。
2. 准入时决定 MoE binding、KV binding 和 token 切分。
3. 控制面发起 KV 迁移，数据面完成搬运。
4. 调度决定被降成每个实例的路由元数据。
5. 实例按 \((M, N)\) 查找并重放已经捕获好的 CUDA Graph。
6. 执行时 master 把 Q 送到 KV binding；每个成员只对自己的 KV 分片做 paged attention，把 partial 输出和 LSE 直接写回 master；master 合并后再进 MoE。

decode 过程中不再搬 KV。KV 在准入时放到位，之后每步只搬 Q 和 partial 结果。这一点是后面整个数据通路的前提。

代码里这条生命周期落在：

- 准入与放置：`nanodeploy/router/request_router.py`、`admission_planner.py`，以及 C++ `sp_state_manager.cpp` 的 `can_allocate`。
- 每步组 batch、生成 Q/Res 矩阵：`scheduler.cpp` 的 `schedule()`。
- Attention 数据通路：`nanodeploy/layers/attention.py` 的 `FlashMLAImpl.forward`。
- 图的选择和元数据注入：`nanodeploy/worker/sp_graph_policy.py`。

## 4. 控制面

### 4.1 为什么要全局视图

主流系统把调度器和执行器焊在每个 DP 实例里。DCP 做不到这样，原因有两个。EP 的 dispatch/combine 每步锁步，不平衡立刻变成 straggler，所以每步都要看见所有实例的 batch。KV 要切到 token 粒度才能吃掉碎片，本地调度器看不见别的实例的空闲块。

`decentral-aug` 的实现比论文 Algorithm 1 的「一个进程看完全部 GPU」更松一点，和论文第 7 节的分层设计一致：

- `RequestRouter` 看每个 engine 的 `LoadSnapshot`，决定请求进哪个 engine。一个 engine 对应一个节点内的 SP 域。
- engine 内部的 C++ `SPStateManager` 决定 master 和 token 切分。
- `DecodeCoordinator` 只广播 wave 的开始和结束，让多个 engine 一起醒、一起停。

### 4.2 双均衡调度

论文 Algorithm 1 对每个请求产出 \((n^\*, \mathcal{P}_r, m_r, \textit{Split}_r)\)。符号是：\(\ell_r\) 序列长度，\(K_s\) 实例 \(s\) 的 KV 负载，\(B_s\) 本步把 MoE 放在 \(s\) 上的请求数，\(B^{(n)}\) 是节点 \(n\) 上这些 \(B_s\) 的和，\(K_{\max}\) 是单实例 KV 容量。

#### 4.2.1 活跃请求的 MoE 重绑定

请求结束后 \(B_s\) 会歪。活跃请求可以在已经存在的 \(\mathcal{P}_r\) 里面改 \(m_r\)，因为 KV 不用搬。算法按 \(|\mathcal{P}_r|\) 从小到大处理：选择少的请求先放，每个请求放到自己 \(\mathcal{P}_r\) 里当前 \(B_s\) 最小的实例。

公开代码里，decode 步上没有对已在跑的请求重写 `master_sp_idx` 的路径。`Scheduler::_schedule_decode` 沿用准入时写死的 master，只做 `can_append` / `may_append` 和抢占。batch 均衡主要发生在准入时选 master，而不是每步在旧的 KV 集合里重选。这是论文算法和这条分支之间最明显的差别。

#### 4.2.2 选节点和 CP 度

新请求先去 \(B^{(n)}\) 最小的节点。代码里 `RequestRouter._candidate_engine_ids` 在 `least_batch` 下按 `(running + 尚未提交的准入数, engine_id)` 排序，注释写明这是在对齐集中式调度器的节点顺序。另有 `least_cache`（按估计剩余块）和 `least_projected_load`（把 Attention token 量和 batch 归一化后比较，避免手写一个 token/请求的换算系数）。

CP 度来自离线的 \(\textsc{Bucket}(\ell_r)\)，再被节点内实例数截断。离线过程是扫序列长度和候选 CP 度，量 DCP 下的分布式 Attention 延迟（计算加通信），每个长度区间取延迟最小的度数。这个映射跟模型和机器有关，跟线上长度分布无关。

`ae_scripts/algorithm/search_sp_buckets.py` 把这件事做成动态规划。输入是一张网格：每个 `seq_len`、每个 `cp_size` ∈ {1…8} 的端到端 master 侧 p50 延迟。目标是加权 regret 最小，约束是 SP 度随长度单调不减，允许跳过某些度数，每个桶至少 3 个采样点。输出是若干区间，例如 `SP=k: [low, high]`。运行时策略字符串是 `k:low-high;...`，由 `SPStateManager::parse_bucket_policy` 和 `AdmissionPlanner._parse_bucket_policy` 解析。查表函数是 `select_bucket_sp_size` / `_bucket_sp_size`：长度落进区间就用该 SP 度，否则退回按 segment 估出来的度数。

除了 bucket，代码还有两条对照路径：

- `fixed_sp_size > 0`：全体请求同一个度数，用来复现均匀 CP。
- `dynamic_sp_size_strategy == "legacy"`：用 `segment_size` 把序列切成段，再按段数决定用几个 rank。这不是论文里的 Bucket。

#### 4.2.3 节点内放置和 token 切分

实现把 KV binding 限制在一个 engine 内，避免 Q/Res/LSE 走节点间关键路径。论文说抽象上允许跨节点，评测没有做。

节点内的顺序和 Algorithm 1 的 9–12 行一致，实现在 `SPStateManager::can_allocate`，Python 侧 `AdmissionPlanner._plan_legacy` 是同一套逻辑的影子，用来在真正提交前判断可不可行：

1. **选 master。** `sp_master_selector` 默认是 LeastBatch：`master_seq_counts_` 最小的 rank。也可以 RoundRobin 或 LeastCache（空闲块最多）。论文正文固定用 LeastBatch。
2. **选其余成员。** master 以外的 rank 按空闲块从多到少排，取前 \(k_r-1\) 个。这就是论文的 `SelectSmallestKV`：KV 负载最小，也就是空闲块最多。
3. **切 token。** 打开 `enable_non_uniform_split` 时走 water-filling。代码注释写的是 “Water-filling / Valley-filling”。它在参与者里找一个水位，使最富的若干 rank 分掉全部 token 之后，剩余空闲 token 尽量拉平，然后把整数余数摊回最富的 rank。论文的 WaterFill 写成最小化 \(\max_s(K_s+\textit{Split}_r[s])\)。两者在容量接近时结果接近；实现直接拉平的是剩余空闲，不是绝对 KV 负载。`fixed_sp_size > 0` 时改为均匀切。两者都关时，按 segment 均分。
4. **master 必须留下一个 token 槽。** 序列里已经包含即将被第一次 decode 消费的那个采样 token，它的 KV 槽必须在 master 上。water-fill 有时会把 master 分到 0。代码会从分到 token 最多的成员上挪 1 个回来。论文算法没有写这一条，这是实现为了 decode 能在 master 上 append 而加的约束。
5. **`CanAllocate`。** 每个分到 token 的 rank 都要有足够空闲块，还要留出 `reserved_blocks_per_req` 乘以该 rank 上 master 请求数的预留。非 master 成员还受 `max_num_recv_seqs` 限制。过不了就留在等待队列，下一轮再试。`AdmissionPlanner._fits_lifetime` 再检查 master 上能否装下后续生成 token（补全长度向上取整）。失败则这个 engine 本轮不收。

#### 4.2.4 路由表

放置确定之后，`Scheduler::schedule` 把每个请求降成两张计数矩阵，维度是 `[dp][src_sp][dst_sp]`：

- `sp_q_matrix[master][participant]++`：master 要把 Q 送给这个成员。
- `sp_res_matrix[participant][master]++`：成员要把 partial 送回 master。

只有真正跨了多于一个 rank 的请求才计入。CP 度为 1 的请求不产生跨卡项。数据面再把这些计数展开成二值 mask 和 slice 下标：`q_mask`、`res_lse_mask`、`q_slice_get/fill`、`res_slice_*`。论文 Figure 9 的例子（5 个请求、4 个实例，实例 2 上 \(M=2\)、\(N=3\)）就是这张表的一个具体形状。

`schedule()` 还有一个执行侧的细节：每个 DP 的序列列表会按 SP rank 滤一遍，`filtered_dp_sp_seqs` 只保留 `master_sp_idx == 本 rank` 的请求。远程请求不会进入伙伴的本地 batch 去跑 MoE。伙伴卡上多出来的是 attention 行，不是 MoE 行。某个 SP rank 本步没有任何 master 请求时，会塞一条 dummy sequence，让集体通信的参与者集合保持完整。

## 5. 数据面

### 5.1 Q 路由，而不是搬 KV

每个 decode step 的四相，对应 `FlashMLAImpl.forward` 的 decode 分支：

1. **投影和 Q 路由。** master 上已经算好 Q。`copy_batch_indexed_triton` 按 `q_slice_*` 把本卡真实的 Q 行拷进发送缓冲，再 `q_buffer.all_to_all_ll`。`hao_basic` 走 `dst_row_indices`，按目标行直写；`nccl` 走 `q_mask` 和 `q_offsets`。offsets 的语义是每个源 rank 只发送自己的有效 Q 行，接收侧仍保持 `[sp_size, max_bs, ...]` 的视图，展平后前 `attention_compute_bs` 行是排紧的有效 Q。设计说明在仓库 `docs/DeepSeek_SP_HaoBasic_Offsets_Design.md`：offsets 只用于 Q 的 non-transpose all-to-all，Res 和 LSE 不传 offsets。
2. **本地 paged attention。** `q = q[:attention_compute_bs]`，配上截断后的 `context_lens_for_attn` 和 `block_tables`，调用 `flash_mla.flash_mla_with_kvcache`。MLA 要求 `num_kv_heads == 1`，KV 是压缩 latent，不是完整的 K 和 V。tile scheduler 元数据在进 transformer 层之前准备一次，各层复用。
3. **Res/LSE 送回。** partial 输出和 FP32 的 LSE 按 `res_slice_*` 拷进缓冲，两次 `all_to_all_ll(..., is_transpose=True)`。LSE 保持 FP32，因为下一步要用它重建全局 softmax 权重，量化会改变合并结果。
4. **LSE 合并。** `inter_rank_gqa_fwd_batch_decode_combine_kv` 在 master 上把各分片合成一条完整 Attention 输出，形状裁回本卡的 master batch。合并公式就是 Flash-Decoding 那一套：全局 LSE 是各分片 LSE 的 logsumexp，输出是 \(O_i \cdot \exp(\mathrm{LSE}_i - \mathrm{LSE}_{\mathrm{global}})\) 的和。合并之后才进入 MoE。MoE 的 all-to-all 仍是 DeepEP，和这条 CP 路由是两条独立的通信。

GQA 路径在同一个文件的 `FlashAttentionImpl` 里，结构相同，Attention 核换成 `flash_attn_with_kvcache`。论文的主结果是 DeepSeek-V3 和 Kimi-K2，都走 MLA。

同一条请求的 MoE 只在 master 上做一次。伙伴卡没有这条请求的 hidden state，也不会把它送进自己的专家。

### 5.2 动态路由怎么进静态 CUDA Graph

每步的 \((M, N)\) 都在变：谁是 master、谁持有哪些 KV 分片，都会改路由和 Attention batch。CUDA Graph 要求 launch 形状和缓冲地址固定。如果对每个 \(M\) 都按最坏的 \(N = M \cdot W_{cp}\) 去捕获，短执行会被 pad 到很大的 Attention batch，重新掉进 2.3.2 的 kernel 效率问题。每步在 CPU 上现组 kernel，又会把 launch 和同步放回 decode 关键路径。

Algorithm 2 的 AOT 图引擎因此预捕获一组二维桶 \((\hat M, \hat N)\)，共享一块够最大桶用的缓冲，每个桶只是这块缓冲上的固定地址视图。运行时选能盖住当前 \((M, N)\) 的最小桶，异步写入输入和路由元数据，然后重放。

`sp_graph_policy.py` 就是这个查找：

- `select_decode_graph_master_bs` 在 `graph_master_rank_bs` 里取大于等于实际 master batch 的最小桶。`nccl` 或固定 SP 度时，master 桶还要盖住通信 batch。
- `select_decode_graph_bucket` 再在 `sp_graph_map[master_bs]` 里取大于等于实际 `attention_compute_bs` 的最小 Attention 桶。
- `copy_decode_context_to_graph_vars` 把 input ids、positions、slot mapping、context lengths、两张 mask、block table、Q/Res 的 slice 下标写进图的持久张量，并调用 `update_sp_graph_metadata` 做融合的路由更新。

论文测得：32 卡上调度加元数据传输占迭代时间的 1.17%–2.61%；选图是字典查找，小于 2 µs；一次元数据注入大约 20 µs。48 张图、每卡 5.32 GiB，比 vLLM 32DP 多出来的大约 4.30 GiB，约是 H200 HBM 的 3%。vLLM 的 4DP8CP 是 68 张图、5.17 GiB，内存量级相同。

### 5.3 路由通信后端

NCCL 这类稠密集合通信要按固定布局做 all-to-all，DCP 的传输是稀疏、不对称的，空缺只能用零填充。DeepEP 一类 MoE 库假设固定宽度的 Top-K。DCP 里同一 batch 的请求 CP 度不同，扇出不固定，硬套 Top-K 就要 pad 到最大 CP 度，或者拆成多次通信。

DCP 的拓扑在这一步开始前已经由控制面定死，不依赖 MoE gate 的运行时输出。后端把路由表当成一张二维 mask：每个数据块读自己的 mask 行，通过 NVLink（节点内）或 IB（节点间）直接写到对方预分配的接收缓冲，接收方轮询本地到达标志。Figure 11 就是这个结构。

NanoDeploy 没有在仓库里重写这个 kernel。节点内直写在 DLSlime 的 `hao_basic`，NanoDeploy 的 `HaoAllToAllBufferAdapter`（`nanodeploy/worker/sp_backend.py`）负责把 mask、offsets、`dst_row_indices` 传下去。`nccl` backend 留着，就是第 6.6 节那个消融。README 把 DLSlime 钉在 `feat/hao-basic-alltoall-offsets` 的 `4f1fce6`，并且要求 `BUILD_INTRA_OPS=ON`；PyPI 上的默认包没有这套节点内算子。

## 6. Evaluation

### 6.1 设置

集群是 H200。每节点 8 卡，NVLink 全连接、双向 900 GB/s，8 张 50 GB/s RDMA 网卡。模型是 DeepSeek-V3 和 Kimi-K2。Attention 用 FlashMLA，MoE all-to-all 用 DeepEP，专家都是宽 EP。NanoCP 还支持 multi-step：一次调度决定复用若干个 decode step。

Attention 侧的 vLLM 基线：

- 32DP，LeastBatch 和 LeastCache 两种策略。
- 均匀 CP：16DP2CP、8DP4CP、4DP8CP。vLLM 这条均匀 CP 按 head 维切分，走的是 Helix 的均匀 decode-CP 数据通路。Helix 论文里的 HOP-B 没有可用开源实现，所以没比。

主指标是 TPOT，SLO 为 50 ms。报告的是 99% 请求满足该 SLO 时能撑住的最大请求率。复现脚本在 `ae_scripts/`，每个 `fig*/` 有自己的命令和硬件要求。

这里有一个和 NanoCP 自身机制不同的事实：论文拿来当基线的 vLLM CP 是按头切的；NanoCP 自己是按序列 token 切 KV，再用 LSE 合并。两者都叫 CP，切的维度不一样。

### 6.2 微基准

4 个节点，每卡 64 条 2048 token 的短请求，每节点再放 1、3、5 或 7 条 512K 的长请求。因为 EP 锁步，他们报告最慢实例的逐层时间，拆成 Attention、MoE all-to-all、CP 通信和其他。

Figure 13：四种长请求数量下 NanoCP 的总延迟都最低。CP 通信远小于 4DP8CP，因为只有会造成 Attention 失衡的请求才跨实例。最大 Attention 延迟也最小：相对 32DP，长请求 KV 被摊开；相对 4DP8CP，没有把整组 batch 都塞进每个 CP 成员。长请求变多之后，继续分 KV 的收益变小，因为已经没什么单点 straggler 可消。

### 6.3 端到端

Figure 12 是六组负载上的 SLO 达成率、平均 TPOT、P99 TPOT。混合负载上差距最大。

DeepSeek-V3，99% SLO 达成：

- Issue 1%：可持续请求率是最强 vLLM 基线的 3.27 倍。
- Issue 5%：2.11 倍。
- 在基线 P99 TPOT 到达 100 ms 的请求率上，NanoCP 把 P99 降到原来的 1/1.94 和 1/2.01。

Kimi-K2：

- Issue 1% / 5% 的可持续请求率是最强基线的 1.88 倍和 2.20 倍。
- 基线平均 TPOT 到 100 ms 的点上，平均 TPOT 加速 1.87–2.07 倍。
- P99 加速 1.79–2.12 倍。

纯 ShareGPT-4o 上，NanoCP 比 vLLM 32DP 略低一点延迟。绝大多数请求 CP 度为 1，选择性 DCP 的空转代价很小。纯 GitHub Issues 上，NanoCP 和 4DP8CP 接近：请求率很低时，CP8 的额外通信不大，而且已经把 straggler 消掉了。DCP 的收益集中在「多数很短、少数极长」的混合，而不是两个极端。

### 6.4 负载均衡

低负载，用变异系数（标准差除以均值）看 32 个实例：

- batch CV：NanoCP 7.34%，LeastBatch 3.24%，LeastCache 59.90%。NanoCP 接近 LeastBatch，远离 LeastCache。
- KV CV：NanoCP 51.15%，LeastBatch 156.77%，LeastCache 87.30%。两个维度上都比「只优化另一维」的策略更齐。

高负载看的是队头阻塞。vLLM 32DP 经常在节点总空闲仍然很多时卡住队头请求。NanoCP 主要在节点总空闲快用尽时才排队，因为一条长请求可以吃多个实例上的碎片。

### 6.5 逐实例延迟分解

Figure 15 把 32 张卡的逐层时间拆开，负载是 ShareGPT-4o 以及 1%、5% 混合，配置是 NanoCP、4DP8CP、32DP-LeastBatch、32DP-LeastCache。

Issue 1% 上，LeastBatch 的 Attention 最大/中位数是 811.3 / 111.8 µs，差 7.26 倍；LeastCache 的 Attention 峰值仍有 986 µs。NanoCP 把最大 Attention 压到 267.7 µs，最大与中位数的差距缩小 84% 以上。

MoE dispatch+combine 的峰值：LeastBatch 1132.9 µs，LeastCache 1342.7 µs，NanoCP 350.9 µs。CP 通信平均 60.4 µs，4DP8CP 是 629.8 µs，少 90.41%。

运行时（Figure 19）峰值只有 2279 个请求里的 25 个（1.1%）在用跨实例 CP，DCP all-to-all 峰值 35.5 µs。大部分时间系统看起来就像普通 DP，只在少数长请求上付钱。

### 6.6 消融

DeepSeek-V3、Issue 1%、40 req/s：

| 配置 | 平均 / P99 TPOT |
| --- | --- |
| 完整 NanoCP | 49.08 / 52.07 ms |
| 把 DCP 换成均匀 CP8 | 109.97 / 126.77 ms |
| DCP 的路由后端换成 NCCL | 412.65 / 732.34 ms |
| DCP 但用分段 CUDA Graph | 54.80 / 57.88 ms |

均匀 CP8 在高请求率下掉下去，因为短请求也付通信，而且有效 Attention batch 变大。NCCL 掉得最狠，稠密集合通信和这种稀疏不对称传输不匹配；Figure 18 在 Q all-to-all 上，路由后端相对 NCCL 有 1.05–2.03 倍加速，小 batch 优势最大。分段 CUDA Graph 比整图重放高大约 11–12%，说明按 \((M, N)\) 捕获完整一步是有测量得到的收益的，但量级远小于换通信后端。

### 6.7 开销

调度器、图内存和通信后端的数字已经在 5.2、5.3 和 6.6 里对过。这里只补一点：DCP 把控制面开销从 32DP 的 1.17%–1.77% 提高到 1.42%–2.61%，最多多 0.84 个百分点。相对 2–3 倍的吞吐，这笔开销不是瓶颈。

## 7. Discussion

**和计算通信重叠的关系。** NanoFlow 和 DeepSeek 的 Two-Batch Overlap 把一个 batch 拆成微批，让计算和通信重叠，但请求还是放在原来的实例上。变长负载造成的实例级 straggler 会把可重叠的窗口压短。Figure 20：ShareGPT-4o、100 req/s 时，TBO 把 vLLM 的平均 TPOT 从 82.33 ms 降到 56.70 ms，P99 从 93.61 ms 降到 67.67 ms。Issue 1%、10 req/s 时反过来，平均从 37.99 ms 升到 64.49 ms，P99 从 51.93 ms 升到 92.75 ms。长请求 straggler 限制了重叠，微批本身又多一层开销。

两者可以叠。按 DeepSeek 的 TBO 流水，把 Q-Route 和 Res/LSE-Route 插在 Attention 两侧，得到

`(SHARED + ATTN-0, Dispatch) → MLP → (Q-Route + ATTN-1 + Res/LSE-Route, Combine)`。

论文把联合调优留作后续工作，评测里的 NanoCP 没有开 TBO。

**部署范围。** MLA、GQA、MHA 在 DP–EP 下都会遇到同样的实例级失衡。加大 TP 消不掉它：GQA 的 KV 头很少，TP 超过 KV 头数就要复制 KV；MLA 按头切会复制共享 latent。节点内放得下更多 DP 实例的 SuperNode，才能让 DCP 在更大的高带宽域里均衡。当前实现的 KV binding 不出节点。

**规模。** 更大部署可以分层：轻量全局路由按聚合负载把请求分到 GPU 组，组内调度器做细粒度 DCP。他们用只跑调度器的 CPU 模型比了集中式和分散式准入。4 个逻辑节点上两者接近（Issue 1% / R90 是 5.5 ms 对 5.0 ms）。32 个逻辑节点上，分散式把准入从 119.5 ms 降到 13.6 ms（8.80 倍），另一组从 109.3 ms 降到 19.0 ms（5.76 倍）。`decentral-aug` 这个分支名指的就是这条路：`RequestRouter` 做组间放置，engine 内的 C++ 调度器做组内 DCP。

## 8. Related Work

论文把相关工作分成三组，边界划得很明确。

**服务侧 CP。** 训练里的 CP 用来拉长序列；服务侧的 LoongServe、InfiniteLLM、Meta 的 CP 推理设计、Medha、Helix 用它池化显存或缓解队头阻塞。LoongServe 在组粒度调 CP 度，Helix 全体请求同一个度。它们都没有在同一步里独立选 MoE binding 和 KV binding。

**MoE 系统。** kernel、通信库、专家放置和训练时负载均衡优化的是专家内部或单次通信的效率。MegaScale-Infer 把 Attention 和 FFN 拆开，但假设两侧执行时间是均衡的。NanoCP 优化的是锁步 EP 之下、DP 实例之间的 straggler。

**请求调度。** vLLM、SGLang、LMDeploy、Sarathi 把 Attention、MoE 通信和 KV 绑在一个 DP 实例。Llumnix 加了跨实例迁移。DistServe、Splitwise、Mooncake 拆的是 prefill 和 decode。SBS 的 decode 调度同时看 batch 和 KV，但一条请求仍然只去一个实例。这些系统都保留「一个请求对一个实例」的绑定。

## 9. Conclusion

论文的结论可以收成三件同时成立的事：batch 接近 LeastBatch，KV 明显好于 LeastBatch，跨实例通信只发生在大约 1% 的请求上。在 50 ms TPOT、99% 达成的口径下，DeepSeek-V3 的 Issue 1% 和 Issue 5% 上，可持续请求率分别是最强 vLLM 基线的 3.27 倍和 2.11 倍。

如果把论文算法和 `decentral-aug` 对齐着读，实现完整覆盖的是：按长度桶选择 CP 度、节点内 LeastBatch 选 master、按空闲块选 KV 成员、water-fill 切 token、Q 与 Res/LSE 分离路由、按 \((M, N)\) 选预捕获的图、DLSlime 直写。论文 Algorithm 1 开头那段「每个 decode step 在已有 KV 集合内重绑活跃请求的 MoE」在这条分支的 decode 调度里没有对应实现；batch 均衡发生在准入，而不是每步重绑。

## 代码索引

| 论文概念 | 路径 |
| --- | --- |
| 跨 engine 选节点 | `nanodeploy/router/request_router.py` |
| 准入影子与 bucket / water-fill | `nanodeploy/router/admission_planner.py` |
| 节点内放置 | `csrc/nanodeploy/scheduler/sp_state_manager.cpp` |
| 每步调度与 Q/Res 矩阵 | `csrc/nanodeploy/scheduler/scheduler.cpp` |
| MLA / GQA 的 Q 路由与 LSE 合并 | `nanodeploy/layers/attention.py` |
| \((M, N)\) 图选择与元数据注入 | `nanodeploy/worker/sp_graph_policy.py` |
| DLSlime / NCCL 适配 | `nanodeploy/worker/sp_backend.py` |
| 长度桶离线搜索 | `ae_scripts/algorithm/search_sp_buckets.py` |
| Q offsets 设计说明 | `docs/DeepSeek_SP_HaoBasic_Offsets_Design.md` |
| 多 engine wave 同步 | `nanodeploy/engine/decode_coordinator.py` |
