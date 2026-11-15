# NanoCP 与 head-TP 性能对比：调研结论与实现方案

更新时间：2026-11-13（第二轮：补充 NanoEngine `decentral-aug` 源码细读）。  
工作区：`/Users/ly/Documents/code/vllm`。  
源码快照：GitHub API 树 `decentral-aug` @ `8abce29`（见 `local_research/.nanocp_tree.json`，仅供本地检索，非 vendor）。  
对照材料：[head_tp_design_status.md](head_tp_design_status.md)、[agent/operator_refactor.md](agent/operator_refactor.md)、[qwen35_head_tp_feasibility.md](qwen35_head_tp_feasibility.md)。

---

## 1. 背景与对比目标

### 1.1 本仓 head-TP 基线（被对比方）

| 项 | 设定 |
| --- | --- |
| 模型切片 | Qwen3.5 Full Attention：16 Q / 2 KV，`head_dim=256`，bf16，paged `block_size=16` |
| 部署 | 两卡 **DP**（`tensor_parallel_size=1`），权重完整；协作度 `d=2`（**不是** `tensor_parallel_size`） |
| 切分 | 长请求：owner 持 KV0 + Q[0:8)，partner 持 KV1 + Q[8:16)；短请求本地全头 |
| KV | paged；长行按 tile 再切 KV 以提高 SM 占用 |
| 通信 | 算子内融合：每 decode 步交换**当前 token** 的 Q/K/V；结果 **按头拼接**，**无 LSE** |
| 已测环境 | 2×A100-SXM4-40GB（NVLink）；混合超长尾相对单卡启发式 split 墙钟比约 **0.23–0.42**（见 [head_tp_design_status.md](head_tp_design_status.md) 注释表） |

### 1.2 对比对象 NanoCP 想回答的问题

两边针对的是同一类生产痛点（见本地 [head_tp_design_status.md](head_tp_design_status.md) §1）：**DPA+EP** 下 attention 耗时随 KV 增长、MoE 集体随 batch 增长，长请求制造 **attention straggler**。  
对比要回答：

1. **算子级**：在相同 batch 形状、相同 KV 长度分布下，**按头 TP** vs **按序列 CP（NanoCP 数据面）** 谁更接近单卡一半时间、墙钟比如何？
2. **系统级（可选，后阶段）**：混合 1%/5% 长请求时，动态 CP 调度相对「仅 head-TP / 仅本地 DP」的 TPOT 尾延迟。

本文件优先保证 **(1)** 可复现；**(2)** 在 NanoCP 全栈不能直跑时记为后续工作。

### 1.3 公平性原则

| 维度 | 约定 |
| --- | --- |
| 比较层级 | 第一阶段：**单层 Full Attention decode**（与 [operator_refactor.md](agent/operator_refactor.md) 微基准一致）；不把 MoE dispatch/combine 算进 attention 对比 |
| 短请求 | 两边均 **CP/协作度 = 1**，全本地 |
| 长请求 | head-TP：`d=2` 按头；NanoCP 等价：**`d=2` 按序列**（token 维切 KV） |
| 精度 | bf16；GQA ratio 8:1 与 Qwen3.5 一致 |
| 单卡 baseline | 同一输入、同一 kernel 后端家族（FA2 paged varlen），**全部 KV 与全部头在一张卡** |

---

## 2. NanoCP 名称澄清与候选选定

### 2.1 业界「NanoCP」主要指什么

在论文、博客与本地设计文档中，**NanoCP** 特指：

> **请求级动态上下文并行（Dynamic Context Parallelism, DCP）**，用于 **MoE 的 DP–EP decode**：把每个请求的 **MoE 绑定** 与 **KV 绑定** 解耦；短请求 CP=1，长请求按 KV 体量选择 CP 度，KV 与 attention 计算跨 DP 实例切分，部分 attention 输出用 **LSE 合并**。

权威出处：

- 论文：[NanoCP: Taming Load Imbalance in Data-Expert Parallel Decoding under Variable-Length Workloads](https://arxiv.org/abs/2605.21100)（文中亦称 *Request-Level Dynamic Context Parallelism*）
- 本地图示：[images/request_level_cp.svg](images/request_level_cp.svg)（NanoCP）vs [images/request_level_tp.svg](images/request_level_tp.svg)（本方案 head-TP）

官方声称的开源实现（论文脚注）：

- 仓库：[nano-infra/NanoEngine](https://github.com/nano-infra/NanoEngine)，分支 **`decentral-aug`**
- 包名：**NanoDeploy**（`import nanodeploy`），非本仓 vLLM 主线

### 2.2 易混淆名称（**不是** 本次对比的「NanoCP」）

| 名称 | 含义 | 与 NanoCP 关系 |
| --- | --- | --- |
| **nano-vllm** | 教学向极简 vLLM 复现（~500 行） | NanoCP **实现栈的依赖之一**（论文 §6.1），**不是** NanoCP 本身 |
| **vLLM uniform DCP / `--decode-context-parallel-size`** | 组内**固定** CP 度，KV 沿序列交错分片 + LSE | NanoCP 论文中的 **对照基线**（4DP8CP 等），非 DCP |
| **Helix**（[arXiv:2507.07120](https://arxiv.org/abs/2507.07120)，vLLM RFC/#34018） | 静态 **TPA×KVP** 网格，A2A + LSE | 双维切分；**不是** 请求级动态 CP |
| **USP / ring attention** | 训练或长 prefill 的序列并行 | decode 一步通信模式不同；可作 **LSE 合并** 参考，不作 NanoCP 替代品 |
| **SGLang dynamic DCP**（[PR #36191](https://github.com/sgl-project/sglang/pull/36191)） | MLA、运行时 TP/DCP 切换 | 工业实现接近，但 **Qwen3.5 GQA + 本仓 head-TP 栈未对齐** |

### 2.3 选定结论

**对比基准选定：NanoCP = `nano-infra/NanoEngine@decentral-aug` 所实现的 DCP 全栈**（调度 + Q-Route/Res-Route + FlashMLA paged attention + LSE 合并）。

若该栈无法在本地硬件跑通，则采用下文 **§5「最小可用 decode CP」**：数据面复现论文 Fig.10 四阶段（Q 路由 → 分片 paged attention → 回传 partial+LSE → LSE 合并），**不**复现 NanoDeploy 调度/MoE/Ray/AOT 全栈。

---

## 3. 开源可用性判定

### 3.1 仓库与许可

| 项 | 结论 |
| --- | --- |
| URL | https://github.com/nano-infra/NanoEngine/tree/decentral-aug |
| 许可 | **MIT**（Copyright 2025 Jinming Ma） |
| 活跃度 | 面向 **ASPLOS AE** 的 artifact 分支；公开 star/fork 很少；文档以 `ae_scripts/` 复现为主 |
| 文档入口 | 根 [README.md](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/README.md)（NanoDeploy）、[ae_scripts/README.md](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/ae_scripts/README.md) |

### 3.2 技术栈与硬件（论文 + AE 脚本）

| 能力 | NanoCP 开源栈 | 本 head-TP 对比需求 |
| --- | --- | --- |
| GPU | **H200** 集群（论文）；`basic-test`：**2 节点 × 8 H200** | **2×A100-40GB** |
| CUDA / PyTorch | AE Docker：**CUDA 12.9**，**PyTorch 2.10.0+cu129** | 本地可能是 CUDA 12/13；**未验证 cu129 与 A100 组合** |
| Attention | **FlashMLA**（DeepSeek MLA） | **FA2 paged GQA**（Qwen3.5） |
| MoE | **DeepEP** + DeepGEMM + 自定义路由（**DLSlime**，须 `BUILD_INTRA_OPS=ON`） | 微基准 **不含 MoE** |
| 通信 | NVLink + **RDMA/NVSHMEM**（DeepEP 构建依赖主机 NVSHMEM） | 仅需 **2 卡 NVLink P2P** |
| 编排 | **Ray** 多节点 | 单机 2 进程即可 |
| 模型 | **DeepSeek-V3**、**Kimi-K2** 权重 | **Qwen3.5-35B**（GQA，非 MLA） |
| paged KV | 论文 Fig.10：**paged attention**（各 KV 分片） | 一致（语义可对齐） |
| varlen decode | 动态 `(M,N)` batch + 路由表 | 需要（NanoCP 强项） |
| bf16 | 是（AE 环境） | 是 |
| GQA | 开源含 `FlashAttentionImpl`（`nanodeploy/layers/attention.py`）与 MLA 共用 SP A2A；**AE/生产默认 FlashMLA** | **必须 GQA 16/2** |
| 请求级 CP 度 | 是（DCP 核心） | 仅需 `d=2` 长请求 |

### 3.3 总判定：**不能作为与 head-TP 同机、同形状的直接对比基准**

**结论：部分能用（论文级复现 / MLA 服务），不能在不改 head-TP 主线的前提下，把 Qwen3.5 head-TP 与 NanoCP 在同一台 2×A100 上做公平、低成本的 apple-to-apple 对比。**

**主要卡点：**

1. **模型与 kernel 族不匹配**：实现绑在 FlashMLA + DeepSeek/Kimi，无文档化的 Qwen3.5 GQA paged 路径。  
2. **硬件与规模**：AE 最小多节点 H200×16；非单节点 2×A100。  
3. **依赖过重**：DeepEP、DLSlime 定制 all-to-all、Ray、修改版 vLLM（[FirwoodLin/vllm@ae-repro-clean](https://github.com/FirwoodLin/vllm/tree/ae-repro-clean)）— 与本地 `local_research/vendor/flash-attention` **head-tp-paged** 分支无交集。  
4. **对比层级不一致**：NanoCP 端到端含调度、KV 迁移、MoE 批次均衡；head-TP 当前优势证据在 **算子微基准**。  
5. **CUDA 13**：AE 钉死 **cu129**；若本地为 CUDA 13.x，需自行验证 PyTorch/FlashMLA/DeepEP 轮子，**无官方矩阵**。

**「几乎能用」的含义（仅当你具备 H200 多节点且目标是复现论文，而非 Qwen head-TP）：** 可按 AE 流程跑 NanoDeploy vs 修改 vLLM CP8，**不能**直接套 Qwen 形状。

### 3.4 若仍要尝试官方栈（与 head-TP 并行、不耦合）

以下步骤 **不修改** 本仓 vLLM 主线；建议在独立目录克隆。

```bash
# 1) 克隆（与 head-TP FA fork 分开）
git clone --branch decentral-aug --depth 1 \
  https://github.com/nano-infra/NanoEngine.git /path/to/NanoEngine

# 2) 按仓库 README 编译 DLSlime / DeepEP / FlashMLA / DeepGEMM
#    scripts/reinstall_deepgemm_deepep_aug.zsh

# 3) 安装 nanodeploy（见 pyproject / Docker）
# 4) 配置 ae_scripts/paths.env（DeepSeek-V3 权重、dataset CSV）
# 5) 两节点 16×H200：Ray 集群 + python3 ae_scripts/basic-test/run_basic_test.py
```

与 head-TP 测试形状 **对齐方式（仅概念）**：

- NanoCP 用 **CSV 轨迹**（`prompt_len`/`output_len`）；head-TP 用 **合成 batch**（`mix_*`）。对齐需把后者转成「等价总 token 数 / 长行数」，而非同一 JSONL。  
- 论文混合负载：ShareGPT 短 + GitHub Issues 长（1%/5%）；本地 eval 计划见 [qwen35_head_tp_eval_plan.md](qwen35_head_tp_eval_plan.md) §2.1（短 1024、长 8192/32768），**长度尺度与论文 512K 长上下文不同**，算子微基准应优先用用户指定的 **64k–256k** 形状。

---

## 4. NanoCP 数据面机理（实现调研摘要）

以下内容来自 [论文 HTML](https://arxiv.org/html/2605.21100) §4–5，用于 §5 最小实现。

### 4.1 切分与调度（控制面）

- 每请求选 **MoE 绑定**（单例，均衡 dispatch/combine batch）与 **KV 绑定**（多实例集合 \(\mathcal{P}_r\)，CP 度 = \(|\mathcal{P}_r|\)）。  
- 长请求 token 在 \(\mathcal{P}_r\) 上 **WaterFill**：最小化 \(\max_s (K_s + \text{Split}_r[s])\)。  
- 下发 **Q-Route**（\(W_{cp} \times M\)）与 **Res-Route**（\(W_{cp} \times N\)）稀疏掩码表。

### 4.2 Decode 一步通信（数据面，Fig.10；源码印证见 §11）

| 阶段 | 论文描述 | NanoEngine 实现（`nanodeploy/layers/attention.py`） |
| --- | --- | --- |
| 1 | MoE 卡投影 Q，**Q-Route** | `q_buffer.all_to_all_ll(...)`：`mask=q_mask` 或 `dst_row_indices`（`hao_basic`/DLSlime）；payload `[active_master_bs, H×D]` |
| 2 | 各 KV 卡 paged attention → \(O\) + **LSE** | MLA：`flash_mla.flash_mla_with_kvcache(..., return 含 lse)`；GQA：`flash_attn_with_kvcache(..., return_softmax_lse=True)` |
| 3 | **Res-Route** 回 MoE 卡 | `res_buffer` / `lse_buffer` 各一次 `all_to_all_ll(..., is_transpose=True, mask=res_lse_mask)`；LSE **保持 fp32** 再通信 |
| 4 | **LSE 合并** | `inter_rank_gqa_fwd_batch_decode_combine_kv`（`nanodeploy/kernels/attention.py` Triton） |

**单步通信次数（`use_sp_a2a=True`，典型 CP≥2 长请求）**：**3 次 masked all-to-all**（Q 非转置 + O 转置 + LSE 转置），外加本地 Triton 合并；默认生产路径为 **DLSlime `hao_basic`**（可选 NCCL 回退 `sp_backend=nccl`，见 `nanodeploy/worker/sp_backend.py`）。

**历史 KV**：各 `sp_rank` 独立 `block_table` + `context_len(sp_rank)=num_dispatched_tokens[sp_rank]`；decode 步只 `append_token` 到 master 的 `input_ids`，各 rank 本地 `num_dispatched_tokens[sp_rank]++`（`model_runner_utils.cpp::update_seqs_inner_loop`），**不按步搬整段 KV**。

与 **uniform vLLM DCP** 差异（论文 §6.1）：固定 CP8 让**每个 CP rank 跑全 CP 组 batch**，放大 attention batch、通信 ~630 µs 级；NanoCP **仅长请求** 跨实例，通信 ~60 µs 级（H200 测量）。

### 4.3 与 ring / USP / Helix 的关系

| 机制 | 切分维 | 合并 | 动态度 |
| --- | --- | --- | --- |
| Ring / USP 类 | 序列（常固定组） | LSE 或等价 | 通常 **组级静态** |
| vLLM uniform DCP | 序列（交错 token） | LSE | **组级静态** |
| Helix KVP×TPA | 序列 × 头 | A2A + LSE | **静态网格** |
| **NanoCP DCP** | 序列（按请求） | LSE | **请求级动态** |
| **head-TP** | **头（KV 头组）** | **拼接** | 请求级 `d≤H` |

### 4.4 与 head-TP 的本质差别（Qwen3.5，`d=2`）

| | head-TP | NanoCP 式 decode CP |
| --- | --- | --- |
| 切什么 | KV **头**（各卡见到 **全序列** 的半组头） | KV **序列**（各卡见到 **半段** 的全部头） |
| Q 分布 | 各卡只算一半 **Q 头** | 各卡 typically **全部 Q 头**（对本地 KV 分片） |
| 每步通信 payload | 当前 token：**半组 Q + 对应 K/V** | **Q（较大）+ partial O + LSE** |
| 合并 | 无 softmax 跨卡；按头拼 | **必须 LSE** |
| 历史 KV | 结对后 **按头存放**于两卡 | **按 token 范围** 分布于两卡 |
| GQA `H=2,d=2` | 自然用满 KV 头维度 | 仍有意义：每卡 KV token 数约 **½**，attention FLOPs/带宽约 **½**；但 **Q 不缩小**，且 **LSE 合并开销** |

**重要约束**（[qwen35_head_tp_feasibility.md](qwen35_head_tp_feasibility.md) §1）：**2 卡不能同时做「头切 + 序列切」**（Q 头与上下文不匹配会错）。对比必须是 **两种独立模式**，不是融合算子。

### 4.5 GQA、`d=2` 时的带宽与 SM 特征（对比直觉）

- **head-TP**：读 KV 字节 \(\propto L \cdot (H/d) \cdot D\)；单序列常仅 **1 个 KV 头 CTA**，需 **KV tile split** 才能喂满 SM（见 [head_tp_design_status.md](head_tp_design_status.md) §5）。  
- **decode CP**：读 KV \(\propto (L/d) \cdot H \cdot D\)，字节上与 head-TP 同阶减半；但 **16 个 Q 头** 对分片做 attention，算子 **batch 有效头数更高**，可能更易占满 SM，代价是 **LSE 状态与 Q 路由**。  
- **合并**：head-TP 每步只拼 **8 个 Q 头向量**；CP 需 **16 头 × LSE 标量** 的数值稳定合并。

### 4.6 KV 分片：NanoCP 开源 vs vLLM DCP（易混点）

| | NanoEngine（NanoCP 实现） | vLLM DCP（本仓） |
| --- | --- | --- |
| 放置语义 | **连续 token 段** + 按 rank **压紧** 的本地页表（`num_dispatched_tokens[rank]` 为本地 KV 长度） | **交错**：token \(i\) → rank \(i \bmod d\)（`vllm/v1/worker/gpu/cp_utils.py`） |
| 动态 CP 度 | 入队时 `SPStateManager::can_allocate` 决定 `active_ranks`；可 **WaterFill** 或 **bucket 策略** | 进程级固定 `decode_context_parallel_size`；**无请求级开关** |
| block 粒度 | 调度 `kvcache_block_size`（`Sequence::block_size` 运行期同步） | `block_size` + `cp_interleave` |

算子公平对比应优先对齐 **NanoCP 连续分片 + 压紧 paged KV**；vLLM 固定 DCP 仅作 **§14 第二对照**，不当作 NanoCP 替身。

---

## 5. 在 Qwen3.5 上复现 NanoCP 思路（本调研的主目标）

官方 NanoEngine 绑的是 **MLA + FlashMLA + H200**。本仓要复现的不是那一套二进制，而是 **同一套请求级 decode CP 语义**，落到 **Qwen3.5 Full Attention（16Q/2KV，GQA，paged）** 上，再与 head-TP 对拍。

### 5.1 思路层（与 GPU 无关，必须对齐 NanoCP）

| 层 | NanoCP 语义 | Qwen3.5 复现 |
| --- | --- | --- |
| 请求级并行度 | 短请求 CP=1；长请求 CP≥2 | 同；微基准固定长请求 `cp=2` |
| KV 归属 | 按 **序列连续段** 切开，各 rank **压紧** 本地 paged 表 | 同 §13.2；**不要**用 vLLM 交错 DCP 当替身 |
| Decode 一步 | master 产 Q → 路由到持 KV 的卡 → 本地 attention → 回传 O+LSE → **LSE 合并** | 同 §13.3；2 卡可用 P2P，语义对齐官方 masked A2A |
| 合并 | online-softmax 的 log-sum-exp | `merge_attn_states` / NanoCP Triton `inter_rank_gqa_*` |
| 与 head-TP 的差别 | 切序列、要 LSE | 切 KV 头、拼接无 LSE（对照方，不是 CP 复现本身） |

MoE / DeepEP / Ray / 动态 WaterFill **不属于算子级复现**；那是系统层。算子对比只要：同一 batch 形状下，「连续段 CP + LSE」相对「单卡全长」和相对「head-TP」。

### 5.2 GPU 差异主要在算子实现（你的理解正确）

跨卡复现时，下面几块 **保持同一套接口与数值约定**，只换本地 kernel：

```text
layout_cp / 路由表 / P2P·NCCL 载荷形状 / LSE 合并
        │
        ▼
  attention_backend(GPU)   ←── 唯一强依赖架构的部分
        │
        ▼
  O_i, LSE_i  (bf16 / fp32)
```

| GPU / 环境 | 建议的本地 attention | 说明 |
| --- | --- | --- |
| **NVIDIA A100**（当前 head-TP 测试机） | **FA2 paged varlen**（本仓 `vendor/flash-attention` / `flash_attn_varlen_func` + `block_table`） | 已与 head-TP 同后端；须能取出 **softmax LSE** |
| **NVIDIA H100 / 更新 arch** | FA2 仍可用；若已接入 FA3/FA4 paged 且支持 LSE，可换同一接口 | 换后端后应用 §13.4 重新对正确性，再比墙钟 |
| **官方 NanoCP 环境（H200）** | FlashMLA | **仅 MLA**；Qwen3.5 GQA **不能**用这条当复现算子 |
| **AMD ROCm** | `rocm_aiter_fa` / 已有 `merge_attn_states` 路径 | 控制面仍可复用；只换 FA 实现 |

**不要**为了「对齐 GPU」去换 KV 切分语义（连续段 → 交错）或合并公式；那会变成另一种 CP，不再是 NanoCP 思路复现。

接口约定（所有 GPU 共用）：

```text
attn(q, k_cache, v_cache, block_table, seqused_k) -> (out, lse)
  q:          [B_local, H_q, D]     # decode 一步，B_local 为本 rank 要算的长行（+可选短行）
  k/v_cache:  paged，本 rank 连续段
  out:        [B_local, H_q, D] bf16
  lse:        [B_local, H_q]    fp32
```

A100 上若现有 FA2 入口不直接返回 LSE，用 wrapper（二次 kernel / 本仓 DCP 已有取 LSE 路径）补齐，**不要**为此改成 MLA。

### 5.3 实现落点（规格见 §13；算子细则见独立文档）

**算子实现计划（合并算子重点）**：[nanocp_operator_impl_plan.md](nanocp_operator_impl_plan.md)。

在 `local_research/bench_decode_cp/`：

`layout_cp.py` → `attn_local.py`（**可插拔 backend**）→ `comm_cp.py` → **`merge_cp.py`** → `reference.py`

- 默认 backend = FA2 paged（A100 / 与 head-TP 对齐）；merge 默认薄封装 `merge_attn_states`。
- 与 **head-tp-paged** 共用 §6 矩阵；**不**改 vLLM 主线、**不**依赖 FlashMLA/DeepEP。
- 换 GPU 时：只加/换 `attn_local` backend + 重编对应扩展；布局、通信与 merge 接口不变。

### 5.4 和「直接跑 NanoEngine」的关系

| | 官方 NanoEngine | Qwen3.5 思路复现（本方案） |
| --- | --- | --- |
| 模型 | DeepSeek/Kimi MLA | Qwen3.5 GQA 16/2/256 |
| Attention 算子 | FlashMLA | FA2（或其它 GQA paged） |
| 调度 / MoE | Ray + DeepEP 全栈 | 微基准固定 cp=2，不算 MoE |
| 可比性 | 论文系统数字 | **与 head-TP 同机、同形状** 的算子数字 |

结论：GPU 不同 → **换 paged GQA attention 实现**；NanoCP 要复现的是 **连续段 KV + Q 路由 + LSE 合并**，不是 FlashMLA 本身。

---

## 11. NanoEngine 源码实现细节（分支 `decentral-aug`）

路径前缀：[nano-infra/NanoEngine@decentral-aug](https://github.com/nano-infra/NanoEngine/tree/decentral-aug)。

### 11.1 请求级 CP 度：谁决定、何时变

| 机制 | 位置 | 要点 |
| --- | --- | --- |
| 入队选 SP 度 | `csrc/nanodeploy/scheduler/sp_state_manager.cpp` `can_allocate()` | `target_num_ranks` 由 `segment_size` 与 `num_tokens` 推导；支持 `fixed_sp_size`、`dynamic_sp_bucket_policy`（`sp:low-high;...`）、**WaterFill**（`enable_non_uniform_split`） |
| Master | `SPMasterSelector` | `master_sp_idx`：decode 仅 master 产 `input_ids`（`prepare_decode_cpp`） |
| 调度周期 | `nanodeploy/engine/scheduler.py` + `scheduler.cpp` | placement 在 admission 写入 `num_dispatched_tokens[]`；非每 token 重算 |
| 可观测性 | `scheduler.cpp` | `sp_size_hist_per_dp`、`sp_q_matrix` / `sp_res_matrix` |

### 11.2 KV 分片与 block_table

- `num_dispatched_tokens[sp_idx]`：该 rank 本地 KV token 数（`sequence.cpp` `context_len`）。
- Uniform：按 rank 顺序切 `segment_size` 倍数（`sp_state_manager.cpp`）；WaterFill：按空闲 block 拉平（§421–504）。
- 每 rank 独立 `sp_block_table[sp_idx]`；decode 打包 `block_table(ACTIVE, sp_rank)`（`model_runner_utils.cpp`）。

### 11.3 Decode 一步 API 链

1. **`prepare_decode_cpp`**：`use_sp_a2a`；`q_mask`/`res_lse_mask`；`q_slice_*`；`q_offsets`；`context_lens_for_attn`。
2. **`FlashAttentionImpl` / `FlashMLAImpl`**：`flash_attn_with_kvcache` 或 `flash_mla_with_kvcache`，`return_softmax_lse`。
3. **`sp_buffer.all_to_all_ll`** ×3（Q；O `is_transpose=True`；LSE 同），见 `nanodeploy/layers/attention.py`。
4. **`inter_rank_gqa_fwd_batch_decode_combine_kv`**（`nanodeploy/kernels/attention.py`）。

通信后端：`nanodeploy/worker/sp_backend.py`（`hao_basic`/DLSlime 或 `nccl`）；缓冲区 `nanodeploy/worker/sp_context.py`。

### 11.4 MLA 绑定 vs 可复用部分

- **不可用（算子基线）**：FlashMLA、`deepep_moe`、Ray 执行器、DLSlime（可换 NCCL）。
- **可复用语义**：`prepare_decode_cpp` 路由、`inter_rank_gqa` 合并、GQA 版 `FlashAttentionImpl` SP 外壳。

### 11.5 官方 profiling

`ae_scripts/microbench/attention/benchmark_flashmla.py`；`ae_scripts/fig19/benchmark_q_res_lse_latency.py`；`fig6/.../run_flashmla_dcp_bench.py`。

---

## 12. 可迁移性评估表

| 组件 | 判定 | 说明 |
| --- | --- | --- |
| CP 度 bucket / WaterFill | **仅作参考** | v0 固定 CP=2、连续对半 |
| 路由表（mask/slice/offsets） | **可改编** | §13；2 卡可 P2P 代替 DLSlime |
| 连续分片 + 压紧 paged KV | **可改编** | 与 head-TP 同 `block_size=16` |
| LSE 合并（Triton / merge_attn_states） | **可直接用** | MIT Triton 或 `vllm/v1/attention/ops/merge_attn_states.py` |
| Q 当前步 / KV append | **可改编** | 对标 `prepare_decode_cpp` + `slot_mapping` |
| 历史 KV 迁移 | **仅作参考** | decode 步不搬；head-TP 按头迁不纳入 CP |
| FA2 paged（替代 FlashMLA） | **可改编** | 需 `return_softmax_lse` 或 wrapper |
| FlashMLA / MLA metadata | **不可用** | `num_kv_heads==1` |
| MoE / DeepEP / Ray / CUDA Graph | **不可用** | 算子对比剔除 |
| DLSlime `hao_basic` | **不可用**（微基准） | 用 NCCL 或 P2P |
| Profiling（event+墙钟） | **可直接用** | 与 head-TP 一致 |

---

## 13. NanoCP 对齐基线 v0 规格

**目标**：在 **Qwen3.5 GQA** 上复现 NanoEngine **decode SP 数据面**（§5）；默认 2×A100 + FA2 paged（无 FlashMLA/DeepEP）。其它 GPU 只换 §5.2 的 `attention_backend`。

### 13.1 角色

- `rank0` = master（MoE 侧 Q 投影）；`rank1` = partner。仅**长行** `SP=2`；短行仅 rank0 本地。

### 13.2 长行布局

- 全长 \(L\)：`L0=\lfloor L/2\rfloor`，`L1=L-L0`；rank0 压紧 KV 长 `L0`（全局前缀），rank1 长 `L1`（全局 `[L0,L)`）。
- 与 head-TP 共用同一 batch 表；`row_id` 标注 `head_tp` / `decode_cp`；短行两边 rank0 全长。

### 13.3 单 decode 步

| 步 | 行为 | 通信 |
| --- | --- | --- |
| 1 | master 算 Q `[bs,16,256]` | — |
| 2 | 长行 Q → partner | 1× P2P / 子集 A2A |
| 3 | 各 rank FA2 paged → `O_i,LSE_i` | — |
| 4 | partner → master 合并 LSE | 1× P2P |
| 短行 | rank0 全长 FA2 | 0 |

官方为 3× masked all-to-all；v0 用 2× P2P，payload 量级对齐。

### 13.4 正确性

- Ref：单卡全长 FA2。
- \(\max|O-O_{\text{ref}}|\le 10^{-2}\)（bf16）；\(\max|\text{LSE}-\text{LSE}_{\text{ref}}|\le 10^{-3}\)（fp32，若可导出）。

### 13.5 故意不对齐（仍算公平算子对比）

| 不对齐 | 原因 |
| --- | --- |
| FlashMLA / `num_splits` | 改用 FA2 paged |
| 动态 CP、MoE 均衡 | 固定 `cp=2` |
| DLSlime 三连 A2A | 2×P2P，字节量同级 |
| vLLM 交错 KV | NanoCP 为连续段 |
| CUDA Graph / Ray / H200 | eager、2×A100 |

仍公平：同一 \(L,H,D\)、block=16、decode 一步语义、同一 LSE 公式。

### 13.6 计时

与 `operator_refactor.md`：`warmup=3`，`repeat=8`，event + 墙钟。

---

## 14. 本仓 vLLM 固定 DCP=2（第二对照）

- **布局**：交错 CP（`cp_utils`），**非** NanoCP 连续段 → 不能作 v0 替身。
- **合并**：`dcp_a2a_lse_reduce`（`vllm/v1/attention/ops/dcp.py`）与 NanoCP 同类。
- **用法**：需 `tp=2,dcp=2` 单独 harness，命名 `vllm_dcp2_interleave`；与 `tensor_parallel_size=1` 的 head-TP **不同进程**。
- **vs 动态 NanoCP**：vLLM 全局固定 CP、全 batch 参与；NanoCP 按请求选度且仅长请求 A2A。

---

## 6. 公平对比实验矩阵

### 6.1 算子微基准用例（与 head-TP 已有/规划一致）

命名约定：`mix_{短条数}x{短长}x{长条数}x{长长}`；`n{k}×{L}` 为 `k` 条等长序列长度 `L`。  
seq 范围 **64–262144**（262144 = 256k）。batch 为 **混合长短** 时，短序列长度默认 **512**（与 [head_tp_design_status.md](head_tp_design_status.md) 注释表一致）。

| 用例 ID | 组成（行数 × seq len） | 目的 |
| --- | --- | --- |
| `mix_32x512_2x256k` | 32×512 + 2×262144 | 极长尾 + 大量短；启发式 split 敏感 |
| `mix_64x1024_4x128k` | 64×1024 + 4×131072 | 中等 batch + 多条 128k |
| `n4x262144` | 4×262144 | 纯长、少 batch |
| `n16x262144` | 16×262144 | 长 batch 占满 SM 趋势 |
| `n32x262144` | 32×262144 | 强带宽/并行度 |
| `mix_512x512_16x256k` | 512×512 + 16×262144 | 高短批 + 少数超长 |
| `mix_1024x1024_4x256k` | 1024×1024 + 4×262144 | 短批极大、长条少 |
| `128x32768` | 128×32768 | 纯长中等 seq（operator_refactor 表） |
| `256x32768` | 256×32768 | 高并行度中等 seq |

**协作/CP 规则**：

- 长行（上表中 **>512** 或显式 128k/256k 行）：**head-TP `d=2`** 或 **decode CP `d=2`**。  
- 短行：**本地**（两边 CP/协作度 = 1）。

### 6.2 每用例必跑配置

| 配置 ID | head-TP | decode CP（最小实现） |
| --- | --- | --- |
| `base` | 单卡，全头全 KV | 单卡，全序列全头 |
| `para_nocomm` | 两卡各算一半 **无通信**（split） | 两卡各算一半序列 **无通信**（仅用于算力上限） |
| `para_comm` | 两卡 **算子内通信**（std） | 两卡 **Q + LSE 合并通信** |
| `wall` | 墙钟比 = `wall(para_comm) / wall(base)` | 同上 |

报告列与 [operator_refactor.md](agent/operator_refactor.md) §2 对齐：`base0, split0/1, std0/1, wall_ratio, half_std`。

### 6.3 单卡 baseline 定义（两边统一）

- **同一输入张量 / 同一 paged KV 填充**（prefill 已由 host 写入 block pool）。  
- **单 decode 步**（`seqlen` 为当前上下文长度，`query` 长度 1）。  
- **不计** QKV 投影、RoPE、O 投影。  
- **warmup=3，repeat=8，取中位数**（与现有 head-TP 测量一致）。

### 6.4 端到端矩阵（第二阶段，可选）

复用 [qwen35_head_tp_eval_plan.md](qwen35_head_tp_eval_plan.md)：`head_tp_matrix` 配置 A/C，长比例 1%/5%，**不等价**于 NanoCP 论文 512K Issue 轨迹；若要做系统级 NanoCP 对照，需 H200 + NanoEngine **或** 在 vLLM 上实现「长请求才 CP」的调度原型（超出本文件最小 CP 范围）。

---

## 7. 风险与开放问题

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| 官方 NanoCP 无法在 A100+Qwen 运行 | §3.3 | 以 §5 最小 CP + 论文数字作 **定性** 对照 |
| LSE 在 FA2 paged 路径缺失 | 实现复杂度 | 先 Triton merge + 分片 attention 拆步；或与 vLLM DCP op 对拍 |
| 连续切分 vs 交错切分 | 与 vLLM DCP 行为差 | 文档标注；长 seq 性能趋势仍可比 |
| 墙钟含 launch 间隙 | 短 kernel 上 head-TP 已观测 ~0.11–0.15 ms | 同时报 event 与墙钟 |
| MoE 气泡未纳入微基准 | NanoCP 优势部分来自 **batch 均衡** | 端到端单列阶段；微基准仅 **attention 上限** |
| CUDA 13 / PyTorch 版本漂移 | FlashMLA/DeepEP 无保证 | 最小 CP **仅用 FA2 + NCCL P2P** |
| 开源 NanoEngine 维护 | 0 star、AE 专用 | 最小 CP **自包含** 在 `local_research` |

**开放问题：**

1. FA2 paged 是否稳定返回 **per-head LSE**（否则 v0 用 wrapper + `merge_attn_states` 对拍）。  
2. `mix_*` 长行 CP 时，短行是否占满 partner SM（NanoCP 用 `master_sp` batching + 稀疏 mask 缓解）。  
3. head-TP **按头迁 KV** vs CP **连续段初次放置**—端到端待 §6.4。  
4. 第三条曲线：**`vllm_dcp2_interleave`**（§14），与 **`nanocp_v0_contiguous`** 分列。

---

## 8. 参考链接

### NanoCP 与官方实现

- 论文：https://arxiv.org/abs/2605.21100  
- 论文 HTML：https://arxiv.org/html/2605.21100  
- 代码：https://github.com/nano-infra/NanoEngine/tree/decentral-aug  
- 关键源码（raw）：[`sp_state_manager.cpp`](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/csrc/nanodeploy/scheduler/sp_state_manager.cpp)、[`model_runner_utils.cpp`](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/csrc/nanodeploy/worker/model_runner_utils.cpp)、[`attention.py`](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/nanodeploy/layers/attention.py)、[`kernels/attention.py`](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/nanodeploy/kernels/attention.py)、[`sp_backend.py`](https://github.com/nano-infra/NanoEngine/blob/decentral-aug/nanodeploy/worker/sp_backend.py)  
- AE 脚本说明：https://github.com/nano-infra/NanoEngine/blob/decentral-aug/ae_scripts/README.md  
- 笔记摘要：https://lukebest.github.io/papers/nanocp-dynamic-context-parallelism-moe  

### 对照系统与 CP 背景

- vLLM Context Parallel：https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/  
- vLLM DCP for MLA（PR）：https://github.com/vllm-project/vllm/pull/23734  
- Helix 论文：https://arxiv.org/abs/2507.07120  
- Helix vLLM RFC：https://github.com/vllm-project/vllm/issues/34018  
- Flash-decoding / LSE（PyTorch Blog，NanoCP 引用）：https://pytorch.org/blog/flash-decoding/  

### 本仓

- [head_tp_design_status.md](head_tp_design_status.md)  
- [agent/operator_refactor.md](agent/operator_refactor.md)  
- [qwen35_head_tp_feasibility.md](qwen35_head_tp_feasibility.md)  
- [qwen35_head_tp_eval_plan.md](qwen35_head_tp_eval_plan.md)  
- [agent/README.md](agent/README.md)  

---

## 9. 执行清单（建议顺序）

1. **确认硬件**：2×A100 NVLink P2P 可用（head-TP 已用）。  
2. **head-TP**：在 `head-tp-paged` 分支跑 §6.1 全矩阵，归档 `base/split/std/wall`。  
3. **decode CP**：按 **§13 v0** 在 `local_research/bench_decode_cp/` 实现并跑同一矩阵。  
4. **出表**：按用例对比 `wall_ratio`、`|wall_cp - wall_htp|/wall_htp`；附单卡正确性日志。  
5. **（可选）** 有 H200 多节点时再跑 NanoEngine `basic-test`，仅作 **趋势校验**，不与 A100 数字直接比。  

---

## 10. 一句话结论（给决策者）

**在 Qwen3.5 上复现 NanoCP，指的是请求级连续段 decode CP + LSE，不是跑官方 FlashMLA 栈；GPU 差异主要换 paged GQA attention 后端（A100 默认 FA2），布局/路由/合并跨卡共用。v0 规格见 §5 / §13；vLLM 交错 DCP 仅作第二对照。**
