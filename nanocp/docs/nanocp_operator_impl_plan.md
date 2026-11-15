# NanoCP 思路复现：算子实现计划（合并算子重点）

更新：2026-11-14。对照：[nanocp_comparison_plan.md](nanocp_comparison_plan.md) §5 / §13、本仓 `vllm/v1/attention/ops/merge_attn_states.py`、FA2 `mha_varlen_fwd` 返回值。

目标：在 **Qwen3.5 GQA（16Q / 2KV / dim256 / bf16 / paged block=16）** 上先做完 **算子级** decode CP，再与 head-TP 同机对比。本文件只谈算子与 harness；不做调度、MoE、动态 CP 度、历史 KV 搬运。

---

## 1. 算子范围总览

```text
                    ┌─────────────────────────────────────────┐
  master Q          │  通信（P2P，第一版不进 FA2）              │
  [B_long, 16, 256]─┼─► partner                                │
                    └─────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   FA2 paged 本段 KV              FA2 paged 本段 KV
   → O0, LSE0                     → O1, LSE1
         │                              │
         │◄──── P2P: O1 + LSE1 ─────────┘
         ▼
   ┌──────────────────┐
   │  merge 算子       │  ← 本文重点
   │  (LSE 加权合并)   │
   └────────┬─────────┘
            ▼
         O_final [B_long, 16, 256]
```

短行：只在 master 上跑全长 FA2，**不进 merge**。

| 模块 | 职责 | 第一版落点 |
| --- | --- | --- |
| `layout_cp` | 长行连续对半、压紧 `block_table` / `seqused_k` | Python |
| `attn_local` | paged decode → `(O, LSE)` | 现有 FA2 varlen |
| `comm_cp` | Q 下发、O+LSE 回传 | `torch` P2P / NCCL |
| **`merge_cp`** | 两段 partial → 最终 O（可选最终 LSE） | **复用 / 薄封装** `merge_attn_states` |
| `harness` | 正确性 + 与 head-TP 同形状计时 | `local_research/bench_decode_cp/` |

---

## 2. 合并算子：数学与约定

### 2.1 公式（与 NanoCP / vLLM 一致）

对每个 query token、每个 Q head，两段 partial `(O_a, ℓ_a)`、`(O_b, ℓ_b)`（ℓ 为自然对数 LSE）：

```text
m   = max(ℓ_a, ℓ_b)
α_a = exp(ℓ_a - m)
α_b = exp(ℓ_b - m)
s   = α_a + α_b
O   = (α_a / s) · O_a + (α_b / s) · O_b
ℓ   = m + log(s)          # 若需要写出合并后 LSE
```

数值注意（与本仓 Triton 实现一致，见 `triton_merge_attn_states.py`）：

1. **先算 scale，再乘 O**；禁止 `O * exp(ℓ - m)` 与另一路分别乘完再除（易炸）。
2. FA2 在「空序列 / 全 mask」时 LSE 可能是 **`+inf`**，FA3 倾向 **`-inf`**。合并前把 **`+inf` → `-inf`**，与 prefix/suffix merge 对齐。
3. 若 `m == -inf`（两路都空）：`O = 0`，`ℓ = -inf`，避免 `0/0 → NaN`。
4. `O_a/O_b` 以 **fp32 累加**，最后再 cast 回 bf16。

两段在 CP 里的语义：

| 符号 | 含义（长行连续对半） |
| --- | --- |
| `a` / prefix | rank0 持有的全局前缀 `[0, L0)` |
| `b` / suffix | rank1 持有的全局后缀 `[L0, L)` |

顺序可对调，但 **scale 公式对称**；实现里固定「a=本卡前缀、b=伙伴后缀」，避免和通信 buffer 搞反。

### 2.2 为什么这是「算子重点」

- 本地 FA2：**已有**；CP 只改 `block_table` / `seqused_k`。
- 通信：第一版 **Python P2P** 即可，不阻塞对比。
- **Merge** 决定正确性门是否过、是否引入额外 kernel 时间；且 head-TP **没有**这一步（按头拼接），两边差异会体现在 merge + 多一倍 Q 头计算上。

---

## 3. 合并算子设计

### 3.1 推荐方案：薄封装本仓 `merge_attn_states`

**不要第一版重写 CUDA。** 直接调用：

- `vllm.v1.attention.ops.merge_attn_states.merge_attn_states`
- 底层优先 CUDA，否则 Triton（`triton_merge_attn_states.py`）

该 API 已实现 §2.1，并处理 `+inf` / 双空 / fp32 scale。

#### 张量布局契约（必须一次定死）

Decode 一步、长行 batch `B`、头数 `H=16`、`D=256`：

| 张量 | 形状 | dtype | 备注 |
| --- | --- | --- | --- |
| `prefix_output` / `suffix_output` / `output` | `[B, H, D]` | bf16 | token 维 = 长行条数（每行 decode 1 token） |
| `prefix_lse` / `suffix_lse` / `output_lse` | `[H, B]` | fp32 | **头在前**；与 merge API 一致 |
| `prefill_tokens_with_context` | `None` 或 `B` | — | decode CP 全部 token 都要 merge，传 `None` |

**禁止**把 FA2 原始 LSE 布局直接塞进 merge，必须先做 §3.2 的规范化。

#### 封装建议

```python
# local_research/bench_decode_cp/merge_cp.py

def merge_cp_decode(
    out: torch.Tensor,          # [B, H, D] bf16, preallocated
    o0: torch.Tensor,           # [B, H, D] rank0 partial
    lse0: torch.Tensor,         # [B, H] or [H, B] — 见 normalize
    o1: torch.Tensor,
    lse1: torch.Tensor,
    out_lse: torch.Tensor | None = None,  # [H, B] optional
) -> None:
    lse0_h = normalize_lse_to_head_major(lse0)  # [H, B]
    lse1_h = normalize_lse_to_head_major(lse1)
    merge_attn_states(
        out, o0, lse0_h, o1, lse1_h,
        output_lse=out_lse,
        prefill_tokens_with_context=None,
    )
```

约束（来自现有 assert）：

- `o0.stride(1) == o1.stride(1)`（head stride 一致；两边都用 contiguous `[B,H,D]`）。
- 第一版不做 FP8 `output_scale`。

### 3.2 LSE 从 FA2 到 merge 的布局

FA2 `mha_varlen_fwd` 在 **GQA decode（`max_seqlen_q==1` → ngroups swap）** 后返回的 `softmax_lse` 布局会变（见 `flash_api.cpp` ~982–994）：

| 路径 | 内核写出习惯 | 返回给 Python 的典型形态 |
| --- | --- | --- |
| `num_splits == 1` | 交换后按 `[h_kv, B·ngroups]` 一类中间态 | reshape/transpose 后接近 `[H_q, B]` 的「canonical」 |
| `num_splits > 1` | combine 后另一套 | 同样收成 `[H_q, B]` 一类 |

**实现要求：**

1. `attn_local` 封装必须 **单元测试钉死**：对已知输入打印 `lse.shape` / `stride`，断言规范化后为 **`[16, B]` contiguous 或至少 stride 可读**。
2. `normalize_lse_to_head_major`：
   - 若已是 `[H, B]`：直接用（或 `.contiguous()`）。
   - 若是 `[B, H]`：`.transpose(0, 1).contiguous()`。
   - 若是 FA2 在 swap 后的 `[H_q, B]` 已满足：透传。
3. 与 head-TP 计时对比时，**LSE 规范化 + merge 的时间算进 CP 路径**（event 包住 merge kernel）。

### 3.3 备选方案（何时用）

| 方案 | 何时 | 说明 |
| --- | --- | --- |
| **A. `merge_attn_states`（默认）** | A100 / 与 vLLM 同机 | 零维护；已处理 FA2 `+inf` |
| **B. NanoCP Triton `inter_rank_gqa_fwd_batch_decode_combine_kv`** | 要逐行对齐官方数字 / 论文附录 | MIT；需改到我们的 `[B,H,D]`+`[H,B]`；作为正确性对照，不是默认性能路径 |
| **C. 自写 CUDA** | 证明 merge 成为热点且 A/B 不够 | 仅在 profile 后；接口仍保持 §3.1 |
| **D. `dcp_a2a_lse_reduce`** | 多 rank（>2）allgather 式 | v0 只有 2 路 pairwise merge，**不必**上全量 DCP reduce |

**第一版禁止**在 FA2 split-KV combine 里「顺便」做跨卡 merge：那是单卡 `num_splits` 的 partial，与跨卡序列分段不是同一层。

### 3.4 合并算子与通信的边界

```text
计时区间建议（para_comm）：

  t0:  clear / 准备
  t1:  P2P 发 Q（可与后续重叠的版本放到二期）
  t2:  两卡 FA2
  t3:  P2P 收 O1+LSE1
  t4:  merge_cp          ← 单独打一段 event，便于看 merge 占比
  t5:  结束
```

Payload（每长行）：

- 回传：`O1` = `16×256×2` B ≈ 8 KiB；`LSE1` = `16×4` B = 64 B。相对 FA2 读半段 256k KV，通信很小；**merge 本身是算子瓶颈的可能性低于 FA2**，但仍要正确且可测。

第一版：**先收齐再 merge**（简单）。二期再考虑 merge 与其它行的 FA2 流水（非必须）。

### 3.5 正确性测试（合并专项）

按优先级：

1. **单卡两段 fake partial**  
   手工构造 `O0,LSE0,O1,LSE1`，与 fp64 公式比；覆盖：一路 `-inf`、一路 `+inf`（模拟 FA2）、双 `-inf`、普通值。

2. **单卡真 FA2 两段**  
   同一序列切成 `[0,L0)` / `[L0,L)` 两次 FA2，`merge_cp` vs 全长 FA2；门：`max|O-O_ref| ≤ 1e-2`（bf16），可选 `|LSE-LSE_ref| ≤ 1e-3`。

3. **两卡端到端**  
   与 (2) 同一数据；额外断言短行逐位等于本地 FA2；伙伴不写短行输出。

4. **与 head-TP 共用形状**  
   `check_mix` 式 `[64,96,128,17]`：前三行 CP 长行，末行短行；不要求与 head-TP 输出相等（语义不同），只要求各自对单卡 ref 过门。

---

## 4. 本地 attention 算子需求（支撑 merge 的输入）

### 4.1 接口

```text
attn_local(q, k_cache, v_cache, block_table, seqused_k, ...) -> (out, lse)
  q:           [B, 16, 256] bf16          # 本 rank 本步要算的行（长行；短行仅 master）
  k/v_cache:   paged [pages, 16, 2, 256]
  block_table: [B, max_pages]             # 只含本段页，压紧
  seqused_k:   [B] int32                  # 本段 token 数（L0 或 L1，短行=全长）
  out:         [B, 16, 256] bf16
  lse:         规范化前任意；经 normalize → [16, B] fp32
```

### 4.2 实现要点

- 调用现有 `_vllm_fa2_C.varlen_fwd` / 与 head-TP 测试相同的包装；`max_seqlen_q=1` 走 GQA swap。
- **必须**使用返回的第二个张量 LSE；在 harness 里加 `assert lse is not None`。
- `num_splits`：与 head-TP 对比时 **base 与 CP 本地 attn 约定一致**（都启发式或都强制 1）；merge 之前本地 attn 可以是 split-KV，但 LSE 必须是 **该段上已经 combine 完的最终 LSE**，不能把 split partial 交给跨卡 merge。
- 空段（`L1=0` 极端）：该侧 LSE 填 `-inf`，O 填 0，仍走 merge（测双空分支）。

### 4.3 页布局（给 attn 的输入约定）

长行全长 `L`，`L0 = L // 2`，`L1 = L - L0`：

| rank | 逻辑区间 | `seqused_k` | `block_table` |
| --- | --- | --- | --- |
| 0 | `[0, L0)` | `L0` | 仅前缀页，行内从 0 压紧 |
| 1 | `[L0, L)` | `L1` | 仅后缀页，行内从 0 压紧 |

页内容与 head-TP **同一套随机种子生成的逻辑 KV** 切开，保证 ref 可比。

---

## 5. 通信算子（最小）

第一版 **不** 融进 FA2：

| 步骤 | 内容 | dtype / 形状 |
| --- | --- | --- |
| Q 下发 | master → partner | `[B_long, 16, 256]` bf16 |
| 回传 | partner → master | `O` 同上 + `LSE` `[16, B_long]` fp32（建议打包成一次发送） |

短行不发送。可用 `torch.distributed` 或两进程 `cudaMemcpyPeer`（与 head-TP 测试同类 peer access）。

二期可选：把 Q 发送提前到 FA2 之前与其它工作重叠；**不改变 merge 接口**。

---

## 6. 目录与里程碑

建议路径：`local_research/bench_decode_cp/`

```text
bench_decode_cp/
  README.md                 # 如何跑 check / perf
  layout_cp.py              # 切段、建 paged 表
  attn_local.py             # FA2 包装 + LSE normalize
  merge_cp.py               # §3 封装
  comm_cp.py                # P2P
  reference.py              # 单卡全长 FA2
  test_merge_unit.py        # §3.5.1
  test_merge_fa2_halves.py  # §3.5.2
  test_e2e_correctness.py   # §3.5.3
  bench_perf.py             # 与 head-TP 同矩阵墙钟
```

| 里程碑 | 完成标准 |
| --- | --- |
| **M1 合并单测** | `test_merge_unit` 全过；含 `+inf`/`-inf`/双空 |
| **M2 单卡两段=全长** | `test_merge_fa2_halves` 在 64/96/128/1k/8k 上过 abs 门 |
| **M3 两卡正确性** | e2e 过门；短行精确 |
| **M4 性能表** | 与 head-TP 同一 §6 用例打出 `base / cp_comm / wall_ratio`；报表含 **merge event 占比** |

---

## 7. 和非算子工作的切割

| 做（算子对比需要） | 不做（对比后再说） |
| --- | --- |
| 连续段 layout + FA2 + LSE | WaterFill / 动态 cp 度 |
| `merge_cp` 封装 | 改 vLLM 主线 forward |
| P2P Q / O+LSE | DeepEP / Ray / FlashMLA |
| 与 head-TP 同形状计时 | 端到端 TPOT、ShareGPT 混合 |

---

## 8. 风险（合并相关）

1. **LSE 布局弄错**：silent 错合并；用 M1+M2 钉死 normalize。  
2. **FA2 `+inf` 未转 `-inf`**：scale NaN；必须走现有 `merge_attn_states` 或同等逻辑。  
3. **把 split-KV 的 partial LSE 拿去跨卡 merge**：数值错；本地 attn 出口必须是段内最终 LSE。  
4. **与 head-TP 比墙钟时 base 不一致**：两边 `base` 都用同一启发式约定（见既有复测）。  
5. **merge 被误优化进「不算时间」**：CP 路径必须包含 merge。

---

## 9. 一句话

算子对比的核心新件是 **LSE 合并**：默认 **薄封装本仓 `merge_attn_states`**，输入统一成 `O:[B,16,256]` + `LSE:[16,B]`；本地 FA2 只负责产出两段 partial，通信第一版用 P2P，做完 M1–M4 再与 head-TP 对表。
