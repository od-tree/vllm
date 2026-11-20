# 双卡 GQA decode 对比

在同一进程内比较 DP、请求级 head-TP 和本地 NanoCP-style CP 数据面。三者使用同一个 FA2 fork。上游 NanoEngine 的单请求复现入口见[项目说明](../README.md)。

双卡实测数据、回传修正与负载扫描见[RTX 5090 复测记录](../docs/results/5090-matched/README.md)。

### 本版测试配套的源码

以下路径相对于两个 vLLM 工作目录的父目录。FA fork 和 NanoEngine 都是独立 Git checkout。复现实验时应核对各自的提交。

| 默认路径 | 来源和版本 | 用途 |
| --- | --- | --- |
| `vllm/` | `od-tree/vllm`，`dtp_test`，`c4fe69224c1f5edcb37bb5546441a3a5af8fc593` | 共享 tile planner 与 head-TP 框架代码 |
| `vllm/local_research/vendor/flash-attention/` | `od-tree/flash-attention`，`5090`，`1b2248bcfe4d4d5b010b506e3f2a6a395e2f5df7` | 编译 FA2 扩展并提供 head-TP 测试执行器 |
| `NanoEngine/` | `nano-infra/NanoEngine`，`8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5` | 运行作者原生基线时使用 |

## 输入和输出

`workloads.py` 定义 owner 原有序列、前几条需要协作、partner 原有序列。每个 case 在 CPU 上为两个 rank 各生成一次 bf16 的 Q 和 paged K/V，再给各方法做物理分片。模型形状为 16 Q / 2 KV、head_dim 256、page 16。当前矩阵的协作请求归 rank 0 所有，两个 rank 均可持有本地请求。公开比较限于这个双卡布局。

例如 `mix_1024x1024_4x256k` 的 rank 0 有 4 条长请求和 1024 条短请求，rank 1 另有 1024 条短请求。CP 伙伴在同一次 FA 里计算长请求后缀和自己的短请求，只向 rank 0 回传长行 O/LSE。head-TP 伙伴在同一次 FA 里计算长请求的第二个 KV 头和自己的短请求。DP 在各自设备上算原批次。三者按本地长度和实际头数调用相同的 tile planner。

## 运行

在双卡 CUDA 环境下从本目录运行，使用安装了修改版 FA2 和 Triton 的解释器。默认查找同级 `vllm` checkout 和它的 `local_research/vendor/flash-attention`。布局不同时可设置 `VLLM_ROOT` 和 `FLASH_ATTN_ROOT`。以下示例假定已激活对应虚拟环境。

```bash
python test_workloads.py
python test_e2e_correctness.py
python bench_perf.py --case paired_smoke --check-only
python bench_perf.py --quick --check
python bench_perf.py --method cp-fa2-default --check --rounds 30 --profile-rounds 3
python bench_perf.py --case mix_1024x1024_4x256k --rounds 30 --profile-rounds 5
python bench_compute.py --case mix_1024x1024_4x256k --rounds 30 --profile-rounds 5
python bench_partner_load.py
python bench_partner_load.py --cp-owner-percent 50 75 90
python bench_partner_load.py --equal-tokens
python bench_partner_load.py --equal-tokens --cp-owner-percent 75 90
```

- `bench_perf.py` 默认对十个性能 case 运行 DP、head-TP、CP 三种方法，输出 JSON Lines。`--method cp-fa2-default` 单独测本地 CP 使用 FA2 内置 split 启发式的结果；保留相同的路由、归约与整图计时，并关闭自定义 CTA 列表。作者的 FA3 + DLSlime 默认实现使用 `run_upstream_gqa.sh`。初始化 OOM、缺少 CUPTI 活动或运行错误都会使比较失败。
- `bench_upstream_5090.py` 直接调用固定提交的上游 GQA forward，走 5090 的 FA2 默认 split 和 memcpy，检查 owner 最终输出以及伙伴 O/LSE 的每个返回元素。默认保留入口 NCCL 同步，`--entry-sync graph` 对齐本地整图计时边界。上游单请求和双卡矩阵的运行命令见[默认实现补测](../docs/results/5090-matched/native-default.md)。
- `--check-only` 捕获及重放实际执行图，然后按原始请求顺序将两个 owner 的输出和 CPU FP32 GQA 对拍。`--check` 在同一次测试中再做性能测量。默认性能模式省去参考计算。仅计算模式下，CP 和 head-TP 的输出分别是 token 部分结果和 head 分片，因此通过完整路径校验最终输出。
- `bench_compute.py` 使用相同输入和 tile，只测两卡本地 attention，Q/K/V 已就位。完整路径包含方法各自的路由、通信、卡内归约和输出拼接或合并。
- `bench_partner_load.py` 固定 owner 为 4×65,536 加 128×256。默认改变伙伴卡 512-token 请求的数量，`--equal-tokens` 固定伙伴卡有 131,072 个 token。每次独立建图，三次重复轮换方法顺序。`--cp-owner-percent` 指定由 owner 保留的历史上下文百分比，按整页向下取整，余下的 token 分给伙伴。
- `test_workloads.py` 在 CPU 逐 token 校验 CP 两侧的 Q/K/V、head-TP 两侧的打包 K/V 和 rank 1 的独立短行。`test_comm_route.py`、`test_tile_plan.py`、`test_merge_unit.py` 定位路由、tile 和归约错误。`test_merge_fa2_halves.py` 使用单卡 FA2 检查切分与合并。

## 计时边界

每种方法先准备 KV、页表和 tile，预热并捕获一张跨卡 CUDA Graph。根流等两卡完成后才记录结束。各方法在相同轮数下报告毫秒数。

| JSON 字段 | 含义 |
| --- | --- |
| `graph_median_ms`、`graph_samples_ms` | 完整算子图的 GPU makespan 中位数和原始样本 |
| `host_median_ms`、`host_samples_ms` | 同轮 CPU 提交、事件记录、等待图完成的墙钟 |
| `kernels` | 单独用 CUPTI 对同一张图采样，按 GPU 和原始 kernel/拷贝名称报告累计耗时中位数及 launch 数 |
| `stages` | 第二张诊断图以可在 replay 后读取的 external CUDA events 记录各 rank 各阶段中位数。各流独立计时，跨度可包含等待 |
| `rank_requests`、`rank_tokens`、`n_long`、`seeds` | 分片前两卡各自的输入规模、协作请求数与随机种子 |

PCIe memcpy 路径可分别看到 head-TP 的发送、Q/K/V 传输、attention、结果传输和拼接。peer 直写路径的交换与 attention 融在一个 kernel 中，按实际融合阶段报告。CP 分别记录 Q 路由、两侧 attention、回传准备、O 回传、LSE 回传和合并。卡内 split-KV combine 在 kernel 明细中可单独识别。两卡并行执行，阶段与 kernel 数值不能直接相加推算整图延迟。

这里的完整用时从 Q 与含当前 token 的 KV 已就位开始，到两边 owner 的 attention 输出就绪为止。head-TP 自身协议再次发送当前 K/V 的成本计入。prefill、初始 KV 放置、QKV/O 投影、MoE 和服务调度在此窗口之外。上游原生单请求 benchmark 与这套固定输入的本地比较分别报告；旧表中 workload 或 tile 策略不同的数值需重新测量。

## 双 Hopper 上的 FA3 head-TP

`head-fa3` 将跨卡请求的 KV 头各半放置。每个本地 KV 头在 FA3 输入中占一行，与它配对的 Q 头使用 PackGQA。短请求留在 owner 且保留全部头。图内计时包含 Q/K/V 打包与发送、当前 token 的 KV 写入、两卡 FA3、返回 O 与按头拼接。图外只做初始 KV 放置、页表及 split 计划构造。

用 FA fork 中含 `row_split_counts` 参数的 `_vllm_fa3_C` 构建，CUDA arch 选择 Hopper。vLLM 源码构建可将 `VLLM_FLASH_ATTN_SRC_DIR` 指向 FA fork。独立 benchmark 可用 `FA3_EXTENSION` 指向编译好的共享库，或使用环境里的 `vllm_flash_attn._vllm_fa3_C`。设置 `VLLM_ROOT` 指向含新 planner 的 vLLM 仓库。

```bash
python test_head_tp_fa3.py -v
python bench_perf.py --method head-fa3 --shape qwen3 --case paired_smoke --check-only
python bench_perf.py --method head-fa3 --shape qwen3 --case mix_1024x1024_4x256k \
  --fa3-policy sweep --check --rounds 30 --profile-rounds 3
```

- `auto` 调用同一 head-TP 数据路径上的 FA3 内置 split。它保留该 fork 在超过 992 个 FA3 输入行时关闭自动 split 的规则。此处是每卡本地 KV 头分组数。和原生 NanoCP GQA 的 batch 行数要分别统计。
- `legacy` 复用 FA2 实验的长度与波次规则，计算 KV block 时取当前 FA3 的 BlockN。Hopper BF16 paged decode 的 dim 128 用 128，dim 256 用 80。
- `minimax` 用单卡预算 `max(SM 数, 未切分工作单元数)`，求这一预算下的最短最大 KV chunk。大 batch 在这个预算下会得到原始未切分结果，可以用 `sweep` 看额外工作单元的收益。
- `sweep` 对 FA3 默认、旧规则和两卡各自的多档 CTA 预算做配对重放。记录同一输入的整图、kernel、分阶段时间以及当前扫描的最短中位数。三个自定义策略受 `--fa3-max-splits` 约束。JSON 的 `tile_stats` 含有效工作单元数、最大 chunk 和 FA3 为整个 batch 预留的 FP32 partial buffer 字节数。

### Tile 的决策目标

固定每卡的 paged decode tile 形状后，设请求 i 在该卡有 `h[i]` 个 KV 头、`b[i] = ceil(本地 KV 长度 / BlockN)` 个 block。若要求每个工作单元最多读 `t` 个 block，最少需要 `s[i] = ceil(b[i]/t)` 个 split。检查所有 `s[i]` 是否在上限内，再检查 `sum(h[i]*s[i])` 是否不超过预算。可行性随 `t` 单调，二分即可得到预算内的精确最小最大 chunk。复杂度为每卡 O(batch × log(max_blocks))。GPU 调度元数据还会删掉因向上取整产生的空 split。

最快的预算由目标机上的带宽、CTA 固定开销、卡内归约和 slower rank 决定。以整图 makespan 校准，优先记录 attention 与 combine 毫秒数后再拟合。单卡时间可用显存读取量除以实测有效带宽给下界，再用每块开销和固定 CTA 开销估算持久化调度的长尾，加上 split combine 成本。在线使用训练负载的校准表或模型选择少数预算，并在留出工作负载验证和离线扫描的差距。比较策略时固定 FA3 版本、页布局、逻辑输入和通信，额外报告估算时间和实际图时间。

例子用 132 个 SM、dim 128 的 Qwen3，在一个 rank 上处理 4 条 64K 请求的半头和 1024 条 512-token 的全头请求。旧规则给长行每头 3 个 split，最长 chunk 为 171 个 block。给未切分的 4104 个工作单元加半个 SM 波次的预算时，二分得到每头 9 个 split、最长 57 个 block，实际共 4168 个工作单元。该结果描述调度量，Hopper 实际毫秒数需用以上脚本测量。

开发机验证包括 SM90a 编译与扩展注册、双 5090 上的独立 CUDA split 元数据探针，以及使用 PyTorch GPU 数学替代 SM90 主 kernel 的双卡图正确性校验。完整 FA3 attention 的结果与性能由 Hopper 测试检查。
