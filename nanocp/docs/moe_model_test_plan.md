# MoE decode 模型选型、开发验证与正式测试计划

双 RTX 5090 用于开发、正确性和服务跑通。当前已在开发机跑通 Qwen3-30B-A3B-Instruct-2507-FP8 的 DPA+EP。下文的显存预算和缩短输入仅用于开发验证。正式平台确定后，按该平台的显存、互连和正式请求分布重选测试长度，独立测量未修改 vLLM、动态 head-TP 与 NanoCP 对照的性能。核心指标是 DP-attention 慢卡缩短了多少 decode step 时间，以及生成阶段的用户延迟是否改善。[开发机运行记录](results/5090-matched/qwen3-2507-dpa-ep.md)

## 模型与容量

| 候选 checkpoint | 全上下文 GQA 层 / 总层 | Q / KV / head dim | 配置上下文上限 | 用途 |
| --- | ---: | --- | ---: | --- |
| ibm-granite/granite-3.1-3b-a800m-instruct | 32 / 32 | 24 / 8 / 64 | 131072 | 首选。BF16 权重、40 个专家选 8、每个请求可均分为两组各 4 个 KV 头。先用它完成有限显存下的真实服务实验。[IBM 配置](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct/blob/main/config.json) |
| Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | 48 / 48 | 32 / 4 / 128 | 262144 | 第二候选。3.3B 激活参数，128 个专家选 8，权重 FP8、实验 KV 统一 BF16。先验证当前 vLLM + sm120 的 FP8 MoE、EP 和显存余量，再定测试规模。[模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) · [配置](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/blob/61c2ee12e2e018cd6cdcaa3713e6ffd19e7b8086/config.json) |
| Qwen/Qwen3.5-35B-A3B | 10 / 40 | 16 / 2 / 256（全注意力层） | 262144 | 留作已有算子回归。30 层线性 attention 也消耗 decode 时间，不能把全注意力单层的收益直接外推到整模。[配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json) |
| openai/gpt-oss-20b | 12 / 24 | 64 / 8 / 64 | 131072 | 12 层使用 128-token 滑窗。优先级低于上述两个全 GQA 模型。[配置](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) |

未分片时，一个请求每个历史 token 的全层 BF16 KV 字节数等于 2 × 层数 × KV heads × head dim × 2。Granite 为 64 KiB/token，单条 32K、64K、128K 分别为 2、4、8 GiB；Qwen3 为 96 KiB/token，对应 3、6、12 GiB。K 按 1024 计。权重、CUDA Graph、MoE workspace、其他活跃行和分配粒度还需额外显存。DP 下原始请求的 KV 驻留在 owner，不能用两卡显存之和估单条请求容量。若 owner 仍保留整份 KV，还要记伙伴的副本。实验从 8K、32K 开始，逐档检查 vLLM 报告的每 rank KV 容量，预留输出 token 和背景请求空间。

## 测试计划

1. **启动与原生 profile。** 从研究分支的上游基点 `b8c04dd3b4` 建独立、干净的 vLLM baseline；记录源码、权重 revision、驱动及实际加载的 wheel、attention/MoE 后端。两卡设 TP=1、DP=2、启用 EP，检查两 rank 实际持有的专家、dispatch 和 combine，先完成短请求和长短并发的生成。Granite 先跑 BF16；Qwen3 做相同的小规模预检。保留生产式 CUDA Graph，分别记录权重、KV blocks、图与工作区占用。[vLLM EP 部署](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
2. **测 GQA 是否占慢卡关键路径。** 每个模型先测长请求 8K、32K、64K，各生成 128 token；伙伴分别为空、带一批 512-token 短请求、与 owner 近似等载。短请求数从 8、32 试起，按实际 KV 预算确定并冻结。另跑双方均短的对照。将 prefill、混合 step、纯 decode 分开，分别记录两卡 GQA 主 kernel 与归约的绝对毫秒数、慢卡 GQA 占纯 decode step 的比例、整步时间、进入同层 EP dispatch 的到达差和 EP 自身时间。跨卡用对齐的 profiler 时间线；各卡本地 CUDA event 仅计算本卡间隔。正式延迟另跑无 profiler 版本。模型选择看自然负载上可缩短的毫秒数和出现频率，同时保留均衡负载的结果。
3. **同输入算子上限。** 按候选真实形状生成一次双卡逻辑 Q/K/V、页表和原始 batch，再派生本地 DP、head-TP、NanoCP GQA 移植版和同 tile CP 布局。覆盖两卡角色互换、伙伴忙、非整页、单长行与多长行。所有有效行对 CPU FP32 参考，独立校验回传头的顺序、当前 token KV 与伙伴自己的短请求。预热、独立建图至少 3 次、每图 30 replay，轮换方法顺序。输出整图 makespan、主机时间、两卡 kernel、各阶段耗时和原始样本。NanoCP 在 5090 使用作者 forward + FA2 默认 split + 适配传输，表中明确标注。[既有 NanoCP 对照](results/5090-matched/native-default.md)
4. **估计开发上限。** 用上述耗时在冻结的请求到达序列上做事后最优选择，按实际 step 频率比较普通 DP 和可选 head-TP。结合实测非 attention 时间，单计初次 KV 放置或拷贝、每步 Q/K/V 和结果传输、预计可摊销步数，避免重计已包含在整图中的通信。分别报告无成本理想上限与扣除成本后的估计。旧规格七档等权示例只有约 1.1% 的 DP/head-TP 整图空间，大 batch 的单格加速不能代替真实负载。[旧负载扫描](results/5090-matched/README.md)
5. **端到端验收。** 用同一份到达时间、prompt token IDs、输出长度和随机种子重放。首选 Azure 公开 trace 的连续时间段，按时间拆开调参和留出段，保留原长短分布，用固定种子生成缺失的输入内容，披露截断及到达率缩放。另做可控的长短偏斜实验。原生分发用于主表，强制绑定 owner 只用于诊断。取低、中、近饱和三档到达率，每档至少三次配对运行并轮换顺序。公开 trace 的上下文不足以触发协作时，单独如实报告覆盖率。[Azure trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
6. **报告与归因。** 主表为上游 vLLM、同构建关闭 head-TP、符合形状条件即协作、按负载动态协作。后三者用相同后端和 tile，另在混合负载切换一次默认与自定义 tile。各方法使用相同显存预算、分页、CUDA Graph、前缀缓存配置、EP 通信后端与调度限制。报告每请求 TPOT 的 P50/P95、流式 token 间隔的 P99、TTFT、输出 tokens/s 和完成率；同时给出两卡进入 EP 的到达差、协作命中率、选择开销及 KV 搬运量。报告配对差和跨运行波动，保留收益为负的场景。进入 EP 更整齐而 TPOT 无改善时，只支持机制层结论。

## 开发计划与交付顺序

| 阶段 | 实现位置与工作 | 完成判据 |
| --- | --- | --- |
| P0 基线与通信 | 复用 vllm/head_tp_eval 的到达序列与 runner，先做上述原生 profile。并行验证与 vLLM worker 一致的双进程传输、固定缓冲和图重放。单进程 memcpy 依赖同进程设备地址，双进程方案优先小测现有 NCCL 通道的收发、建图、空伙伴和有背景请求的情况。 | 实际 EP 分片、生成与图模式可验证；记录两卡传输的完整往返代价。若双进程开销过大，重估收益。 |
| P1 模型规格与算子 | 在 nanocp/bench_decode_cp 和 vLLM 对应 head-TP runner 里传递 Q/KV/dim、真实 scale、GQA 分组及 tile 宽度。现有 layout 固定 16/2/256，attn_local 将 scale 固定为 1/16，bench_perf 的 FP32 参考则从 dim 推导。Granite 的实际 scale 是 1/64，两条路径均须与模型配置对齐，还须验证 3 个 Q 对 1 个 KV 的排列。 | 旧规格回归、新规格 DP/TP/CP 对拍、多次 graph replay 均通过；得到实测协作成本曲线和留出 trace 的收益上限。 |
| P2 接真实 forward | 在目标模型 RoPE 后、输出投影前接入 paged head-TP。复用 vllm/vllm/model_executor/layers/head_tp_attn.py 的头映射和 v1/head_tp_kv.py 的 KV 管理，核对后端真实布局。先固定一条长请求和一个伙伴，在 decode 开始前一次分配伙伴 KV；保持请求归属与 MoE expert 路由。 | 强制协作能完成整模生成；相同 token 前缀逐步比较 logits 与原生数值波动，检查两卡短请求、页复用、取消、抢占与释放。记录实际进入 paged kernel 的层数，分开列 eager 和 graph。 |
| P3 动态准入 | 在 v1/head_tp_sched.py 及 scheduler 处按两卡行数、上下文页数、可用 KV、预测慢卡耗时选本地或协作。首版只在 prefill 到 decode 边界决策，按请求固定到结束，避免来回迁移。保留强制本地与强制协作模式。 | 每项决策可重放，只用决策时可知的数据；伙伴忙、无空页或无收益时正常本地执行。 |
| P4 端到端与复核 | 跑冻结主表；在第二模型上至少复核一组均衡、一组偏斜和一个留出 trace 条件，按可用资源决定是否扩大。 | 可从配对运行重现主表；区分 tile、协作准入和实际 MoE 同步贡献。 |

第一轮交付限于 P0、P1。建议在开发集预先约定一个值得整合的收益幅度，例如偏斜场景端到端预测可降至少 5%，并且幅度超过基线重复运行的波动；这是投入决策的参考值。若只有人工极端负载看到空间，先收窄工作负载声明。若原生模型确有明显 attention 慢卡，但固定两卡按头均分计入通信后仍几乎无收益，停止扩展该策略，回到已有的可变比例 CP 对照。
