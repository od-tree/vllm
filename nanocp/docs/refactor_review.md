# GQA MoE decode 的 baseline、重构顺序与选题判断

2026-11-18。目标环境为节点内 DP-attention + EP、两张 GPU、单 token decode。文中 head-TP 指只把选中请求的 KV 头和对应 Q 头分给伙伴，模型权重仍按原 DP/EP 配置部署。建议保留这条执行路径，先验证请求准入和 KV 放置的收益，再决定是否投入整机集成。

## 重构进度（2026-11-19）

下文审阅表记录修改前的工作树。现已统一 DP、head-TP、CP 的输入、tile planner、整图与分阶段计时，修正伙伴本地 batch 缺失、head-TP 部分页打包、eager owner 恢复全头时的 KV 更新和剩余输出长度估计。CPU 布局、路由、归约和 tile 回归已通过。双卡数值对拍、CUPTI 明细及新性能表待 CUDA 主机验证。调度 profile 的加载、生效计数、跨进程 graph 和服务层对比仍是独立工作。运行入口与测量边界见[双卡算子测试](../bench_decode_cp/README.md)。

## NanoCP 基线的边界

公开代码已经有 GQA 数据面。固定提交的 FlashAttentionImpl 在第 89 行写新 KV，第 164 行前完成 Q 路由，随后按本地 page table 调用 flash_attn_with_kvcache 取得 O/LSE，第 287 行合并。FlashMLAImpl 是另一条实现。作者用 H200 上的 DeepSeek-V3、Kimi-K2 评价系统，并为 NanoCP 和 vLLM 使用等价的本地 attention 和 MoE 通信 kernel。这支持复用成熟 GQA attention。仅从代码存在性无法推导 Qwen3.5 混合层、sm_120、跨进程通信可以直接跑通。[固定提交源码](https://github.com/nano-infra/NanoEngine/blob/8abce29fe3ba0af3cd785b5e533b5b4b22ae70d5/nanodeploy/layers/attention.py#L64-L297) · [论文 §4–6 与 artifact](https://arxiv.org/html/2605.21100v2)

| 对照 | 应实现的能力 | 可支持的结论 |
| --- | --- | --- |
| DP + EP | LeastBatch 和 LeastCache 放置，所有请求 degree=1，原生后端与 per-row split 均可选 | 比普通放置和单卡内调优多得多少 |
| 固定 CP2 | 全部请求按 token 分，Q 路由、partial O/LSE 和归约 | 固定协作成本 |
| NanoCP-style GQA 数据面 | 给定相同批次与参与 rank，选中请求做 CP，其余本地。统一设备、通信能力、graph 和计时边界 | 分头与分 token 的数据面差异 |
| NanoCP-style GQA 策略 | 在前一行上加入按本机 profile 搜索的长度桶，准入选 master/伙伴，按现有负载、空闲页分 token，考虑 MoE 侧 batch | 在约定的单节点 GQA 部署上比较策略。说明遵循论文算法还是工件分支，列出差异 |
| 本文策略 | 同一批候选、容量和 profile 预算，引入伙伴后台工作与放置成本，允许选本地、CP、合法的 head-TP | 通过逐项去掉成本项和 head 选项量出各自贡献 |

作者原生本地 kernel、移植到目标硬件的默认 kernel、共享调优后的 kernel 要区分命名。主表使用每条路径在同一校准集和预算下的最佳可用配置，另报默认配置。让 CP 和 DP 都能使用通用的 per-row split、尾波处理、graph 和打包传输。各自可选不同 split，合并成本写入选择目标。只对 head-TP 有意义的分头页和按头拼回保留给 head-TP，只对 CP 有意义的可变 token 切分、partial/LSE 归约保留给 CP。若一个现成的 FlashInfer 后端更快且支持目标形状，应作为本地后端参照。试图靠降级 CP 来展示 TP 价值，无法区分算子与系统贡献。

## 对当前数值的判断

[工作记录 §5–8](work_record.md) 的 A100 同机表是旧版 CP，缺少对称 tile 的同机整表。5090 用的是 sm80 PTX JIT、PCIe 拷贝与 graph，结果覆盖一个头配置、两卡、预置 KV 的单层 attention。大多数 case 只在一个 owner 上产生请求，不能代替两边同时收请求和进入 EP 的服务测量。

- 在 16Q/2KV、d=2 下，设每头每 token 的 K+V 共 b 字节。均分长请求在 head-TP 上每卡读 Lb，在 CP 上每卡读 (L/2)×2b。只看历史 KV 主读量，两种方案没有倍数差。切头能少传某些 Q/O，并省跨卡 softmax 合并，但卡内为补 CTA 而增加的 split 会扩大本地归约。用实际 DRAM 字节、持续带宽、每个 SM 的活跃 CTA 和 kernel 时间检验带宽假说。单凭 CTA 数或理论带宽作不了这个判定。
- 5090 上 256×32768 的带通信时间为 5.478 对 6.128 ms，head-TP 低 10.6%。mix_128 是 0.584 对 0.502 ms，head-TP 高 16.3%。其 owner 卡内 FA2 combine 为 0.102 对 0.051 ms，解释了 0.082 ms 总差中的一部分。CP 的跨卡 combine_kv 仅 0.001–0.009 ms。两个 combine 应分别计数。
- mix_1024 在 head-TP 侧为 owner 的 4 长 + 1024 短再加 partner 的 1024 短，CP 侧为全局 4 长 + 1024 短。两边 fork/join 正确也无法补救全局工作量不同。CP 的 compute-only 脚本同样没有接收 partner_seqs。
- 去掉这行，其余九种形状各执行一步，按记录中已取整的中位数求和。固定 CP 为 20.175 ms，固定 head-TP 为 19.660 ms。每行任选较快者且选择零成本，合计 19.349 ms，分别节省 4.09% 和 1.58%。这是这组输入、这两种整步执行器的示例上限，不涵盖重新准入、不等分 CP、批内混合或真实到达过程。九行中 head-TP 三胜一平五负。正式结果需要独立重复和按真实 step 频率加权。
- 单层收益进入整机还要乘关键路径中 Full Attention 的时间占比。若占比为 f、该段延迟减少 r，固定批次、其余耗时不变时整步降幅为 f×r。排队和准入变化可能改变尾延迟，必须另测。优先采集这个占比以及各 rank 进入 EP 前的等待。

## 代码审阅与风险

范围为相对共同 vLLM main b8c04dd3b4 的两个当前工作树和未跟踪研究文件，以及 ignored vendor 内相对 af5422c 的 FA2 fork。vLLM dtp_test 为 92cd045f30，CP nanocp_5090 为 cf481f9b49，FA2 5090 为 39ac30e。eval 与当前 head_tp_eval 文件树一致。另一个 nanocp 分支含通信回退、graph 与饱和尾波变体，重构时要从算法差分吸收，避免用分支名隐式代表硬件。两个 FA 子模块含本地元数据脏状态。

| 模块及改动 | 核对结果与处置 |
| --- | --- |
| vLLM request、scheduler output、engine/core、outputs、gpu_worker、forward_context | 已有计划交换、idle rank 参与、完成与失败回报。新增显式 off，避免原始 DP 基线进入额外 collective。采集各 rank 等待与控制面时间。|
| v1/head_tp_sched.py、core/sched/scheduler.py | 默认 HeadTPProfile 为空，查找未见正常入口加载测量表。符合模型条件时仍会交换状态。head_tp_attn_us 有消费者和默认零值，未找到生产写入端。先做配置加载、单位和生效计数，服务启动验证 off/local/active 真正走哪条路径。|
| 同一调度器的进入和退出 | LengthBucketEMA 已有冷启动和按 prompt/prefix 分桶的完成长度均值。命中桶时直接返回历史总输出均值，用作剩余量会高估已生成较久的请求。用已存活到当前长度的样本估条件剩余，至少比较固定寿命和简单截断估计。_benefit 是跨剩余步累计量，_pair_cost 加一次伙伴延迟。应把伙伴的逐步背景负载纳入双卡关键路径，再与一次性搬迁比较。|
| v1/head_tp.py、worker/gpu/head_tp.py、test_head_tp_tiles | 框架对饱和长行另做尾波搜索，当前 FA2 GPU bench 的 _tile_plan 和 CP 工作树的 plan_tiles 到饱和均保留 split=1。现有 vLLM 测试的 [262144,131072,512,512] 期望 [72,36,1,1]。独立执行当前预算函数得到长行 [71,36]，含短行的预算是 107。统一实现后同时测未饱和、刚好饱和、刚超一波和 split 上限。|
| head_tp_kv.py、KV connector/offload、shared region、manager/spec、prefix tests | CPU block 映射、pin/unpin、复制和追平可复用。helper 私有缓存按 example 形状申请，独立于 BlockPool，需实测驻留与峰值、计入 admission，验证回收。不要把 microbench 的紧凑页直接记为服务容量收益。|
| head_tp_attn.py、qwen3_next.py、gpu/model_runner.py、eager/schedule tests | owner 对 active 行只更新自己的 head，helper 更新另一份；_maybe_exit 可以清掉计划。静态路径上未见退出前把协作期间新增的 helper KV 补回 owner，恢复全头时有读旧 KV 风险。优先做 local→active→local 多步 logits 对拍，短期可让布局活到请求结束。active 强制 eager 并跳过 compiled/graph，SDPA 仅适合作数值参照。|
| head_tp_eval 的 trace/arrivals、dataset/scheme registry、runner、summarize 和测试 | 保留 trace 生成、重放和汇总。runner 连接已运行的 server，config 标签本身不能切换其策略。把服务实际后端、配置、profile 哈希、激活请求数和退回次数写入结果。|
| FA fork 的 CMake/setup、Python/C++ op 注册、Flash params、paged 寻址、exchange、launch/combine、splice 和 GPU tests | 提取已验证的页格式、算子和传输适配，补齐当前未跟踪 splice 源码后固定可重建版本。benchmark 用环境变量逐次改变 phase、直接握有两张卡的指针和流。接入 vLLM 时要验证真实多进程 IPC/collective 和 graph 生命周期。仅在 bench graph replay 成功不能证明此路径成立。|
| CP layout、attn_local、decode_cp、merge、copy_idx、hao_basic、comm、IPC 和测试 | 保留为参考执行器。当前 page_split 固定约一半。run_routed 以单一 moe_rank 发起，最后 combine_kv 的输入覆盖整个 batch，短行在 partner 掩码为零。跨卡未传该行与 master kernel 未访问该行是两件事。计时中 CP 读预先构造的 KV，head-TP 还包含本步 K/V 传输及 helper 写入。增设同口径 append + attention，以及不含 append 的纯 attention 对照。|
| CP bench_perf、未跟踪 bench_compute、文档、图与上游 tree 快照 | 统一 case、种子、双卡背景工作、设备流和统计。compute 脚本动态导入特定目录下的测试模块，改为导入固定版本执行器。设计图和旧表保留作来源，不能替代运行清单。|

正确性至少覆盖非页对齐、空 segment、非连续合作行、两边同时有 owner、helper 忙/闲、多个连续 token、新 KV 的可见性、请求结束和 graph 重放。单步绝对误差加入 finite 检查，当前 CP 测试的 max_abs > threshold 对 NaN 不会触发。多步以独立 dense/reference 或原生 vLLM 比较 logits。此次在本地 CPU 运行 CP tile 脚本通过；另单独执行了 vLLM 的预算函数。未运行 CUDA、多进程服务或完整 pytest。

## 建议的重构顺序

1. **锁定工件和计时边界。** 一个 vLLM 主工作树承载正常入口、调度器与统一 benchmark，FA fork 单独钉版本。第二份完整 vLLM checkout 暂留作历史参照，CP 执行器迁到共同 benchmark/backend。统一 workload 描述为每个请求的 owner、长度、页位置和参与 rank。每次运行断言全局请求集、输出 token 数与两卡本地背景完全一致。保存原始 samples、fork 版本与脏补丁、显存和传输方式。
2. **先测可获得的收益。** 在相同 5090 机器先做 local/CP2/head2 × 默认/调优。对 partner 背景从空闲扫到接近 owner，同批长短、多种长度、长请求比例 0/1/5/20%，CP 可按页不等分。测预置 KV 的整步、首次放置、append、CPU 准备和 CUDA graph replay，并在 A100 NVLink 上重跑代表性反转点。冷启动和稳态分开；各方法交错顺序、跨进程多次独立重复，报告配对差和置信区间。
3. **先试准入，再试中途迁移。** 在 prefill 输出放置或 decode 准入时选择布局，请求结束前固定，以免引入重分片状态机。用简单查表估各 rank 在当前批次的整步时间，包含通信与本地归约，施加容量约束。只有预计剩余步数 × 每步整组节省超过额外放置或迁移成本，才选协作；区分 CPU 命中、GPU 已驻留和直接落位。在同一控制器里只允许 local/CP，以及允许 local/CP/head，各跑一次。用未来真实长度只算离线上界，线上仅用决策时可见的量。
4. **按收益决定是否接入服务。** 在独立 trace 上，先比较上述上界、NanoCP-style 长度桶、只看负载的 DP、静态策略及现成后端，确认有超过重复测量波动的余量，再接入一个能放入设备的 GQA MoE 和真实 EP 通信。模型及量化需明列，先 profile Full Attention、GDN、专家与通信占比。主指标选固定短请求 P95/P99 TPOT 目标下的 goodput，辅报分长短 TPOT、TTFT、吞吐、队列、峰值 KV 和抢占。固定到达序列、输出长度、显存预算与拓扑。单层结果和服务结果分别报告。
5. **停止条件。** 若真实批次上 local/CP/head 的理想选择相对 local/CP 已无显著空间，保留 head-TP 为消融，不做运行中 TP↔CP 转换。若 local/CP 的收益也被强调优 DP 或 admission-aware CP 吸收，把结论缩至 GQA paged decode 的 kernel 实验，先对 FlashInfer 复核再决定是否作为独立论文。

## 相关工作与备选切口

- **最优先验证伙伴负载能否改变一次准入决定。** NanoCP 已有双均衡、离线长度桶、按 token 分片和 KV 放置。可争取的差异是同一长度在“伙伴已有短请求 / 空闲”“CPU prefix 命中 / GPU 冷搬”下，最优选择是否稳定反转。用同一 trace 在这两类状态分别和可变比例 CP 对照。Gimbal 已用后端压力指导 DP 路由，所以要补上只用相同压力做整请求放置的对照。[NanoCP](https://arxiv.org/html/2605.21100v2) · [Gimbal](https://arxiv.org/abs/2606.15177)
- **备选是降低卡内 split 合并成本。** 比较现实现与按行实际 split 归约、让 split=1 的短行直写，仅在有收益的 shape 启用。按中间张量读写字节和 graph 总耗时验收，给 CP/DP 使用同一优化。FlashInfer 已有动态 CTA 工作计划和 contraction，单纯按长度分 tile 很难独立成文。[FlashInfer](https://arxiv.org/abs/2501.01005)
- **暂缓大范围运行中切换。** Helix 已提供按 token 切 KV 与不同 FFN 并行布局的强静态参照。SPLASH 已研究带转换成本的 attention 布局切换，Moebius 已在 MoE 上做运行中 EP/TP 切换。本文可能成立的差异是一个 decode step 内，仅选中 GQA 请求借用伙伴，其他请求与 EP 权重放置照常，并按可用 KV 来源决定是否值得合作。需要前述 oracle 和真实 trace 才知道价值。[Helix](https://arxiv.org/abs/2507.07120) · [SPLASH](https://arxiv.org/abs/2609.37626) · [Moebius](https://arxiv.org/abs/2606.26607)
