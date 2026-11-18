# 双卡 GQA decode 对比

在同一进程内比较 DP、请求级 head-TP 和本地 NanoCP-style CP 数据面。三者使用同一个 FA2 fork。上游 NanoEngine 的单请求复现入口见[项目说明](../README.md)。

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
python bench_perf.py --case mix_1024x1024_4x256k --rounds 30 --profile-rounds 5
python bench_compute.py --case mix_1024x1024_4x256k --rounds 30 --profile-rounds 5
```

- `bench_perf.py` 默认对十个性能 case 运行三种方法，输出 JSON Lines。用 `--method dp|head|cp` 可单独运行。初始化 OOM、缺少 CUPTI 活动或运行错误都会使比较失败。
- `--check-only` 捕获及重放实际执行图，然后按原始请求顺序将两个 owner 的输出和 CPU FP32 GQA 对拍。`--check` 在同一次测试中再做性能测量。默认性能模式省去参考计算。仅计算模式下，CP 和 head-TP 的输出分别是 token 部分结果和 head 分片，因此通过完整路径校验最终输出。
- `bench_compute.py` 使用相同输入和 tile，只测两卡本地 attention，Q/K/V 已就位。完整路径包含方法各自的路由、通信、卡内归约和输出拼接或合并。
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
