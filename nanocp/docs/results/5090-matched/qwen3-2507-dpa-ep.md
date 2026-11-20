# Qwen3-30B-A3B-Instruct-2507-FP8 双 RTX 5090 DPA + EP 验证

2026-11-19 在两张 32 GB RTX 5090 上完成了真实权重加载、CUDA Graph 捕获与并发生成。研究 fork 源码为 c4fe69224c1f5edcb37bb5546441a3a5af8fc593，使用其上游基点 b8c04dd3b455affd8d91f0ab436b815491c9af38 的 cu130 扩展。两提交之间的服务端改动在 Python 层。启动日志已打印 fork 的实际导入路径。Torch 2.14.0+cu130，驱动 615.71.09。Qwen 官方 ModelScope 仓库的四个权重分片与该仓库文件清单的 SHA-256 全部一致。

服务配置为 TP=1、DP=2、EP=2，模型长度上限 4096，max-num-seqs=8、max-num-batched-tokens=2048、gpu-memory-utilization=0.87。两卡都运行完整 attention，每个 EP rank 分到 128 个专家中的 64 个；日志报告 linear placement、FLASH_ATTN attention、DeepGEMM FP8 linear/MoE 和 AgRsAll2AllManager。独立双进程 NCCL all-reduce、all-to-all 校验通过。每卡模型加载占用 15.74 GiB，各引擎报告 KV cache 容量 118,720 tokens。首次启动包含 Torch、FlashInfer、DeepGEMM 编译和 CUDA Graph 捕获，首次请求仍触发了个别 Triton kernel JIT。

| 请求 | DP0 | DP1 |
| --- | --- | --- |
| 同时提交的问答，temperature=0 | 17 + 25 得到 42，自然停止 | 法国首都得到 Paris，自然停止 |
| 同时提交的指定长度输入 | 输入 2048 tokens，生成 64 | 输入 128 tokens，生成 64 |

四次调用均收到 HTTP 200。通过请求头 X-data-parallel-rank 指定目标，生成长度按服务端 usage 检查。当前结果仅验证功能，尚无长上下文与稳态吞吐测量。

服务器上复用已校验的模型、隔离环境与当前 fork 的启动方式如下。服务仅绑定本机回环地址。

```bash
cd /root/bench-runs/qwen-ep-6Vwgas
bash serve.sh
```

另开服务器终端运行 `cd /root/bench-runs/qwen-ep-6Vwgas && runtime/bin/python smoke.py`。同目录的 `serve-default.log`、`smoke-default.json`、`mixed-smoke.json`、`nccl.log` 和 `download.log` 保留启动记录、请求结果、通信与文件校验记录。
