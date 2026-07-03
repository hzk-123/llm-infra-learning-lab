# Day 45：Streaming Client、TTFT 和 TPOT

## 目标

Day44 测的是 non-streaming 完整响应耗时：

```text
non_stream_latency
```

Day45 开始使用 streaming 请求，把一次生成拆成更细的时间指标：

```text
TTFT
TPOT
total_latency
output tokens/s
```

## 为什么需要 streaming

Non-streaming 请求会等模型完整生成完，再一次性返回 response。

这样只能知道：

```text
从发起请求到完整响应返回用了多久。
```

但真实 LLM serving 更关心：

```text
用户多久看到第一个 token？
后续 token 生成得有多快？
```

所以需要 streaming。

## 指标定义

### TTFT

TTFT = Time To First Token。

在 Day45 里定义为：

```text
first_content_time - request_start
```

它表示：

```text
客户端发出请求后，第一次收到 assistant 内容片段用了多久。
```

TTFT 主要受这些影响：

```text
HTTP 请求开销
prompt tokenization
prefill
scheduler 排队
首次 decode
server flush
```

### TPOT

TPOT = Time Per Output Token。

在 Day45 里近似为：

```text
(request_end - first_content_time) / max(completion_tokens - 1, 1)
```

它表示：

```text
第一个内容 token 到达之后，后续每个输出 token 平均耗时。
```

TPOT 主要反映 decode 阶段速度。

### total_latency

```text
request_end - request_start
```

它是完整 streaming 请求结束的总耗时。

## 为什么 Day45 仍然不是正式 benchmark

Day45 只跑一个请求。

它用于验证：

```text
streaming client 能工作。
TTFT 能被记录。
TPOT 能被计算。
usage 能被读取。
```

正式 benchmark 需要：

```text
warmup
多次重复
p50 / p95
不同 prompt 长度
不同 output 长度
不同并发
```

这些后面再做。

## 前置条件

先启动 vLLM server：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
tmux new -s vllm-day43
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

另开一个 SSH 终端运行：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
python exercises/08_serving_vllm/day45_streaming_ttft_tpot.py
```

## 输出重点

重点看：

```text
TTFT
TPOT
total_latency
prompt_tokens
completion_tokens
total_tokens
stream_content_chunks
assistant content
```

## 你应该掌握

学完 Day45 后，你应该能解释：

```text
streaming response 是服务端不断推送数据块。
TTFT 代表用户看到第一个内容的等待时间。
TPOT 代表后续输出 token 的平均生成间隔。
non_stream_latency 不能拆出 TTFT / TPOT。
第一次请求可能受 warmup / Triton JIT 影响，不能直接当正式 benchmark。
```

