# Daily Log

Date: 2026-07-03

## Topic

Day45：Streaming Client，记录 TTFT / TPOT。

这一天的目标是：

```text
使用 streaming 请求调用本地 vLLM server。
记录 TTFT。
记录 TPOT。
记录 total_latency。
记录 token usage。
理解 streaming 指标和 non-streaming latency 的区别。
```

## Ran

服务器环境：

```text
conda env: llm-sprint
project: ~/llm-infra-learning-lab
server: vLLM on http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
```

运行命令：

```text
python exercises/08_serving_vllm/day45_streaming_ttft_tpot.py
```

## Client Config

脚本配置：

```text
base_url: http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
max_tokens: 96
temperature: 0.2
timeout: 60.0s
request_stream_usage: True
```

其中：

```text
request_stream_usage=True
```

表示请求中带了：

```text
stream_options: {"include_usage": true}
```

这样 streaming 结束时可以拿到：

```text
prompt_tokens
completion_tokens
total_tokens
```

## Streaming Metrics

输出：

```text
status: ok
response_id: chatcmpl-9ab3c16201dbbce3
finish_reason: stop
TTFT: 0.056s
TPOT: 0.0020s/token
total_latency: 0.138s
decode_after_first: 0.082s
output tokens/s: 312.08
prompt_tokens: 56
completion_tokens: 43
total_tokens: 99
raw_stream_events: 45
content_chunks: 42
TPOT uses completion_tokens: True
```

## Metric Explanation

### TTFT

本次：

```text
TTFT = 0.056s
```

含义：

```text
从 Python client 发起请求，到第一次收到 assistant 内容片段，用了 56 ms。
```

TTFT 主要包含：

```text
HTTP 请求开销
server 接收请求
prompt tokenization
prefill
scheduler 调度
第一次 decode
第一个 streaming chunk 返回
```

这次 prompt 很短，模型也很小，所以 TTFT 很低。

### TPOT

本次：

```text
TPOT = 0.0020s/token
```

含义：

```text
第一个内容片段返回之后，后续每个输出 token 平均约 2 ms。
```

脚本计算方式是：

```text
TPOT = decode_after_first / max(completion_tokens - 1, 1)
```

本次：

```text
decode_after_first = 0.082s
completion_tokens = 43
```

所以近似：

```text
0.082 / 42 = 0.00195s/token
```

也就是输出里的：

```text
0.0020s/token
```

### total_latency

本次：

```text
total_latency = 0.138s
```

含义：

```text
从请求开始，到整个 streaming 响应结束，总共用了 138 ms。
```

它和 Day44 的 `non_stream_latency` 不一样。

Day45 streaming 可以拆出：

```text
TTFT
TPOT
decode_after_first
```

Day44 non-streaming 只能知道完整响应用了多久。

### output tokens/s

本次：

```text
output tokens/s = 312.08
```

它是：

```text
completion_tokens / total_latency
```

这是单请求粗略吞吐，不是并发 benchmark 的 aggregate throughput。

后面并发实验中，还要区分：

```text
single-request output tokens/s
aggregate output tokens/s
QPS
p50 / p95 latency
```

## Stream Event Explanation

输出：

```text
raw_stream_events = 45
content_chunks = 42
completion_tokens = 43
```

含义：

```text
raw_stream_events
  客户端收到的 SSE JSON event 数量。

content_chunks
  其中真正包含 assistant content 的 chunk 数量。

completion_tokens
  server usage 统计中的生成 token 数。
```

它们不一定完全相等。

原因是：

```text
有些 stream event 可能只包含 role、finish_reason 或 usage。
一个 content chunk 也不一定严格等于一个 tokenizer token。
```

所以脚本优先使用：

```text
completion_tokens
```

来计算 TPOT。

## Prompt And Output

Prompt：

```text
在 LLM 推理服务性能指标中，请用两句话分别解释 TTFT 和 TPOT，不要展开其他概念。
```

模型输出：

```text
TTFT（True Positive Rate）：表示模型预测结果为正类的概率。
TPOT（Tuned Parameter Optimization）：是一种优化算法，用于找到最优的参数组合，以提高模型性能。
```

## Bad Case

这次模型回答是错误的。

正确解释应该是：

```text
TTFT = Time To First Token，表示请求发出后到第一个输出 token 返回所需时间。
TPOT = Time Per Output Token，表示生成阶段每个输出 token 的平均耗时。
```

模型错误地把：

```text
TTFT
```

解释成了：

```text
True Positive Rate
```

又把：

```text
TPOT
```

解释成了：

```text
Tuned Parameter Optimization
```

这说明：

```text
服务指标采集链路成功，不代表模型知识回答正确。
```

Day44 和 Day45 都出现了类似现象：

```text
模型能生成。
API 能返回。
指标能记录。
但回答内容可能不可靠。
```

这就是后面必须做 eval / bad case 分析的原因。

## What I Actually Verified

这一天实际验证的是：

```text
Python client 可以发起 streaming chat completion。
客户端可以逐行解析 SSE data。
客户端可以记录第一个 content chunk 到达时间。
客户端可以计算 TTFT。
客户端可以读取 completion_tokens。
客户端可以估算 TPOT。
客户端可以记录 total_latency。
```

## What I Did Not Prove Yet

Day45 还不能说明：

```text
模型在稳定状态下 TTFT 就是 0.056s。
TPOT 稳定就是 0.0020s/token。
output tokens/s 稳定就是 312.08。
这个服务并发能力有多强。
长 prompt 下 TTFT 如何变化。
长 output 下 TPOT 如何变化。
prefix cache 是否有效。
```

原因：

```text
这里只跑了一次请求。
没有 warmup 批次。
没有重复采样。
没有 p50 / p95。
没有并发。
没有 workload 分组。
```

## Key Takeaway

Day45 的核心理解：

```text
Streaming 让客户端能观察生成过程。
TTFT 衡量第一个内容 token 到达时间。
TPOT 衡量后续输出 token 的平均生成间隔。
usage 字段提供 token 统计。
content_chunks 和 tokenizer tokens 不一定一一对应。
单次结果只能说明链路跑通，不能代表稳定性能。
```

## Day45 Conclusion

Day45 已通过。

当前完成：

```text
Streaming client 跑通。
TTFT 记录成功。
TPOT 估算成功。
token usage 读取成功。
发现模型回答 bad case。
```

下一步建议：

```text
不要急着并发压测。
先做 Day46：warmup + repeated single-request benchmark。
目标是对同一个请求重复多次，记录 p50 / p95 latency、TTFT、TPOT。
```

