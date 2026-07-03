# Daily Log

Date: 2026-07-03

## Topic

Day44：vLLM Python Smoke Client。

这一天的目标是：

```text
不用 curl，而是用 Python 脚本调用本地 vLLM OpenAI-compatible API。
验证 /v1/models。
验证 /v1/chat/completions。
记录完整响应 latency 和 token usage。
观察 vLLM server 日志。
```

Day44 不做 streaming，所以这一天不测 TTFT / TPOT。

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
python exercises/08_serving_vllm/day44_python_smoke_client.py
```

## Client Config

脚本配置：

```text
base_url: http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
max_tokens: 128
temperature: 0.7
timeout: 60.0s
```

含义：

```text
base_url
  本地 vLLM server 地址。

model
  vLLM 对外暴露的 served model name。

max_tokens
  本次最多生成 128 个 token。

temperature
  采样温度，影响输出随机性。

timeout
  Python client 最多等待 60 秒。
```

## /v1/models Result

输出：

```text
latency: 0.015s
model_count: 1
requested_model_found: True
model_ids: ['qwen2.5-0.5b-instruct']
```

说明：

```text
Python client 能访问 /v1/models。
vLLM server 已经注册模型。
脚本请求的 model name 和 server 暴露的 model id 一致。
```

这里的 `0.015s` 只是模型列表接口耗时，不是模型推理耗时。

## /v1/chat/completions Result

输出：

```text
status: ok
request_id: chatcmpl-a916fead7e3c2ead
model: qwen2.5-0.5b-instruct
finish_reason: stop
non_stream_latency: 0.981s
prompt_tokens: 41
completion_tokens: 99
total_tokens: 140
```

说明：

```text
Python client 成功触发了一次 chat completion。
server 返回了正常 response。
finish_reason=stop 表示模型自然停止生成。
usage 字段能返回 token 统计。
```

这次完整响应耗时：

```text
non_stream_latency = 0.981s
```

注意：

```text
这是 non-streaming 完整响应耗时。
它不是 TTFT。
它不是 TPOT。
```

它大致包含：

```text
HTTP 请求开销
prefill
decode
response JSON 返回
Python client 解析
```

## Prompt And Output

Prompt：

```text
你好，用三句话解释 vLLM 是什么。
```

模型输出：

```text
VLLM 是一种基于深度学习的智能语言生成模型，它能够通过大量的文本数据进行自我学习和优化，从而在特定领域或场景中提供准确、流畅的语言输出。它的主要特点包括强大的自然语言处理能力、高精度的文本生成能力和广泛的适用性。此外，VLLM 还支持多模态信息融合，可以理解并整合图片、音频等多种形式的信息，进一步提升了其在实际应用中的表现力和适应性。
```

## Bad Case

这个回答能生成，但内容不严谨。

问题在于：

```text
vLLM 不是一个语言生成模型。
vLLM 是一个大模型推理和 serving 框架。
```

更准确的说法应该是：

```text
vLLM 是一个高吞吐的大模型推理服务框架。
它通过 PagedAttention、continuous batching、KV cache 管理等机制提升 serving 效率。
它提供 OpenAI-compatible API，方便把本地大模型部署成可调用服务。
```

这说明 Day44 有两个结论：

```text
工程链路成功。
模型回答质量不一定可靠，需要评测和 bad case 分析。
```

这是后面做模型服务评测时必须记住的点。

## vLLM Server Logs

vLLM 日志：

```text
GET /v1/models HTTP/1.1 200 OK
Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel.
POST /v1/chat/completions HTTP/1.1 200 OK
Avg prompt throughput: 4.1 tokens/s
Avg generation throughput: 9.9 tokens/s
Running: 0 reqs
Waiting: 0 reqs
GPU KV cache usage: 0.0%
Prefix cache hit rate: 0.0%
```

解释：

```text
GET /v1/models 200 OK
  模型列表接口请求成功。

POST /v1/chat/completions 200 OK
  聊天生成接口请求成功。

Triton kernel JIT compilation during inference
  第一次碰到某个 shape/config 时，Triton kernel 临时编译。
  这会带来一次 latency spike。
  后面做 benchmark 前需要 warmup，不能直接统计第一次请求。

Avg prompt throughput: 4.1 tokens/s
  最近一个统计窗口内的 prompt/prefill 处理吞吐。

Avg generation throughput: 9.9 tokens/s
  最近一个统计窗口内的生成吞吐。

Running: 0 reqs, Waiting: 0 reqs
  打印日志时已经没有请求在跑，也没有请求排队。

GPU KV cache usage: 0.0%
  请求结束后 KV cache 占用释放或当前无活跃请求。

Prefix cache hit rate: 0.0%
  本次没有重复 prefix 的 workload，所以 prefix cache 没有命中。
```

## What I Actually Verified

这一天实际验证的是：

```text
Python 脚本能访问本地 vLLM server。
/v1/models 能返回模型列表。
/v1/chat/completions 能返回完整生成结果。
响应里能拿到 request_id、finish_reason、usage。
客户端能记录 non_stream_latency。
vLLM server 端能打印请求日志和吞吐日志。
```

## What I Did Not Measure Yet

Day44 没有测：

```text
TTFT
TPOT
p50 / p95 latency
并发 QPS
多请求 output tokens/s
长 prompt
长 output
prefix cache 命中效果
多卡 tensor parallel
```

这些需要后续 streaming client 和 benchmark runner。

## Key Takeaway

Day44 的核心理解：

```text
curl 和 Python client 调的是同一套 OpenAI-compatible API。
non-streaming 请求只能得到完整响应时间。
usage 字段是后面计算吞吐和分析请求成本的基础。
第一次请求可能受到 Triton JIT / warmup 影响。
工程链路成功不等于模型回答一定正确。
```

## Day44 Conclusion

Day44 已通过。

当前状态：

```text
vLLM server 正常运行。
Python smoke client 调用成功。
模型列表接口成功。
聊天生成接口成功。
完整响应 latency 和 token usage 已记录。
发现一个模型回答 bad case：把 vLLM 错说成语言模型。
```

下一步建议：

```text
先复盘 non-streaming latency 的边界。
然后进入 Day45：streaming client，记录 TTFT / TPOT。
```

