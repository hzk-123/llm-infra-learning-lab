# Day 44：vLLM Python Smoke Client

## 目标

Day43 已经用 `curl` 验证了 vLLM server：

```text
/v1/models
/v1/chat/completions
```

Day44 的目标是把这个验证动作变成一个 Python 脚本。

这一天不做并发压测，也不做 streaming。

只做：

```text
用 Python 调用本地 vLLM OpenAI-compatible API。
记录一次请求的完整响应 latency。
读取 prompt_tokens、completion_tokens、total_tokens。
打印模型输出内容。
```

## 为什么要先写 smoke client

后面做 benchmark 前，必须先确认：

```text
Python 程序能连上 vLLM server。
模型名正确。
请求 JSON 格式正确。
响应 JSON 能正常解析。
usage 字段能正常读取。
```

如果这些没跑通，直接做 TTFT / TPOT / 并发压测会很乱。

## Day44 测的是什么

Day44 脚本是 non-streaming 请求。

也就是说，客户端会等模型完整生成完，再一次性收到 response。

因此 Day44 记录的是：

```text
non_stream_latency
```

它大致包含：

```text
HTTP 请求开销
prefill 时间
decode 时间
response JSON 返回时间
```

它不是：

```text
TTFT
TPOT
```

TTFT / TPOT 需要 streaming client，在后面的 Day45 再做。

## 前置条件

服务器上需要先启动 vLLM：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
tmux new -s vllm-day43
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

另开一个 SSH 终端运行 Day44：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
python exercises/08_serving_vllm/day44_python_smoke_client.py
```

## 输出重点

重点看：

```text
models endpoint latency
chat completion latency
prompt_tokens
completion_tokens
total_tokens
finish_reason
assistant content
```

如果输出里有：

```text
status: ok
```

并且能打印 assistant content，就说明 Python client 链路成功。

## 你应该掌握

学完 Day44 后，你应该能解释：

```text
curl 请求和 Python HTTP client 请求本质上是同一个 API。
/v1/models 用于确认服务可访问、模型已注册。
/v1/chat/completions 用于真正触发一次生成。
non-streaming latency 是完整响应时间，不等于 TTFT。
usage 里的 token 统计是后续 benchmark 的基础。
```

