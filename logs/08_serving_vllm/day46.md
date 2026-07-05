# Daily Log

Date: 2026-07-03

## Topic

Day46：Warmup + Repeated Single-request Benchmark。

这一天的目标是：

```text
先 warmup。
再对同一个请求重复运行多次。
统计 TTFT / TPOT / total_latency / output tokens/s 的 min、avg、p50、p95、max。
```

这仍然是单请求 benchmark，不是并发压测。

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
python exercises/08_serving_vllm/day46_repeated_single_request_benchmark.py
```

## Config

脚本配置：

```text
base_url: http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
max_tokens: 96
temperature: 0.2
timeout: 60.0s
warmup_runs: 2
measured_runs: 8
sleep_between_runs: 0.2s
request_stream_usage: True
```

含义：

```text
warmup_runs = 2
  前 2 次请求只用于预热，不进入最终统计。

measured_runs = 8
  正式统计 8 次单请求 streaming 调用。

temperature = 0.2
  低温采样，但仍然可能导致每次输出长度不同。
```

## Preflight

`/v1/models` 检查：

```text
latency: 0.004s
requested_model_found: True
model_ids: ['qwen2.5-0.5b-instruct']
```

说明：

```text
vLLM server 正常运行。
模型名正确。
Python benchmark client 可以访问本地 API。
```

## Warmup

warmup 结果：

```text
run 1: TTFT 0.044s, TPOT 0.0027s/token, total_latency 0.143s, completion_tokens 38
run 2: TTFT 0.021s, TPOT 0.0025s/token, total_latency 0.102s, completion_tokens 34
```

观察：

```text
第 1 次 warmup 的 TTFT 明显高于第 2 次。
```

这说明 warmup 是有意义的。

第一次请求可能受到：

```text
Triton JIT
CUDA graph shape 覆盖
缓存冷启动
server 内部状态初始化
```

等因素影响。

所以后面的正式统计排除了 warmup。

## Measured Runs

正式统计 8 次：

```text
run 1: TTFT 0.019s, TPOT 0.0025s/token, total_latency 0.110s, output tok/s 346.71, completion_tokens 38
run 2: TTFT 0.019s, TPOT 0.0030s/token, total_latency 0.112s, output tok/s 284.65, completion_tokens 32
run 3: TTFT 0.019s, TPOT 0.0025s/token, total_latency 0.087s, output tok/s 320.77, completion_tokens 28
run 4: TTFT 0.021s, TPOT 0.0024s/token, total_latency 0.087s, output tok/s 323.48, completion_tokens 28
run 5: TTFT 0.019s, TPOT 0.0024s/token, total_latency 0.084s, output tok/s 331.38, completion_tokens 28
run 6: TTFT 0.019s, TPOT 0.0032s/token, total_latency 0.136s, output tok/s 280.02, completion_tokens 38
run 7: TTFT 0.019s, TPOT 0.0032s/token, total_latency 0.146s, output tok/s 281.41, completion_tokens 41
run 8: TTFT 0.021s, TPOT 0.0024s/token, total_latency 0.087s, output tok/s 323.03, completion_tokens 28
```

可以看到：

```text
TTFT 很稳定，大多在 0.019s 左右。
TPOT 在 0.0024s/token 到 0.0032s/token 之间。
total_latency 会跟 completion_tokens 变化。
```

## Summary

正式统计结果：

```text
TTFT:
  min = 0.0185s
  avg = 0.0194s
  p50 = 0.0190s
  p95 = 0.0209s
  max = 0.0209s

TPOT:
  min = 0.0024s/token
  avg = 0.0027s/token
  p50 = 0.0025s/token
  p95 = 0.0032s/token
  max = 0.0032s/token

total_latency:
  min = 0.0845s
  avg = 0.1061s
  p50 = 0.0984s
  p95 = 0.1457s
  max = 0.1457s

output tokens/s:
  min = 280.0177
  avg = 311.4335
  p50 = 321.9035
  p95 = 346.7132
  max = 346.7132
```

## Metric Explanation

### TTFT

本次 TTFT 平均：

```text
avg TTFT = 0.0194s
```

含义：

```text
请求发出后，平均约 19.4ms 收到第一个 assistant content chunk。
```

这个值比 Day45 的单次 TTFT：

```text
0.056s
```

更低、更稳定。

原因可能是：

```text
Day46 前面先做了 warmup。
server 已经处于热状态。
Triton JIT / CUDA graph 等一次性开销不再明显影响正式统计。
```

### TPOT

本次 TPOT 平均：

```text
avg TPOT = 0.0027s/token
```

含义：

```text
第一个内容 chunk 到达后，后续每个输出 token 平均约 2.7ms。
```

TPOT 的 p95 是：

```text
0.0032s/token
```

说明尾部较慢的单次请求，每个 token 约 3.2ms。

### Total Latency

本次 total latency 平均：

```text
avg total_latency = 0.1061s
```

含义：

```text
从请求开始到完整 streaming 响应结束，平均约 106.1ms。
```

p95 是：

```text
0.1457s
```

说明这 8 次里，尾部较慢请求接近 146ms。

### Output Tokens/s

本次平均：

```text
avg output tokens/s = 311.4335
```

注意：

```text
这是单请求 output tokens/s。
不是系统并发 aggregate tokens/s。
```

后续并发 benchmark 中，需要区分：

```text
单请求 output tokens/s
整体 aggregate output tokens/s
QPS
p50 / p95 latency
```

## Important Caveat

这次每次输出长度不完全一致：

```text
completion_tokens: 28 到 41
```

原因是：

```text
temperature=0.2 仍然不是完全确定性输出。
模型可能自然 stop。
不同回答长度会影响 total_latency 和 output tokens/s。
```

所以这次 benchmark 已经比单次 smoke test 更可信，但还不是严格的性能对比实验。

如果后面要更严格，可以考虑：

```text
temperature=0
固定 prompt
固定 max_tokens
让请求尽量都以 length 结束
多跑一些 runs，例如 30 或 50 次
```

## First Measured Output

第一次正式测量的输出：

```text
Kv缓存在LLM推理中用于存储和快速检索模型参数、中间结果等关键数据；通过高效的数据组织与访问，显著提升推理效率和性能。
```

这个回答仍然不够严谨。

KV cache 更准确的解释是：

```text
KV cache 存储的是 Transformer attention 中历史 token 的 Key / Value。
它避免 decode 阶段重复计算旧 token 的 K/V，从而提升自回归生成效率。
```

模型回答里说：

```text
存储模型参数、中间结果
```

这不准确。

因此继续记录 bad case：

```text
模型服务能跑通，指标能采集，但小模型对专业概念的回答仍可能不可靠。
```

## What I Actually Verified

这一天实际验证的是：

```text
benchmark 前可以执行 warmup。
warmup 结果可以排除在正式统计之外。
可以重复进行 single-request streaming 调用。
可以记录每次 TTFT、TPOT、total_latency、output tokens/s。
可以计算 avg、p50、p95。
可以观察 completion_tokens 变化对 total_latency 的影响。
```

## What I Did Not Prove Yet

Day46 还没有证明：

```text
服务的并发能力。
QPS。
多请求 aggregate tokens/s。
长 prompt 下 TTFT 如何变化。
长 output 下 TPOT 如何变化。
prefix cache 对 TTFT 的影响。
不同 max_model_len / gpu_memory_utilization 的影响。
```

因为当前仍然是：

```text
单请求。
短 prompt。
小模型。
低重复次数。
无并发。
```

## Key Takeaway

Day46 的核心理解：

```text
warmup 可以降低一次性编译/冷启动开销对统计的影响。
重复测量比单次 smoke test 更可信。
p50 代表典型表现。
p95 代表尾部较慢表现。
输出长度会影响 total_latency，所以做严格对比时要控制 completion_tokens。
```

## Day46 Conclusion

Day46 已通过。

当前完成：

```text
Warmup + repeated single-request benchmark 跑通。
TTFT / TPOT / total_latency 的 avg、p50、p95 已记录。
发现 completion_tokens 波动会影响 total_latency。
再次发现模型专业概念回答不稳定。
```

下一步建议：

```text
Day47：构造 short prompt / long prompt / long output 三类 workload。
先不并发，分别测单请求 p50 / p95。
观察长 prompt 对 TTFT、长 output 对 total_latency / TPOT 的影响。
```

