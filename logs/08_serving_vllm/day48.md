# Daily Log

Date: 2026-07-06

## Topic

Day48：Concurrent Workload Benchmark。

这一天的目标是：

```text
在同一个 vLLM server 上测试不同 workload 的并发表现。
比较 concurrency = 1 / 2 / 4 / 8。
观察 QPS、aggregate output tokens/s、TTFT p95、latency p95 的变化。
```

这是第一次真正做并发 benchmark。

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
python exercises/08_serving_vllm/day48_concurrent_workload_benchmark.py
```

## Config

脚本配置：

```text
base_url: http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
temperature: 0.0
timeout: 120.0s
concurrency_levels: [1, 2, 4, 8]
requests_per_level: 8
warmup_runs_per_workload: 1
workloads: short_prompt, long_prompt, long_output
request_stream_usage: True
```

这次固定：

```text
同一个 server
同一个模型
同一张 GPU
同一组 workload
```

只改变：

```text
concurrency
```

## Workloads

三类 workload：

```text
short_prompt:
  max_tokens = 48
  prompt_chars = 16
  目标：短输入、短输出。

long_prompt:
  max_tokens = 48
  prompt_chars = 2891
  目标：长输入、短输出。

long_output:
  max_tokens = 192
  prompt_chars = 29
  目标：短输入、长输出。
```

## Preflight

`/v1/models` 检查：

```text
latency: 0.009s
requested_model_found: True
model_ids: ['qwen2.5-0.5b-instruct']
```

说明：

```text
vLLM server 正常运行。
模型名正确。
benchmark client 可以访问本地 API。
```

## Warmup

warmup 结果：

```text
short_prompt:
  TTFT = 0.0448s
  latency = 0.1124s
  completion_tokens = 26

long_prompt:
  TTFT = 0.0320s
  latency = 0.0620s
  completion_tokens = 16

long_output:
  TTFT = 0.0139s
  latency = 0.2439s
  completion_tokens = 113
```

warmup 不进入正式统计。

## Result Summary

### short_prompt

```text
concurrency 1:
  QPS = 14.35
  agg output tokens/s = 373.00
  TTFT p95 = 0.0176s
  latency p95 = 0.0704s

concurrency 2:
  QPS = 25.14
  agg output tokens/s = 653.69
  TTFT p95 = 0.0267s
  latency p95 = 0.0805s

concurrency 4:
  QPS = 41.51
  agg output tokens/s = 1063.78
  TTFT p95 = 0.0398s
  latency p95 = 0.1003s

concurrency 8:
  QPS = 64.53
  agg output tokens/s = 1653.50
  TTFT p95 = 0.0582s
  latency p95 = 0.1186s
```

观察：

```text
QPS 随 concurrency 明显上升。
aggregate output tokens/s 明显上升。
TTFT p95 和 latency p95 也随 concurrency 上升。
```

这说明：

```text
吞吐提高了，但单请求尾延迟变差。
```

这是 serving benchmark 里非常常见的 tradeoff。

### long_prompt

```text
concurrency 1:
  QPS = 17.01
  agg output tokens/s = 272.21
  TTFT p95 = 0.0275s
  latency p95 = 0.0680s

concurrency 2:
  QPS = 27.80
  agg output tokens/s = 444.79
  TTFT p95 = 0.0425s
  latency p95 = 0.0742s

concurrency 4:
  QPS = 35.88
  agg output tokens/s = 574.04
  TTFT p95 = 0.0701s
  latency p95 = 0.1168s

concurrency 8:
  QPS = 51.58
  agg output tokens/s = 825.34
  TTFT p95 = 0.1081s
  latency p95 = 0.1495s
```

观察：

```text
long_prompt 的 TTFT p95 对 concurrency 更敏感。
concurrency 8 时 TTFT p95 到 108.1ms。
```

原因可能是：

```text
long_prompt 的 prompt_tokens = 1673。
长 prompt 的 prefill 更重。
并发下多个长 prompt 同时进入调度，prefill 压力更明显。
```

这和之前的理论对应：

```text
长 prompt 主要影响 prefill 和 TTFT。
并发长 prompt 会放大 prefill 压力。
```

### long_output

```text
concurrency 1:
  QPS = 3.53
  agg output tokens/s = 399.21
  TTFT p95 = 0.0182s
  latency p95 = 0.3246s

concurrency 2:
  QPS = 6.76
  agg output tokens/s = 804.41
  TTFT p95 = 0.0226s
  latency p95 = 0.3325s

concurrency 4:
  QPS = 15.35
  agg output tokens/s = 1826.41
  TTFT p95 = 0.0317s
  latency p95 = 0.2691s

concurrency 8:
  QPS = 29.81
  agg output tokens/s = 3546.89
  TTFT p95 = 0.0224s
  latency p95 = 0.2634s
```

观察：

```text
long_output 的 QPS 随 concurrency 明显提升。
aggregate output tokens/s 从 399.21 提升到 3546.89。
latency p95 没有随 concurrency 明显恶化，甚至在 concurrency 4/8 更低。
```

这很有意思。

可能原因：

```text
long_output 每个请求有较长 decode 阶段。
多个 decode 请求一起跑时，vLLM continuous batching 能更好地把 decode step 组成 batch。
批处理让 GPU 利用率更高。
固定开销被更多 tokens 摊薄。
```

这说明：

```text
对长输出 workload，并发可能显著提升整体吞吐。
```

但注意：

```text
这只是当前小模型、当前请求数量、当前服务器配置下的结果。
不能直接泛化到所有模型和所有并发。
```

## Cross-workload Comparison

### QPS

concurrency 8：

```text
short_prompt QPS = 64.53
long_prompt  QPS = 51.58
long_output  QPS = 29.81
```

解释：

```text
short_prompt 单请求最短，所以 QPS 最高。
long_prompt prefill 重，QPS 低一些。
long_output decode 长，每个请求生成更多 token，所以 QPS 最低。
```

### Aggregate output tokens/s

concurrency 8：

```text
short_prompt agg output tokens/s = 1653.50
long_prompt  agg output tokens/s = 825.34
long_output  agg output tokens/s = 3546.89
```

解释：

```text
long_output 虽然 QPS 低，但每个请求输出 token 多。
所以 aggregate output tokens/s 最高。
```

这就是为什么要同时看：

```text
QPS
aggregate output tokens/s
latency
completion_tokens
```

不能只看一个指标。

### Tail latency

concurrency 8：

```text
short_prompt latency p95 = 0.1186s
long_prompt  latency p95 = 0.1495s
long_output  latency p95 = 0.2634s
```

解释：

```text
long_output 生成 token 多，所以 total latency p95 最高。
long_prompt 的 TTFT p95 高，但输出短，所以 total latency 仍低于 long_output。
```

## Success Rate

所有组合：

```text
success = 100%
```

说明：

```text
concurrency=1/2/4/8 下，当前小模型服务稳定返回。
没有 timeout。
没有 HTTP error。
没有请求失败。
```

这是并发 benchmark 的基本验收。

## What I Actually Verified

这一天实际验证的是：

```text
Python client 可以并发发起 streaming 请求。
vLLM server 能处理 concurrency=1/2/4/8。
每个 workload 都能统计 QPS。
每个 workload 都能统计 aggregate output tokens/s。
每个 workload 都能统计 TTFT / TPOT / latency p50/p95。
并发提升能明显提高吞吐。
并发也可能让尾延迟变差。
不同 workload 的 QPS 和 token throughput 不能混为一谈。
```

## What I Did Not Prove Yet

Day48 还没有证明：

```text
更高 concurrency 下的极限吞吐。
不同 max_model_len 的影响。
不同 gpu_memory_utilization 的影响。
prefix cache 的真实收益。
量化模型的显存和吞吐变化。
多卡 tensor parallel 的收益。
更大模型如 1.5B / 7B 的服务表现。
```

因为当前实验固定：

```text
Qwen2.5-0.5B
单卡
默认 vLLM 参数
concurrency 最高 8
每组 8 个请求
```

## Key Takeaway

Day48 的核心理解：

```text
并发 benchmark 要同时看吞吐和延迟。
QPS 上升不代表用户体验一定更好。
aggregate output tokens/s 高不代表请求完成更快。
p95 latency 是观察尾部慢请求的重要指标。
不同 workload 必须分开看。
long prompt 更容易影响 TTFT。
long output 更容易影响 total latency 和 aggregate token throughput。
```

## Day48 Conclusion

Day48 已通过。

当前完成：

```text
short_prompt / long_prompt / long_output 三类 workload 的并发 benchmark。
concurrency=1/2/4/8 全部跑通。
所有组合 success rate = 100%。
观察到并发提升 QPS 和 aggregate output tokens/s。
观察到 short/long_prompt 的 p95 latency 随并发上升。
观察到 long_output 在并发下 token throughput 显著提升。
```

下一步建议：

```text
Day49：vLLM 参数实验。
先选择一个 workload，例如 long_prompt。
对比 max_model_len / gpu_memory_utilization / prefix cache 开关。
不要同时改太多变量。
```

