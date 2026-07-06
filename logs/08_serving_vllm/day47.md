# Daily Log

Date: 2026-07-05

## Topic

Day47：Short Prompt / Long Prompt / Long Output 单请求 workload benchmark。

这一天的目标是：

```text
把请求拆成不同 workload。
分别测 short_prompt、long_prompt、long_output。
观察 prompt 长度、输出长度对 TTFT / TPOT / total latency 的影响。
```

这仍然不是并发 benchmark。

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
python exercises/08_serving_vllm/day47_workload_single_request_benchmark.py
```

## Config

脚本配置：

```text
base_url: http://127.0.0.1:8000
model: qwen2.5-0.5b-instruct
temperature: 0.0
timeout: 90.0s
warmup_runs_per_workload: 1
measured_runs_per_workload: 5
sleep_between_runs: 0.2s
request_stream_usage: True
```

这次设置：

```text
temperature = 0.0
```

目的是让输出尽量稳定，减少采样随机性对 benchmark 的影响。

## Workloads

三类 workload：

```text
short_prompt:
  prompt_chars = 16
  max_tokens = 48
  目标：短输入、短输出。

long_prompt:
  prompt_chars = 2891
  max_tokens = 48
  目标：长输入、短输出。

long_output:
  prompt_chars = 29
  max_tokens = 192
  目标：短输入、长输出。
```

## Preflight

`/v1/models` 检查：

```text
latency: 0.014s
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

### short_prompt warmup

```text
TTFT: 0.257s
TPOT: 0.0026s/token
total_latency: 0.323s
prompt_tokens: 38
completion_tokens: 26
finish_reason: stop
```

### long_prompt warmup

```text
TTFT: 0.048s
TPOT: 0.0027s/token
total_latency: 0.088s
prompt_tokens: 1673
completion_tokens: 16
finish_reason: stop
```

### long_output warmup

```text
TTFT: 0.028s
TPOT: 0.0029s/token
total_latency: 0.349s
prompt_tokens: 49
completion_tokens: 113
finish_reason: stop
```

观察：

```text
short_prompt 的 warmup TTFT 明显偏高。
```

这说明 warmup 的确会吸收一些冷启动 / shape 覆盖 / kernel 编译等额外开销。

所以正式 summary 不包含 warmup。

## Workload Summary

正式测量结果：

```text
short_prompt:
  avg prompt tokens = 38.0
  avg completion tokens = 26.0
  TTFT p50 = 0.0210s
  TTFT p95 = 0.0237s
  TPOT p50 = 0.0032s/token
  TPOT p95 = 0.0041s/token
  latency p50 = 0.1000s
  latency p95 = 0.1230s
  avg output tokens/s = 254.93

long_prompt:
  avg prompt tokens = 1673.0
  avg completion tokens = 16.0
  TTFT p50 = 0.0332s
  TTFT p95 = 0.0360s
  TPOT p50 = 0.0040s/token
  TPOT p95 = 0.0045s/token
  latency p50 = 0.0916s
  latency p95 = 0.1017s
  avg output tokens/s = 185.29

long_output:
  avg prompt tokens = 49.0
  avg completion tokens = 113.0
  TTFT p50 = 0.0177s
  TTFT p95 = 0.0199s
  TPOT p50 = 0.0027s/token
  TPOT p95 = 0.0030s/token
  latency p50 = 0.3255s
  latency p95 = 0.3516s
  avg output tokens/s = 342.79
```

## Observations

### 1. Long prompt 的 TTFT 更高，但没有高很多

对比：

```text
short_prompt:
  prompt_tokens = 38
  TTFT p50 = 0.0210s

long_prompt:
  prompt_tokens = 1673
  TTFT p50 = 0.0332s
```

prompt tokens 从：

```text
38 -> 1673
```

增加很多，但 TTFT 只从：

```text
21.0ms -> 33.2ms
```

原因可能是：

```text
模型很小：Qwen2.5-0.5B。
单请求，没有排队。
使用 FlashAttention / vLLM 优化后，1673 tokens 的 prefill 对这张 4090 压力不大。
long_prompt 生成很短，只有 16 tokens。
```

所以这次实验支持：

```text
长 prompt 会提高 TTFT。
```

但在这个小模型和单请求条件下，提升幅度不大。

### 2. Long output 明显提高 total latency

对比：

```text
short_prompt:
  completion_tokens = 26
  latency p50 = 0.1000s

long_output:
  completion_tokens = 113
  latency p50 = 0.3255s
```

completion tokens 从：

```text
26 -> 113
```

total latency 从：

```text
100ms -> 325.5ms
```

这符合预期：

```text
输出越长，decode step 越多，总耗时越高。
```

### 3. TPOT 在不同 workload 下有差异

对比：

```text
short_prompt TPOT p50 = 0.0032s/token
long_prompt  TPOT p50 = 0.0040s/token
long_output  TPOT p50 = 0.0027s/token
```

`long_prompt` 的 TPOT 略高。

可能原因：

```text
decode 每一步都需要 attend 到已有 KV cache。
long_prompt 的 cache length 更长。
所以每个 decode step 读取的 K/V 更长，TPOT 可能变高。
```

这和之前 Day35 的 decode attention scaling 对应：

```text
cache_len 越长，单步 decode 需要读的 K/V 越多。
```

### 4. output tokens/s 不能孤立比较

结果：

```text
short_prompt avg output tokens/s = 254.93
long_prompt  avg output tokens/s = 185.29
long_output  avg output tokens/s = 342.79
```

这里 `long_output` 最高，不代表它“整体更快”。

原因是：

```text
output tokens/s = completion_tokens / total_latency
```

长输出时，固定开销被更多 tokens 摊薄，所以 output tokens/s 可能更高。

因此后面写报告时不能只看一个指标。

要同时看：

```text
TTFT
TPOT
total_latency
completion_tokens
output tokens/s
```

## Sample Outputs And Bad Cases

### short_prompt

输出：

```text
VLLM是基于Transformer架构的超大规模语言模型，能够理解和生成各种复杂和长篇复杂的文本内容。
```

问题：

```text
还是把 vLLM 说成了语言模型。
```

正确说法：

```text
vLLM 是大模型推理服务框架，不是模型本身。
```

### long_prompt

输出：

```text
KV cache 在 decode 阶段的主要解决问题是提高查询速度。
```

这个回答过于模糊。

更准确的说法：

```text
KV cache 缓存历史 token 的 Key / Value，避免 decode 阶段重复计算旧 token 的 K/V。
```

### long_output

输出中包含：

```text
使用更高效的模型架构和参数设置。
调整推理环境的资源配置。
增加数据量以提升模型训练速度。
```

问题：

```text
把训练数据量和推理服务优化混在了一起。
```

这再次说明：

```text
小模型输出可以用于测试 serving 链路，但不适合直接当专业答案。
```

## What I Actually Verified

这一天实际验证的是：

```text
可以构造多类 workload。
可以分别对每类 workload 做 warmup 和 repeated single-request benchmark。
可以比较不同 prompt length 和 output length 下的 TTFT / TPOT / latency。
可以看到 long output 明显提高 total latency。
可以看到 long prompt 对 TTFT 有影响，但在当前小模型下影响不大。
```

## What I Did Not Prove Yet

Day47 没有证明：

```text
并发下的 QPS。
并发下的 p95 latency。
continuous batching 效果。
prefix cache 命中效果。
多卡 tensor parallel 效果。
更大模型下的长 prompt prefill 压力。
```

因为当前条件是：

```text
单请求。
小模型。
无并发。
固定 server 参数。
```

## Key Takeaway

Day47 的核心理解：

```text
不同 workload 必须分开测。
长 prompt 主要影响 prefill 和 TTFT。
长 output 主要增加 decode steps 和 total latency。
TPOT 会受 cache length 影响。
output tokens/s 不能脱离 completion_tokens 和 total_latency 单独解读。
```

## Day47 Conclusion

Day47 已通过。

当前完成：

```text
short_prompt / long_prompt / long_output 三类 workload 已跑通。
每类 workload 都有 warmup 和 repeated measured runs。
已得到每类 workload 的 TTFT / TPOT / latency p50/p95。
已观察到 long_output 明显增加 total latency。
已观察到 long_prompt 的 TTFT 更高但幅度不大。
继续记录模型专业回答 bad case。
```

下一步建议：

```text
Day48：单机并发 benchmark。
在 short_prompt / long_prompt / long_output 上分别测试 concurrency=1/2/4/8。
观察 QPS、aggregate output tokens/s、p95 latency 的变化。
```

