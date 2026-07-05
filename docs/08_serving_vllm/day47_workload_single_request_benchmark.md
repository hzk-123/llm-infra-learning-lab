# Day 47：Short / Long Prompt / Long Output Workload

## 目标

Day46 对同一个请求重复测量，得到单请求 p50 / p95。

Day47 开始引入 workload，但仍然不做并发。

这一天只比较三类单请求：

```text
short_prompt
long_prompt
long_output
```

目标是观察：

```text
长 prompt 是否更容易提高 TTFT。
长 output 是否更容易提高 total_latency。
不同 workload 的 TPOT 是否有明显差异。
```

## 三类 workload

### short_prompt

短输入、短输出。

用于观察基本请求延迟。

### long_prompt

长输入、短输出。

用于观察 prompt / prefill 对 TTFT 的影响。

### long_output

短输入、长输出。

用于观察 decode 阶段对 total latency 的影响。

## 为什么仍然不做并发

如果现在直接并发，会同时混入：

```text
workload 差异
scheduler 排队
batching
KV cache 竞争
HTTP 并发开销
```

很难判断到底是谁造成指标变化。

所以 Day47 先固定：

```text
concurrency = 1
```

只看 workload 自身差异。

## 运行方式

确保 vLLM server 正在运行。

然后：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
python exercises/08_serving_vllm/day47_workload_single_request_benchmark.py
```

可选保存 CSV：

```bash
python exercises/08_serving_vllm/day47_workload_single_request_benchmark.py \
  --out-csv results/08_serving_vllm/day47_workload_single_request.csv
```

## 输出重点

重点看每类 workload 的：

```text
prompt_tokens
completion_tokens
TTFT p50 / p95
TPOT p50 / p95
total_latency p50 / p95
output tokens/s
```

## 你应该掌握

学完 Day47 后，你应该能解释：

```text
TTFT 更容易受到 prompt 长度和 prefill 影响。
total latency 同时受到 TTFT 和输出长度影响。
TPOT 更接近 decode 阶段每 token 速度。
不同 workload 要分开测，不能混成一个平均值。
单请求 workload 对比是并发 benchmark 之前的必要准备。
```

