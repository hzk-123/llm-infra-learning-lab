# Day 48：Concurrent Workload Benchmark

## 目标

Day47 比较了不同 workload 在单请求下的表现。

Day48 开始做第一版并发 benchmark：

```text
concurrency = 1 / 2 / 4 / 8
workload = short_prompt / long_prompt / long_output
```

关注指标：

```text
QPS
aggregate output tokens/s
TTFT p50 / p95
TPOT p50 / p95
total latency p50 / p95
success rate
```

## 为什么这一步重要

LLM serving 的价值不只是单请求快。

真实服务更关心：

```text
并发上来后，吞吐能否提高？
p95 latency 是否变差？
长 prompt 是否更容易拖慢 TTFT？
长 output 是否占用更多 decode 资源？
```

Day48 是第一次观察 vLLM continuous batching 在真实服务里的表现。

## Day48 仍然不做什么

Day48 不改 vLLM 参数：

```text
不改 max_model_len
不改 gpu_memory_utilization
不测 prefix cache
不测量化
不测多卡
```

只固定当前 server，观察 workload 和 concurrency 的影响。

## 运行方式

确保 vLLM server 正在运行：

```bash
curl http://127.0.0.1:8000/v1/models
```

然后运行：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
python exercises/08_serving_vllm/day48_concurrent_workload_benchmark.py
```

可选保存 CSV：

```bash
python exercises/08_serving_vllm/day48_concurrent_workload_benchmark.py \
  --out-csv results/08_serving_vllm/day48_concurrent_workload.csv
```

## 输出重点

重点看：

```text
QPS 是否随 concurrency 上升
aggregate output tokens/s 是否随 concurrency 上升
TTFT p95 是否变差
latency p95 是否变差
不同 workload 的差异
```

## 你应该掌握

学完 Day48 后，你应该能解释：

```text
QPS 是请求吞吐。
aggregate output tokens/s 是整体输出 token 吞吐。
p95 latency 比平均值更能反映慢请求体验。
并发提升吞吐的同时，可能让单请求 latency 变差。
不同 workload 不能混在一起平均，必须分开看。
```

