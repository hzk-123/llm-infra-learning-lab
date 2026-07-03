# Day 46：Warmup + Repeated Single-request Benchmark

## 目标

Day45 只跑了一次 streaming 请求，说明链路能采集：

```text
TTFT
TPOT
total_latency
token usage
```

Day46 要把单次 smoke test 升级成重复测量：

```text
先 warmup。
再正式跑多次单请求。
统计 avg / p50 / p95。
```

这仍然不是并发压测。

Day46 只回答一个问题：

```text
同一个请求，在单并发条件下，指标大概稳定在什么范围？
```

## 为什么要 warmup

第一次请求可能包含额外开销：

```text
Triton JIT
CUDA graph 未覆盖 shape
缓存冷启动
Python / HTTP 连接初始化
```

Day45 已经看到过：

```text
Triton kernel JIT compilation during inference
```

所以正式统计前，需要先跑几次 warmup。

warmup 的结果用于让系统进入相对稳定状态，不参与最终统计。

## 为什么要重复运行

单次结果不可靠。

真实服务里，指标会受这些因素影响：

```text
调度
GPU 状态
server 后台统计
网络和本机环回开销
生成长度
采样结果
```

重复多次后，可以看：

```text
平均值 avg
中位数 p50
尾部 p95
最小值 min
最大值 max
```

## Day46 仍然不做什么

Day46 不做：

```text
并发 QPS
多 workload
长 prompt 对比
长 output 对比
prefix cache 对比
多卡对比
```

这些后面再做。

## 运行方式

先确认 vLLM server 正在运行。

然后运行：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
python exercises/08_serving_vllm/day46_repeated_single_request_benchmark.py
```

如果想保存 CSV：

```bash
python exercises/08_serving_vllm/day46_repeated_single_request_benchmark.py \
  --out-csv results/08_serving_vllm/day46_single_request.csv
```

`results/` 是运行产物目录，已被 `.gitignore` 忽略，不会提交到 Git。

## 输出重点

重点看：

```text
warmup_runs
measured_runs
TTFT avg / p50 / p95
TPOT avg / p50 / p95
total_latency avg / p50 / p95
output tokens/s avg / p50 / p95
finish_reason
completion_tokens
```

## 你应该掌握

学完 Day46 后，你应该能解释：

```text
为什么 benchmark 前要 warmup。
为什么不能拿单次 TTFT / TPOT 当结论。
p50 表示典型请求表现。
p95 表示尾部较慢请求表现。
单请求 benchmark 只能看单并发稳定性，不能代表系统并发吞吐。
```

