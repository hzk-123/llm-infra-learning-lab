# Daily Log

Date: 2026-06-17

## Learned

- benchmark 结果要同时看单请求指标和整体吞吐指标。
- `TTFT` 主要受 prompt length / prefill 影响。
- `TPOT` 表示 decode 阶段每生成一个 token 的平均耗时。
- `total_latency` 大致可以理解为 `TTFT + output_tokens * TPOT`。
- `output tokens/s` 是单请求 decode 速度。
- `QPS` 和 `aggregate output tokens/s` 是 workload 级别的整体吞吐。
- 并发升高时，整体吞吐可能变高，但单请求延迟也可能变差。

## Ran

```text
python exercises\05_inference_generation\day25_toy_benchmark_runner.py
```

## My Explanation

这次 Day25 不是跑真实模型，而是用 toy benchmark 模拟不同 workload 的推理指标。

脚本测试了 3 个并发级别：

```text
concurrency = 1
concurrency = 4
concurrency = 8
```

每个并发级别下都有 4 类请求：

```text
short_prompt: prompt_tokens=32, output_tokens=32
long_prompt:  prompt_tokens=2048, output_tokens=32
long_output:  prompt_tokens=32, output_tokens=256
mixed:        prompt_tokens=512, output_tokens=128
```

在 `concurrency=1` 时，`long_prompt` 的 TTFT 是 `1.206s`，明显高于 `short_prompt` 的 `0.098s`：

```text
short_prompt TTFT: 0.098s
long_prompt  TTFT: 1.206s
```

原因是 `long_prompt` 有 2048 个 prompt tokens。prefill 阶段需要处理整个 prompt，所以 prompt 越长，首 token 延迟通常越高。

`long_output` 的 TTFT 不高，只有 `0.098s`，但 total latency 达到 `9.058s`：

```text
long_output TTFT: 0.098s
long_output TPOT: 0.035s/token
long_output output_tokens: 256
long_output total_latency: 9.058s
```

这是因为它输出 token 多。decode 是一个 token 一个 token 地生成，所以长输出会拉长总耗时。

单请求的 total latency 可以近似理解为：

```text
total_latency = TTFT + output_tokens * TPOT
```

以 `long_output` 为例：

```text
0.098 + 256 * 0.035 = 9.058
```

这个结果和脚本输出一致。

并发升高后，单请求指标变差：

```text
concurrency=1: short_prompt total_latency = 1.218s
concurrency=4: short_prompt total_latency = 1.510s
concurrency=8: short_prompt total_latency = 1.899s
```

这是因为 toy 模型里加入了 concurrency slowdown，用来模拟并发升高后调度、计算、显存等资源竞争带来的额外开销。

但整体吞吐变好了：

```text
concurrency=1: QPS = 0.44, aggregate output tokens/s = 49.46
concurrency=4: QPS = 1.42, aggregate output tokens/s = 159.55
concurrency=8: QPS = 2.26, aggregate output tokens/s = 253.65
```

这说明服务系统里经常存在一个核心 tradeoff：

```text
更高并发 -> 更高整体吞吐
更高并发 -> 更差单请求延迟
```

所以做推理服务压测时，不能只看一个指标。

如果只看 QPS，可能忽略用户等待时间变长。
如果只看单请求 latency，可能忽略服务整体吞吐能力。

真正的 benchmark 应该同时报告：

```text
TTFT
TPOT
total latency
QPS
tokens/s
p50 / p95 latency
GPU memory
success rate
```

这次脚本还有一个重要限制：它是 toy benchmark。这里的 `estimated_wall_time` 是根据模拟 latency 算出来的，不是真实服务的实际耗时。

后面接入真实 vLLM server 后，wall time、TTFT、TPOT、QPS 都应该从真实请求时间中采集。

## Next

- 进入真实 benchmark 的准备阶段：学习如何用 Python 记录 request start、first token time、end time。
