# Daily Log

Date: 2026-06-17

## Learned

- 并发 benchmark 要区分整体吞吐和单请求体验。
- `wall_time` 表示跑完整批请求花了多久。
- `QPS = total_requests / wall_time`。
- `aggregate output tokens/s = total_output_tokens / wall_time`。
- `avg latency` 表示平均单请求延迟。
- `p95 latency` 表示 95% 请求以内的最大延迟，更能反映慢请求和尾延迟。
- 并发升高可能提升整体吞吐，但也可能让单请求延迟变差。

## Ran

```text
python exercises\05_inference_generation\day27_concurrent_benchmark.py
```

## My Explanation

这次 Day27 用同一批 12 个请求，分别测试了 3 个并发级别：

```text
concurrency = 1
concurrency = 2
concurrency = 4
```

输出的聚合指标是：

```text
concurrency=1:
  wall_time = 4.244s
  QPS = 2.83
  output tokens/s = 45.24
  avg TTFT = 0.083s
  avg latency = 0.353s
  p95 latency = 0.606s

concurrency=2:
  wall_time = 3.453s
  QPS = 3.48
  output tokens/s = 55.61
  avg TTFT = 0.100s
  avg latency = 0.558s
  p95 latency = 1.023s

concurrency=4:
  wall_time = 2.141s
  QPS = 5.61
  output tokens/s = 89.69
  avg TTFT = 0.116s
  avg latency = 0.581s
  p95 latency = 1.040s
```

从整体吞吐看，并发越高，wall time 越低：

```text
4.244s -> 3.453s -> 2.141s
```

同样是 12 个请求，`concurrency=4` 比 `concurrency=1` 更快跑完整批请求。

QPS 也随并发提高：

```text
2.83 -> 3.48 -> 5.61
```

这说明并发能提升服务整体吞吐。

但是单请求体验变差了。平均延迟从 `0.353s` 上升到 `0.581s`：

```text
avg latency:
0.353s -> 0.558s -> 0.581s
```

平均 TTFT 也上升：

```text
avg TTFT:
0.083s -> 0.100s -> 0.116s
```

这说明并发请求之间存在资源竞争。即使整体跑得更快，每个请求自己感受到的等待时间可能变长。

最值得关注的是 p95 latency：

```text
p95 latency:
0.606s -> 1.023s -> 1.040s
```

p95 比 avg latency 更能反映慢请求。比如 `concurrency=4` 时：

```text
avg latency = 0.581s
p95 latency = 1.040s
```

如果只看平均值，会觉得请求平均 0.58 秒完成；但 p95 告诉我们，尾部慢请求可能已经超过 1 秒。

这就是线上服务为什么常看 p95 / p99，而不是只看平均延迟。

样例请求中，`long_output` 的 latency 大约是 `1.04s`：

```text
long_output latency: about 1.04s
```

它输出 32 个 tokens，比 `long_prompt` 的 8 个 tokens 更多，所以 decode 时间更长，成为慢请求来源之一。

`long_prompt` 的 TTFT 更高：

```text
long_prompt TTFT: about 0.187s
long_output TTFT: about 0.078s
```

这符合之前的结论：

```text
long prompt -> higher TTFT
long output -> higher total latency
```

Day27 的关键结论是：

```text
concurrency up -> wall_time down
concurrency up -> QPS up
concurrency up -> avg latency / p95 latency may go up
```

所以推理服务压测不是追求单一指标最大，而是在吞吐和延迟之间找平衡。

## Next

- 进入真实 HTTP benchmark 的准备：先写一个本地 fake streaming server，再用 benchmark client 采集 TTFT、TPOT、latency、QPS。
