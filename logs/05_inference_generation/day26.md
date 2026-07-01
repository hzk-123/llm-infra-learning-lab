# Daily Log

Date: 2026-06-17

## Learned

- 真实 benchmark 不是只靠公式估算，而是通过时间戳采集指标。
- `request_start` 表示客户端发出请求的时间。
- `prefill_done` 表示 prompt 处理完成的时间。
- `first_token_time` 表示第一个生成 token 到达的时间。
- `request_end` 表示整个请求结束的时间。
- `TTFT = first_token_time - request_start`。
- `total_latency = request_end - request_start`。
- `TPOT` 可以用 decode 阶段耗时除以输出 token 数估算。

## Ran

```text
python exercises\05_inference_generation\day26_request_timing.py
```

## My Explanation

这次 Day26 从 toy 公式推进到了真实 benchmark 的核心思路：用时间戳记录请求生命周期。

脚本模拟了 3 类请求：

```text
short_prompt: prompt_tokens=32,  output_tokens=8
long_prompt:  prompt_tokens=512, output_tokens=8
long_output:  prompt_tokens=32,  output_tokens=32
```

输出结果里，`short_prompt` 的指标是：

```text
TTFT: 0.054s
TPOT: 0.016s/token
total_latency: 0.163s
output tokens/s: 64.25
```

`long_prompt` 的指标是：

```text
TTFT: 0.153s
TPOT: 0.016s/token
total_latency: 0.263s
output tokens/s: 64.28
```

它和 `short_prompt` 的 output_tokens 都是 8，但是 `long_prompt` 的 TTFT 更高。

原因是 `long_prompt` 有 512 个 prompt tokens，prefill 阶段更长。输出中的事件时间也验证了这一点：

```text
request_start:    0.000s
prefill_done:     0.138s
first_token_time: 0.153s
request_end:      0.263s
```

这里可以拆成：

```text
prefill cost approx = prefill_done - request_start
                   = 0.138s

first decode step approx = first_token_time - prefill_done
                         = 0.015s

TTFT = first_token_time - request_start
     = 0.153s
```

所以 TTFT 不是单纯的 prefill 时间，它还包含“生成第一个 token”的那一步 decode 时间。

`long_output` 的指标是：

```text
TTFT: 0.046s
TPOT: 0.016s/token
total_latency: 0.557s
output tokens/s: 60.83
```

它的 prompt 很短，所以 TTFT 不高；但是 output_tokens 是 32，比另外两个请求的 8 更长，因此 total latency 更高。

这说明：

```text
long prompt  -> 更高 TTFT
long output  -> 更高 total latency
```

最后的整体指标是：

```text
total_requests: 3
total_output_tokens: 48
measured_wall_time: 0.983s
QPS: 3.05
aggregate output tokens/s: 48.82
```

这里的 `measured_wall_time` 是真实计时得到的，不是手动估算。它表示整个 benchmark 从开始到结束花了多久。

当前脚本是串行执行 3 个请求，所以 wall time 接近三个请求耗时之和。

真实服务压测中，如果请求并发发出，wall time 不再是简单相加，而会受到服务端调度、batching、KV cache、GPU 利用率等因素影响。

Day26 的关键收获是：真实 benchmark 的底层也是这几个时间点。

```text
request_start
first_token_time
request_end
```

有了这些时间戳，就能计算：

```text
TTFT
TPOT
total latency
QPS
tokens/s
```

后面接真实 vLLM/OpenAI-compatible server，本质上就是把这里的 `time.sleep()` 换成 HTTP streaming response。

## Next

- 进入并发 benchmark：用多个请求同时运行，观察 wall time、QPS 和单请求 latency 的变化。
