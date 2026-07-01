# Daily Log

Date: 2026-06-17

## Learned

- 推理 benchmark 不能只测一种 prompt，需要覆盖不同请求类型。
- `short_prompt` 主要检查基础延迟。
- `long_prompt` 主要给 prefill 和 TTFT 施压。
- `long_output` 主要给 decode、TPOT、tokens/s 施压。
- `mixed` 更接近真实服务流量。
- concurrency 增大会带来调度压力和 KV cache 显存压力。
- 当前脚本里的 pressure estimate 是单请求压力；实际并发总压力还要乘以 concurrency。

## Ran

```text
python exercises\05_inference_generation\day24_benchmark_design.py
```

## My Explanation

这次 benchmark 设计了 4 类 workload：

```text
short_prompt:  prompt_tokens=32,   max_new_tokens=32
long_prompt:   prompt_tokens=2048, max_new_tokens=32
long_output:   prompt_tokens=32,   max_new_tokens=256
mixed:         prompt_tokens=512,  max_new_tokens=128
```

`short_prompt` 的压力估计是：

```text
prefill_pressure: 32
decode_pressure: 32
kv_cache_pressure: 64
```

它 prompt 和输出都短，适合检查基础服务延迟。

`long_prompt` 的压力估计是：

```text
prefill_pressure: 2048
decode_pressure: 32
kv_cache_pressure: 2080
```

它 prompt 很长、输出较短，所以主要测试 prefill 阶段和 TTFT。长 prompt 需要先处理大量输入 token，因此首 token 延迟通常会更高。

`long_output` 的压力估计是：

```text
prefill_pressure: 32
decode_pressure: 256
kv_cache_pressure: 288
```

它 prompt 短、输出长，所以主要测试 decode 阶段、TPOT 和 tokens/s。输出越长，decode step 越多。

`mixed` 的压力估计是：

```text
prefill_pressure: 512
decode_pressure: 128
kv_cache_pressure: 640
```

它同时包含中等长度 prompt 和中等长度输出，更接近真实服务中的混合流量。

脚本分别展示了 concurrency 为 1、4、8 时的 workload。当前每条请求的 pressure estimate 没有变化，因为这些数值是单请求视角：

```text
kv_cache_pressure = prompt_tokens + max_new_tokens
```

但真实服务里，总压力会随着并发增大。例如 `long_prompt` 的单请求 KV cache pressure 是 2080：

```text
concurrency=1 -> total pressure approx 2080
concurrency=4 -> total pressure approx 8320
concurrency=8 -> total pressure approx 16640
```

这说明 concurrency 不只是让请求更多，也会让调度和 KV cache 显存压力变大。

这次 Day 24 的核心是：设计 benchmark 时要把请求分型，才能判断瓶颈来自哪里。

```text
long_prompt -> prefill / TTFT
long_output -> decode / TPOT / tokens/s
mixed       -> 真实流量近似
concurrency -> 调度和显存压力
```

## Next

- 进入 toy benchmark runner：给这些 workload 补上模拟的 TTFT、TPOT、latency、tokens/s、QPS 指标。
