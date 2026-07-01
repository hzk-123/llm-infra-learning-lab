# Daily Log

Date: 2026-06-16

## Learned

- TTFT 表示从请求开始到第一个输出 token 出来的时间，主要反映 prefill 成本。
- TPOT 表示 decode 阶段平均每生成一个 token 的时间。
- output tokens/s 表示每秒生成多少输出 token。
- total latency 是整个请求从开始到结束的总耗时。
- QPS 表示服务每秒完成多少请求，衡量请求吞吐。
- prompt 越长，通常 TTFT 越高；output 越长，通常 decode time 和 total latency 越高。

## Ran

```text
python exercises\05_inference_generation\day23_inference_metrics.py
```

## My Explanation

这次模拟了 3 类请求：

```text
short_prompt
long_prompt
long_output
```

`short_prompt` 的结果：

```text
prompt_tokens: 64
output_tokens: 20
TTFT: 0.200s
decode_time: 1.000s
TPOT: 0.050s/token
output tokens/s: 20.00
total_latency: 1.200s
```

`long_prompt` 的结果：

```text
prompt_tokens: 4096
output_tokens: 20
TTFT: 1.800s
decode_time: 1.000s
TPOT: 0.050s/token
output tokens/s: 20.00
total_latency: 2.800s
```

`short_prompt` 和 `long_prompt` 的 output tokens 都是 20，decode time 也都是 1.000s，所以 TPOT 都是：

```text
0.050s/token
```

但 `long_prompt` 的 TTFT 从 `0.200s` 增加到 `1.800s`。这说明长 prompt 主要增加 prefill 成本，让第一个 token 更晚出来。

`long_output` 的结果：

```text
prompt_tokens: 64
output_tokens: 100
TTFT: 0.200s
decode_time: 5.000s
TPOT: 0.050s/token
output tokens/s: 20.00
total_latency: 5.200s
```

它的 prompt 不长，所以 TTFT 仍然是 `0.200s`。但 output tokens 从 20 增加到 100，所以 decode time 从 1.000s 增加到 5.000s，total latency 也变成 5.200s。

TPOT 和 output tokens/s 的关系是：

```text
TPOT = 0.050s/token
tokens/s = 1 / 0.050 = 20 tokens/s
```

聚合吞吐结果：

```text
total_requests: 3
wall_time: 5.200s
QPS: 0.58
aggregate output tokens/s: 26.92
```

这里 QPS 衡量的是请求吞吐，output tokens/s 衡量的是 token 吞吐。二者不是同一个东西：一个请求可能输出很少 token，也可能输出很多 token。

核心区别：

```text
TTFT: 首 token 延迟，主要看 prefill
TPOT: 每 token decode 延迟
tokens/s: token 生成吞吐
QPS: 请求完成吞吐
```

## Next

- 学 benchmark 设计：如何构造 short prompt、long prompt、long output、并发请求来测试推理服务。
