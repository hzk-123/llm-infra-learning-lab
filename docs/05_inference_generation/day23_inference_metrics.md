# Day 23: Inference Metrics

今天只理解推理服务最常见的几个指标：

```text
TTFT
TPOT
tokens/s
QPS
total latency
```

## TTFT

TTFT = Time To First Token

含义：

```text
从请求发出，到收到第一个生成 token 的时间
```

TTFT 主要受 prefill 影响。

长 prompt 会让 TTFT 变高，因为模型需要先处理完整 prompt。

## TPOT

TPOT = Time Per Output Token

含义：

```text
decode 阶段平均每生成一个 token 花多久
```

如果生成了 `N` 个 output tokens：

```text
TPOT = decode_time / N
```

## tokens/s

tokens/s 表示每秒生成多少 output tokens。

单请求近似：

```text
tokens_per_second = output_tokens / decode_time
```

它和 TPOT 互为倒数：

```text
tokens/s = 1 / TPOT
```

注意单位要一致。

## Total Latency

总延迟：

```text
total_latency = TTFT + decode_time
```

或者：

```text
请求开始到请求结束的总时间
```

## QPS

QPS = Queries Per Second

含义：

```text
服务每秒完成多少个请求
```

QPS 更偏服务吞吐，不是单请求指标。

## 今天你要能回答

- TTFT 衡量什么？
- 为什么长 prompt 会增加 TTFT？
- TPOT 衡量什么？
- tokens/s 和 TPOT 的关系是什么？
- QPS 和 tokens/s 有什么区别？
