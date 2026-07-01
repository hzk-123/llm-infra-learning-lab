# Day 24: Benchmark Design

今天只理解一个问题：

> 如何设计推理服务 benchmark，才能知道瓶颈在哪里？

## 为什么不能只测一个 prompt

真实推理服务会遇到不同请求：

- 短 prompt，短输出
- 长 prompt，短输出
- 短 prompt，长输出
- 长 prompt，长输出
- 混合请求
- 并发请求

只测一个 prompt，无法判断问题来自 prefill、decode、KV cache、还是调度。

## 四类基础 workload

### 1. short_prompt

特点：

```text
prompt_tokens 少
max_new_tokens 少
```

主要用于检查基础服务延迟。

### 2. long_prompt

特点：

```text
prompt_tokens 多
max_new_tokens 少
```

主要测试 prefill 和 TTFT。

### 3. long_output

特点：

```text
prompt_tokens 少
max_new_tokens 多
```

主要测试 decode、TPOT、tokens/s。

### 4. mixed

特点：

```text
不同 prompt 长度和输出长度混在一起
```

更接近真实线上服务。

## 并发

同一组 workload 要在不同 concurrency 下测试：

```text
1, 2, 4, 8, 16 ...
```

并发越高，调度和 KV cache 显存压力越大。

## 每条请求需要记录什么

```text
request_id
workload_type
prompt_tokens
max_new_tokens
concurrency
```

真实 benchmark 后还要补：

```text
TTFT
TPOT
tokens/s
total_latency
success
error
```

## 今天你要能回答

- short prompt 主要测什么？
- long prompt 主要测什么？
- long output 主要测什么？
- mixed workload 为什么重要？
- concurrency 会给系统带来什么压力？
