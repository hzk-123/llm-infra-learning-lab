# Day 41：Prefix Cache Toy

## 目标

今天只看一个问题：

```text
多个请求有相同 prompt 前缀时，为什么可以减少重复 prefill？
```

Day34 学过：

```text
prefill 处理完整 prompt。
prompt 越长，TTFT 通常越高。
```

Prefix cache 的直觉是：

```text
如果一个新请求的 prompt 前缀，之前已经算过 KV cache，
那么这部分前缀不需要重新 prefill。
```

## Toy 设定

脚本把 prompt token 按固定大小切成 blocks：

```text
block_size = 2
```

例如：

```text
[10, 11, 12, 13, 201, 202]
```

会切成：

```text
[10, 11] [12, 13] [201, 202]
```

prefix cache 只缓存“从 prompt 开头开始的完整 block 前缀”。

也就是：

```text
[10, 11]
[10, 11, 12, 13]
[10, 11, 12, 13, 201, 202]
```

## 为什么必须是前缀

Prefix cache 不是看中间某一段 token 是否出现过。

它要求：

```text
新请求从开头开始的一段 token，和旧请求从开头开始的一段 token 完全一致。
```

例如：

```text
req_a = [10, 11, 12, 13, 201, 202]
req_b = [10, 11, 12, 13, 301, 302]
```

它们共享前 4 个 token：

```text
[10, 11, 12, 13]
```

所以 `req_b` 可以复用这 4 个 token 对应的 prefix KV cache。

## 重点观察

运行脚本后看这几个字段：

```text
reused_prefix_tokens
new_prefill_tokens
cache_entries_after
```

含义是：

- `reused_prefix_tokens`：命中的前缀 token 数。
- `new_prefill_tokens`：还需要重新 prefill 的 token 数。
- `cache_entries_after`：处理完当前请求后，prefix cache 里有多少个前缀块。

## 你应该掌握

学完 Day41 后，你应该能解释：

```text
prefix cache 减少的是重复 prompt 前缀的 prefill 计算。
它要求 token 前缀完全一致，不是语义相似。
它通常按 KV block / page 粒度复用。
它主要影响 TTFT，因为它减少了新请求的 prefill 工作量。
它不等于让 decode 阶段完全免费，后续新 token 仍然要继续 decode。
```

