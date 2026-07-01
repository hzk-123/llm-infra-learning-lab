# Daily Log

Date: 2026-07-01

## Learned

- Prefix cache 复用的是完全一致的 token 前缀，不是语义相似的 prompt。
- Prefix cache 主要减少重复 prefill 计算，因此主要影响 TTFT。
- Prefix cache 通常按 block/page 粒度复用 KV cache。
- 如果新请求的开头 token 和缓存里的前缀完全一致，就可以少算这部分 prefill。
- 如果 prompt token 从开头就不同，或者只是中间片段相同，就不能当作 prefix cache 命中。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day41_prefix_cache_toy.py
```

## My Explanation

这次 Day41 学的是 prefix cache。

配置是：

```text
block_size = 2
request_count = 4
```

脚本里每个 prompt 都是一串 token id：

```text
req_a = [10, 11, 12, 13, 201, 202]
req_b = [10, 11, 12, 13, 301, 302]
req_c = [10, 11, 999, 998]
req_d = [700, 701, 702, 703]
```

因为 `block_size=2`，所以 prompt 会按 2 个 token 一组切成 block。

例如 `req_a`：

```text
[10, 11] [12, 13] [201, 202]
```

prefix cache 保存的是从开头开始的完整 token 前缀：

```text
[10, 11]
[10, 11, 12, 13]
[10, 11, 12, 13, 201, 202]
```

## Request-by-request

### req_a

`req_a` 是第一个请求：

```text
req_a = [10, 11, 12, 13, 201, 202]
```

此时 prefix cache 还是空的，所以不能复用：

```text
reused_prefix_tokens = 0
new_prefill_tokens = 6
```

处理完 `req_a` 后，缓存里新增了 3 个完整前缀：

```text
[10, 11]
[10, 11, 12, 13]
[10, 11, 12, 13, 201, 202]
```

### req_b

`req_b` 是：

```text
req_b = [10, 11, 12, 13, 301, 302]
```

它和 `req_a` 的前 4 个 token 完全一样：

```text
[10, 11, 12, 13]
```

所以它可以复用 4 个 token 的 prefix cache：

```text
reused_prefix_tokens = 4
new_prefill_tokens = 2
```

也就是说，`req_b` 的前 4 个 token 不需要重新 prefill，只需要对后面的：

```text
[301, 302]
```

做新的 prefill。

### req_c

`req_c` 是：

```text
req_c = [10, 11, 999, 998]
```

它只和之前请求共享前 2 个 token：

```text
[10, 11]
```

所以：

```text
reused_prefix_tokens = 2
new_prefill_tokens = 2
```

这里说明 prefix cache 命中可以是部分前缀命中，不一定要整个 prompt 都一样。

### req_d

`req_d` 是：

```text
req_d = [700, 701, 702, 703]
```

它从开头就和之前缓存的前缀不同，所以完全不能复用：

```text
reused_prefix_tokens = 0
new_prefill_tokens = 4
```

## Summary

没有 prefix cache 时，所有 prompt 都要完整 prefill：

```text
without prefix cache = 20 tokens
```

开启 prefix cache 后，只需要新算没有命中的部分：

```text
with prefix cache = 14 tokens
```

节省的 prefill token 数是：

```text
saved prefill tokens = 6
saved ratio = 30.00%
```

这里的 6 个 token 来自：

```text
req_b 复用 4 个 token
req_c 复用 2 个 token
```

所以：

```text
4 + 2 = 6
```

## Key Takeaway

Prefix cache 的核心不是“这个问题看起来像不像”，而是：

```text
token 前缀是否完全一致。
```

它省掉的是重复 prompt prefix 的 prefill 计算。

因此它主要优化：

```text
TTFT
```

因为 TTFT 很大程度上受 prompt prefill 影响。

它不等于让整个请求免费，也不等于让 decode 阶段免费。

后续新 token 仍然需要继续 decode。

## Next

- 进入 Day42：vLLM readiness checklist。
- 目标是检查从 toy serving 过渡到真实 vLLM 服务前，服务器、模型、命令、指标、日志该准备什么。

