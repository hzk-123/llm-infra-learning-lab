# Day 35：Decode Attention Scaling

## 目标

理解 decode 阶段为什么每一步只处理 1 个新 token，但仍然会随着 KV cache 变长而变重。

Day34 学的是 prefill：

```text
scores.shape = [B, H, T, T]
```

Day35 学的是 decode 单步：

```text
scores.shape = [B, H, 1, T]
```

这里的 `T` 是当前 KV cache length。

## Decode 阶段做什么

Decode 阶段每次只输入一个新 token：

```text
q_new.shape = [B, H_q, 1, D]
```

历史 K/V 来自 KV cache：

```text
k_cache.shape = [B, H_kv, T, D]
v_cache.shape = [B, H_kv, T, D]
```

如果是 GQA，attention 计算时会把 K/V 逻辑上 repeat 到 `H_q`：

```text
k_for_attention.shape = [B, H_q, T, D]
v_for_attention.shape = [B, H_q, T, D]
```

然后当前新 token 的 Q 会 attend 到全部历史 K：

```text
scores = q_new @ k_for_attention.transpose(-2, -1)
scores.shape = [B, H_q, 1, T]
```

## 和 Prefill 的区别

Prefill 是：

```text
[B, H, T, T]
```

Decode 单步是：

```text
[B, H, 1, T]
```

所以：

```text
prefill attention scores 随 T 平方增长
decode 单步 attention scores 随 T 线性增长
```

但是 decode 会发生很多步。生成 100 个 token，就要做 100 次 decode step。

所以 decode 阶段的总耗时主要影响：

```text
TPOT
total latency
output tokens/s
```

## 运行时重点看什么

运行脚本时重点观察：

- `cache_len`
- `q_new.shape`
- `k_cache.shape`
- `k_for_attention.shape`
- `scores.shape`
- `score_elements`
- `stored_kv_elements`
- `repeated_kv_elements`
- `score_matmul_ms`
- `attention_output_ms`

## 你应该形成的理解

完成 Day35 后，你应该能说清楚：

```text
decode 每步只处理 1 个新 token。
但是新 token 要 attend 到全部历史 KV cache。
所以单步 scores.shape 是 [B, H, 1, T]。
T 越长，单步 decode 要读的 K/V 越多。
decode 单步随 T 线性增长，不像 prefill 是 T*T。
长输出和长上下文都会让 decode 阶段压力变大。
```

