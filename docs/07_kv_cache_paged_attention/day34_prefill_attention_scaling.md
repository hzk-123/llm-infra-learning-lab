# Day 34：Prefill Attention Scaling

## 目标

理解为什么 prompt 越长，prefill 阶段越慢，TTFT 通常越高。

Day33 看的是 KV cache 如何按 `[B, H_kv, T, D]` 增长。  
Day34 看的是 prefill 阶段 attention scores 为什么会按 `[B, H, T, T]` 增长。

## Prefill 阶段做什么

Prefill 输入完整 prompt：

```text
x.shape = [B, T, n_embd]
```

投影出 Q/K/V 后，attention 需要计算：

```text
scores = q @ k.transpose(-2, -1)
```

如果：

```text
q.shape = [B, H, T, D]
k.shape = [B, H, T, D]
```

那么：

```text
scores.shape = [B, H, T, T]
```

这里最重要的是最后两个维度：

```text
T x T
```

这意味着 prompt 长度翻倍时，attention scores 的元素数量大约变成 4 倍。

## 为什么长 prompt 会提高 TTFT

TTFT 是 first token latency。

在生成第一个 token 之前，模型必须先完成 prompt 的 prefill：

```text
prompt -> Q/K/V projection -> attention scores -> softmax -> attention output -> logits
```

prompt 越长，prefill 阶段要处理的 token 越多。

其中 attention scores 是 `[T, T]` 级别增长，所以长 prompt 会显著增加 prefill 压力。

## 运行时重点看什么

运行脚本时重点观察：

- `seq_len`
- `q.shape`
- `k.shape`
- `scores.shape`
- `score_elements`
- `score_memory`
- `score_elements` 相对 `T=64` 的增长倍数
- `score_matmul_ms`

## 你应该形成的理解

完成 Day34 后，你应该能说清楚：

```text
prefill 处理完整 prompt。
prefill attention scores 的 shape 是 [B, H, T, T]。
T 翻倍，scores 元素数大约变成 4 倍。
长 prompt 会让 prefill 更重，因此 TTFT 更高。
```

## 注意

本脚本在 CPU 上运行，耗时会有噪声。

所以不要只看时间，更要看确定性的规模指标：

```text
score_elements
score_memory
relative_elements
```

