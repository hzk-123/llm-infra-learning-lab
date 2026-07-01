# Day 31：MHA / MQA / GQA

## 目标

理解现代 LLM attention 里两个很容易混淆的概念：

```text
num_heads
num_kv_heads
```

它们不一定相等。

Day13 学过普通 Multi-Head Attention，也就是 MHA。今天继续看真实大模型里常见的 MQA / GQA。

## 三种 Attention Head 设计

### MHA：Multi-Head Attention

```text
num_q_heads = num_kv_heads
```

每个 query head 都有自己对应的 key/value head。

例如：

```text
num_q_heads = 8
num_kv_heads = 8
```

### MQA：Multi-Query Attention

```text
num_kv_heads = 1
```

所有 query heads 共享同一组 key/value head。

例如：

```text
num_q_heads = 8
num_kv_heads = 1
```

### GQA：Grouped-Query Attention

```text
1 < num_kv_heads < num_q_heads
```

多个 query heads 分成若干组，每组共享一个 key/value head。

例如：

```text
num_q_heads = 8
num_kv_heads = 2
```

这表示每 4 个 query heads 共享 1 个 KV head。

## 为什么 GQA / MQA 能省显存

推理时 KV cache 只存 `K` 和 `V`，不存 `Q`。

KV cache 的简化公式是：

```text
bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_element
```

注意这里是：

```text
num_kv_heads
```

不是：

```text
num_q_heads
```

所以只要减少 `num_kv_heads`，KV cache 显存就会下降。

## Shape 重点

脚本里会使用：

```text
batch_size = 1
seq_len = 4
num_q_heads = 8
head_dim = 6
```

所以 `q.shape` 始终是：

```text
[1, 8, 4, 6]
```

但 `k/v.shape` 会变化：

```text
MHA: [1, 8, 4, 6]
GQA: [1, 2, 4, 6]
MQA: [1, 1, 4, 6]
```

真实计算 attention 时，GQA/MQA 的 K/V 可以逻辑上 repeat 到 query heads 数量：

```text
k_for_attention.shape = [1, 8, 4, 6]
v_for_attention.shape = [1, 8, 4, 6]
```

但 KV cache 里真正保存的仍然是较小的 `[B, H_kv, T, D]`。

## 你应该形成的理解

完成 Day31 后，你应该能说清楚：

```text
MHA: 每个 Q head 对应一个 KV head。
MQA: 所有 Q heads 共享一个 KV head。
GQA: 一组 Q heads 共享一个 KV head。

Q 只在当前 forward/decode step 里临时使用。
K/V 会写入 KV cache，并在后续 decode step 反复读取。
因此减少 num_kv_heads 可以直接减少 KV cache 显存。
```

