# Daily Log

Date: 2026-06-24

## Learned

- MHA、MQA、GQA 的核心区别在于 `num_kv_heads` 是否等于 `num_q_heads`。
- MHA：每个 Q head 都有自己的 K/V head。
- MQA：所有 Q heads 共享同一个 K/V head。
- GQA：一组 Q heads 共享一个 K/V head。
- Q 是当前 forward/decode step 里的临时 tensor。
- K/V 会写入 KV cache，并在后续 decode step 中反复读取。
- KV cache 显存主要由 `num_kv_heads` 决定，而不是 `num_q_heads`。

## Ran

```text
python exercises\06_realistic_transformer_components\day31_mha_mqa_gqa.py
```

## My Explanation

这次 Day31 对比了三种 attention head 设计：

```text
MHA: num_q_heads=8, num_kv_heads=8
GQA: num_q_heads=8, num_kv_heads=2
MQA: num_q_heads=8, num_kv_heads=1
```

输入 tensor 是：

```text
x.shape: [1, 4, 48]
```

这里：

```text
batch_size = 1
seq_len = 4
n_embd = 48
num_q_heads = 8
head_dim = 6
```

因为：

```text
n_embd = num_q_heads * head_dim = 8 * 6 = 48
```

### MHA

MHA 的配置是：

```text
num_q_heads = 8
num_kv_heads = 8
query heads per KV head = 1
```

所以 Q/K/V 都有 8 个 heads：

```text
q.shape: [1, 8, 4, 6]
k.shape: [1, 8, 4, 6]
v.shape: [1, 8, 4, 6]
```

投影矩阵 shape 是：

```text
q_proj.weight: [48, 48]
k_proj.weight: [48, 48]
v_proj.weight: [48, 48]
```

KV cache 显存估算是：

```text
24.00 KiB
```

### GQA

GQA 的配置是：

```text
num_q_heads = 8
num_kv_heads = 2
query heads per KV head = 4
```

这表示每 4 个 Q heads 共享 1 个 K/V head。

Q 仍然有 8 个 heads：

```text
q.shape: [1, 8, 4, 6]
```

但是 K/V 只有 2 个 heads：

```text
k.shape: [1, 2, 4, 6]
v.shape: [1, 2, 4, 6]
```

对应的 K/V 投影矩阵也变小：

```text
k_proj.weight: [12, 48]
v_proj.weight: [12, 48]
```

因为：

```text
num_kv_heads * head_dim = 2 * 6 = 12
```

为了和 8 个 Q heads 做 attention，K/V 可以逻辑上 repeat：

```text
k_for_attention.shape: [1, 8, 4, 6]
v_for_attention.shape: [1, 8, 4, 6]
```

但要注意：KV cache 里真正存的是原始的：

```text
k.shape: [1, 2, 4, 6]
v.shape: [1, 2, 4, 6]
```

所以 GQA 的 KV cache 显存是：

```text
6.00 KiB
```

相当于 MHA 的：

```text
0.25x
```

也就是只需要四分之一。

### MQA

MQA 的配置是：

```text
num_q_heads = 8
num_kv_heads = 1
query heads per KV head = 8
```

这表示所有 8 个 Q heads 共享同一个 K/V head。

Q 还是 8 个 heads：

```text
q.shape: [1, 8, 4, 6]
```

但 K/V 只有 1 个 head：

```text
k.shape: [1, 1, 4, 6]
v.shape: [1, 1, 4, 6]
```

K/V 投影矩阵进一步变小：

```text
k_proj.weight: [6, 48]
v_proj.weight: [6, 48]
```

KV cache 显存是：

```text
3.00 KiB
```

相当于 MHA 的：

```text
0.12x
```

### 总结对比

这次输出的 KV cache 对比是：

```text
MHA: 24.00 KiB, 1.00x
GQA:  6.00 KiB, 0.25x
MQA:  3.00 KiB, 0.12x
```

核心原因是 KV cache 公式里使用的是 `num_kv_heads`：

```text
bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_element
```

而不是 `num_q_heads`。

这里前面的 `2` 表示：

```text
K cache + V cache
```

所以减少 `num_kv_heads` 会直接降低 KV cache 显存。

Day31 的核心结论是：

```text
MHA: K/V heads 和 Q heads 一样多，KV cache 最大。
MQA: 所有 Q heads 共享一个 K/V head，KV cache 最小。
GQA: 介于 MHA 和 MQA 之间，是现代 LLM 常用折中方案。
```

从推理优化角度看，GQA 很重要，因为 decode 阶段会反复读取 KV cache。减少 KV cache 不只是省显存，也可能改善长上下文推理中的显存带宽压力。

## Next

- 进入 Day32：把 RMSNorm、RoPE、GQA、SwiGLU 组合成一个 tiny Llama-style block。
