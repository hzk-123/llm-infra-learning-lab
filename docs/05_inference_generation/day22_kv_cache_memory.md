# Day 22: KV Cache Memory

今天只理解一个问题：

> KV cache 到底占多少显存？

## KV cache 存什么

KV cache 存的是每一层 attention 里的历史：

```text
Key cache
Value cache
```

所以公式里会有一个 `2`：

```text
2 = K + V
```

## 单层 KV cache shape

先用常见布局理解：

```text
K shape: [batch_size, seq_len, num_kv_heads, head_size]
V shape: [batch_size, seq_len, num_kv_heads, head_size]
```

其中：

- `batch_size`: 同时服务多少条请求。
- `seq_len`: 当前缓存的 token 数。
- `num_kv_heads`: K/V 的 head 数。
- `head_size`: 每个 head 的维度。

## 总显存公式

所有层的 KV cache 近似显存：

```text
bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_size * bytes_per_element
```

其中：

- `2`: K 和 V。
- `num_layers`: Transformer 层数。
- `bytes_per_element`: dtype 大小，比如 FP16/BF16 是 2 bytes，FP32 是 4 bytes。

## 为什么推理服务容易被 KV cache 卡住

KV cache 会随着这些因素线性增长：

```text
batch_size
seq_len
num_layers
num_kv_heads
head_size
```

所以长上下文和高并发会显著增加显存占用。

## MHA / GQA / MQA

不同模型的 K/V head 数可能不同：

- MHA: `num_kv_heads = num_attention_heads`
- GQA: `num_kv_heads < num_attention_heads`
- MQA: `num_kv_heads = 1`

GQA/MQA 可以减少 KV cache 显存。

## 今天你要能回答

- KV cache 为什么公式里有 `2`？
- `seq_len` 增大时，KV cache 怎么变化？
- `batch_size` 增大时，KV cache 怎么变化？
- FP16 和 FP32 的 KV cache 差多少？
- GQA/MQA 为什么能省 KV cache 显存？
