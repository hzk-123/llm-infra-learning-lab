# Day 2: Embedding

今天只理解一个概念：

> token id 只是一个整数编号，不能直接表达语义。Embedding 会把每个 token id 映射成一个可训练向量。

## 从 Day 1 接上

Day 1 的输入是：

```text
batch_x shape: [batch_size, block_size]
```

例如：

```text
batch_x = [
  [10, 11, 12],
  [11, 12, 13],
]
```

这里的 `10`、`11`、`12` 都只是 token 编号。

## Embedding 做什么

代码：

```python
embedding = torch.nn.Embedding(vocab_size, n_embd)
x = embedding(input_ids)
```

如果：

```text
input_ids shape: [batch_size, block_size]
```

那么：

```text
embedding output shape: [batch_size, block_size, n_embd]
```

意思是：

```text
每个 token id -> 一个 n_embd 维向量
```

## 今天你要能回答

- `vocab_size` 是什么？
- `n_embd` 是什么？
- 为什么输入 shape 是 `[batch_size, block_size]`？
- 为什么输出 shape 多了一维？
- `embedding.weight` 的 shape 是什么？

## 暂时不要学

- attention
- Transformer
- RoPE
- KV cache
- vLLM
