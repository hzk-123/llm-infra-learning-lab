# Day 17: Tiny Transformer Language Model

今天把前面学过的组件组合成一个真正的 Tiny Transformer LM。

## 组件

Tiny Transformer LM 包含：

```text
token embedding
positional embedding
Transformer Block
final LayerNorm
LM head
```

## 数据流

输入：

```text
input_ids shape: [batch_size, block_size]
```

token embedding：

```text
tok_emb shape: [batch_size, block_size, n_embd]
```

positional embedding：

```text
pos_emb shape: [block_size, n_embd]
```

相加：

```text
x = tok_emb + pos_emb
x shape: [batch_size, block_size, n_embd]
```

经过 Transformer Block：

```text
x shape: [batch_size, block_size, n_embd]
```

经过 LM head：

```text
logits shape: [batch_size, block_size, vocab_size]
```

## LM Head 是什么

LM head 是一个线性层：

```python
nn.Linear(n_embd, vocab_size)
```

它把每个 token 的 hidden vector 转换成对整个词表的预测分数。

## Loss

和 Day 3 一样：

```text
logits: [batch_size, block_size, vocab_size]
labels: [batch_size, block_size]
```

计算 cross entropy 前要 flatten：

```text
logits_flat: [batch_size * block_size, vocab_size]
labels_flat: [batch_size * block_size]
```

## 今天你要能回答

- Tiny Transformer LM 包含哪些组件？
- token embedding 和 positional embedding 怎么组合？
- Transformer Block 的输入输出 shape 是什么？
- LM head 的作用是什么？
- 为什么 logits shape 是 `[batch_size, block_size, vocab_size]`？
