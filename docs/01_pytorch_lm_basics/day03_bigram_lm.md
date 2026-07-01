# Day 3: Bigram Language Model

今天只理解一个概念：

> 用 token embedding 直接预测下一个 token。

Bigram LM 是最小的语言模型。它不看复杂上下文，只根据当前位置的 token 预测下一个 token。

## 从 Day 1 和 Day 2 接上

Day 1:

```python
x = chunk[:-1]
y = chunk[1:]
```

Day 2:

```python
embedding = nn.Embedding(vocab_size, n_embd)
embedded = embedding(x)
```

Day 3:

```python
token_embedding_table = nn.Embedding(vocab_size, vocab_size)
logits = token_embedding_table(x)
loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
```

## Shape

如果：

```text
x shape: [batch_size, block_size]
y shape: [batch_size, block_size]
```

那么：

```text
logits shape: [batch_size, block_size, vocab_size]
```

这里最后一维 `vocab_size` 表示：

> 对每个位置，模型都给词表里的每个 token 一个分数。

## 为什么 cross entropy 要 reshape

`F.cross_entropy` 需要：

```text
input:  [N, C]
target: [N]
```

其中：

- `N` 是要预测的位置数量
- `C` 是类别数量，也就是 `vocab_size`

所以要把：

```text
logits: [batch_size, block_size, vocab_size]
y:      [batch_size, block_size]
```

变成：

```text
logits_flat: [batch_size * block_size, vocab_size]
y_flat:      [batch_size * block_size]
```

## 今天你要能回答

- `logits` 是什么？
- 为什么 `logits` 的最后一维是 `vocab_size`？
- `cross_entropy` 在这里计算什么？
- 为什么要 `view(-1, vocab_size)`？
- 为什么 Bigram LM 很弱？
