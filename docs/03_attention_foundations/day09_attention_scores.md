# Day 9: Attention Scores

今天只理解 self-attention 的第一层直觉：

> 每个 token 会和其它 token 做相似度打分，然后根据分数决定应该关注谁。

## 从 Day 8 接上

Day 8 中我们得到了每个 token 的输入表示：

```text
x shape: [batch_size, block_size, n_embd]
```

其中 `x` 已经包含：

```text
token information + position information
```

Day 9 先用最简单的方式计算 attention score：

```python
scores = x @ x.transpose(-2, -1)
```

## Shape

如果：

```text
x shape: [batch_size, block_size, n_embd]
```

那么：

```text
scores shape: [batch_size, block_size, block_size]
```

含义：

```text
scores[b, i, j] = 第 b 条样本中，第 i 个 token 对第 j 个 token 的相似度分数
```

## Softmax

原始 scores 还不是概率，所以要做：

```python
weights = softmax(scores, dim=-1)
```

这样每一行权重加起来等于 1。

然后：

```python
out = weights @ x
```

含义是：

```text
每个 token 根据 attention weights，对所有 token 的表示做加权求和。
```

## 今天暂时不学

- Query
- Key
- Value
- causal mask
- multi-head attention

今天只理解：

```text
similarity scores -> softmax weights -> weighted sum
```

## 今天你要能回答

- 为什么 `scores.shape` 是 `[batch_size, block_size, block_size]`？
- `scores[b, i, j]` 表示什么？
- softmax 后为什么每行加起来是 1？
- `weights @ x` 得到的是什么？
