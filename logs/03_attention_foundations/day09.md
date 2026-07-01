# Daily Log

Date: 2026-06-10

## Learned

- Attention 的第一步是计算 token 和 token 之间的相似度分数。
- 输入 `x.shape` 是 `[batch_size, block_size, n_embd]`。
- `scores.shape` 是 `[batch_size, block_size, block_size]`，表示每个 token 对每个 token 的相似度。
- `softmax` 会把 raw scores 转换成 attention weights。
- `weights` 的每一行加起来等于 1。
- `out = weights @ x` 会对 token 表示做加权求和，输出 shape 回到 `[batch_size, block_size, n_embd]`。

## Ran

```text
python exercises\day09_attention_scores.py
```

## My Explanation

这次输入是：

```text
x.shape: [1, 4, 3]
```

表示 1 条样本，4 个 token，每个 token 是 3 维向量。

通过：

```python
scores = x @ x.transpose(-2, -1)
```

得到：

```text
scores.shape: [1, 4, 4]
```

这里最后两个维度是 `4 x 4`，表示 4 个 token 两两之间的相似度分数。例如：

```text
scores[0, 1, 2] = 1.1389809846878052
```

含义是第 0 条样本中，第 1 个 token 对第 2 个 token 的相似度分数。

raw scores 还不是概率，所以需要：

```python
weights = softmax(scores, dim=-1)
```

softmax 后每一行权重加起来都是 1：

```text
row sums:
tensor([[1., 1., 1., 1.]])
```

这表示每个 token 会把自己的注意力分配给所有 token。

最后：

```python
out = weights @ x
```

得到：

```text
out.shape: [1, 4, 3]
```

这说明 attention 的输出仍然是每个 token 一个 3 维表示，只不过每个 token 的表示已经融合了其它 token 的信息。

当前 Day 9 还没有引入 Q/K/V，也没有 causal mask。今天只是理解 attention 最核心的直觉：

```text
相似度打分 -> softmax 权重 -> 加权求和
```

## Next

- 进入 Q/K/V：用三组线性层分别产生 Query、Key、Value，再计算标准 self-attention。
