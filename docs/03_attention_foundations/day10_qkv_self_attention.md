# Day 10: Q/K/V Self-Attention

今天只理解一个升级：

> 标准 self-attention 不直接用 `x` 和 `x` 做相似度，而是先把 `x` 投影成 Query、Key、Value。

## 从 Day 9 接上

Day 9 用最简单的方式计算 attention：

```python
scores = x @ x.transpose(-2, -1)
weights = softmax(scores)
out = weights @ x
```

Day 10 改成标准形式：

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

scores = q @ k.transpose(-2, -1) / sqrt(head_size)
weights = softmax(scores)
out = weights @ v
```

## Q/K/V 的直觉

可以先这样理解：

- Query：当前位置想找什么信息。
- Key：每个位置提供什么索引特征，供别人匹配。
- Value：真正被加权汇总的信息内容。

打分用的是：

```text
Query 和 Key 的相似度
```

输出用的是：

```text
attention weights 加权汇总 Value
```

## Shape

如果：

```text
x shape: [batch_size, block_size, n_embd]
```

线性投影后：

```text
q shape: [batch_size, block_size, head_size]
k shape: [batch_size, block_size, head_size]
v shape: [batch_size, block_size, head_size]
```

attention 分数：

```text
scores shape: [batch_size, block_size, block_size]
```

输出：

```text
out shape: [batch_size, block_size, head_size]
```

## 为什么要除以 `sqrt(head_size)`

点积维度越大，score 的数值可能越大。

如果 score 太大，softmax 会变得非常尖锐，训练不稳定。

所以标准 attention 会做缩放：

```python
scores = scores / sqrt(head_size)
```

## 今天暂时不学

- causal mask
- multi-head attention
- Transformer block
- KV cache

## 今天你要能回答

- Q、K、V 分别是什么直觉？
- 为什么 score 用 `q @ k.T`？
- 为什么输出用 `weights @ v`？
- 为什么要除以 `sqrt(head_size)`？
- `scores.shape` 为什么还是 `[batch_size, block_size, block_size]`？
