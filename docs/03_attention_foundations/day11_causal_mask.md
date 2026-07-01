# Day 11: Causal Mask

今天只理解一个概念：

> Decoder-only language model 训练时，当前位置不能看到未来 token。

## 为什么需要 causal mask

语言模型要做 next-token prediction：

```text
看到过去 token -> 预测下一个 token
```

如果第 1 个位置在训练时可以看到第 2、第 3、第 4 个 token，那它就等于偷看答案。

所以 self-attention 必须加 mask：

```text
token 0 只能看 token 0
token 1 可以看 token 0, 1
token 2 可以看 token 0, 1, 2
token 3 可以看 token 0, 1, 2, 3
```

## 下三角 mask

对于 `block_size = 4`：

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

1 表示可以看，0 表示不能看。

## 怎么应用 mask

先计算 scores：

```python
scores = q @ k.transpose(-2, -1) / sqrt(head_size)
```

然后把未来位置设置成 `-inf`：

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

再做 softmax：

```python
weights = softmax(scores, dim=-1)
```

这样未来位置的 softmax 权重会变成 0。

## 今天你要能回答

- 为什么 language model 不能看到未来 token？
- causal mask 为什么是下三角？
- 为什么要把 masked 位置设置成 `-inf`？
- softmax 后未来位置的权重为什么是 0？
- causal self-attention 和 Day 10 的 self-attention 有什么区别？
