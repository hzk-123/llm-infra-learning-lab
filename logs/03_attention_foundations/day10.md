# Daily Log

Date: 2026-06-11

## Learned

- 标准 self-attention 会先把输入 `x` 投影成 Query、Key、Value。
- Q 和 K 用来计算 attention scores。
- V 是真正被 attention weights 加权汇总的信息。
- attention score 的计算是 `q @ k.transpose(-2, -1)`。
- 标准 attention 会把 scores 除以 `sqrt(head_size)`，避免 softmax 过于尖锐。
- softmax 后每一行 attention weights 加起来等于 1。
- 输出 `out` 是 `weights @ v`，不是 `weights @ x`。

## Ran

```text
python exercises\day10_qkv_self_attention.py
```

## My Explanation

这次输入是：

```text
x.shape: [1, 4, 6]
```

表示 1 条样本，4 个 token，每个 token 是 6 维表示。

三个投影层分别是：

```text
q_proj.weight.shape: [3, 6]
k_proj.weight.shape: [3, 6]
v_proj.weight.shape: [3, 6]
```

因为 `nn.Linear(n_embd, head_size)` 的权重 shape 是 `[out_features, in_features]`，这里就是把 6 维输入投影成 3 维 Q/K/V。

投影后：

```text
q.shape: [1, 4, 3]
k.shape: [1, 4, 3]
v.shape: [1, 4, 3]
```

然后计算：

```python
scores = q @ k.transpose(-2, -1)
```

其中：

```text
q:              [1, 4, 3]
k.transpose:    [1, 3, 4]
scores:         [1, 4, 4]
```

所以 `scores.shape` 是 `[1, 4, 4]`，表示 4 个 token 两两之间的 attention 打分。

例如：

```text
scores[0, 1, 2] = 0.5179912447929382
```

表示第 0 条样本中，第 1 个 token 的 Query 和第 2 个 token 的 Key 的匹配分数。

缩放后：

```text
scaled_scores[0, 1, 2] = 0.2990624010562897
```

这里除以的是：

```text
sqrt(head_size) = sqrt(3)
```

缩放的目的是避免 scores 数值过大，让 softmax 更稳定。

softmax 后：

```text
weights.shape: [1, 4, 4]
row sums: [[1.0000, 1.0000, 1.0000, 1.0000]]
```

说明每个 token 都把注意力权重分配给 4 个 token，并且每一行权重之和是 1。

最后：

```python
out = weights @ v
```

得到：

```text
out.shape: [1, 4, 3]
```

这说明每个 token 的输出表示是对所有 token 的 V 做加权求和后得到的。输出维度是 `head_size = 3`，因为 V 的最后一维是 3。

标准 self-attention 的核心公式就是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

## Next

- 学 causal mask：让当前位置只能看自己和之前的 token，不能偷看未来 token。
