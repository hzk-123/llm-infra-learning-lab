# Day 12: CausalSelfAttention Module

今天只做一件事：

> 把 Q/K/V self-attention 和 causal mask 封装成一个可复用的 PyTorch module。

## 从 Day 10 和 Day 11 接上

Day 10:

```text
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
scores = q @ k.T / sqrt(head_size)
weights = softmax(scores)
out = weights @ v
```

Day 11:

```text
scores = scores.masked_fill(mask == 0, -inf)
```

Day 12:

```python
class CausalSelfAttention(nn.Module):
    ...
```

## 为什么要封装成 Module

因为后面的 Transformer block 会反复使用 causal self-attention。

封装后，我们可以像普通层一样调用：

```python
attn = CausalSelfAttention(...)
out = attn(x)
```

## Module 里有什么

需要保存这些层：

```python
self.q_proj = nn.Linear(n_embd, head_size, bias=False)
self.k_proj = nn.Linear(n_embd, head_size, bias=False)
self.v_proj = nn.Linear(n_embd, head_size, bias=False)
```

还需要保存 causal mask：

```python
self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
```

## register_buffer 是什么

`mask` 不是可训练参数，但它属于模型状态。

所以用 `register_buffer` 保存它。

这样模型 `.to(device)` 时，mask 也会跟着移动到对应设备。

## 今天你要能回答

- 为什么要继承 `nn.Module`？
- `__init__` 里保存了什么？
- `forward` 里做了什么？
- `register_buffer` 和 `nn.Parameter` 有什么区别？
- 为什么输出 shape 是 `[batch_size, block_size, head_size]`？
