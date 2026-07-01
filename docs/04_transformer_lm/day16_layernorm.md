# Day 16: LayerNorm

今天只理解一个概念：

> LayerNorm 会对每个 token 的 hidden vector 做归一化，让训练更稳定。

## LayerNorm 作用在哪一维

输入：

```text
x shape: [batch_size, block_size, n_embd]
```

LayerNorm 通常写成：

```python
nn.LayerNorm(n_embd)
```

它会对最后一维 `n_embd` 做归一化。

也就是每个 token 自己的向量会被归一化。

## Shape 不变

LayerNorm 输入：

```text
[batch_size, block_size, n_embd]
```

LayerNorm 输出：

```text
[batch_size, block_size, n_embd]
```

它不会改变 shape。

## Pre-Norm Transformer Block

现代 decoder-only Transformer 常用 pre-norm：

```python
x = x + attention(ln1(x))
x = x + feed_forward(ln2(x))
```

也就是先 norm，再进入子层，然后 residual add。

## 为什么需要 LayerNorm

深层网络中，hidden states 的数值分布可能变得不稳定。

LayerNorm 可以让每个 token 的 hidden vector 保持更稳定的均值和方差。

这通常能让训练更稳定。

## 今天你要能回答

- LayerNorm 归一化哪一维？
- 为什么 LayerNorm 不改变 shape？
- `nn.LayerNorm(n_embd)` 里的 `n_embd` 是什么意思？
- pre-norm block 的结构是什么？
- LayerNorm 和 residual connection 怎么配合？
