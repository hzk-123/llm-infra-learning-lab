# Day 14: Feed Forward Network

今天只理解一个概念：

> Feed Forward Network 对每个 token 的表示独立做非线性变换。

## Transformer Block 里有两类子层

第一类：

```text
Multi-Head Attention
```

作用是：

```text
在 token 之间混合信息
```

第二类：

```text
Feed Forward Network
```

作用是：

```text
对每个 token 自己的表示做非线性变换
```

## Shape

如果输入是：

```text
x shape: [batch_size, block_size, n_embd]
```

Feed Forward 输出仍然是：

```text
out shape: [batch_size, block_size, n_embd]
```

它不会改变 batch size，也不会改变 sequence length。

## 常见结构

最简单的 FFN：

```python
nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
)
```

第一层把维度扩展到 `4 * n_embd`。

第二层再投影回 `n_embd`。

## 为什么需要非线性

如果模型只有线性层，表达能力有限。

ReLU/GELU/SwiGLU 这类非线性激活可以让模型表示更复杂的函数。

今天先用 ReLU，后面再了解 GELU/SwiGLU。

## 今天你要能回答

- Feed Forward 和 Attention 的区别是什么？
- 为什么 FFN 不改变 `[batch_size, block_size]`？
- 为什么中间维度通常扩展到 `4 * n_embd`？
- 为什么需要激活函数？
- FFN 是跨 token 混合信息，还是逐 token 处理？
