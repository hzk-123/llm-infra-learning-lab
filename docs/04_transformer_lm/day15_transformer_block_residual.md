# Day 15: Transformer Block With Residual Connection

今天只理解一个升级：

> 把 Multi-Head Attention 和 Feed Forward Network 组合成一个最小 Transformer Block，并加入 residual connection。

## 从前两天接上

Day 13：

```text
Multi-Head Attention
```

作用：

```text
让 token 之间交换信息
```

Day 14：

```text
Feed Forward Network
```

作用：

```text
对每个 token 的表示做非线性变换
```

Day 15：

```text
Transformer Block = Attention + FFN + residual connection
```

## Residual Connection

Residual connection 写成：

```python
x = x + attention(x)
x = x + feed_forward(x)
```

也叫 skip connection。

它让模型在每一层都保留原始输入信息，同时叠加新学到的变换。

## 为什么需要 residual connection

如果没有 residual connection，深层网络训练更难。

Residual connection 可以让梯度更容易往前传播，也让模型更容易学习“在原表示基础上做修改”。

## Shape

Transformer Block 输入：

```text
x shape: [batch_size, block_size, n_embd]
```

输出：

```text
out shape: [batch_size, block_size, n_embd]
```

为了能做：

```python
x + attention(x)
```

attention 的输出 shape 必须和 `x` 一样。

为了能做：

```python
x + feed_forward(x)
```

FFN 的输出 shape 也必须和 `x` 一样。

## 今天暂时不学

- LayerNorm
- dropout
- full Transformer LM

## 今天你要能回答

- Transformer Block 里包含哪两个主要子层？
- residual connection 是什么？
- 为什么 residual connection 要求输入输出 shape 一样？
- 为什么 residual connection 有助于训练深层网络？
