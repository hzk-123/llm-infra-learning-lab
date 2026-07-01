# Daily Log

Date: 2026-06-12

## Learned

- 最小 Transformer Block 可以由 Multi-Head Attention 和 Feed Forward Network 组成。
- Attention 负责在 token 之间混合信息。
- FFN 负责对每个 token 的表示做非线性变换。
- Residual connection 会把子层输出加回当前表示。
- Residual connection 要求相加的两个 tensor shape 完全一致。

## Ran

```text
python exercises\day15_transformer_block_residual.py
```

## My Explanation

这次输入是：

```text
x.shape: [2, 4, 8]
```

表示 2 条样本，每条 4 个 token，每个 token 是 8 维表示。

经过 Multi-Head Attention 后：

```text
attn_out.shape: [2, 4, 8]
```

经过 Feed Forward 后：

```text
ffn_out.shape: [2, 4, 8]
```

最终输出也是：

```text
out.shape: [2, 4, 8]
```

因为 attention 和 FFN 的输出 shape 都和输入 `x` 一样，所以可以做 residual connection：

```python
x = x + attn_out
x = x + ffn_out
```

这次程序也验证了：

```text
x and attn_out same shape: True
x and ffn_out same shape: True
```

说明 residual 相加在 shape 上是合法的。

这个 block 里有两个主要子模块：

```text
attn -> MultiHeadAttention
ffn  -> FeedForward
```

整体流程是：

```text
先用 attention 让 token 之间交换信息
再用 FFN 对每个 token 自身表示做非线性变换
每个子层都通过 residual connection 加回主路径
```

Residual connection 的意义是让模型保留原来的表示，同时叠加新学到的变化。这样深层网络会更容易训练。

## Next

- 加入 LayerNorm，理解为什么 Transformer Block 需要归一化来稳定训练。
