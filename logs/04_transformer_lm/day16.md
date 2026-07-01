# Daily Log

Date: 2026-06-12

## Learned

- `nn.LayerNorm(n_embd)` 会对每个 token 的最后一维 hidden vector 做归一化。
- LayerNorm 不改变 tensor shape。
- LayerNorm 会让每个 token 向量的均值接近 0，标准差接近 1。
- 现代 decoder-only Transformer 常用 pre-norm 结构。
- Pre-norm 的写法是先 LayerNorm，再进入 attention 或 FFN，最后做 residual add。

## Ran

```text
python exercises\day16_layernorm.py
```

## My Explanation

这次输入是：

```text
x.shape: [2, 4, 8]
```

经过 LayerNorm 后：

```text
x_norm.shape: [2, 4, 8]
```

说明 LayerNorm 不改变 batch size、sequence length，也不改变 embedding 维度。

对同一个 token 来看，LayerNorm 前：

```text
mean: 10.003459930419922
std:  7.01130485534668
```

LayerNorm 后：

```text
mean: -7.450580596923828e-08
std:   0.9999998211860657
```

这里的 mean 接近 0，std 接近 1。`-7.45e-08` 基本可以看成 0，是浮点数误差。

这说明 LayerNorm 是对每个 token 自己的 8 维 hidden vector 做归一化，让 hidden state 的数值分布更稳定。

这次 Transformer Block 使用的是 pre-norm 结构：

```python
x = x + self.attn(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

模块结构是：

```text
ln1  -> LayerNorm
attn -> MultiHeadAttention
ln2  -> LayerNorm
ffn  -> FeedForward
```

含义是：

```text
先归一化，再进入 attention，然后 residual add；
再归一化，再进入 FFN，然后 residual add。
```

LayerNorm 的作用不是跨 token 混合信息，而是稳定每个 token 的 hidden vector。它和 residual connection 配合，可以让更深的 Transformer 更容易训练。

## Next

- 把 token embedding、positional embedding、Transformer Block 和 LM head 组合起来，形成一个 Tiny Transformer Language Model。
