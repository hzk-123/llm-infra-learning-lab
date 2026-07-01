# Daily Log

Date: 2026-06-17

## Learned

- LayerNorm 和 RMSNorm 都是对每个 token 的 hidden vector 做归一化。
- LayerNorm 会减去均值，再除以标准差。
- RMSNorm 不减去均值，只根据 root mean square 做缩放。
- 两者都会保持 tensor shape 不变。
- RMSNorm 参数更少，没有 bias，常见于 Llama / Qwen 风格的大模型 block。

## Ran

```text
python exercises\06_realistic_transformer_components\day28_rmsnorm.py
```

## My Explanation

这次 Day28 对比了 LayerNorm 和 RMSNorm。

输入 tensor 的 shape 是：

```text
x.shape: [2, 4, 8]
```

这里含义是：

```text
batch_size = 2
seq_len = 4
n_embd = 8
```

经过 LayerNorm 和 RMSNorm 后，shape 都没有变化：

```text
x_layer_norm.shape: [2, 4, 8]
x_rms_norm.shape:   [2, 4, 8]
```

这说明归一化层不会改变 batch 维度、序列长度，也不会改变 hidden size。它只是改变每个 token vector 内部的数值分布。

参数 shape 是：

```text
layer_norm.weight.shape: [8]
layer_norm.bias.shape:   [8]
rms_norm.weight.shape:   [8]
```

LayerNorm 有 `weight` 和 `bias`，RMSNorm 这里只有 `weight`。

归一化前，一个 token 的统计值是：

```text
mean: 10.0035
std:  7.0113
rms:  12.2159
```

这个 token 的数值整体偏大，均值大约是 10，标准差大约是 7。

经过 LayerNorm 后：

```text
mean: -0.0000
std:   1.0000
rms:   1.0000
```

这说明 LayerNorm 做了两步：

```text
1. subtract mean
2. divide by std
```

所以输出的均值接近 0，标准差接近 1。

经过 RMSNorm 后：

```text
mean: 0.8189
std:  0.5740
rms:  1.0000
```

这里最关键的是：RMSNorm 后 `rms` 变成了 1，但 `mean` 没有变成 0。

原因是 RMSNorm 不做中心化，不会减去均值。它只按 RMS 进行缩放：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

所以 RMSNorm 的核心不是让 mean=0，而是让向量整体尺度稳定。

这次可以总结成：

```text
LayerNorm: 中心化 + 缩放
RMSNorm:   只缩放，不中心化
```

和 Day16 的关系：

```text
Day16: 教学版 Transformer block 使用 LayerNorm
Day28: Llama/Qwen 风格 block 更常使用 RMSNorm
```

后面实现 tiny Llama-style block 时，会把之前的：

```text
x = x + attention(LayerNorm(x))
x = x + ffn(LayerNorm(x))
```

替换成类似：

```text
x = x + attention(RMSNorm(x))
x = x + swiglu(RMSNorm(x))
```

Day28 的重点不是重新学归一化，而是理解真实大模型中常见的 RMSNorm 和教学版 LayerNorm 的差异。

## Next

- 进入 Day29：SwiGLU，理解 Llama/Qwen 风格 FFN 中的 gate/up/down projection。
