# Day 28：RMSNorm

## 目标

理解为什么 Llama / Qwen 这类现代大模型经常使用 RMSNorm，而不是之前教学版 Transformer 里常见的 LayerNorm。

## 核心概念

LayerNorm 会对每个 token 的 hidden vector 做两件事：

```text
1. 减去均值
2. 除以标准差
```

也就是：

```text
y = (x - mean) / std * weight + bias
```

RMSNorm 更简单。它不减去均值，只用 root mean square 对 token vector 做缩放：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

所以 RMSNorm 的重点是：

```text
不中心化，只按 RMS 缩放
```

它和 LayerNorm 一样不会改变 tensor shape，但计算更简单，也更常见于 Llama / Qwen 风格的 Transformer block。

## 运行时重点看什么

运行脚本时重点观察：

- 输入 `x.shape`
- `LayerNorm` 输出 shape
- `RMSNorm` 输出 shape
- 归一化前一个 token 的 `mean / std / rms`
- 经过 LayerNorm 后一个 token 的 `mean / std / rms`
- 经过 RMSNorm 后一个 token 的 `mean / std / rms`
- 两种 norm 的可训练参数 shape

## 你应该形成的理解

完成 Day28 后，你应该能说清楚：

```text
LayerNorm 会对每个 token vector 做中心化和缩放。
RMSNorm 不做中心化，只根据 RMS 做缩放。
两者都保持 shape 不变。
RMSNorm 更简单，是 Llama/Qwen 等现代 LLM 中常见的归一化方式。
```

## 和前面 Day16 的关系

Day16 学的是 LayerNorm：

```text
TransformerBlock:
  x = x + attention(LayerNorm(x))
  x = x + ffn(LayerNorm(x))
```

Day28 学的是 RMSNorm。后面实现 tiny Llama-style block 时，会变成类似：

```text
LlamaStyleBlock:
  x = x + attention(RMSNorm(x))
  x = x + swiglu(RMSNorm(x))
```

所以 Day28 不是重新学归一化，而是把教学版 Transformer 里的 LayerNorm 替换成更接近真实大模型的 RMSNorm。
