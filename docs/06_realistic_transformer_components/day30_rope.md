# Day 30：RoPE

## 目标

理解 Llama / Qwen 等现代 LLM 常用的位置编码方式：RoPE，也就是 Rotary Positional Embedding。

前面 Day08 学过 learned positional embedding：

```text
x = token_embedding + position_embedding
```

Day30 学的是另一种方式：

```text
先得到 Q/K
再对 Q/K 按位置做旋转
```

RoPE 不是直接加到 token embedding 上，而是作用在 attention 里的 `q` 和 `k` 上。

## 核心概念

Attention score 来自：

```text
scores = q @ k.transpose(-2, -1)
```

如果要让 attention 感知位置信息，就可以把位置信息注入到 `q` 和 `k` 里。

RoPE 的做法是：根据 token 的 position，对 `q/k` 的 hidden 维度做二维旋转。

直觉上可以理解为：

```text
同一个 token 在不同位置，会得到不同角度的 q/k。
attention 用旋转后的 q/k 计算相似度。
```

## Shape 重点

在真实多头 attention 里，`q/k` 通常会整理成：

```text
[batch_size, num_heads, seq_len, head_dim]
```

RoPE 的 `cos/sin` 通常是：

```text
[seq_len, head_dim]
```

应用到 `q/k` 时，会 broadcast 成：

```text
[1, 1, seq_len, head_dim]
```

这样每个 batch、每个 head 都使用同一套 position 旋转规则。

## 运行时重点看什么

运行脚本时重点观察：

- `q.shape`
- `k.shape`
- `cos.shape`
- `sin.shape`
- position 0 上 RoPE 前后向量是否基本不变
- position 1 上 RoPE 前后向量是否发生变化
- RoPE 前后 vector norm 是否基本保持一致

## 你应该形成的理解

完成 Day30 后，你应该能说清楚：

```text
RoPE 是一种作用在 Q/K 上的位置编码。
它不会改变 q/k 的 shape。
它会根据 position 对 q/k 做旋转。
position 0 的旋转角度为 0，所以向量基本不变。
更靠后的位置旋转角度不同，所以 q/k 数值会变化。
旋转通常保持向量范数，但改变方向。
```

## 和后续内容的关系

Day31 会继续看 MHA / MQA / GQA。

RoPE 和 GQA 都会出现在真实 Llama/Qwen 风格 attention 里：

```text
x -> q/k/v projection
q/k -> apply RoPE
k/v -> write KV cache
q @ cached_k -> attention scores
```

