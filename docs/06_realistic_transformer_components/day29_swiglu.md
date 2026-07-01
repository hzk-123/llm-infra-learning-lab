# Day 29：SwiGLU

## 目标

理解 Llama / Qwen 风格 Transformer block 中常见的 FFN 结构：SwiGLU。

Day14 学过教学版 FFN：

```text
Linear(n_embd -> 4 * n_embd)
ReLU
Linear(4 * n_embd -> n_embd)
```

Day29 学的是更接近现代 LLM 的 FFN：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

## 核心概念

SwiGLU 有三个线性层：

```text
gate_proj: n_embd -> hidden_dim
up_proj:   n_embd -> hidden_dim
down_proj: hidden_dim -> n_embd
```

其中：

- `gate_proj` 产生门控信号。
- `up_proj` 产生被门控的内容。
- `silu(gate) * up` 表示用 gate 控制哪些信息通过。
- `down_proj` 把 hidden_dim 再投影回 n_embd。

## 为什么叫门控

普通 ReLU FFN 大致是：

```text
hidden = relu(W1 x)
out = W2 hidden
```

SwiGLU 是：

```text
hidden = silu(W_gate x) * (W_up x)
out = W_down hidden
```

也就是说，`gate` 分支会影响 `up` 分支的每个 hidden 维度。

你可以直觉理解为：

```text
up 分支提供内容
gate 分支决定内容通过多少
```

## 运行时重点看什么

运行脚本时重点观察：

- `x.shape`
- `gate.shape`
- `up.shape`
- `hidden.shape`
- `out.shape`
- `gate_proj / up_proj / down_proj` 的参数 shape
- 普通 ReLU FFN 和 SwiGLU FFN 的结构差异

## 你应该形成的理解

完成 Day29 后，你应该能说清楚：

```text
Attention 负责 token 之间的信息混合。
FFN 负责对每个 token 的 hidden vector 做非线性变换。
SwiGLU 是现代 LLM 常用的 FFN 结构。
它用 gate_proj 和 up_proj 两条分支，再通过 silu(gate) * up 做门控。
输出 shape 仍然回到 [B, T, n_embd]，这样才能和 residual connection 相加。
```

