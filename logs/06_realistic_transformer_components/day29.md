# Daily Log

Date: 2026-06-22

## Learned

- 普通 ReLU FFN 使用一条 hidden 分支：`relu(W1 x)`。
- SwiGLU FFN 使用两条 hidden 分支：`gate_proj(x)` 和 `up_proj(x)`。
- SwiGLU 的核心计算是 `hidden = silu(gate) * up`。
- `up` 分支提供内容，`gate` 分支控制这些内容通过多少。
- `down_proj` 会把 hidden 维度投影回 `n_embd`，这样输出才能和 residual connection 相加。
- SwiGLU 是 Llama / Qwen 风格 Transformer block 中常见的 FFN 结构。

## Ran

```text
python exercises\06_realistic_transformer_components\day29_swiglu.py
```

## My Explanation

这次 Day29 对比了普通 ReLU FFN 和 SwiGLU FFN。

输入 tensor 的 shape 是：

```text
x.shape: [2, 4, 8]
```

含义是：

```text
batch_size = 2
seq_len = 4
n_embd = 8
```

普通 ReLU FFN 的输出 shape 是：

```text
relu_out.shape: [2, 4, 8]
```

它的结构是：

```text
Linear(8 -> 32)
ReLU
Linear(32 -> 8)
```

参数量是：

```text
parameter_count: 552
```

SwiGLU FFN 的中间结果是：

```text
gate.shape:   [2, 4, 32]
up.shape:     [2, 4, 32]
hidden.shape: [2, 4, 32]
out.shape:    [2, 4, 8]
```

它的结构是：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

其中参数 shape 是：

```text
gate_proj.weight: [32, 8]
up_proj.weight:   [32, 8]
down_proj.weight: [8, 32]
```

这说明 SwiGLU 不是一条普通的 MLP 分支，而是两条分支：

```text
gate branch: 控制信号
up branch:   内容信号
```

中间的逐元素相乘：

```text
hidden = silu(gate) * up
```

表示 gate 分支会控制 up 分支每个 hidden 维度的通过程度。

从一个 token 的前 8 个 hidden 维度可以看到：

```text
gate[0, 0, :8]:
[-0.8606, -0.7603, 1.0496, -0.4574, 0.0853, -1.6264, 0.6386, 0.2850]

silu(gate)[0, 0, :8]:
[-0.2558, -0.2422, 0.7774, -0.1773, 0.0445, -0.2673, 0.4180, 0.1627]

up[0, 0, :8]:
[0.6186, -0.0232, -0.5379, -0.2803, -0.3099, 1.3108, -0.6106, 0.6530]

hidden[0, 0, :8]:
[-0.1582, 0.0056, -0.4182, 0.0497, -0.0138, -0.3503, -0.2552, 0.1062]
```

比如第一个维度：

```text
silu(gate) = -0.2558
up = 0.6186
hidden = -0.2558 * 0.6186 approx -0.1582
```

这就说明 hidden 是 gate 和 up 逐元素相乘得到的，不是简单过一个 ReLU。

SwiGLU 的参数量是：

```text
parameter_count: 768
```

比这个例子里的 ReLU FFN 参数量 `552` 更多，因为它多了一条 `gate_proj` 分支。

但是两者的最终输出 shape 都保持为 `[2, 4, 8]`：

```text
ReLU FFN keeps shape: True
SwiGLU FFN keeps shape: True
```

这是 Transformer block 里非常重要的要求，因为 FFN 输出要和原来的 hidden state 做 residual addition：

```text
x = x + ffn(norm(x))
```

如果 FFN 最终没有投影回 `n_embd`，就无法和原来的 `x` 相加。

Day29 的核心结论是：

```text
ReLU FFN:
  hidden = relu(W1 x)
  out = W2 hidden

SwiGLU FFN:
  gate = W_gate x
  up = W_up x
  hidden = silu(gate) * up
  out = W_down hidden
```

SwiGLU 仍然是在每个 token 内部独立做非线性变换，不负责 token 之间的信息混合。token 之间的信息混合仍然是 attention 的职责。

## Next

- 进入 Day30：RoPE，理解为什么现代 LLM 不只用 learned positional embedding，而是把旋转位置编码作用在 Q/K 上。
