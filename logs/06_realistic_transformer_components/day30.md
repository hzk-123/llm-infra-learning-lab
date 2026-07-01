# Daily Log

Date: 2026-06-23

## Learned

- RoPE 是 Rotary Positional Embedding，常见于 Llama / Qwen 风格的大模型。
- RoPE 不是直接加到 token embedding 上，而是作用在 attention 的 `Q` 和 `K` 上。
- RoPE 不改变 `Q/K` 的 shape。
- RoPE 根据 position 对 `Q/K` 做旋转，让 attention score 带上位置信息。
- position 0 的旋转角度为 0，所以向量基本不变。
- 后续 position 会有不同旋转角度，所以向量方向会变化。
- RoPE 通常保持向量 norm 基本不变，但改变向量方向。

## Ran

```text
python exercises\06_realistic_transformer_components\day30_rope.py
```

## My Explanation

这次 Day30 学的是 RoPE，也就是旋转位置编码。

输入 tensor 是：

```text
x.shape: [1, 4, 12]
```

含义是：

```text
batch_size = 1
seq_len = 4
n_embd = 12
```

脚本先用 `q_proj` 和 `k_proj` 得到 Q/K，再 reshape 成多头 attention 常见的格式：

```text
q.shape: [1, 2, 4, 6]
k.shape: [1, 2, 4, 6]
```

这里含义是：

```text
batch_size = 1
num_heads = 2
seq_len = 4
head_dim = 6
```

经过 RoPE 后，shape 没有变化：

```text
q_rope.shape: [1, 2, 4, 6]
k_rope.shape: [1, 2, 4, 6]
```

这说明 RoPE 不改变 tensor 结构，只改变 Q/K 内部的数值。

RoPE cache 的 shape 是：

```text
cos.shape: [4, 6]
sin.shape: [4, 6]
```

这里 `[4, 6]` 对应：

```text
seq_len = 4
head_dim = 6
```

每个 position 都有一组 `cos/sin`，用于旋转该位置上的 Q/K。

position 0 的结果是：

```text
cos[0]: [1, 1, 1, 1, 1, 1]
sin[0]: [0, 0, 0, 0, 0, 0]
```

这表示 position 0 的旋转角度为 0。

所以 position 0 上的 q 向量经过 RoPE 前后完全一样：

```text
before: [-0.5176, 0.0387, -0.3756, 1.2573, 0.9862, -0.2214]
after:  [-0.5176, 0.0387, -0.3756, 1.2573, 0.9862, -0.2214]
allclose: True
```

position 1 的 `cos/sin` 不再是全 1 和全 0：

```text
cos[1]: [0.5403, 0.5403, 0.9989, 0.9989, 1.0000, 1.0000]
sin[1]: [0.8415, 0.8415, 0.0464, 0.0464, 0.0022, 0.0022]
```

所以 position 1 上的 q 向量经过 RoPE 后发生变化：

```text
before: [-0.5023, 0.2278, -0.4772, -1.0487, -1.3094, -0.3506]
after:  [-0.4631, -0.2996, -0.4280, -1.0697, -1.3086, -0.3534]
allclose: False
```

但 RoPE 前后的 norm 基本一致：

```text
before norm: 1.8625308275
after norm:  1.8625307083
```

这说明 RoPE 更像是在改变向量方向，而不是改变向量长度。

可以这样理解 RoPE：

```text
普通 learned position embedding:
  x = token_embedding + position_embedding

RoPE:
  q = q_proj(x)
  k = k_proj(x)
  q = rotate_by_position(q)
  k = rotate_by_position(k)
```

RoPE 作用在 `Q/K` 上，是因为 attention score 是由 `Q` 和 `K` 算出来的：

```text
scores = q @ k.transpose(-2, -1)
```

如果 Q/K 中包含位置信息，那么 attention score 自然也会受到位置信息影响。

Day30 的核心结论是：

```text
RoPE 不改变 Q/K shape。
RoPE 根据 position 旋转 Q/K。
position 0 基本不变。
后续 position 会改变方向。
向量 norm 基本保持不变。
```

## Next

- 进入 Day31：MHA / MQA / GQA，理解 `num_heads` 和 `num_kv_heads` 的区别，以及为什么 GQA 可以减少 KV cache 显存。
