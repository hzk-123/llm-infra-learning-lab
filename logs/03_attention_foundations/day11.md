# Daily Log

Date: 2026-06-11

## Learned

- Decoder-only language model 不能让当前位置看到未来 token。
- Causal mask 是下三角矩阵，表示每个位置只能看自己和之前的位置。
- 被 mask 的未来位置会被填成 `-inf`。
- `softmax(-inf)` 后对应权重会变成 0。
- 加 causal mask 后，self-attention 才适合做 next-token prediction。

## Ran

```text
python exercises\day11_causal_mask.py
```

## My Explanation

这次先算出未加 mask 的 attention scores：

```text
scores.shape: [1, 4, 4]
```

表示 1 条样本中，4 个 token 两两之间的 attention 分数。

causal mask 是一个下三角矩阵：

```text
[[1., 0., 0., 0.],
 [1., 1., 0., 0.],
 [1., 1., 1., 0.],
 [1., 1., 1., 1.]]
```

含义是：

```text
token 0 只能看 token 0
token 1 可以看 token 0, 1
token 2 可以看 token 0, 1, 2
token 3 可以看 token 0, 1, 2, 3
```

应用 mask 后，未来位置被填成 `-inf`：

```text
[[ 0.3081,    -inf,    -inf,    -inf],
 [-0.7376, -0.0789,    -inf,    -inf],
 [-0.6364, -0.3281,  0.3119,    -inf],
 [-0.7234, -0.0804,  0.2840, -0.1064]]
```

再经过 softmax，未来位置的权重变成 0：

```text
[[1.0000, 0.0000, 0.0000, 0.0000],
 [0.3410, 0.6590, 0.0000, 0.0000],
 [0.2023, 0.2754, 0.5223, 0.0000],
 [0.1334, 0.2538, 0.3654, 0.2473]]
```

每一行仍然加起来等于 1：

```text
row sums: [[1.0000, 1.0000, 1.0000, 1.0000]]
```

最关键的验证是：

```text
weights[0, 0, 1:] = [0., 0., 0.]
weights[0, 1, 2:] = [0., 0.]
```

这说明 token 0 不能看 token 1、2、3，token 1 不能看 token 2、3。未来信息确实被屏蔽了。

如果不加 causal mask，模型训练时就能偷看未来 token，相当于提前看到了答案，这不符合 next-token prediction 的任务要求。

## Next

- 把 Q/K/V self-attention 和 causal mask 封装成一个 `CausalSelfAttention` 模块。
