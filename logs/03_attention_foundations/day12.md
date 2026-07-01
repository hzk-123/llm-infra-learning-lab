# Daily Log

Date: 2026-06-11

## Learned

- `CausalSelfAttention` 可以把 Q/K/V projection、scaled dot-product attention 和 causal mask 封装成一个可复用的 `nn.Module`。
- `q_proj`、`k_proj`、`v_proj` 是可训练参数。
- causal mask 不是可训练参数，但属于模块状态，所以用 `register_buffer` 保存。
- `named_parameters()` 会列出可训练参数。
- `named_buffers()` 会列出非训练但随模型保存和移动设备的状态。
- causal mask 生效后，未来 token 的 attention weight 是 0。

## Ran

```text
python exercises\day12_causal_self_attention_module.py
```

## My Explanation

这次输入是：

```text
x.shape: [2, 4, 6]
```

表示 2 条样本，每条 4 个 token，每个 token 是 6 维向量。

`CausalSelfAttention` 内部有三组线性投影：

```text
q_proj.weight.shape: [3, 6]
k_proj.weight.shape: [3, 6]
v_proj.weight.shape: [3, 6]
```

它们把每个 token 从 `n_embd = 6` 投影到 `head_size = 3`。因此输出是：

```text
out.shape: [2, 4, 3]
```

attention weights 的 shape 是：

```text
weights.shape: [2, 4, 4]
```

表示每条样本中，4 个 token 两两之间的 attention 权重。

第 0 条样本的权重是：

```text
[[1.0000, 0.0000, 0.0000, 0.0000],
 [0.5715, 0.4285, 0.0000, 0.0000],
 [0.3555, 0.3535, 0.2910, 0.0000],
 [0.1757, 0.2708, 0.2802, 0.2732]]
```

这说明 causal mask 生效了：

```text
weights[0, 0, 1:] = [0., 0., 0.]
weights[0, 1, 2:] = [0., 0.]
```

也就是 token 0 不能看未来 token，token 1 也不能看 token 2 和 token 3。

这次还验证了参数和 buffer 的区别。

可训练参数：

```text
q_proj.weight [3, 6]
k_proj.weight [3, 6]
v_proj.weight [3, 6]
```

非训练 buffer：

```text
mask [4, 4]
```

`mask` 不需要梯度更新，但它属于模型的一部分。使用 `register_buffer` 后，模型移动到 GPU 或保存 state dict 时，mask 会跟着一起处理。

## Next

- 进入 Multi-Head Attention：并行使用多个 attention head，再把结果拼接起来。
