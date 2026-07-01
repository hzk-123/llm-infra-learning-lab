# Daily Log

Date: 2026-06-12

## Learned

- Multi-Head Attention 会并行运行多个 causal self-attention head。
- 每个 head 在自己的子空间里计算 attention。
- 多个 head 的输出会沿 embedding 维度拼接。
- 拼接后通常会经过一个 output projection，把多个 head 的信息重新混合。
- 通常设置 `num_heads * head_size = n_embd`，这样拼接后的维度仍然是 `n_embd`。

## Ran

```text
python exercises\day13_multi_head_attention.py
```

## My Explanation

这次输入是：

```text
x.shape: [2, 4, 8]
```

表示 2 条样本，每条 4 个 token，每个 token 是 8 维表示。

模型设置：

```text
num_heads: 2
head_size: 4
```

因为：

```text
n_embd = 8
num_heads = 2
head_size = n_embd / num_heads = 4
```

每个 attention head 的输出都是：

```text
head_outputs[0].shape: [2, 4, 4]
head_outputs[1].shape: [2, 4, 4]
```

这表示每个 head 都会为每个 token 输出一个 4 维表示。

然后沿最后一维拼接：

```text
concat.shape: [2, 4, 8]
```

两个 4 维 head 拼在一起，重新变成 8 维。

最后经过 output projection：

```text
proj.weight.shape: [8, 8]
out.shape: [2, 4, 8]
```

`proj` 的作用是把多个 head 拼接后的信息重新混合，同时保持输出维度仍然是 `n_embd = 8`。

这次模块结构里有两个主要子模块：

```text
heads -> ModuleList
proj  -> Linear
```

`heads` 保存多个 attention head，`proj` 负责最终投影。

Multi-Head Attention 和 Single-Head Attention 的核心区别是：

```text
Single-head: 一个 attention 子空间
Multi-head: 多个 attention 子空间并行，然后拼接和混合
```

## Next

- 进入 Feed Forward Network：每个 token 在 attention 后还需要经过一个 MLP 进行非线性变换。
