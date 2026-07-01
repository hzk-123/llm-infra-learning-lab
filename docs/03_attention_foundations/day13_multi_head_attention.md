# Day 13: Multi-Head Attention

今天只理解一个升级：

> Multi-head attention 会并行使用多个 attention head，然后把每个 head 的输出拼接起来。

## 从 Day 12 接上

Day 12 的单头 attention：

```text
x:   [batch_size, block_size, n_embd]
out: [batch_size, block_size, head_size]
```

Day 13 的多头 attention：

```text
head_1_out: [batch_size, block_size, head_size]
head_2_out: [batch_size, block_size, head_size]
...
concat:     [batch_size, block_size, num_heads * head_size]
```

通常设置：

```text
num_heads * head_size = n_embd
```

这样 concat 后还是 `n_embd` 维。

## 为什么需要多个 head

一个 attention head 只能在一个子空间里计算注意力。

多个 head 可以让模型同时学习不同类型的关系。

例如：

- 一个 head 关注局部相邻 token。
- 一个 head 关注句法结构。
- 一个 head 关注长距离依赖。

这里先只理解直觉，不需要过度解释每个 head 实际学到了什么。

## 输出投影

拼接后通常还会接一个线性层：

```python
proj = nn.Linear(n_embd, n_embd)
out = proj(concat)
```

这一步把多个 head 的信息重新混合。

## 今天你要能回答

- 单头 attention 输出 shape 是什么？
- 多头 attention 为什么要 concat？
- 为什么通常让 `num_heads * head_size = n_embd`？
- 输出投影 `proj` 的作用是什么？
- multi-head attention 和 single-head attention 的核心区别是什么？
