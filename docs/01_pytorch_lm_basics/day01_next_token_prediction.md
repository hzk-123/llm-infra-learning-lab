# Day 1: Next-Token Prediction

今天只理解一个概念：

> LLM 训练时，本质是在做“根据前面的 token 预测下一个 token”。

## 最小例子

假设一段 token id 是：

```text
[10, 11, 12, 13]
```

训练输入 `x` 是：

```text
[10, 11, 12]
```

训练目标 `y` 是：

```text
[11, 12, 13]
```

所以模型要学：

```text
看到 10       -> 预测 11
看到 10,11    -> 预测 12
看到 10,11,12 -> 预测 13
```

在代码里通常写成：

```python
x = chunk[:-1]
y = chunk[1:]
```

这就是 `loss shift` 的最小形式。

## 今天你要能回答

- `chunk` 是什么？
- `x = chunk[:-1]` 是什么意思？
- `y = chunk[1:]` 是什么意思？
- 为什么 `x` 和 `y` 长度一样？
- batch 之后，`x.shape` 和 `y.shape` 是什么？

## 今天先不要学

- Transformer
- attention
- KV cache
- vLLM
- SFT / LoRA
- 量化

这些后面都会来，但不是今天。
