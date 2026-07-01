# Day 19: Generation Mechanics

今天只理解生成时的两个问题：

1. 为什么只取最后一个位置的 logits？
2. 为什么要裁剪到最近 `block_size` 个 token？

## 训练时

训练时，模型一次看到一整段：

```text
input_ids shape: [batch_size, block_size]
```

输出：

```text
logits shape: [batch_size, block_size, vocab_size]
```

这表示每个位置都在预测下一个 token。

## 生成时

生成时，我们已经有一段 context：

```text
[h, e, l, l, o]
```

我们只需要预测“下一个 token”。

所以模型虽然输出所有位置的 logits：

```text
logits shape: [batch_size, current_length, vocab_size]
```

但真正用的是最后一个位置：

```python
last_logits = logits[:, -1, :]
```

因为最后一个位置对应“根据当前完整 context 预测下一个 token”。

## 为什么要裁剪到 block_size

模型的位置 embedding 只支持固定长度：

```text
position_embedding size = block_size
```

如果当前序列长度超过 `block_size`，模型无法直接处理更长位置。

所以生成时会裁剪：

```python
context = ids[:, -block_size:]
```

含义：

```text
只保留最近 block_size 个 token 作为上下文
```

## 生成循环

生成一个 token：

```text
context -> logits -> last_logits -> probabilities -> sample next token
```

循环多次：

```text
把 next token 拼到 ids 后面，再继续生成
```

## 今天你要能回答

- 训练时为什么每个位置都有 logits？
- 生成时为什么只用最后一个位置的 logits？
- `logits[:, -1, :]` 的 shape 是什么？
- 为什么要用 `ids[:, -block_size:]`？
- 生成是并行的还是自回归的？
