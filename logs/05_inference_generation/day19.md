# Daily Log

Date: 2026-06-15

## Learned

- 训练时模型会对每个位置输出 next-token logits。
- 生成时只需要预测当前序列的下一个 token，所以只取最后一个位置的 logits。
- 如果当前序列长度超过 `block_size`，生成时只保留最近 `block_size` 个 token 作为上下文。
- 生成是自回归的：每次生成一个 token，再把它拼接到已有序列后面。
- `logits[:, -1, :]` 表示取最后一个位置对整个词表的预测分数。

## Ran

```text
python exercises\day19_generation_mechanics.py
```

## My Explanation

这次输入序列是：

```text
ids: [1, 2, 3, 4, 5, 6]
ids.shape: [1, 6]
```

但是模型的 `block_size = 4`，所以生成时会裁剪上下文：

```python
context = ids[:, -block_size:]
```

得到：

```text
context: [3, 4, 5, 6]
context.shape: [1, 4]
```

这表示模型这一步只能看到最近 4 个 token。

模型对这 4 个位置都会输出 logits：

```text
logits.shape: [1, 4, 10]
```

其中 `10` 是 `vocab_size`。虽然有 4 个位置的 logits，但生成下一个 token 时只需要最后一个位置的预测：

```python
last_logits = logits[:, -1, :]
```

所以：

```text
last_logits.shape: [1, 10]
```

这表示当前完整 context 的最后一个位置，对词表中 10 个 token 的预测分数。

本次采样得到：

```text
next_id: [5]
next_id.shape: [1, 1]
```

然后把它拼接到原序列后面：

```text
new_ids: [1, 2, 3, 4, 5, 6, 5]
new_ids.shape: [1, 7]
```

这就是自回归生成：

```text
当前 tokens -> 预测 1 个 next token -> 拼接 -> 再预测下一个
```

这个过程和训练不同。训练时可以并行计算所有位置的 loss；生成时必须一个 token 一个 token 地串行生成。

## Next

- 学习 prefill 和 decode：第一次处理完整 prompt 是 prefill，后续每次生成一个 token 是 decode。
