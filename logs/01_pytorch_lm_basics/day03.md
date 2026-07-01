# Daily Log

Date: 2026-05-20

## Learned

- Bigram Language Model 用当前 token 预测下一个 token。
- `x.shape` 和 `y.shape` 都是 `[batch_size, block_size]`。
- `logits.shape` 是 `[batch_size, block_size, vocab_size]`。
- `logits` 的最后一维是词表大小，表示模型给每个候选 token 的分数。
- `cross_entropy` 会把模型预测的 logits 和真实 next-token target 做比较。

## Ran

```text
python exercises\day03_bigram_lm.py
```

## My Explanation

这次输入 `x` 的 shape 是 `[2, 3]`，表示 2 条样本、每条 3 个 token。模型输出 `logits` 的 shape 是 `[2, 3, 20]`，表示每个位置都对 20 个 token id 给出一个预测分数。

例如第一个位置的输入 token id 是 `10`，真实目标是 `11`，但当前未训练模型预测出的 token id 是 `7`。这说明模型现在还没有学会正确的 next-token 关系。

loss 是 `4.405989170074463`，它表示当前模型预测分布和真实目标之间的差距。后面训练时，目标就是通过反向传播降低这个 loss。

## Next

- 学 Bigram LM 的训练循环：`loss.backward()`、`optimizer.step()`、`optimizer.zero_grad()`。
