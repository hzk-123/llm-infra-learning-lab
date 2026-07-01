# Daily Log

Date: 2026-05-20

## Learned

- 训练循环的目标是不断降低 loss，让模型预测更接近真实的 next-token target。
- Day 4 的 Bigram LM 主要更新的是 `token_embedding_table.weight`。
- `optimizer.zero_grad()` 用来清空上一轮留下的梯度，因为 PyTorch 默认会累加梯度。
- `loss.backward()` 会根据当前 loss 计算模型参数的梯度。
- `optimizer.step()` 会根据梯度更新模型参数。
- loss 从 `3.7881` 降到 `0.0241`，说明模型已经在这个小训练集上学会了 next-token 关系。

## Ran

```text
python exercises\day04_train_bigram.py
```

## My Explanation

训练前，模型还没有学过这个 next-token 规律，所以第一个位置的输入 token id 是 `10`，真实目标是 `11`，但模型预测成了 `1`。

训练过程中，每一步都会先计算 logits 和 loss，然后清空旧梯度，执行反向传播，再用 optimizer 更新参数：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

随着训练进行，loss 明显下降：

```text
step 000 | loss 3.7881
step 020 | loss 0.7581
step 040 | loss 0.0960
step 060 | loss 0.0400
step 080 | loss 0.0292
step 100 | loss 0.0243
```

训练后，第一个位置的预测从 `1` 变成了 `11`，和真实目标 `11` 一致。这说明 optimizer 已经把 embedding 表里的参数调整到更偏向正确 next-token target。

这个例子很小，所以模型可以很快把训练样本记住。真实 LLM 训练也是同一个基本闭环，只是数据更大、模型更复杂、训练更难。

## Next

- 学会从文本构造 token id 数据，而不是手写 `[10, 11, 12]`。
- 进入字符级 tokenizer：把字符串变成 token id，再用 Bigram LM 训练。
