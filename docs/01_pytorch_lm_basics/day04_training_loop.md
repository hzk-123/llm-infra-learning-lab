# Day 4: Training Loop

今天只理解一个概念：

> 训练循环会反复计算 loss、反向传播梯度、更新参数，让模型预测越来越接近目标。

## 从 Day 3 接上

Day 3 中模型会输出：

```text
logits: [batch_size, block_size, vocab_size]
loss:   一个标量
```

但 Day 3 还没有更新模型参数。

Day 4 开始训练：

```python
logits, loss = model(x, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 三个核心动作

### 1. `optimizer.zero_grad()`

清空上一次反向传播留下的梯度。

PyTorch 默认会累加梯度，所以每次更新前通常要清零。

### 2. `loss.backward()`

根据 loss 计算每个可训练参数的梯度。

### 3. `optimizer.step()`

根据梯度更新参数。

对于 Bigram LM，主要更新的是：

```text
model.token_embedding_table.weight
```

## 今天你要能回答

- 为什么要先 `zero_grad()`？
- `backward()` 做了什么？
- `optimizer.step()` 更新了什么？
- 为什么 loss 会下降？
- 如果学习率太大，可能会发生什么？
