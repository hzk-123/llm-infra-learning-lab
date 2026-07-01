# Day 6: Train Bigram LM On Text

今天把 Day 1 到 Day 5 串起来：

```text
text -> encode -> token ids -> x/y -> Bigram LM -> generate ids -> decode -> text
```

## 你已经学过的部分

Day 1:

```python
x = chunk[:-1]
y = chunk[1:]
```

Day 2:

```python
nn.Embedding(...)
```

Day 3:

```python
logits = model(x)
loss = cross_entropy(logits, y)
```

Day 4:

```python
zero_grad -> backward -> step
```

Day 5:

```python
encode: text -> token ids
decode: token ids -> text
```

## Day 6 新增

今天新增的是：

1. 从真实文本构造训练数据。
2. 每次随机采样一个 batch。
3. 训练 Bigram LM。
4. 从一个起始 token 开始生成新 token。
5. 把生成的 token ids decode 回文本。

## Bigram LM 的局限

Bigram LM 只根据当前 token 预测下一个 token。

它不知道更长上下文，所以生成文本会比较混乱。但它已经具备语言模型训练的最小闭环。

## 今天你要能回答

- `data = torch.tensor(encode(text))` 做了什么？
- `get_batch` 怎么构造 `x` 和 `y`？
- 为什么训练 loss 会下降？
- `generate` 为什么一次生成一个 token？
- 为什么 Bigram LM 生成效果有限？
