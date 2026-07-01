# Day 7: MLP Language Model

今天只理解一个升级：

> Bigram LM 只看当前 token；MLP LM 可以看一小段上下文来预测下一个 token。

## 从 Bigram 到 MLP

Day 6 的 Bigram LM：

```text
current token -> next token
```

Day 7 的 MLP LM：

```text
context tokens -> next token
```

例如：

```text
context: [h, e, l, l]
target:  o
```

模型看到 `hell`，目标是预测下一个字符 `o`。

## Shape

如果：

```text
x shape: [batch_size, block_size]
y shape: [batch_size]
```

经过 embedding：

```text
emb shape: [batch_size, block_size, n_embd]
```

然后把上下文里的 embedding 拼平：

```text
emb_flat shape: [batch_size, block_size * n_embd]
```

最后 MLP 输出：

```text
logits shape: [batch_size, vocab_size]
```

## 为什么 y 是 `[batch_size]`

Day 7 里，每条样本只预测 context 后面的一个 token。

所以：

```text
x: [h, e, l, l]
y: o
```

不是每个位置都预测，而是整个 context 预测一个 next token。

## MLP LM 的局限

- context 长度固定。
- 没有 attention。
- block_size 之外的信息看不到。
- 不能像 Transformer 那样灵活建模任意位置之间的关系。

但它能帮助我们理解：

- context window
- embedding flatten
- hidden layer
- logits
- cross entropy
- generation with a rolling context

## 今天你要能回答

- MLP LM 和 Bigram LM 的区别是什么？
- 为什么 `x.shape` 是 `[batch_size, block_size]`？
- 为什么 `y.shape` 是 `[batch_size]`？
- 为什么要把 embedding flatten？
- 为什么生成时要保留最近 `block_size` 个 token？
