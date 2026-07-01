# Daily Log

Date: 2026-06-10

## Learned

- MLP Language Model 不再只看当前 token，而是看固定长度的上下文来预测下一个 token。
- `block_size = 4` 表示模型每次看 4 个 token 作为 context。
- Day 7 中 `x.shape` 是 `[batch_size, block_size]`，`y.shape` 是 `[batch_size]`。
- `logits.shape` 是 `[batch_size, vocab_size]`，表示每条 context 只预测一个 next token。
- embedding 后会把 `[batch_size, block_size]` 变成 `[batch_size, block_size, n_embd]`，再 flatten 成 `[batch_size, block_size * n_embd]` 输入 MLP。
- MLP LM 比 Bigram LM 更强，因为它能利用一小段上下文；但它仍然没有 attention，也不能处理任意长上下文。

## Ran

```text
python exercises\day07_mlp_lm.py
```

## My Explanation

这次模型的核心结构是：

```text
token_embedding.weight: [16, 8]
net.0.weight:           [64, 32]
net.0.bias:             [64]
net.2.weight:           [16, 64]
net.2.bias:             [16]
```

其中 `vocab_size = 16`，`n_embd = 8`，`block_size = 4`。所以每个 token 会先变成 8 维向量，4 个 token 的上下文 flatten 后就是：

```text
block_size * n_embd = 4 * 8 = 32
```

这也对应第一层线性层的输入维度：

```text
net.0.weight: [64, 32]
```

它把 32 维的上下文表示映射到 64 维 hidden layer，最后再映射回 `vocab_size = 16` 个 token 的预测分数：

```text
net.2.weight: [16, 64]
```

本次 batch 的形状是：

```text
x.shape:      [2, 4]
y.shape:      [2]
logits.shape: [2, 16]
```

这说明每条样本输入 4 个 token，目标只预测 1 个 next token。和 Day 6 的 Bigram 不同，MLP LM 是：

```text
context tokens -> next token
```

训练前生成文本几乎是随机字符：

```text
'hellldknlinrritkhanfdststolstaarfdo\n dfakksm\noemsdi\nstdhrfe en\na\nmlht\nktko\nds\nnhk\nrh'
```

训练过程中 loss 明显下降：

```text
step 0000 | loss 2.7645
step 0200 | loss 0.1509
step 0400 | loss 0.4498
step 0600 | loss 0.0033
step 0800 | loss 0.3399
step 1000 | loss 0.0226
```

loss 不是严格单调下降，因为每一步随机采样 batch，而且数据集很小。但整体已经从 2.7 左右下降到很低，说明模型学到了训练文本中的局部上下文模式。

训练后生成文本明显更像训练数据：

```text
'hello model learns\nhello rd model\nmodel\nmodel\nmodel\nmodel\nmodel\nmodel\nmodel\nmo'
```

它已经能生成 `hello`、`model`、`learns` 这类更完整的片段，比 Bigram LM 更稳定。不过它也出现了反复输出 `model` 的现象，说明模型只是记住了固定窗口内的局部模式，并没有真正理解长距离结构。

这个实验完成了从 Bigram 到 MLP LM 的升级：

```text
Bigram: 当前 token -> next token
MLP LM: 固定长度 context -> next token
```

## Next

- 学习 positional information：为什么只把 token embedding flatten 还不够。
- 进入 self-attention 前的准备：理解每个 token 位置为什么需要位置信息。
