# Day 18: Train Tiny Transformer LM On Text

今天把 Day 17 的 Tiny Transformer LM 放到真实字符文本上训练。

## 完整闭环

```text
text
-> char tokenizer
-> token ids
-> x/y batches
-> Tiny Transformer LM
-> logits
-> cross entropy loss
-> backward + optimizer step
-> generate token ids
-> decode text
```

## 和 Day 6 的区别

Day 6 是 Bigram LM：

```text
current token -> next token
```

Day 18 是 Tiny Transformer LM：

```text
context tokens -> causal self-attention -> next-token logits at every position
```

Transformer 能通过 causal self-attention 在上下文窗口内混合 token 信息。

## 关键 shape

输入 batch：

```text
x shape: [batch_size, block_size]
y shape: [batch_size, block_size]
```

模型输出：

```text
logits shape: [batch_size, block_size, vocab_size]
```

计算 loss 前 flatten：

```text
logits_flat shape: [batch_size * block_size, vocab_size]
labels_flat shape: [batch_size * block_size]
```

## Generate

生成时每次只预测一个新 token：

```text
当前 context -> logits -> sample next token -> 拼到序列后面
```

如果序列长度超过 `block_size`，就只保留最近的 `block_size` 个 token 作为 context。

## 今天你要能回答

- Tiny Transformer LM 和 Bigram LM 的主要区别是什么？
- 为什么 `logits.shape` 是 `[batch_size, block_size, vocab_size]`？
- 为什么训练时每个位置都有 target？
- 生成时为什么要裁剪到最近 `block_size` 个 token？
- loss 下降说明了什么？
