# Day 8: Positional Embedding

今天只理解一个概念：

> Token embedding 告诉模型“这个 token 是什么”，positional embedding 告诉模型“这个 token 在哪个位置”。

## 为什么需要位置信息

Embedding 只根据 token id 查表。

如果同一个 token id 出现在不同位置，它查到的 token embedding 是一样的。

例如：

```text
token id 7 at position 0
token id 7 at position 2
```

如果只有 token embedding，它们的向量完全一样。

但语言中位置很重要：

```text
I love AI
AI love I
```

token 一样，顺序不同，意思就不同。

所以模型还需要 position embedding。

## Shape

输入：

```text
input_ids shape: [batch_size, block_size]
```

token embedding：

```text
token_emb shape: [batch_size, block_size, n_embd]
```

position ids：

```text
position_ids shape: [block_size]
```

position embedding：

```text
pos_emb shape: [block_size, n_embd]
```

二者相加：

```text
x = token_emb + pos_emb
x shape: [batch_size, block_size, n_embd]
```

PyTorch 会自动 broadcast `pos_emb` 到 batch 维度。

## 今天你要能回答

- token embedding 表示什么？
- positional embedding 表示什么？
- 为什么同一个 token 在不同位置需要不同表示？
- `position_embedding.weight` 的 shape 是什么？
- 为什么 `token_emb + pos_emb` 后 shape 不变？

## 之后会发生什么

Transformer 会把包含 token 信息和位置信息的 `x` 送进 self-attention。

今天还不学 attention。
