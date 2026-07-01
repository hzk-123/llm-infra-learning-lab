# Daily Log

Date: 2026-05-20

## Learned

- `vocab_size` 是词表大小，表示有多少种 token id。
- `n_embd` 是每个 token id 被映射成向量后的维度。
- `nn.Embedding(vocab_size, n_embd)` 内部有一个形状为 `[vocab_size, n_embd]` 的可训练参数表。
- 输入 shape 是 `[batch_size, block_size]`，经过 embedding 后变成 `[batch_size, block_size, n_embd]`。
- 同一个 token id 会查到同一行 embedding vector。

## Ran

```text
python exercises\day02_embedding.py
```

## My Explanation

Embedding 是一个可训练的查表模块。输入里的每个 token id 原本只是一个整数编号，经过 embedding 后，每个 token id 会被替换成一个 `n_embd` 维向量。

例如输入 shape 是 `[2, 3]`，表示 2 条样本、每条 3 个 token。当 `n_embd = 4` 时，每个 token 变成 4 维向量，所以输出 shape 变成 `[2, 3, 4]`。

## Next

- 学最小 Bigram Language Model：用 embedding 输出 logits，并开始计算 next-token prediction 的 loss。
