# Daily Log

Date: 2026-06-10

## Learned

- Token embedding 表示 token 是什么。
- Positional embedding 表示 token 在哪个位置。
- 只有 token embedding 时，同一个 token id 在不同位置查到的向量是一样的。
- 加上 positional embedding 后，同一个 token id 在不同位置的最终表示会不同。
- `token_emb + pos_emb` 后 shape 不变，仍然是 `[batch_size, block_size, n_embd]`。

## Ran

```text
python exercises\day08_positional_embedding.py
```

## My Explanation

这次输入的 shape 是：

```text
input_ids.shape: [2, 4]
```

表示 2 条样本，每条样本有 4 个 token。

token embedding 表的 shape 是：

```text
token_embedding.weight.shape: [10, 6]
```

其中 `10` 是 `vocab_size`，表示有 10 种 token id；`6` 是 `n_embd`，表示每个 token 会被映射成 6 维向量。

position embedding 表的 shape 是：

```text
position_embedding.weight.shape: [4, 6]
```

其中 `4` 是 `block_size`，表示有 4 个位置；每个位置也对应一个 6 维向量。

经过 token embedding 后：

```text
token_emb.shape: [2, 4, 6]
```

位置 id 是：

```text
position_ids: [0, 1, 2, 3]
```

经过 position embedding 后：

```text
pos_emb.shape: [4, 6]
```

把二者相加：

```text
x = token_emb + pos_emb
x.shape: [2, 4, 6]
```

这里 `pos_emb` 会自动 broadcast 到 batch 维度，所以最终 shape 不变。

这次最关键的观察是：

```text
token_emb[0, 0] == token_emb[0, 2]: True
x[0, 0] == x[0, 2]: False
```

因为 `input_ids[0, 0]` 和 `input_ids[0, 2]` 都是 token id `7`，所以只看 token embedding 时它们完全一样。但它们处在不同位置：一个在 position 0，一个在 position 2。加上 positional embedding 后，最终表示不同。

这说明模型不仅需要知道“这个 token 是什么”，还需要知道“这个 token 在哪里”。这为后面的 self-attention 做准备。

## Next

- 进入 self-attention 的第一步：理解 Query、Key、Value 之前，先理解 token 之间如何做相似度打分。
