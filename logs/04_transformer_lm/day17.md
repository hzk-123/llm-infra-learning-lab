# Daily Log

Date: 2026-06-12

## Learned

- Tiny Transformer LM 由 token embedding、positional embedding、Transformer Block、final LayerNorm 和 LM head 组成。
- `input_ids` 和 `labels` 的 shape 都是 `[batch_size, block_size]`。
- `logits` 的 shape 是 `[batch_size, block_size, vocab_size]`。
- LM head 把每个位置的 hidden vector 映射成对整个词表的预测分数。
- Tiny Transformer LM 已经具备 decoder-only language model 的核心前向结构。

## Ran

```text
python exercises\day17_tiny_transformer_lm.py
```

## My Explanation

这次输入是：

```text
input_ids.shape: [2, 4]
labels.shape:    [2, 4]
```

表示 2 条样本，每条样本长度是 4。

token embedding 表是：

```text
token_embedding.weight.shape: [16, 8]
```

其中 `16` 是 `vocab_size`，`8` 是 `n_embd`。它把每个 token id 映射成 8 维向量。

position embedding 表是：

```text
position_embedding.weight.shape: [4, 8]
```

其中 `4` 是 `block_size`，表示 4 个位置；每个位置也有一个 8 维向量。

token embedding 和 positional embedding 相加后，进入 Transformer Block。Block 内部包含：

```text
Multi-Head Attention
Feed Forward
Residual Connection
LayerNorm
```

之后经过 final LayerNorm，再经过 LM head：

```text
lm_head.weight.shape: [16, 8]
```

LM head 把 8 维 hidden vector 映射成 16 个词表 token 的分数。

最终输出：

```text
logits.shape: [2, 4, 16]
```

含义是：2 条样本、每条 4 个位置、每个位置都对 16 个 token 给出预测分数。

本次 loss 是：

```text
loss: 2.9095940589904785
```

说明现在只是前向计算和 loss 计算，还没有训练。后面训练时会通过反向传播更新 embedding、attention、FFN、LayerNorm 和 LM head 的参数。

模型模块结构是：

```text
token_embedding    -> Embedding
position_embedding -> Embedding
block              -> TransformerBlock
ln_f               -> LayerNorm
lm_head            -> Linear
```

完整数据流是：

```text
input_ids
-> token embedding
-> + positional embedding
-> Transformer Block
-> final LayerNorm
-> LM head
-> logits
-> cross entropy loss
```

## Next

- 在真实字符文本上训练 Tiny Transformer LM，并生成文本。
