# Daily Log

Date: 2026-06-14

## Learned

- Tiny Transformer LM 已经可以在真实字符文本上训练。
- 完整训练链路是 `text -> tokenizer -> token ids -> x/y batches -> Transformer LM -> logits -> loss -> optimizer step`。
- `logits.shape` 是 `[batch_size, block_size, vocab_size]`，表示每个位置都对词表中的每个 token 给出预测分数。
- 生成时模型只保留最近 `block_size` 个 token 作为上下文。
- Tiny Transformer 比 Bigram 和 MLP LM 更强，因为它通过 causal self-attention 在上下文窗口内混合 token 信息。

## Ran

```text
python exercises\day18_train_tiny_transformer_text.py
```

## My Explanation

这次训练的是项目里的第一个 Tiny Transformer Language Model。

模型配置是：

```text
vocab_size: 19
data length: 141
block_size: 8
n_embd: 32
num_heads: 4
num_layers: 2
parameter count: 26771
```

其中：

- `vocab_size = 19` 表示字符级词表里有 19 种 token。
- `block_size = 8` 表示模型最多看最近 8 个 token 作为上下文。
- `n_embd = 32` 表示每个 token 的 hidden vector 是 32 维。
- `num_heads = 4` 表示 multi-head attention 有 4 个 head。
- `num_layers = 2` 表示堆叠了 2 个 Transformer Block。
- `parameter count = 26771` 表示这个小模型有 26771 个可训练参数。

单个 batch 的 shape 是：

```text
x.shape:      [2, 8]
y.shape:      [2, 8]
logits.shape: [2, 8, 19]
```

这说明每个 batch 有 2 条样本，每条样本长度 8；模型在每个位置都输出 19 个 token 的预测分数。

训练前生成文本几乎是随机字符：

```text
'hello mvmflvxst e dvcirnncncet axdsfmnrrexrktvasc avn vsdm mnv mdxninr\nfsrhietxxmraxon xf lxd ciatvnc tx '
```

训练过程中 loss 明显下降：

```text
initial loss: 3.142853021621704
step 0000 | loss 3.1534
step 0200 | loss 0.4992
step 0400 | loss 0.3863
step 0600 | loss 0.3016
step 0800 | loss 0.3071
step 1000 | loss 0.3452
```

loss 从 3.1 左右降到 0.3 左右，说明 Tiny Transformer 已经学到了训练文本中的 next-token 模式。

训练后生成文本明显变好：

```text
'hello llm\nhello infra\nllm infra learns\nhello model learns context\ntransformer learns tokens\ntransformer l'
```

它不再是随机字符，而是生成了类似训练语料的结构和短语，例如：

```text
hello llm
hello infra
llm infra learns
transformer learns tokens
```

这说明 token embedding、positional embedding、causal self-attention、FFN、LayerNorm 和 LM head 已经形成了一个可训练的 decoder-only language model 最小闭环。

和之前模型的区别：

```text
Bigram LM: 只看当前 token
MLP LM: 看固定窗口，但没有 attention
Tiny Transformer LM: 用 causal self-attention 在上下文窗口内混合 token 信息
```

## Next

- 学习 generation 的细节：为什么生成时每次只取最后一个位置的 logits，以及为什么要裁剪到最近 `block_size` 个 token。
