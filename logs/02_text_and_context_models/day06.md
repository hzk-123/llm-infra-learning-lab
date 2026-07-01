# Daily Log

Date: 2026-06-08

## Learned

- 现在已经把前五天的内容串成了一个完整语言模型闭环。
- 文本先经过字符级 tokenizer 变成 token ids。
- `get_batch` 会从一长串 token ids 中随机切片，构造 `x` 和 `y`。
- Bigram LM 用当前 token 预测下一个 token。
- 训练后 loss 明显低于初始值，说明模型学到了一些局部字符转移规律。
- Bigram LM 不理解长上下文，所以生成文本仍然混乱。

## Ran

```text
python exercises\day06_train_bigram_text.py
```

## My Explanation

这次输入不再是手写的 token id，而是真实文本：

```text
hello llm
hello infra
llm infra learns
hello model
model learns tokens
```

程序先用字符级 tokenizer 构造词表，得到：

```text
vocab_size: 16
data length: 71
```

然后把文本 encode 成 token ids。前 30 个 token ids 是：

```text
[6, 4, 9, 9, 12, 1, 9, 9, 10, 0, 6, 4, 9, 9, 12, 1, 7, 11, 5, 13, 2, 0, 9, 9, 10, 1, 7, 11, 5, 13]
```

decode 后能还原出对应文本片段：

```text
'hello llm\nhello infra\nllm infr'
```

这说明 tokenizer 到 token ids 的转换是正确的。

训练前生成的文本几乎是随机字符：

```text
'h hln\ndo\nldlhmt\nois n lnolkke\ntk\n\nll olniifnhhhllnrs tkfhohmttn\n ffsisfhmr\ne\n\nhnt'
```

训练过程中 loss 从 `3.6995` 降到了大约 `0.9 ~ 1.0`：

```text
step 000 | loss 3.6995
step 100 | loss 0.8951
step 200 | loss 1.0296
step 300 | loss 1.0391
step 400 | loss 0.8914
step 500 | loss 1.0020
```

loss 没有一直单调下降，是因为每一步都随机采样 batch，而且数据集很小。但整体已经明显低于初始 loss，说明模型确实学到了一些 next-character 规律。

训练后生成文本变得更像训练数据，出现了 `hello`、`learns`、`llm`、`tokens` 等局部模式：

```text
'hearns lodea ins tokearns lllokelelllokelell\nhens ins todelea lo lllm m\nhelokeara'
```

但它仍然不稳定，因为 Bigram LM 只根据当前字符预测下一个字符，不知道更长的上下文。例如它知道 `h` 后面可能接 `e`，`l` 后面可能接 `l` 或 `o`，但不知道整句话应该怎么组织。

这个实验完成了第一个完整闭环：

```text
text -> encode -> token ids -> x/y batches -> training -> generated ids -> decode -> text
```

## Next

- 进入 MLP Language Model：不只看当前 token，而是看一小段上下文来预测下一个 token。
