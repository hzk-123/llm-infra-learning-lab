# Day 5: Character Tokenizer

今天只理解一个概念：

> 模型不能直接处理字符串。Tokenizer 会把文本转换成 token id。

我们先用最简单的字符级 tokenizer：

```text
每个字符 -> 一个 token id
```

## 为什么需要 tokenizer

Day 1 到 Day 4 中，我们手写了 token id：

```text
[10, 11, 12, 13]
```

但真实数据是文本：

```text
hello llm
```

所以需要 tokenizer 做两件事：

```text
encode: text -> token ids
decode: token ids -> text
```

## 字符级 tokenizer 怎么做

1. 找出文本里出现过的所有字符。
2. 给每个字符分配一个整数 id。
3. 用 `stoi` 保存字符到 id 的映射。
4. 用 `itos` 保存 id 到字符的映射。

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
```

## 今天你要能回答

- `vocab_size` 为什么等于字符种类数量？
- `stoi` 是什么？
- `itos` 是什么？
- `encode` 做了什么？
- `decode` 做了什么？
- 字符级 tokenizer 有什么缺点？

## 字符级 tokenizer 的缺点

- 序列会很长。
- 不能直接复用常见词/子词结构。
- 对中文、英文、代码都比较粗糙。

但它非常适合学习 LLM 的最小闭环。
