# Daily Log

Date: 2026-05-25

## Learned

- Tokenizer 的作用是把文本转换成模型能处理的 token id。
- 字符级 tokenizer 会把文本中出现过的每个字符当成一个 token。
- `stoi` 表示 string to integer，也就是字符到 token id 的映射。
- `itos` 表示 integer to string，也就是 token id 到字符的映射。
- `encode` 把文本转换成 token id 列表。
- `decode` 把 token id 列表还原成文本。
- `round trip works: True` 说明文本经过 encode 再 decode 后没有丢失信息。

## Ran

```text
python exercises\day05_char_tokenizer.py
```

## My Explanation

这次输入文本是：

```text
'hello llm\nhello infra'
```

程序先找出文本中出现过的所有字符：

```text
['\n', ' ', 'a', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'r']
```

所以 `vocab_size = 12`，表示这个字符级词表里一共有 12 种 token。每个字符都有一个唯一的 id，例如 `h -> 5`，`e -> 3`，`l -> 7`。

`encode` 后得到一串 token id：

```text
[5, 3, 7, 7, 10, 1, 7, 7, 8, 0, 5, 3, 7, 7, 10, 1, 6, 9, 4, 11, 2]
```

`decode` 后又还原成原文本：

```text
'hello llm\nhello infra'
```

这说明 tokenizer 的 encode/decode 映射是正确的。

字符级 tokenizer 的优点是简单，适合学习 LLM 最小闭环；缺点是序列会比较长，也不能像 BPE/tokenizer 那样利用常见词或子词结构。

## Next

- 把字符级 tokenizer 和 Bigram LM 连接起来：从真实文本构造 `x/y`，训练 Bigram LM 生成字符。
