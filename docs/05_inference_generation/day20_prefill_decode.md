# Day 20: Prefill And Decode

今天只理解两个阶段：

```text
prefill -> decode -> decode -> decode ...
```

## Prefill 是什么

用户给模型一个 prompt：

```text
[1, 2, 3, 4]
```

模型第一次需要处理整个 prompt。

这一步叫：

```text
prefill
```

prefill 的输入长度通常比较长：

```text
prompt length = T
```

模型会为 prompt 中每个位置计算 hidden state 和 logits。

## Decode 是什么

prefill 后，模型开始一个 token 一个 token 地生成。

每次生成新 token 后，下一步只需要处理刚生成的这个 token：

```text
decode step input length = 1
```

这一步叫：

```text
decode
```

## 没有 KV Cache 时

如果没有 KV cache，每次 decode 都要把完整上下文重新送进模型：

```text
step 1: [prompt]
step 2: [prompt + token1]
step 3: [prompt + token1 + token2]
...
```

这样会重复计算很多旧 token。

## 有 KV Cache 时

有 KV cache 后：

```text
prefill: 处理完整 prompt，并缓存历史 K/V
decode: 只处理新 token，并复用历史 K/V
```

今天还不实现 KV cache，只先理解 prefill/decode 的输入长度差异。

## 今天你要能回答

- prefill 处理什么？
- decode 处理什么？
- 为什么生成是自回归的？
- 没有 KV cache 时会重复计算什么？
- 有 KV cache 后为什么 decode 可以只处理新 token？
