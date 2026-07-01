# Day 21: KV Cache Intro

今天只理解一个概念：

> KV cache 缓存历史 token 的 Key 和 Value，让 decode 阶段不必重复计算旧 token 的 K/V。

## 没有 KV cache

生成时如果已有：

```text
[prompt + generated tokens]
```

每一步 decode 都要重新计算整个 context 的 Q/K/V。

这样旧 token 的 K/V 会被反复计算。

## 有 KV cache

prefill 阶段：

```text
完整 prompt -> 计算 K/V -> 存入 cache
```

decode 阶段：

```text
只输入新 token -> 计算这个新 token 的 K/V -> 拼到历史 cache 后面
```

也就是：

```python
k_all = concat(past_k, new_k)
v_all = concat(past_v, new_v)
```

## Decode 时 Q/K/V 的长度不同

decode 阶段只处理一个新 token：

```text
q_new shape: [batch_size, 1, head_size]
```

但它要 attend 到历史所有 token：

```text
k_all shape: [batch_size, total_length, head_size]
v_all shape: [batch_size, total_length, head_size]
```

所以 attention scores 是：

```text
scores shape: [batch_size, 1, total_length]
```

含义：

```text
新 token 对历史所有 token 做 attention
```

## 今天你要能回答

- KV cache 缓存的是哪两个东西？
- prefill 阶段 cache 里有什么？
- decode 阶段为什么只需要输入新 token？
- decode 时 `q_new` 和 `k_all/v_all` 的长度为什么不同？
- KV cache 省掉了什么重复计算？
