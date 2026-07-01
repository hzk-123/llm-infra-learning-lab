# Daily Log

Date: 2026-06-15

## Learned

- KV cache 缓存的是历史 token 的 Key 和 Value。
- Prefill 阶段会处理完整 prompt，并把 prompt 的 K/V 存入 cache。
- Decode 阶段只需要为新 token 计算 Q/K/V。
- 新 token 的 K/V 会拼接到历史 K/V 后面，让 cache 变长。
- Decode 时新 token 的 Query 会 attend 到全部 cached K/V。
- KV cache 避免了每个 decode step 反复计算旧 token 的 K/V。

## Ran

```text
python exercises\05_inference_generation\day21_kv_cache_intro.py
```

## My Explanation

Prefill 阶段输入完整 prompt：

```text
x_prompt.shape: [1, 4, 6]
```

表示 1 条样本，prompt 长度是 4，每个 token 是 6 维表示。

经过 attention 后，输出是：

```text
out_prefill.shape: [1, 4, 3]
```

同时缓存 prompt 的 K/V：

```text
cache['k'].shape: [1, 4, 3]
cache['v'].shape: [1, 4, 3]
```

这里的 `4` 是 prompt length，`3` 是 head_size。

Decode 阶段只输入一个新 token：

```text
x_new.shape: [1, 1, 6]
```

这个新 token 会计算自己的 Q/K/V：

```text
q_new.shape: [1, 1, 3]
k_new.shape: [1, 1, 3]
v_new.shape: [1, 1, 3]
```

然后把新 K/V 拼接到历史 cache 后面：

```text
k_all.shape: [1, 5, 3]
v_all.shape: [1, 5, 3]
new_cache['k'].shape: [1, 5, 3]
new_cache['v'].shape: [1, 5, 3]
```

这里的长度从 4 变成 5，表示 cache 里现在包含 4 个 prompt token 加 1 个新 token。

Decode attention 的权重是：

```text
weights.shape: [1, 1, 5]
```

含义是：当前这个新 token 对 5 个历史/当前 K/V 位置分配 attention 权重。

输出是：

```text
out_decode.shape: [1, 1, 3]
```

表示 decode 阶段只产生这一个新 token 对应的输出表示。

这和没有 KV cache 的区别是：

```text
没有 KV cache: 每一步重新计算整个上下文的 K/V
有 KV cache:   旧 token 的 K/V 复用，只计算新 token 的 K/V
```

所以 KV cache 优化的是 decode 阶段的重复计算。

## Next

- 学 KV cache 显存占用：cache 长度、层数、head 数、head_size、dtype 如何决定显存。
