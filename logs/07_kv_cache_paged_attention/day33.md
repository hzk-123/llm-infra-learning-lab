# Daily Log

Date: 2026-06-29

## Learned

- 真实多头 KV cache 通常按 `[B, H_kv, T, D]` 存储。
- `B` 是 batch size，`H_kv` 是 KV heads 数，`T` 是 cache 当前长度，`D` 是 head_dim。
- Prefill 阶段会一次性把 prompt 的 K/V 写入 cache。
- Decode 阶段每次只追加一个新 token 的 K/V。
- 每 decode 一步，cache 的 `T` 增加 1。
- GQA 下 `H_kv < H_q`，所以真实存储的 KV cache 更小。
- Attention 计算时可以把 K/V 逻辑 repeat 到 `H_q`，但 cache 里不需要真的存重复后的版本。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day33_multi_head_kv_cache_layout.py
```

## My Explanation

这次 Day33 学的是真实多头 KV cache 的 layout 和 append 过程。

配置是：

```text
batch_size = 2
prompt_len = 5
decode_steps = 3
num_q_heads = 4
num_kv_heads = 2
query heads per KV head = 2
head_dim = 4
n_embd = 16
```

因为：

```text
n_embd = num_q_heads * head_dim = 4 * 4 = 16
```

### Prefill

Prefill 输入完整 prompt：

```text
x_prompt.shape: [2, 5, 16]
```

Q 的 shape 是：

```text
q_prompt.shape: [2, 4, 5, 4]
```

这里 Q 有 4 个 heads，也就是 `num_q_heads=4`。

但是 KV cache 里存的是：

```text
k_cache.shape: [2, 2, 5, 4]
v_cache.shape: [2, 2, 5, 4]
```

这里 K/V 只有 2 个 heads，也就是 `num_kv_heads=2`。

所以真实 cache layout 是：

```text
[B, H_kv, T, D] = [2, 2, 5, 4]
```

这和 Day21 的单头 cache 不一样。Day21 是：

```text
[B, T, D]
```

Day33 升级成了真实多头形式：

```text
[B, H_kv, T, D]
```

### Attention 使用时的 repeat

Prefill 后，attention 计算需要让 4 个 Q heads 都能看到 K/V。

所以 K/V 会逻辑上 repeat：

```text
k_for_attention.shape: [2, 4, 5, 4]
v_for_attention.shape: [2, 4, 5, 4]
```

注意这里的关键区别：

```text
真实存储:
  k_cache.shape = [2, 2, 5, 4]
  v_cache.shape = [2, 2, 5, 4]

attention 计算时逻辑展开:
  k_for_attention.shape = [2, 4, 5, 4]
  v_for_attention.shape = [2, 4, 5, 4]
```

cache 里不需要真的存 `[2, 4, 5, 4]`，否则 GQA 就失去省显存的意义。

元素数量也能看出来：

```text
stored cache elements: 160
repeated-for-attention elements: 320
```

因为 `num_q_heads=4`，`num_kv_heads=2`，repeat 后刚好是 2 倍。

### Decode append

Decode step 1 只输入一个新 token：

```text
x_new.shape: [2, 1, 16]
q_new.shape: [2, 4, 1, 4]
k_new.shape: [2, 2, 1, 4]
v_new.shape: [2, 2, 1, 4]
```

这里 Q 还是 4 个 heads，K/V 还是 2 个 heads。

append 到 cache 后：

```text
updated k_cache.shape: [2, 2, 6, 4]
updated v_cache.shape: [2, 2, 6, 4]
cache seq_len: 6
```

原来 prompt_len 是 5，decode 1 步后，cache 长度变成 6。

Decode step 2 后：

```text
updated k_cache.shape: [2, 2, 7, 4]
updated v_cache.shape: [2, 2, 7, 4]
cache seq_len: 7
```

Decode step 3 后：

```text
updated k_cache.shape: [2, 2, 8, 4]
updated v_cache.shape: [2, 2, 8, 4]
cache seq_len: 8
```

这说明每 decode 一个 token，KV cache 的 `T` 维度就增加 1。

### Cache growth

最终增长表是：

```text
prefill:  [2, 2, 5, 4], stored=160, repeated=320
decode_1: [2, 2, 6, 4], stored=192, repeated=384
decode_2: [2, 2, 7, 4], stored=224, repeated=448
decode_3: [2, 2, 8, 4], stored=256, repeated=512
```

stored elements 每步增加：

```text
192 - 160 = 32
224 - 192 = 32
256 - 224 = 32
```

这个 32 来自每步新增一个 token 的 K/V：

```text
2 * B * H_kv * 1 * D
= 2 * 2 * 2 * 1 * 4
= 32
```

前面的 `2` 表示：

```text
K cache + V cache
```

这正好对应 KV cache 显存公式：

```text
2 * batch_size * seq_len * num_kv_heads * head_dim
```

如果加上多层和 dtype，就是 Day22 的完整公式：

```text
2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_element
```

Day33 的核心结论是：

```text
KV cache 真实存储的是 [B, H_kv, T, D]。
GQA 通过减少 H_kv 来减少真实存储。
attention 需要 H_q 个 heads 时，可以逻辑 repeat K/V。
但 cache 里不存 repeated 后的 H_q 版本。
```

这比 Day21 更接近真实 Llama/Qwen/vLLM 里的 KV cache layout。

## Next

- 进入 Day34：prefill attention scaling，观察 prompt 长度变大时 attention scores `[T, T]` 的元素数量和计算压力如何增长。
