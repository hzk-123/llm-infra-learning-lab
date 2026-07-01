# Day 33：Multi-head KV Cache Layout 与 Append

## 目标

把之前 Day21 学过的单头 KV cache：

```text
cache['k'].shape = [B, T, D]
cache['v'].shape = [B, T, D]
```

升级到真实大模型更常见的多头 KV cache：

```text
cache['k'].shape = [B, H_kv, T, D]
cache['v'].shape = [B, H_kv, T, D]
```

这里：

```text
B    = batch_size
H_kv = num_kv_heads
T    = cache sequence length
D    = head_dim
```

## 为什么这一天不再做 no-cache 对比

Day20-Day22 已经学过：

```text
没有 KV cache：decode 会反复处理完整 context
有 KV cache：prefill 缓存 K/V，decode 只追加新 token 的 K/V
KV cache 显存公式和 num_kv_heads 有关
```

Day31 又学过：

```text
MHA / MQA / GQA 的区别是 num_kv_heads 不同
减少 num_kv_heads 可以减少 KV cache 显存
```

所以 Day33 的重点不再是重复“有无 KV cache”，而是看真实布局：

```text
KV cache 到底按什么 shape 存？
decode step 如何 append 新 K/V？
GQA 下 cache 存储 shape 和 attention 计算 shape 有什么区别？
```

## Prefill 和 Decode

Prefill 阶段会一次性处理完整 prompt：

```text
x_prompt.shape = [B, prompt_len, n_embd]
k_prompt.shape = [B, H_kv, prompt_len, D]
v_prompt.shape = [B, H_kv, prompt_len, D]
```

这时 cache 长度是：

```text
cache_seq_len = prompt_len
```

Decode 阶段每次只处理一个新 token：

```text
x_new.shape = [B, 1, n_embd]
k_new.shape = [B, H_kv, 1, D]
v_new.shape = [B, H_kv, 1, D]
```

然后 append 到 cache 后面：

```text
k_cache.shape = [B, H_kv, prompt_len + 1, D]
v_cache.shape = [B, H_kv, prompt_len + 1, D]
```

每 decode 一步，`T` 就增加 1。

## GQA 下的关键区别

以 GQA 为例：

```text
num_q_heads = 4
num_kv_heads = 2
```

KV cache 真正存储：

```text
k_cache.shape = [B, 2, T, D]
v_cache.shape = [B, 2, T, D]
```

但 attention 计算时，4 个 Q heads 都要能看到 K/V。

因此 K/V 会被逻辑上 repeat：

```text
k_for_attention.shape = [B, 4, T, D]
v_for_attention.shape = [B, 4, T, D]
```

重点是：

```text
cache 里不需要真的按 H_q 存一份。
cache 只按 H_kv 存。
```

这就是 GQA 节省 KV cache 显存的核心。

## 运行时重点看什么

运行脚本时重点观察：

- prefill 后 `k_cache/v_cache` 的 shape。
- decode step 里 `k_new/v_new` 的 shape。
- 每次 append 后 cache 的 `T` 是否增加 1。
- `cache_position` 是否从 `prompt_len` 开始递增。
- stored cache shape 和 repeated-for-attention shape 的区别。
- stored cache elements 和 repeated elements 的比例。

## 你应该形成的理解

完成 Day33 后，你应该能说清楚：

```text
真实 KV cache 通常按 [B, H_kv, T, D] 存储。
prefill 一次性写入 prompt 的 K/V。
decode 每步只追加一个新 token 的 K/V。
GQA 下 H_kv 小于 H_q，因此 cache 更小。
attention 计算时可以逻辑上 repeat K/V 到 H_q。
但 cache 里真正存储的仍然是 H_kv 份 K/V。
```
