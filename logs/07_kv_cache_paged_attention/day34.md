# Daily Log

Date: 2026-06-29

## Learned

- Prefill 阶段会一次性处理完整 prompt。
- Attention scores 的 shape 是 `[B, H, T, T]`。
- `T` 是 prompt length，也就是 seq_len。
- 当 `T` 翻倍时，attention scores 的元素数量会变成 4 倍。
- 长 prompt 会显著增加 prefill 阶段的 attention 计算和中间显存压力。
- TTFT 包含 prefill 成本，所以长 prompt 通常会让 TTFT 变高。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day34_prefill_attention_scaling.py
```

## My Explanation

这次 Day34 学的是 prefill attention scaling，也就是 prompt 长度变大时，attention scores 的计算规模如何增长。

配置是：

```text
batch_size = 1
num_heads = 4
head_dim = 32
seq_lens = [64, 128, 256, 512]
dtype = fp32
```

在 prefill 阶段，模型会处理完整 prompt。

Q/K 的 shape 是：

```text
q.shape = [B, H, T, D]
k.shape = [B, H, T, D]
```

其中：

```text
B = batch_size
H = num_heads
T = seq_len
D = head_dim
```

attention scores 来自：

```text
scores = q @ k.transpose(-2, -1)
```

所以 scores 的 shape 是：

```text
scores.shape = [B, H, T, T]
```

这次输出里，不同 seq_len 的 shape 是：

```text
T = 64:
q.shape      = [1, 4, 64, 32]
k.shape      = [1, 4, 64, 32]
scores.shape = [1, 4, 64, 64]

T = 128:
scores.shape = [1, 4, 128, 128]

T = 256:
scores.shape = [1, 4, 256, 256]

T = 512:
scores.shape = [1, 4, 512, 512]
```

最核心的是最后两个维度：

```text
T x T
```

也就是说，prompt 长度越长，每个 token 都要和 prompt 中的其他 token 计算 attention score。

### 元素数量增长

输出中的 score elements 是：

```text
T = 64:  16,384
T = 128: 65,536
T = 256: 262,144
T = 512: 1,048,576
```

相对 `T=64` 的增长倍数是：

```text
T = 64:  1.0x
T = 128: 4.0x
T = 256: 16.0x
T = 512: 64.0x
```

这说明当 `T` 翻倍时：

```text
T: 64 -> 128，score elements: 4x
T: 128 -> 256，score elements: 4x
T: 256 -> 512，score elements: 4x
```

原因是 scores 的最后两维是：

```text
T * T
```

所以：

```text
(2T) * (2T) = 4 * T * T
```

### 显存增长

score memory 也跟着增长：

```text
T = 64:  64.00 KiB
T = 128: 256.00 KiB
T = 256: 1.00 MiB
T = 512: 4.00 MiB
```

这里用的是 fp32，所以每个元素 4 bytes。

例如 `T=512` 时：

```text
score elements = 1,048,576
memory = 1,048,576 * 4 bytes = 4.00 MiB
```

这只是一个很小的 toy 配置：

```text
batch_size = 1
num_heads = 4
head_dim = 32
```

真实大模型会有更多层、更大的 heads、更长上下文，所以 prefill 的压力会更明显。

### 耗时趋势

这次输出的 score matmul ms 是：

```text
T = 64:  0.020 ms
T = 128: 0.051 ms
T = 256: 0.178 ms
T = 512: 0.709 ms
```

softmax ms 是：

```text
T = 64:  0.032 ms
T = 128: 0.080 ms
T = 256: 0.228 ms
T = 512: 0.895 ms
```

CPU timing 会有噪声，但整体趋势是：`T` 变大时，matmul 和 softmax 都变重。

不过这一天最重要的不是具体毫秒数，而是确定性的规模关系：

```text
scores.shape = [B, H, T, T]
T 翻倍 -> score elements 约 4 倍
```

### 和 TTFT 的关系

TTFT 是 first token latency。

生成第一个 token 之前，模型必须先处理完整 prompt：

```text
prompt -> prefill -> first token
```

长 prompt 会让 prefill 更重，所以 TTFT 通常更高。

Day34 的核心结论是：

```text
长 prompt 慢，不只是因为 token 多。
更关键的是 prefill attention scores 是 [T, T] 级别增长。
```

这也解释了为什么 Day24/Day25 的 long_prompt workload 主要用来压测 prefill 和 TTFT。

## Next

- 进入 Day35：decode attention scaling，观察 decode 阶段每步只生成一个 token，但随着 KV cache 变长，每一步要 attend 的历史 K/V 也越来越长。
