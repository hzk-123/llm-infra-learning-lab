# Daily Log

Date: 2026-06-29

## Learned

- Decode 阶段每一步只处理 1 个新 token。
- 新 token 的 Q 会 attend 到全部历史 KV cache。
- Decode 单步 attention scores 的 shape 是 `[B, H_q, 1, T]`。
- `T` 是当前 KV cache length。
- 当 `T` 翻倍时，decode 单步 score elements 也翻倍。
- Decode 单步复杂度随 `T` 线性增长，不像 prefill 的 `[B, H, T, T]` 是平方增长。
- 长上下文会提高每个输出 token 的 decode 成本，因此可能拉高 TPOT。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day35_decode_attention_scaling.py
```

## My Explanation

这次 Day35 学的是 decode attention scaling。

配置是：

```text
batch_size = 1
num_q_heads = 4
num_kv_heads = 2
query heads per KV head = 2
head_dim = 32
cache_lens = [64, 128, 256, 512, 1024]
dtype = fp32
```

Decode 阶段每次只处理一个新 token，所以 Q 的 shape 固定是：

```text
q_new.shape: [1, 4, 1, 32]
```

这里：

```text
B = 1
H_q = 4
new token length = 1
D = 32
```

KV cache 真实存储的是 GQA 形式：

```text
k_cache.shape = [B, H_kv, T, D]
```

例如 `cache_len=64` 时：

```text
k_cache.shape: [1, 2, 64, 32]
```

这里 `H_kv=2`，小于 `H_q=4`。

为了让 4 个 Q heads 都能和 K 做 attention，K 会逻辑上 repeat：

```text
k_for_attention.shape: [1, 4, 64, 32]
```

也就是从：

```text
[B, H_kv, T, D]
```

扩展到：

```text
[B, H_q, T, D]
```

但注意：cache 里真实存储的仍然是 `[B, H_kv, T, D]`，这延续了 Day33 的结论。

### Decode scores shape

Decode 单步 attention score 来自：

```text
scores = q_new @ k_for_attention.transpose(-2, -1)
```

所以 scores 的 shape 是：

```text
[B, H_q, 1, T]
```

这次输出里：

```text
cache_len = 64:
scores.shape = [1, 4, 1, 64]

cache_len = 128:
scores.shape = [1, 4, 1, 128]

cache_len = 256:
scores.shape = [1, 4, 1, 256]

cache_len = 512:
scores.shape = [1, 4, 1, 512]

cache_len = 1024:
scores.shape = [1, 4, 1, 1024]
```

这和 Day34 的 prefill 不一样。

Day34 prefill 是：

```text
scores.shape = [B, H, T, T]
```

Day35 decode 单步是：

```text
scores.shape = [B, H, 1, T]
```

### 元素数量增长

这次 score elements 是：

```text
T = 64:   256
T = 128:  512
T = 256:  1024
T = 512:  2048
T = 1024: 4096
```

每次 cache length 翻倍，score elements 也翻倍：

```text
64 -> 128:   2.0x
128 -> 256:  2.0x
256 -> 512:  2.0x
512 -> 1024: 2.0x
```

原因是 decode 单步只有一个 query token：

```text
1 * T
```

所以它对当前 cache length 是线性增长。

这和 prefill 的：

```text
T * T
```

形成了鲜明对比。

### KV 元素数量

stored KV elements 是：

```text
T = 64:   8192
T = 128:  16384
T = 256:  32768
T = 512:  65536
T = 1024: 131072
```

每次 cache length 翻倍，stored KV elements 也翻倍。

这是因为 KV cache 真实存储规模是：

```text
2 * B * H_kv * T * D
```

这里前面的 `2` 表示 K 和 V 两份 cache。

以 `T=64` 为例：

```text
2 * 1 * 2 * 64 * 32 = 8192
```

和输出一致。

repeated KV elements 是 stored KV elements 的 2 倍：

```text
stored:   8192
repeated: 16384
```

原因是：

```text
H_q = 4
H_kv = 2
```

所以 repeat 后 head 数从 2 变成 4。

### 和 TPOT 的关系

Day35 解释的是 decode 阶段的单步成本。

Decode 阶段每生成一个 token，都要做一次类似：

```text
q_new attends to all cached K/V
```

所以如果上下文越来越长：

```text
T 越大 -> 每步要读的 KV 越多 -> 单 token decode 成本越高
```

这会影响：

```text
TPOT
output tokens/s
total latency
```

Day35 的核心结论是：

```text
prefill: [B, H, T, T]，随 T 平方增长，主要影响 TTFT。
decode:  [B, H, 1, T]，单步随 T 线性增长，主要影响 TPOT。
```

这也说明为什么长上下文不只是 prefill 慢，后续每生成一个 token 也会因为 KV cache 更长而变重。

## Next

- 进入 Day36：PyTorch CUDA memory analysis，在服务器上用真实 GPU 观察 batch、seq_len、dtype 对显存的影响。
