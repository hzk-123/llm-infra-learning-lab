# Daily Log

Date: 2026-06-29

## Learned

- 真实 GPU 显存分析要同时看理论 tensor memory 和 PyTorch 实测 peak allocated。
- Prefill scores 的 shape 是 `[B, H, T, T]`，所以 `seq_len` 对它是平方影响。
- KV cache 的 shape 是 `[B, H_kv, T, D]`，单层显存对 `seq_len` 是线性影响。
- 完整模型 KV cache 需要再乘以 `num_layers`。
- FP32 的理论显存约是 FP16 的 2 倍。
- `peak allocated` 可能大于单个 tensor 的理论显存，因为 q/k/scores/softmax 等中间张量可能同时存在。
- 小 tensor 的 measured peak 还会受到 PyTorch CUDA allocator、临时 buffer、对齐和缓存机制影响。

## Ran

```text
python test.py
```

服务器环境：

```text
GPU: NVIDIA GeForce RTX 4090
GPU total memory: 23.52 GiB
PyTorch version: 2.12.0+cu130
CUDA version: 13.0
```

## My Explanation

这次 Day36 在真实 4090 GPU 上测了 PyTorch CUDA 显存。

实验配置是：

```text
batch_sizes = [1, 2]
seq_lens = [512, 1024, 2048]
dtype_names = ['fp16', 'fp32']
num_heads = 8
num_kv_heads = 2
head_dim = 64
num_layers = 32
```

### Prefill scores

Prefill scores 的 shape 是：

```text
[B, H, T, T]
```

例如 fp16、`B=1` 时：

```text
T=512:  [1, 8, 512, 512]   theoretical = 4.00 MiB
T=1024: [1, 8, 1024, 1024] theoretical = 16.00 MiB
T=2048: [1, 8, 2048, 2048] theoretical = 64.00 MiB
```

当 `T` 从 512 到 1024，长度变 2 倍，但 scores 理论显存从 4 MiB 到 16 MiB，变 4 倍。

当 `T` 从 1024 到 2048，也是 2 倍长度，显存从 16 MiB 到 64 MiB，也是 4 倍。

这验证了：

```text
prefill scores 对 seq_len 是平方增长
```

fp16、`B=2` 时：

```text
T=512:  theoretical = 8.00 MiB
T=1024: theoretical = 32.00 MiB
T=2048: theoretical = 128.00 MiB
```

batch 从 1 到 2，理论显存也变成 2 倍。这说明 prefill scores 对 batch size 是线性增长。

fp32 相比 fp16：

```text
fp16 B=2 T=2048: theoretical = 128.00 MiB
fp32 B=2 T=2048: theoretical = 256.00 MiB
```

fp32 是 fp16 的 2 倍，因为每个元素从 2 bytes 变成 4 bytes。

### measured peak allocated

实测 peak allocated 比 theoretical 更大。

例如：

```text
fp16 B=1 T=2048:
theoretical = 64.00 MiB
peak allocated = 140.12 MiB

fp32 B=2 T=2048:
theoretical = 256.00 MiB
peak allocated = 536.12 MiB
```

这是合理的，因为 prefill 过程中不只有 scores 一个 tensor，还可能同时存在：

```text
q
k
scores
softmax output / attention weights
临时 buffer
```

所以真实 peak allocated 往往比单个 scores tensor 的理论显存更高。

### KV cache one layer

KV cache 单层 shape 是：

```text
2 x [B, H_kv, T, D]
```

前面的 `2` 表示：

```text
K cache + V cache
```

fp16、`B=1` 时：

```text
T=512:  theoretical = 256.00 KiB
T=1024: theoretical = 512.00 KiB
T=2048: theoretical = 1.00 MiB
```

这里 `T` 翻倍，KV cache 单层理论显存也翻倍。

这验证了：

```text
KV cache 对 seq_len 是线性增长
```

fp16、`B=2` 时：

```text
T=512:  theoretical = 512.00 KiB
T=1024: theoretical = 1.00 MiB
T=2048: theoretical = 2.00 MiB
```

batch 从 1 到 2，KV cache 理论显存也变成 2 倍。

### 为什么 KV cache measured peak 比 theoretical 大很多

KV cache one-layer 的 theoretical 很小，例如：

```text
fp16 B=1 T=512: theoretical = 256.00 KiB
peak allocated = 8.38 MiB
```

这个差距不能理解为公式错了。

原因是小规模 tensor 的实测 peak 会受到 PyTorch CUDA allocator、内存对齐、临时 buffer、CUDA 上下文等影响。

所以对于小 tensor：

```text
theoretical memory 更适合看增长规律
measured peak 更适合看真实运行时开销
```

### Full-model KV cache estimates

完整模型 KV cache 需要乘以层数：

```text
2 * B * T * num_layers * H_kv * D * bytes_per_element
```

这次估算使用：

```text
num_layers = 32
H_kv = 2
D = 64
```

fp16 结果：

```text
B=1 T=512:  8.00 MiB
B=1 T=1024: 16.00 MiB
B=1 T=2048: 32.00 MiB

B=2 T=512:  16.00 MiB
B=2 T=1024: 32.00 MiB
B=2 T=2048: 64.00 MiB
```

fp32 结果：

```text
B=1 T=512:  16.00 MiB
B=1 T=1024: 32.00 MiB
B=1 T=2048: 64.00 MiB

B=2 T=512:  32.00 MiB
B=2 T=1024: 64.00 MiB
B=2 T=2048: 128.00 MiB
```

这再次验证：

```text
batch_size 翻倍 -> KV cache 翻倍
seq_len 翻倍 -> KV cache 翻倍
fp32 相比 fp16 -> KV cache 约 2 倍
```

### Day36 核心结论

这次实验把前几天的公式落到了真实 GPU 上：

```text
Prefill scores:
  shape = [B, H, T, T]
  seq_len 是平方影响
  主要影响 TTFT

KV cache:
  shape = [B, H_kv, T, D] per layer
  seq_len 是线性影响
  还要乘 num_layers
  主要影响长上下文和并发下的显存
```

从工程角度看，之后做 vLLM / PagedAttention / 量化 / 多卡时，都要带着这两个问题看：

```text
prefill 中间张量是否太大？
KV cache 是否占用了太多长期显存？
```

## Next

- 进入 Day37：从公式和连续 cache 走向 toy KV block allocator，为理解 PagedAttention 的 block/page 管理做准备。
