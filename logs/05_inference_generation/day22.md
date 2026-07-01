# Daily Log

Date: 2026-06-16

## Learned

- KV cache 显存公式是 `2 * batch_size * seq_len * num_layers * num_kv_heads * head_size * bytes_per_element`。
- 公式里的 `2` 来自 K 和 V 两份 cache。
- KV cache 显存会随着 `batch_size` 和 `seq_len` 线性增长。
- FP32 的 KV cache 显存是 FP16/BF16 的 2 倍。
- GQA/MQA 通过减少 `num_kv_heads` 显著降低 KV cache 显存。
- 长上下文和高并发会让 KV cache 成为推理服务的核心显存瓶颈。

## Ran

```text
python exercises\05_inference_generation\day22_kv_cache_memory.py
```

## My Explanation

KV cache 存的是每一层 attention 里的历史 Key 和 Value，所以公式最前面有一个 `2`：

```text
bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_size * bytes_per_element
```

这次 toy example 是：

```text
batch_size=1
seq_len=5
num_layers=1
num_kv_heads=1
head_size=3
dtype=fp32
```

所以显存是：

```text
2 * 1 * 5 * 1 * 1 * 3 * 4 = 120 bytes
```

脚本输出：

```text
readable: 120.00 B
```

base fp16 example 是：

```text
batch_size=1
seq_len=4096
num_layers=32
num_kv_heads=32
head_size=128
dtype=fp16
```

结果是：

```text
2.00 GiB
```

这说明即使只有 1 个请求、4096 token 上下文，KV cache 也可能占用 2GiB 显存。

当上下文长度翻倍：

```text
seq_len: 4096 -> 8192
```

KV cache 从：

```text
2.00 GiB -> 4.00 GiB
```

说明 `seq_len` 和 KV cache 显存是线性关系。

当 batch size 从 1 变成 4：

```text
batch_size: 1 -> 4
```

KV cache 从：

```text
2.00 GiB -> 8.00 GiB
```

说明并发请求越多，KV cache 显存也会线性增长。

FP32 case：

```text
fp16: 2.00 GiB
fp32: 4.00 GiB
```

说明 FP32 的 KV cache 显存是 FP16/BF16 的 2 倍，因为每个元素从 2 bytes 变成 4 bytes。

GQA-style fewer KV heads case：

```text
num_kv_heads: 32 -> 8
```

KV cache 从：

```text
2.00 GiB -> 512.00 MiB
```

这说明减少 KV heads 可以显著降低 KV cache 显存。GQA/MQA 的工程价值就在这里：attention heads 可以很多，但 K/V heads 更少，从而节省推理时的 cache 显存。

这次实验让我看到：推理服务不是只被模型参数占显存，KV cache 也会因为长上下文和并发快速增长。

## Next

- 学 Prefill/Decode 的性能指标：TTFT、TPOT、tokens/s、QPS。
