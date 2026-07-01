# Day 36：PyTorch CUDA Memory Analysis

## 目标

在真实 GPU 上观察 batch、seq_len、dtype 对显存的影响。

前面几天主要是用公式和 shape 推导：

```text
prefill scores: [B, H, T, T]
decode scores:  [B, H, 1, T]
KV cache:       [B, H_kv, T, D]
```

Day36 开始用 PyTorch 的 CUDA memory API 真实测量：

```text
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()
torch.cuda.memory_reserved()
```

## 为什么要上服务器跑

本机 4070 Laptop 只有 8GB 显存，可以跑小规模实验，但不适合跑大上下文和多 batch。

你的 3 张 4090 24GB 更适合做：

```text
seq_len 增大
batch_size 增大
fp16 vs fp32
KV cache 显存估算
OOM case 记录
```

## 本日实验内容

脚本会测三类实验：

### 1. Prefill scores 显存

构造：

```text
q.shape = [B, H, T, D]
k.shape = [B, H, T, D]
scores = q @ k.transpose(-2, -1)
scores.shape = [B, H, T, T]
```

观察：

```text
T 变大时，scores 显存如何增长
fp16 和 fp32 显存差多少
```

### 2. KV cache 单层显存

构造：

```text
k_cache.shape = [B, H_kv, T, D]
v_cache.shape = [B, H_kv, T, D]
```

观察：

```text
T 变大时，KV cache 单层显存如何增长
```

### 3. KV cache 全模型估算

真实大模型有很多层，所以完整 KV cache 需要乘以 `num_layers`：

```text
2 * B * T * num_layers * H_kv * D * bytes_per_element
```

脚本不会默认直接分配完整 32 层 cache，避免不必要 OOM；它会根据单层结果给出估算。

## 运行时重点看什么

重点观察：

- `CUDA available`
- GPU name
- total memory
- dtype
- seq_len
- score tensor memory
- measured peak allocated
- KV cache one-layer measured memory
- estimated full-model KV cache memory
- 是否出现 OOM

## 你应该形成的理解

完成 Day36 后，你应该能说清楚：

```text
prefill scores 的显存跟 [B, H, T, T] 相关。
KV cache 的显存跟 [B, H_kv, T, D] 和层数相关。
fp32 显存大约是 fp16/bf16 的 2 倍。
batch_size 和 seq_len 都会线性增加 KV cache。
seq_len 会平方增加 prefill scores。
真实服务器实验需要记录 GPU、dtype、shape、峰值显存和 OOM case。
```

## 建议运行位置

优先在 4090 服务器上运行：

```bash
conda activate llm-sprint
cd ~/llm-sprint/llm-infra-learning-lab
python exercises/07_kv_cache_paged_attention/day36_pytorch_cuda_memory_analysis.py
```

