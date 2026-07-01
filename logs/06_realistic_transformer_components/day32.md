# Daily Log

Date: 2026-06-27

## Learned

- Tiny Llama-style block 可以由 `RMSNorm + GQA attention with RoPE + SwiGLU` 组成。
- Block 内部仍然保持 pre-norm + residual 的结构。
- Attention 输出必须回到 `[B, T, n_embd]`，才能和输入 `x` 做 residual addition。
- FFN 输出也必须回到 `[B, T, n_embd]`，才能继续做 residual addition。
- GQA 中 `num_q_heads` 可以大于 `num_kv_heads`，K/V 逻辑上 repeat 后参与 attention。
- RoPE 作用在 Q/K 上，不作用在 V 上。

## Ran

```text
python exercises\06_realistic_transformer_components\day32_tiny_llama_block.py
```

## My Explanation

这次 Day32 把 Day28-Day31 的真实 LLM 组件组合成了一个 tiny Llama-style block。

配置是：

```text
batch_size = 2
seq_len = 5
n_embd = 16
num_q_heads = 4
num_kv_heads = 2
head_dim = 4
hidden_dim = 32
parameter_count = 2336
```

输入 shape 是：

```text
x.shape: [2, 5, 16]
```

含义是：

```text
batch_size = 2
seq_len = 5
n_embd = 16
```

整个 block 的主路径是：

```text
x
-> norm1
-> GQA attention with RoPE
-> residual add
-> norm2
-> SwiGLU FFN
-> residual add
-> out
```

输出里的 block-level shapes 是：

```text
norm1_out.shape:          [2, 5, 16]
attn_out.shape:           [2, 5, 16]
after_attn_residual.shape:[2, 5, 16]
norm2_out.shape:          [2, 5, 16]
ffn_out.shape:            [2, 5, 16]
out.shape:                [2, 5, 16]
```

这说明每个子层最终都会回到 `[B, T, n_embd]`。

这是 residual connection 的前提，因为：

```text
after_attn_residual = x + attn_out
out = after_attn_residual + ffn_out
```

如果 `attn_out` 或 `ffn_out` 的 shape 不是 `[2, 5, 16]`，就无法和 residual 相加。

### Attention 部分

Attention 内部 shape 是：

```text
q.shape: [2, 4, 5, 4]
k.shape: [2, 2, 5, 4]
v.shape: [2, 2, 5, 4]
```

这里：

```text
num_q_heads = 4
num_kv_heads = 2
head_dim = 4
```

所以 Q 有 4 个 heads，而 K/V 只有 2 个 heads。这就是 GQA。

RoPE 作用在 Q/K 上：

```text
q_rope.shape: [2, 4, 5, 4]
k_rope.shape: [2, 2, 5, 4]
```

注意 V 没有 RoPE，因为位置旋转用于影响 attention score，而 attention score 来自 Q 和 K：

```text
scores = q @ k.transpose(-2, -1)
```

为了让 4 个 Q heads 都能和 K/V 做 attention，K/V 会被逻辑上 repeat：

```text
k_for_attention.shape: [2, 4, 5, 4]
v_for_attention.shape: [2, 4, 5, 4]
```

但从 KV cache 角度看，真正需要存储的仍然是较小的 K/V：

```text
k.shape: [2, 2, 5, 4]
v.shape: [2, 2, 5, 4]
```

这也是 GQA 节省 KV cache 显存的关键。

Attention score 的 shape 是：

```text
scores.shape: [2, 4, 5, 5]
weights.shape: [2, 4, 5, 5]
```

含义是：

```text
batch_size = 2
num_q_heads = 4
query_seq_len = 5
key_seq_len = 5
```

最后 attention 结果会从多头格式合并回：

```text
attn_before_out_proj.shape: [2, 5, 16]
attn_out.shape:             [2, 5, 16]
```

### SwiGLU 部分

SwiGLU 内部 shape 是：

```text
gate.shape:       [2, 5, 32]
up.shape:         [2, 5, 32]
ffn_hidden.shape: [2, 5, 32]
```

这里 hidden_dim 是 32，所以 gate/up 两条分支都会先升维到 32。

SwiGLU 的核心计算仍然是：

```text
ffn_hidden = silu(gate) * up
```

然后通过 down projection 回到：

```text
ffn_out.shape: [2, 5, 16]
```

这样 FFN 输出才能和 attention residual 后的 hidden state 相加。

### 模块结构

脚本输出的 child modules 是：

```text
norm1 -> RMSNorm
attn -> GQAAttentionWithRoPE
norm2 -> RMSNorm
ffn -> SwiGLUFeedForward
```

这和现代 Llama/Qwen 风格 block 的核心结构一致：

```text
RMSNorm
GQA + RoPE attention
RMSNorm
SwiGLU FFN
```

Day32 的核心结论是：

```text
教学版 Transformer block:
  LayerNorm + MHA + ReLU FFN

Llama-style block:
  RMSNorm + RoPE + GQA + SwiGLU
```

这一天标志着 Phase 1 完成：你已经不只是知道这些组件的名字，而是把它们在一个 block 里串起来，并看清楚了每一步的 shape。

## Next

- 进入 Phase 2：KV Cache 与推理计算实验。
- Day33：no-cache generation vs KV-cache generation，真实比较有无 KV cache 的生成计算差异。
