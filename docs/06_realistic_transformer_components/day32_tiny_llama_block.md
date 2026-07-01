# Day 32：Tiny Llama-style Block

## 目标

把 Day28-Day31 学过的真实 LLM 组件组合起来，形成一个 tiny Llama-style Transformer block。

这一天不追求训练效果，重点是看清楚结构：

```text
x
-> RMSNorm
-> GQA attention with RoPE
-> residual add
-> RMSNorm
-> SwiGLU FFN
-> residual add
```

## 和之前教学版 TransformerBlock 的区别

Day16-Day18 的教学版 block 大致是：

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + ReLUFeedForward(LayerNorm(x))
```

Day32 的 tiny Llama-style block 变成：

```text
x = x + GQAAttentionWithRoPE(RMSNorm(x))
x = x + SwiGLUFeedForward(RMSNorm(x))
```

核心变化：

```text
LayerNorm -> RMSNorm
ReLU FFN  -> SwiGLU FFN
MHA       -> GQA
learned position embedding / no RoPE -> RoPE on Q/K
```

## Block 结构

本练习使用：

```text
n_embd = 16
num_q_heads = 4
num_kv_heads = 2
head_dim = 4
hidden_dim = 32
seq_len = 5
```

所以：

```text
q.shape = [B, 4, T, 4]
k.shape = [B, 2, T, 4]
v.shape = [B, 2, T, 4]
```

由于 `num_q_heads=4`，`num_kv_heads=2`，所以每 2 个 Q heads 共享一个 K/V head。

attention 计算前，K/V 会逻辑上 repeat：

```text
k_for_attention.shape = [B, 4, T, 4]
v_for_attention.shape = [B, 4, T, 4]
```

但如果进入 KV cache，真正存储的仍然是较小的：

```text
k.shape = [B, 2, T, 4]
v.shape = [B, 2, T, 4]
```

## 运行时重点看什么

运行脚本时重点观察：

- `x.shape`
- `norm1_out.shape`
- `q/k/v` shape
- `q_rope/k_rope` shape
- `k_for_attention/v_for_attention` shape
- `scores.shape`
- `attn_out.shape`
- 第一次 residual 后的 shape
- `gate/up/ffn_hidden` shape
- 第二次 residual 后的 `out.shape`

## 你应该形成的理解

完成 Day32 后，你应该能说清楚：

```text
RMSNorm 稳定每个 token vector 的尺度。
RoPE 把位置信息注入 Q/K。
GQA 减少 K/V heads，从而减少 KV cache 显存。
SwiGLU 是现代 LLM 常见的 FFN 结构。
Residual connection 要求 attention 和 FFN 最终输出都回到 [B, T, n_embd]。
```

Day32 是 Phase 1 的收束。后面 Day33 会进入推理计算实验：no-cache generation vs KV-cache generation。

