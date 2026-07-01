# Milestone Index

Day files are grouped by learning milestone. Each milestone has matching folders under:

```text
docs/
exercises/
logs/
```

## 01 PyTorch LM Basics

Days 01-04:

- Day 01: next-token data
- Day 02: embedding
- Day 03: Bigram LM forward and loss
- Day 04: Bigram LM training loop

Paths:

```text
docs/01_pytorch_lm_basics/
exercises/01_pytorch_lm_basics/
logs/01_pytorch_lm_basics/
```

## 02 Text And Context Models

Days 05-07:

- Day 05: character tokenizer
- Day 06: train Bigram LM on text
- Day 07: MLP LM with fixed context

Paths:

```text
docs/02_text_and_context_models/
exercises/02_text_and_context_models/
logs/02_text_and_context_models/
```

## 03 Attention Foundations

Days 08-13:

- Day 08: positional embedding
- Day 09: attention scores
- Day 10: Q/K/V self-attention
- Day 11: causal mask
- Day 12: CausalSelfAttention module
- Day 13: multi-head attention

Paths:

```text
docs/03_attention_foundations/
exercises/03_attention_foundations/
logs/03_attention_foundations/
```

## 04 Transformer LM

Days 14-18:

- Day 14: feed-forward network
- Day 15: Transformer block with residual connection
- Day 16: LayerNorm
- Day 17: Tiny Transformer LM forward pass
- Day 18: train Tiny Transformer LM on text

Paths:

```text
docs/04_transformer_lm/
exercises/04_transformer_lm/
logs/04_transformer_lm/
```

## 05 Inference Generation

Days 19-27:

- Day 19: generation mechanics
- Day 20: prefill and decode
- Day 21: KV cache intro
- Day 22: KV cache memory estimate
- Day 23: inference metrics
- Day 24: benchmark workload design
- Day 25: toy benchmark runner
- Day 26: request timing with timestamps
- Day 27: concurrent benchmark with ThreadPoolExecutor

Paths:

```text
docs/05_inference_generation/
exercises/05_inference_generation/
logs/05_inference_generation/
```

## 06 Realistic Transformer Components

Days 28-32:

- Day 28: RMSNorm vs LayerNorm
- Day 29: SwiGLU FFN
- Day 30: RoPE
- Day 31: MHA / MQA / GQA shape and KV memory
- Day 32: tiny Llama-style block

Paths:

```text
docs/06_realistic_transformer_components/
exercises/06_realistic_transformer_components/
logs/06_realistic_transformer_components/
```

## 07 KV Cache Paged Attention

Days 33-43:

- Day 33: multi-head KV cache layout and append
- Day 34: prefill attention scaling
- Day 35: decode attention scaling
- Day 36: PyTorch CUDA memory analysis
- Day 37: KV block allocator
- Day 38: toy PagedAttention page table
- Day 39: toy serving scheduler
- Day 40: continuous batching step-level scheduler
- Day 41: prefix cache toy
- Day 42: vLLM readiness checklist
- Day 43: vLLM OpenAI-compatible server prep

Paths:

```text
docs/07_kv_cache_paged_attention/
exercises/07_kv_cache_paged_attention/
logs/07_kv_cache_paged_attention/
```

## 08 Serving vLLM

Days 44-48:

- Day 44: vLLM OpenAI-compatible server
- Day 45: streaming benchmark client
- Day 46: workload benchmark
- Day 47: vLLM parameter experiments
- Day 48: Transformers vs vLLM

Paths:

```text
docs/08_serving_vllm/
exercises/08_serving_vllm/
logs/08_serving_vllm/
```
