# 里程碑索引

这个文件只记录已经创建并正在使用的学习阶段。

原则：

```text
学到哪个阶段，再创建哪个阶段的目录。
不提前维护空目录。
```

每个阶段通常有三类文件：

```text
docs/       概念说明
exercises/  可运行脚本
logs/       学习日志
```

## 01 PyTorch LM Basics

Day01-Day04：

- Day01：next-token 数据构造
- Day02：Embedding
- Day03：Bigram LM forward 和 loss
- Day04：Bigram LM training loop

路径：

```text
docs/01_pytorch_lm_basics/
exercises/01_pytorch_lm_basics/
logs/01_pytorch_lm_basics/
```

## 02 Text And Context Models

Day05-Day07：

- Day05：字符级 tokenizer
- Day06：在文本上训练 Bigram LM
- Day07：固定窗口 MLP LM

路径：

```text
docs/02_text_and_context_models/
exercises/02_text_and_context_models/
logs/02_text_and_context_models/
```

## 03 Attention Foundations

Day08-Day13：

- Day08：Positional Embedding
- Day09：Attention Scores
- Day10：Q/K/V Self-Attention
- Day11：Causal Mask
- Day12：CausalSelfAttention Module
- Day13：Multi-Head Attention

路径：

```text
docs/03_attention_foundations/
exercises/03_attention_foundations/
logs/03_attention_foundations/
```

## 04 Transformer LM

Day14-Day18：

- Day14：Feed-Forward Network
- Day15：Residual Connection
- Day16：LayerNorm
- Day17：Tiny Transformer LM forward
- Day18：训练 Tiny Transformer LM

路径：

```text
docs/04_transformer_lm/
exercises/04_transformer_lm/
logs/04_transformer_lm/
```

## 05 Inference Generation

Day19-Day27：

- Day19：Generation Mechanics
- Day20：Prefill / Decode
- Day21：KV Cache Intro
- Day22：KV Cache Memory
- Day23：Inference Metrics
- Day24：Benchmark Workload Design
- Day25：Toy Benchmark Runner
- Day26：Request Timing
- Day27：Concurrent Benchmark

路径：

```text
docs/05_inference_generation/
exercises/05_inference_generation/
logs/05_inference_generation/
```

## 06 Realistic Transformer Components

Day28-Day32：

- Day28：RMSNorm
- Day29：SwiGLU
- Day30：RoPE
- Day31：MHA / MQA / GQA
- Day32：Tiny Llama-style Block

路径：

```text
docs/06_realistic_transformer_components/
exercises/06_realistic_transformer_components/
logs/06_realistic_transformer_components/
```

## 07 KV Cache Paged Attention

Day33-Day43：

- Day33：Multi-head KV Cache Layout
- Day34：Prefill Attention Scaling
- Day35：Decode Attention Scaling
- Day36：PyTorch CUDA Memory Analysis
- Day37：KV Block Allocator
- Day38：PagedAttention Page Table
- Day39：Toy Serving Scheduler
- Day40：Static vs Continuous Batching
- Day41：Prefix Cache Toy
- Day42：vLLM Readiness Check
- Day43：vLLM Server Prep 和 Smoke Test

路径：

```text
docs/07_kv_cache_paged_attention/
exercises/07_kv_cache_paged_attention/
logs/07_kv_cache_paged_attention/
```

## 下一阶段

Day44 已开始，因此创建 `08_serving_vllm` 阶段目录。

## 08 Serving vLLM

当前阶段：

```text
docs/08_serving_vllm/
exercises/08_serving_vllm/
```

已创建：

```text
Day44：vLLM Python Smoke Client
Day45：Streaming Client，记录 TTFT / TPOT
Day46：Warmup + Repeated Single-request Benchmark
```

后续预计：

```text
Day47：Workload benchmark
Day48：vLLM 参数实验
```
