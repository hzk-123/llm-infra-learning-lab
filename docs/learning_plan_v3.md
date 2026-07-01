# LLM Infra / High-Performance Inference Learning Plan V3

## Target

This project now has two job targets:

- Primary: LLM serving and performance engineering intern.
- Stretch: LLM inference framework / CUDA kernel optimization intern.

The project should grow from small PyTorch experiments into a portfolio that covers:

- realistic Transformer components
- KV cache and attention compute experiments
- toy PagedAttention and continuous batching
- vLLM serving benchmarks
- quantization and tensor parallel inference
- C++ TensorBuffer basics
- CUDA kernels and Nsight analysis
- DeepSpeed / Megatron awareness

Day 01-27 are already complete and should be treated as prerequisites, not repeated.

## Current State

Completed:

```text
Day 01-04: PyTorch next-token data, embedding, Bigram LM, training loop
Day 05-07: tokenizer, text training, MLP LM
Day 08-13: positional embedding, attention, Q/K/V, causal mask, MHA
Day 14-18: FFN, residual, LayerNorm, Tiny Transformer LM
Day 19-23: generation, prefill, decode, KV cache, inference metrics
Day 24-27: workload design, request timing, concurrent benchmark
```

Next rule:

```text
Do not keep expanding shallow fake benchmarks.
Move back into model and inference internals.
```

## Phase 1: Realistic Transformer Components

Goal: move from teaching Transformer blocks toward Llama/Qwen-style components.

Planned days:

- Day 28: RMSNorm vs LayerNorm
- Day 29: SwiGLU FFN
- Day 30: RoPE
- Day 31: MHA / MQA / GQA shape and KV memory
- Day 32: tiny Llama-style block

Acceptance:

- Implement minimal PyTorch versions.
- Print shapes and key statistics.
- Explain why modern LLMs use these components.

## Phase 2: KV Cache And Inference Compute

Goal: measure how KV cache and attention compute affect inference.

Planned days:

- Day 33: multi-head KV cache layout and append `[B, H_kv, T, D]`
- Day 34: prefill attention scaling
- Day 35: decode attention scaling
- Day 36: PyTorch CUDA memory analysis on the server

Acceptance:

- Explain prefill and decode bottlenecks with measured data.
- Connect TTFT / TPOT to attention and KV cache behavior.

## Phase 3: Toy PagedAttention And Continuous Batching

Goal: understand vLLM's core system ideas before reading source code.

Planned days:

- Day 37: KV block allocator
- Day 38: toy PagedAttention page table
- Day 39: toy serving scheduler
- Day 40: continuous batching step-level scheduler
- Day 41: prefix cache toy
- Day 42: vLLM readiness checklist
- Day 43: vLLM OpenAI-compatible server prep

Acceptance:

- Draw request -> scheduler -> KV allocator -> decode step.
- Explain why block-based KV management and dynamic batching improve serving.

## Phase 4: vLLM Serving And Real Benchmark

Goal: replace toy timing with real model serving measurements.

Planned days:

- Day 44: start vLLM OpenAI-compatible server
- Day 45: streaming benchmark client
- Day 46: workload benchmark
- Day 47: vLLM parameter experiments
- Day 48: Transformers vs vLLM

Acceptance:

- Collect real TTFT, TPOT, QPS, output tokens/s, p95 latency, and GPU memory.
- Save benchmark results as CSV or Markdown.

## Phase 5: Quantization And Multi-GPU Inference

Goal: handle common LLM serving tradeoffs around memory, latency, and scale.

Planned days:

- Day 49: quantization basics
- Day 50: deploy a quantized model
- Day 51: vLLM tensor parallel TP=1/2/3
- Day 52: NCCL and multi-GPU debugging notes

Acceptance:

- Explain why quantization saves memory and why it may not always speed up.
- Explain tensor parallel tradeoffs with real measurements.

## Phase 6: C++ Inference Framework Basics

Goal: prepare for inference framework and CUDA work without overclaiming.

Planned days:

- Day 53: C++ / CMake skeleton
- Day 54: CPU TensorBuffer
- Day 55: RAII and ownership
- Day 56: dtype dispatch
- Day 57: CPU add, naive matmul, RMSNorm

Acceptance:

- Build with CMake.
- Implement correctness checks.
- Explain shape, dtype, data pointer, and ownership.

## Phase 7: CUDA Kernels

Goal: implement simple kernels that map to LLM inference operators.

Planned days:

- Day 58: CUDA vector add
- Day 59: memory coalescing
- Day 60: reduction
- Day 61: RMSNorm CUDA
- Day 62: Softmax CUDA
- Day 63: naive matmul -> tiled matmul
- Day 64: INT8 quantization demo

Acceptance:

- Run kernels on the server.
- Compare outputs against PyTorch or CPU references.
- Record latency and correctness.

## Phase 8: Nsight Performance Analysis

Goal: learn how to identify kernel bottlenecks.

Planned days:

- Day 65: Nsight Systems
- Day 66: Nsight Compute
- Day 67: RMSNorm optimization notes
- Day 68: MatMul performance analysis
- Day 69: CUDA kernel report

Acceptance:

- Identify at least one memory-bound kernel.
- Write a short performance report with before/after numbers.

## Phase 9: DeepSpeed / Megatron / Distributed Bonus

Goal: understand training-side infra concepts without making them the main project.

Planned days:

- Day 70: DeepSpeed ZeRO demo
- Day 71: activation checkpointing
- Day 72: Megatron-LM parallelism reading
- Day 73: NCCL collectives

Acceptance:

- Explain ZeRO 1/2/3.
- Explain tensor, pipeline, and sequence parallelism.
- Distinguish training parallelism from inference tensor parallelism.

## Portfolio Output

Final README should show:

- Tiny LLM principles
- KV cache and GQA experiments
- toy PagedAttention and scheduler
- vLLM real benchmark tables
- quantization comparison
- multi-GPU tensor parallel comparison
- C++ TensorBuffer
- CUDA kernels
- Nsight performance analysis
- reproducible commands and bad cases
