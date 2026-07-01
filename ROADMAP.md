# Roadmap

This roadmap keeps the full plan visible while preserving an incremental learning pace.

V3 target:

- Primary: LLM serving and performance engineering intern.
- Stretch: LLM inference framework / CUDA kernel optimization intern.

## Stage 1: PyTorch Foundations

- Tensor shape, dtype, and device
- Dataset and DataLoader
- `nn.Module`
- forward, loss, backward
- optimizer step and zero grad
- checkpoint save/load

Deliverable: a tiny next-token model that can train, save, load, and generate.

## Stage 2: LLM Principles

- next-token prediction
- embedding
- attention
- causal mask
- decoder-only Transformer
- RoPE
- RMSNorm
- SwiGLU

Deliverable: a small decoder-only Transformer implemented in PyTorch.

## Stage 3: Inference Mechanics

- prefill
- decode
- KV cache
- sampling
- TTFT
- TPOT
- KV cache memory estimate

Deliverable: a generation benchmark comparing cache and no-cache decoding.

## Stage 4: Realistic Transformer Components

- RMSNorm
- SwiGLU
- RoPE
- MHA / MQA / GQA
- Llama/Qwen-style block structure

Deliverable: small PyTorch modules that mirror modern LLM building blocks.

## Stage 5: KV Cache And Attention Compute

- no-cache vs KV-cache generation
- multi-head KV cache layout
- prefill attention scaling
- decode attention scaling
- CUDA memory analysis with PyTorch

Deliverable: benchmark tables that connect TTFT / TPOT to attention and KV cache behavior.

## Stage 6: Toy PagedAttention And Scheduling

- KV block allocator
- page table
- request lifecycle
- continuous batching
- prefix cache

Deliverable: a toy serving report that explains vLLM's core system ideas.

## Stage 7: vLLM Serving

- OpenAI-compatible server
- PagedAttention
- continuous batching
- prefix cache
- request lifecycle

Deliverable: vLLM benchmark table for a Qwen-class model.

## Stage 8: Performance Engineering

- QPS
- tokens/s
- p50/p95 latency
- GPU memory
- concurrency
- long-context workload

Deliverable: reproducible benchmark report.

## Stage 9: Quantization and Multi-GPU

- FP16/BF16
- INT8/INT4
- AWQ/GPTQ
- tensor parallel
- NCCL basics

Deliverable: quantization and 1/2/3 GPU serving comparisons.

## Stage 10: C++ And CUDA Systems Track

- C++ / CMake
- TensorBuffer
- RAII and dtype dispatch
- CUDA vector add, reduction, RMSNorm, Softmax, MatMul
- Nsight Systems and Nsight Compute

Deliverable: correctness-checked kernels and a performance report.

## Stage 11: Distributed Bonus Track

- DeepSpeed ZeRO
- Megatron-LM parallelism
- NCCL collectives
- vLLM source reading

Deliverable: notes and small demos that strengthen AI Infra interviews.
