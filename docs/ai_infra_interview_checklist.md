# AI Infra Interview Checklist

## PyTorch

- [ ] Explain Tensor shape, dtype, and device.
- [ ] Explain Dataset and DataLoader.
- [ ] Explain `nn.Module.forward`.
- [ ] Explain loss, backward, optimizer step, and zero grad.
- [ ] Explain checkpoint content.

## LLM Principles

- [ ] Explain next-token prediction.
- [ ] Explain causal mask.
- [ ] Explain attention.
- [ ] Explain RoPE.
- [ ] Explain KV cache.
- [ ] Explain prefill and decode.

## Serving Metrics

- [ ] Explain QPS.
- [ ] Explain tokens/s.
- [ ] Explain TTFT.
- [ ] Explain TPOT.
- [ ] Explain p95 latency.
- [ ] Explain GPU memory bottlenecks.

## vLLM

- [ ] Explain OpenAI-compatible server.
- [ ] Explain continuous batching.
- [ ] Explain PagedAttention.
- [ ] Explain prefix cache.
- [ ] Explain request lifecycle at a high level.

## Quantization

- [ ] Explain FP16/BF16 vs INT8/INT4.
- [ ] Explain why quantization saves memory.
- [ ] Explain why quantization may not always be faster.
- [ ] Explain how to compare quality after quantization.

## Distributed

- [ ] Explain tensor parallel.
- [ ] Explain NCCL all-reduce.
- [ ] Explain why multi-GPU speedup is not linear.
- [ ] Explain DeepSpeed ZeRO at a high level.
- [ ] Explain Megatron tensor/pipeline parallel at a high level.
