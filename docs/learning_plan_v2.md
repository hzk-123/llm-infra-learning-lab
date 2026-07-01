# AI Infra / LLM Inference Optimization Learning Plan V2

## Target Role

Large-model inference serving and performance engineering intern.

Primary job targets:

- LLM inference deployment intern
- AI Infra large-model serving intern
- vLLM / model serving / LLMOps intern
- Large-model performance optimization intern
- Large-model platform backend intern

## Core Track

| Area | Learn | Practice | Output |
| --- | --- | --- | --- |
| Python/Linux/Git | argparse, logging, JSONL, project layout, tmux, nvidia-smi, ps/kill, du/df, Git | Write reproducible scripts and logs | CLI templates and daily logs |
| Data Structures | arrays, hash maps, heap, queue, binary search, graph basics, cache, scheduling | LRU cache, priority queue, batch scheduler | `exercises/scheduler_basics.py` |
| PyTorch | Tensor, shape, dtype, device, Module, DataLoader, loss, backward, optimizer, checkpoint | next-token dataset to TinyLM loop | trainable tiny LM |
| Memory Analysis | batch size, sequence length, activations, optimizer state, CUDA memory stats | compare memory across batch and sequence length | memory table |
| LLM Principles | next-token prediction, embedding, attention, causal mask, Transformer decoder, RoPE, RMSNorm, SwiGLU | implement tiny decoder-only Transformer | tiny Transformer and notes |
| Inference | prefill, decode, KV cache, sampling, TTFT, TPOT | compare generation with and without KV cache | generation benchmark |
| vLLM | OpenAI-compatible server, PagedAttention, continuous batching, prefix cache | deploy Qwen and benchmark requests | vLLM benchmark table |
| Metrics | QPS, tokens/s, TTFT, TPOT, latency, p95, memory, success rate | short, long, concurrent, and RAG-like workload | benchmark CSV |
| Deployment | FastAPI, OpenAI-compatible API, health check, service scripts | wrap inference API and call vLLM | server/client scripts |
| Quantization | FP16/BF16, INT8/INT4, AWQ, GPTQ, distillation and pruning awareness | deploy quantized model and compare | quantization report |
| Multi-GPU | tensor parallel, NCCL, single-vs-multi GPU tradeoffs | 1/2/3 GPU vLLM tensor parallel comparison | multi-GPU table |
| Engineering | logs, monitoring, load test, OOM analysis, reproducibility, bad cases | every experiment has config, command, log, result, conclusion | experiment report |

## Bonus Track

| Area | Learn | Expected Level | Output |
| --- | --- | --- | --- |
| C++ | syntax, RAII, references, pointers, STL, CMake | read simple C++ systems code and write tools | `cpp_basics/` |
| CUDA | grid/block/thread, global/shared memory, warp basics, simple kernels | write vector add, transpose, and simple softmax | `cuda_kernels/` |
| Triton | block programming and benchmark basics | understand common inference kernels | optional small benchmark |
| DeepSpeed | ZeRO 1/2/3, activation checkpointing, config files | run a small demo and explain memory sharding | `distributed/` demo notes |
| Megatron-LM | tensor, pipeline, sequence parallel concepts | explain parallel strategies and launch patterns | Megatron notes |
| NCCL | all-reduce, broadcast, communication bottlenecks | explain why multi-GPU scaling is not linear | NCCL notes |
| Source Reading | vLLM request lifecycle, scheduler, KV cache manager, worker | draw request path from API to token generation | `vllm_source_notes.md` |

## Recommended Order

1. PyTorch + next-token prediction
2. Tiny Transformer
3. prefill / decode / KV cache
4. vLLM serving + benchmark
5. metrics and load testing
6. quantized inference
7. tensor parallel multi-GPU serving
8. C++ basics
9. CUDA basics
10. DeepSpeed / Megatron small demos
11. vLLM source reading and small experiments

## Final Acceptance

You should be able to:

- write a PyTorch next-token training loop independently
- implement a tiny decoder-only Transformer
- explain prefill, decode, KV cache shape, and KV cache memory cost
- start a vLLM OpenAI-compatible server
- write a benchmark for TTFT, TPOT, QPS, tokens/s, p95 latency, and GPU memory
- explain PagedAttention, continuous batching, and prefix cache
- deploy and compare at least one quantized model
- compare 1/2/3 GPU tensor parallel serving on your 4090 server
- write small C++ tools and simple CUDA kernels
- run a DeepSpeed ZeRO demo and explain Megatron parallelism
- produce a reproducible AI Infra project report
