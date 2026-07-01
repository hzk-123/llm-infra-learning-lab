# LLM Infra Learning Lab

This is a fresh, incremental learning project for AI Infra / LLM inference optimization.

The goal is not to build a huge project on day one. The project grows in small steps:

1. PyTorch + next-token prediction
2. Tiny language model training
3. Tiny decoder-only Transformer
4. Prefill / decode / KV cache
5. realistic Llama/Qwen-style Transformer components
6. KV cache, GQA, and attention compute experiments
7. toy PagedAttention and continuous batching
8. vLLM serving and real benchmarking
9. quantized inference and tensor parallel serving
10. C++ TensorBuffer and CUDA kernels
11. Nsight performance analysis
12. DeepSpeed / Megatron bonus modules
13. a resume-ready AI Infra project report

## Current Rule

Learn one small idea, run one small experiment, then add one small piece of code.

Do not jump to vLLM, CUDA, or Megatron before the PyTorch and LLM basics are clear.

Day 01-27 are complete and now serve as prerequisites. Day 28 starts the V3 track by moving from teaching Transformer blocks toward Llama/Qwen-style components.

## Project Layout

```text
exercises/      Small PyTorch, scheduler, cache, and systems exercises
tiny_lm/        Bigram, MLP LM, and Tiny Transformer implementations
inference/      Prefill/decode, KV cache, and generation benchmarks
serving/        FastAPI, vLLM client, and OpenAI-compatible requests
benchmarks/     Concurrency, long-context, prefix-cache, quantization, multi-GPU tests
cpp_basics/     C++ bonus exercises
cuda_kernels/   CUDA bonus kernels
distributed/    DeepSpeed, Megatron, NCCL notes and demos
docs/           Notes, learning plan, experiment templates, source reading
results/        CSV and Markdown benchmark results
logs/           Daily learning logs
```

Day-based files inside `docs/`, `exercises/`, and `logs/` are grouped by milestone:

```text
01_pytorch_lm_basics        Day 01-04
02_text_and_context_models  Day 05-07
03_attention_foundations    Day 08-13
04_transformer_lm           Day 14-18
05_inference_generation     Day 19-27
06_realistic_transformer_components Day 28-32
07_kv_cache_paged_attention Day 33-43
08_serving_vllm             Day 44-48
```

## Start Here

Run Day 1:

```powershell
cd llm-infra-learning-lab
conda activate llm-sprint
python exercises\01_pytorch_lm_basics\day01_next_token_data.py
```

Read:

```text
docs/milestone_index.md
docs/01_pytorch_lm_basics/day01_next_token_prediction.md
docs/learning_plan_v3.md
```

## Daily Output

Every learning day should leave at least one artifact:

- a short log
- a code exercise
- an experiment result
- a note explaining one concept in your own words
- a small benchmark table
