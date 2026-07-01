# Daily Log

Date:

## Learned

-

## Ran

```text

```

## My Explanation

-

## Next


# AI Infra / LLM 推理优化完整学习计划 V2

## Summary

目标定位：**大模型推理服务与性能工程实习生**。  
主线是 PyTorch、LLM 原理、vLLM 推理服务、性能压测、量化、多卡推理；同时把 **C++、CUDA、Megatron、DeepSpeed** 保留并整合为后期加分模块，不一开始硬啃，但不丢掉。

最终项目：`llm-infra-learning-lab`，从最小 PyTorch 练习逐步长成 AI Infra 作品集。

## Core Track

| 模块 | 必学内容 | 实践任务 | 产出 |
|---|---|---|---|
| Python/Linux/Git | argparse、logging、JSONL、项目结构、tmux、nvidia-smi、ps/kill、du/df、Git | 写可复现实验脚本，所有实验有命令和日志 | CLI 模板、daily log、实验记录 |
| 数据结构算法 | 数组、哈希表、堆、队列、排序、二分、图基础；重点理解调度、缓存、批处理 | 实现 LRU cache、priority queue、简单 batch scheduler | `scheduler_basics.py` |
| PyTorch | Tensor、shape、dtype、device、Module、DataLoader、loss、backward、optimizer、checkpoint | 从 next-token dataset 写到 TinyLM 训练 loop | tiny LM 可训练、可保存、可生成 |
| PyTorch 显存分析 | batch size、seq length、activation、optimizer state、`torch.cuda.memory_allocated` | 比较不同 batch/seq_len 显存变化 | 显存实验表 |
| LLM 原理 | next-token prediction、Embedding、Attention、causal mask、Transformer decoder、RoPE、RMSNorm、SwiGLU | 从零实现 tiny decoder-only Transformer | `tiny_transformer.py` + 原理笔记 |
| 推理机制 | prefill、decode、KV Cache、sampling、TTFT、TPOT | 对比有无 KV cache；计算 KV cache 显存 | `generation_benchmark.py` + 推理笔记 |
| vLLM | OpenAI-compatible server、PagedAttention、continuous batching、prefix cache | 部署 Qwen 小模型，写压测脚本 | vLLM benchmark 表 |
| 性能指标 | QPS、tokens/s、TTFT、TPOT、total latency、p50/p95、显存、成功率 | 短 prompt、长 prompt、并发、RAG workload 压测 | `benchmark_results.csv` |
| 模型部署 | FastAPI、OpenAI-compatible API、服务启动、健康检查 | 封装 inference API，调用 vLLM server | `serve.py`、`client.py`、部署文档 |
| 量化压缩 | FP16/BF16、INT8/INT4、AWQ、GPTQ；了解蒸馏、剪枝 | 部署量化模型，对比显存、速度、输出质量 | 量化实验报告 |
| 多卡推理 | tensor parallel、NCCL、单卡/多卡 tradeoff | 3 张 4090 跑 1/2/3 卡 vLLM tensor parallel 对比 | 多卡 benchmark 表 |
| 工程能力 | 日志、监控、压测、OOM 分析、复现实验、bad case | 每个实验有 config、命令、日志、结果、结论 | `experiment_report.md` |

## Bonus Track

这些作为加分模块整合进计划，但顺序靠后，不阻塞主线。

| 加分模块 | 学什么 | 学到什么程度 | 产出 |
|---|---|---|---|
| C++ | 基础语法、RAII、引用/指针、STL、编译、CMake、简单性能意识 | 能读 vLLM/TensorRT-LLM 周边 C++ 代码，能写小工具 | `cpp_basics/` 小练习 |
| CUDA | thread/block/grid、global/shared memory、warp 基础、简单 kernel、memory coalescing | 能写 vector add、matrix transpose、simple softmax；理解 GPU 并行模型 | `cuda_kernels/` 小 kernel + notes |
| Triton 可选 | Triton kernel 基础、block programming、benchmark | 能看懂一些推理优化 kernel 思路 | 简单 matmul/softmax benchmark |
| DeepSpeed | ZeRO 1/2/3、activation checkpointing、配置文件、显存分片 | 能跑一个小模型训练 demo，理解训练侧显存优化 | DeepSpeed demo + ZeRO 对比笔记 |
| Megatron-LM | tensor parallel、pipeline parallel、sequence parallel 概念 | 不要求魔改源码；能解释并行策略和启动方式 | Megatron 源码/配置阅读笔记 |
| NCCL | all-reduce、broadcast、通信瓶颈、常见环境变量 | 能解释多卡为什么不线性加速，能排查基础 NCCL 问题 | NCCL notes + 多卡实验分析 |
| 源码阅读 | vLLM request lifecycle、scheduler、KV cache manager、worker | 能画出一条请求从 API 到生成 token 的路径 | `vllm_source_notes.md` |

## Recommended Order

不要一开始把所有加分项铺开。顺序固定：

1. PyTorch + next-token prediction
2. Tiny Transformer
3. prefill / decode / KV cache
4. vLLM serving + benchmark
5. 性能指标与压测
6. 量化推理
7. 多卡 tensor parallel
8. C++ 基础
9. CUDA 基础
10. DeepSpeed / Megatron 概念与小 demo
11. vLLM 源码阅读与小范围实验

## Project Outputs

`llm-infra-learning-lab` 最终包含：

- `exercises/`：PyTorch、DataLoader、training loop、scheduler、cache。
- `tiny_lm/`：Bigram、MLP LM、Tiny Transformer。
- `inference/`：prefill/decode、KV cache、有无 cache benchmark。
- `serving/`：FastAPI、vLLM client、OpenAI-compatible 请求脚本。
- `benchmarks/`：并发、长上下文、prefix cache、量化、多卡。
- `cpp_basics/`：C++ 小练习。
- `cuda_kernels/`：CUDA 小 kernel。
- `distributed/`：DeepSpeed/Megatron/NCCL 笔记和 demo。
- `docs/`：LLM 原理、vLLM、量化、分布式、源码阅读、实验报告。
- `results/`：CSV/Markdown 实验结果。

## Acceptance Criteria

完成后你应该能：

- 独立写 PyTorch next-token training loop。
- 手写 tiny decoder-only Transformer。
- 解释 prefill、decode、KV cache shape、KV cache 显存公式。
- 启动 vLLM OpenAI-compatible server。
- 写并发 benchmark，统计 TTFT、TPOT、QPS、tokens/s、p95 latency、显存。
- 解释 PagedAttention、continuous batching、prefix cache。
- 部署量化模型并比较显存、速度、输出质量。
- 用 3 张 4090 做 1/2/3 卡 tensor parallel 对比。
- 写基础 C++ 工具和简单 CUDA kernel。
- 跑通一个 DeepSpeed ZeRO 小 demo，读懂 Megatron 并行策略。
- 写出可复现实验报告和 AI Infra 简历项目。

## Assumptions

- 主线仍是 AI Infra / LLM 推理优化，不是 CUDA 专岗或训练框架研发岗。
- C++、CUDA、DeepSpeed、Megatron 是加分项，但会被保留进项目目录和学习路线。
- RAG/SFT/LoRA 暂不作为主线，只在后续作为 workload 或 adapter serving 场景加入。
- 当前第一步仍然是 PyTorch + next-token prediction，不直接跳 vLLM 或 CUDA。

