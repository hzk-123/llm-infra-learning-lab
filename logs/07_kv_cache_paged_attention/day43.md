# Daily Log

Date: 2026-07-01

Updated: 2026-07-02

## Topic

Day43：第一次启动真实 vLLM OpenAI-compatible server。

这一天的目标不是压测，也不是优化性能，而是完成从 toy serving 到真实模型服务的第一步：

```text
启动 vLLM server
加载 Qwen2.5-0.5B-Instruct
验证 /v1/models
验证 /v1/chat/completions
理解启动日志里最重要的阶段
```

## Should I Learn Every Startup Detail Now?

不需要一次性全部学透。

vLLM 启动日志里包含很多层次：

```text
模型结构
权重下载和加载
推理 dtype
FlashAttention
FlashInfer
torch.compile
CUDA graph
KV cache
scheduler
API server
sampling config
Triton JIT
```

这些都重要，但学习优先级不同。

当前阶段必须掌握的是：

```text
模型为什么能被 vLLM 加载
服务为什么能通过 OpenAI-compatible API 被访问
KV cache memory 是什么
max_model_len / gpu_memory_utilization 会影响什么
/v1/models 和 /v1/chat/completions 分别验证什么
启动日志中哪些是正常信息，哪些才是错误
```

现在只需要知道大概作用，后面再深入的是：

```text
torch.compile / Inductor
CUDA graph capture
FlashAttention kernel 细节
FlashInfer sampling kernel
Triton JIT
NCCL / TP / PP / DP
```

以后到 CUDA、Nsight、vLLM 源码阶段，再把这些展开。

不要在 Day43 就试图把所有底层细节啃完，否则会偏离当前目标。

## Environment

服务器环境：

```text
conda env: llm-sprint
python: 3.10.20
torch: 2.11.0+cu130
CUDA: 13.0
vLLM: 0.24.0
GPU: 3 x NVIDIA GeForce RTX 4090 24GB
```

Day43 启动时只使用第 0 张 GPU：

```text
CUDA_VISIBLE_DEVICES=0
```

## Server Config

启动脚本输出的配置：

```text
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
SERVED_MODEL_NAME=qwen2.5-0.5b-instruct
HOST=0.0.0.0
PORT=8000
DTYPE=auto
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.85
CUDA_VISIBLE_DEVICES=0
HF_HOME=/home/hzk/llm-sprint/model_cache/huggingface
HUGGINGFACE_HUB_CACHE=/home/hzk/llm-sprint/model_cache/huggingface/hub
```

这些参数的含义：

```text
MODEL_ID
  Hugging Face 上的真实模型 ID。

SERVED_MODEL_NAME
  对外 API 调用时使用的模型名。
  curl 请求里填写的是这个名字，而不是必须填写 Hugging Face 原始 ID。

HOST=0.0.0.0
  监听所有网卡。
  本机测试仍然可以用 127.0.0.1 访问。

PORT=8000
  API server 监听端口。

DTYPE=auto
  让 vLLM 根据模型和硬件自动选择 dtype。
  本次日志里实际使用的是 torch.bfloat16。

MAX_MODEL_LEN=2048
  单个请求最大上下文长度。
  它会影响 KV cache 容量规划。

GPU_MEMORY_UTILIZATION=0.85
  vLLM 可以使用 GPU 显存的目标比例。
  比例越高，理论上可留给 KV cache 的空间越大，但 OOM 风险也可能更高。

CUDA_VISIBLE_DEVICES=0
  只让 vLLM 看到第 0 张 GPU。

HF_HOME / HUGGINGFACE_HUB_CACHE
  模型下载和缓存目录。
```

## Commands

启动 vLLM server：

```text
conda activate llm-sprint
cd ~/llm-infra-learning-lab
tmux new -s vllm-day43
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

健康检查：

```text
curl http://127.0.0.1:8000/v1/models
```

聊天请求：

```text
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "messages": [
      {"role": "user", "content": "你好，用一句话介绍你自己。"}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

## What Happened During Startup

### 1. vLLM CLI 启动

日志开头显示：

```text
vLLM version 0.24.0
model Qwen/Qwen2.5-0.5B-Instruct
```

说明 vLLM 命令行服务进程已经启动，并读取到了目标模型。

随后日志中出现：

```text
non-default args:
model_tag: Qwen/Qwen2.5-0.5B-Instruct
host: 0.0.0.0
max_model_len: 2048
served_model_name: qwen2.5-0.5b-instruct
gpu_memory_utilization: 0.85
```

这说明启动脚本里的关键参数已经生效。

### 2. 解析模型架构

日志：

```text
Resolved architecture: Qwen2ForCausalLM
Using max model len 2048
```

含义：

```text
vLLM 识别出这个模型是 Qwen2ForCausalLM。
它是 decoder-only causal language model。
也就是我们前面从 Bigram、MLP LM、Tiny Transformer 一路学到的 next-token prediction 类型模型。
```

这里不需要现在读完 Qwen2 源码。

当前要会解释：

```text
Qwen2ForCausalLM 表示这是一个用于自回归生成的 causal LM。
输入 prompt 后，模型逐 token 预测下一个 token。
```

### 3. 初始化 vLLM Engine

日志：

```text
Initializing a V1 LLM engine (v0.24.0)
dtype=torch.bfloat16
max_seq_len=2048
tensor_parallel_size=1
pipeline_parallel_size=1
data_parallel_size=1
device_config=cuda
enable_prefix_caching=True
enable_chunked_prefill=True
```

重点解释：

```text
dtype=torch.bfloat16
  模型权重和计算使用 bf16。
  bf16 比 fp32 省显存，也适合 4090 这类现代 GPU。

max_seq_len=2048
  本次服务最大上下文长度是 2048。

tensor_parallel_size=1
  本次没有做 tensor parallel。
  只用了单卡。

pipeline_parallel_size=1
  本次没有做 pipeline parallel。

data_parallel_size=1
  本次没有做 data parallel。

device_config=cuda
  使用 NVIDIA GPU。

enable_prefix_caching=True
  vLLM 开启了 prefix caching。
  这和 Day41 的 prefix cache toy 对应。

enable_chunked_prefill=True
  vLLM 可以把长 prompt 的 prefill 拆成 chunk，更方便调度和混合处理。
```

### 4. 初始化分布式状态

日志：

```text
world_size=1 rank=0 local_rank=0 backend=nccl
TP rank 0
```

含义：

```text
虽然这次只用了 1 张 GPU，vLLM 仍然会初始化一套分布式通信框架。
world_size=1 表示只有一个 worker。
TP rank 0 表示 tensor parallel 只有一个 rank。
```

当前不需要深入 NCCL。

现在只要知道：

```text
这次是单卡 serving，不是多卡 tensor parallel。
```

等后面 Day51 做 TP=1/2/3，再重新看 NCCL 和 rank。

### 5. Sampling 后端

日志：

```text
Using FlashInfer for top-p & top-k sampling.
```

含义：

```text
生成 token 时，模型会先输出 logits。
然后根据 temperature、top-k、top-p 等采样参数选下一个 token。
vLLM 使用 FlashInfer 来加速 top-k / top-p sampling。
```

当前需要掌握：

```text
top-k / top-p 是采样策略。
FlashInfer 是用于加速采样等推理相关操作的后端。
```

暂时不需要深入 FlashInfer kernel 实现。

### 6. Attention 后端

日志：

```text
Using FLASH_ATTN attention backend
Using FlashAttention version 2
```

含义：

```text
vLLM 在 attention 计算中使用 FlashAttention 2。
它可以更高效地计算 attention，减少中间 attention matrix 的显存读写。
```

这和我们之前学的内容对应：

```text
Day34 prefill attention scaling
  普通 attention scores 是 [B, H, T, T]，长 prompt 下开销很大。

FlashAttention
  不直接把完整 attention matrix 大量落到显存中，而是用更高效的分块计算方式。
```

当前只需知道：

```text
FlashAttention 是优化 attention 计算和显存访问的核心技术。
```

CUDA kernel 细节后面再学。

### 7. 下载和加载模型权重

日志：

```text
Time spent downloading weights: 162.908305 seconds
Checkpoint size: 0.92 GiB
Loading weights took 0.25 seconds
Model loading took 0.93 GiB memory and 168.304875 seconds
```

含义：

```text
第一次启动时，模型权重需要从 Hugging Face 下载。
下载耗时约 162.9 秒。
模型 checkpoint 大小约 0.92 GiB。
加载权重本身很快，约 0.25 秒。
整体模型加载耗时约 168.3 秒，主要时间花在下载。
```

这说明：

```text
第一次启动慢，不一定是 vLLM 慢，也可能是模型下载慢。
模型缓存后，下一次启动通常会更快。
```

### 8. torch.compile 编译

日志：

```text
Using cache directory: /home/hzk/.cache/vllm/torch_compile_cache/...
Dynamo bytecode transform time: 4.85 s
Compiling a graph for compile range (1, 2048) takes 6.72 s
torch.compile took 14.35 s in total
```

含义：

```text
vLLM 使用 torch.compile / Dynamo / Inductor 对部分模型计算图做编译优化。
这样后续推理可以减少 Python 调度开销，并让算子执行更高效。
```

当前只需要知道：

```text
torch.compile 是启动时的一次性优化成本。
第一次启动或新 shape 可能慢一些。
后续推理可能因此更快。
```

暂时不需要深入 Dynamo、Inductor 的内部实现。

### 9. Profiling / Warmup

日志：

```text
Initial profiling/warmup run took 0.23 s
```

含义：

```text
vLLM 会做一些 profiling 和 warmup，用来了解当前模型、硬件、显存状态。
```

这一步和后面的 KV cache 规划有关。

当前要知道：

```text
vLLM 启动时不是简单加载模型就结束，它还会做 profiling。
```

### 10. CUDA Graph Memory Profiling

日志：

```text
Profiling CUDA graph memory
Estimated CUDA graph memory: 0.28 GiB total
CUDA graph pool memory: 0.38 GiB actual
```

含义：

```text
vLLM 会捕获 CUDA graph。
CUDA graph 可以减少推理时反复 launch kernel 的 CPU 开销。
```

当前只需要知道：

```text
CUDA graph 是一种减少 GPU kernel launch overhead 的技术。
它会占用额外显存。
```

CUDA graph 的深入细节放到后面的 CUDA / Nsight 阶段。

### 11. 创建 KV Cache

日志：

```text
Available KV cache memory: 18.08 GiB
GPU KV cache size: 1,579,808 tokens
Maximum concurrency for 2,048 tokens per request: 771.39x
```

这是 Day43 最重要的推理系统信息之一。

含义：

```text
vLLM 在 GPU 上规划了约 18.08 GiB 显存用于 KV cache。
这些 KV cache 空间理论上可以容纳 1,579,808 个 token 的 K/V。
如果每个请求都占满 2048 tokens，理论 KV 容量约等于 771 个请求。
```

但要注意：

```text
Maximum concurrency for 2,048 tokens per request: 771.39x
```

不是说真实系统一定能同时高质量处理 771 个请求。

它主要表示：

```text
从 KV cache 容量角度，最多能放下多少个满长请求。
```

真实并发还会受这些限制：

```text
模型计算吞吐
调度开销
网络和 HTTP 开销
输出长度
batch 形状
GPU 利用率
请求到达模式
```

这和前面几天对应：

```text
Day21: KV cache intro
Day22: KV cache memory formula
Day31: GQA / MQA 降低 KV cache
Day33: multi-head KV cache layout
Day37: KV block allocator
Day40: continuous batching
Day41: prefix cache
```

Day43 是第一次看到真实 vLLM 给出的 KV cache 规划数据。

### 12. 捕获 CUDA Graph

日志：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
Capturing CUDA graphs (decode, FULL)
Graph capturing finished in 2 secs, took 0.38 GiB
```

含义：

```text
vLLM 对一些常见推理形状捕获 CUDA graph。
包括混合 prefill-decode 和 decode 阶段。
```

当前只需要知道：

```text
这是为了减少后续推理时的 kernel launch overhead。
它会消耗一些额外显存。
```

### 13. 启动 API Server

日志：

```text
Starting vLLM server on http://0.0.0.0:8000
Available routes are:
/v1/models
/v1/chat/completions
/v1/completions
/health
/metrics
/tokenize
/detokenize
...
Application startup complete.
```

关键成功标志：

```text
Application startup complete.
```

这说明：

```text
FastAPI / ASGI server 已经启动完成。
OpenAI-compatible API 已经可以访问。
```

## API Validation

### /v1/models

命令：

```text
curl http://127.0.0.1:8000/v1/models
```

返回关键信息：

```text
id: qwen2.5-0.5b-instruct
root: Qwen/Qwen2.5-0.5B-Instruct
max_model_len: 2048
owned_by: vllm
```

说明：

```text
server 已经注册模型。
OpenAI-compatible 模型列表接口可用。
```

### /v1/chat/completions

命令：

```text
curl http://127.0.0.1:8000/v1/chat/completions ...
```

返回内容：

```text
我是来自阿里云的超大规模语言模型，我叫通义千问。
```

返回 usage：

```text
prompt_tokens: 36
completion_tokens: 18
total_tokens: 54
```

说明：

```text
模型能接收 chat messages。
模型能生成回答。
vLLM 能返回 token usage。
```

这就是 Day43 的核心验收。

## Normal Logs vs Things To Watch

### 正常日志

这些都不是错误：

```text
GET / HTTP/1.1 404 Not Found
GET /favicon.ico HTTP/1.1 404 Not Found
```

原因：

```text
vLLM 不是网页首页服务。
根路径 / 没有页面，所以返回 404 正常。
```

正确的 API 路径是：

```text
/v1/models
/v1/chat/completions
/health
/metrics
```

### 需要注意的 warning

日志：

```text
Default vLLM sampling parameters have been overridden by the model's generation_config.json
```

含义：

```text
模型自带 generation_config.json。
其中的 repetition_penalty、temperature、top_k、top_p 等默认采样参数覆盖了 vLLM 默认值。
```

这不是错误。

但是做 benchmark 时，为了实验可控，最好在请求里显式指定：

```text
temperature
top_p
max_tokens
```

如果要强制使用 vLLM 默认 generation config，可以研究：

```text
--generation-config vllm
```

后面 benchmark 阶段再处理。

另一个 warning：

```text
Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel.
This causes a latency spike.
```

含义：

```text
第一次真实请求时，有 Triton kernel 临时 JIT 编译。
这会让某一次请求延迟变高。
```

这说明后续做 latency benchmark 前必须先 warmup。

不能把第一次请求的延迟直接当作稳定性能。

## tmux Operations

### 让 vLLM 继续运行，但退出当前窗口

在 tmux 里按：

```text
Ctrl-b
d
```

这叫 detach。

服务还在后台运行。

### 重新进入 tmux

```text
tmux attach -t vllm-day43
```

### 停止 vLLM server

进入 tmux：

```text
tmux attach -t vllm-day43
```

然后按：

```text
Ctrl-c
```

这会停止 vLLM server。

然后输入：

```text
exit
```

关闭 tmux 里的 shell。

### 强制关闭整个 tmux session

如果不想进入 tmux，也可以：

```text
tmux kill-session -t vllm-day43
```

### 检查是否关闭成功

```text
tmux ls
nvidia-smi
curl http://127.0.0.1:8000/v1/models
```

如果：

```text
tmux session 不存在
GPU 显存释放
curl 连接失败
```

说明服务已经关闭。

### 下次重新启动

```text
conda activate llm-sprint
cd ~/llm-infra-learning-lab
git pull
tmux new -s vllm-day43
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

如果 session 已经存在：

```text
tmux attach -t vllm-day43
```

如果端口 8000 已被旧服务占用：

```text
PORT=8001 bash scripts/start_vllm_server.sh
```

## What I Actually Did

这一天实际完成的是：

```text
1. 在服务器 llm-sprint 环境中启动 vLLM。
2. 使用单张 4090 加载 Qwen2.5-0.5B-Instruct。
3. 让 vLLM 创建 KV cache。
4. 让 vLLM 完成 torch.compile、warmup、CUDA graph capture。
5. 启动 OpenAI-compatible HTTP API server。
6. 用 /v1/models 验证模型注册成功。
7. 用 /v1/chat/completions 验证模型可以生成回答。
8. 确认 response 中包含 token usage。
9. 初步理解 vLLM 启动过程中的关键日志。
```

## What I Did Not Do Yet

Day43 没有做：

```text
没有做并发压测。
没有测 TTFT。
没有测 TPOT。
没有测 p50 / p95 latency。
没有测 QPS。
没有测 output tokens/s。
没有比较 Transformers vs vLLM。
没有使用多卡 tensor parallel。
没有部署量化模型。
没有换 1.5B / 7B 模型。
没有分析 Nsight / CUDA kernel。
```

这些后面再做。

Day43 的边界很清楚：

```text
只验证真实 vLLM server 能跑通。
```

## Key Takeaways

最重要的理解：

```text
vLLM server = 模型加载 + 推理引擎 + KV cache 管理 + scheduler + HTTP API。
```

这不是简单的：

```text
python model.generate()
```

它额外做了很多 serving 层工作：

```text
显存规划
KV cache 预分配
prefix caching
chunked prefill
continuous batching
CUDA graph
API route 管理
token usage 统计
```

现在应该能解释：

```text
为什么启动 vLLM 比普通 Python 脚本复杂。
为什么第一次启动会慢。
为什么 KV cache memory 是 serving 系统的关键资源。
为什么 /v1/models 和 /v1/chat/completions 是最小验收。
为什么后续 benchmark 前需要 warmup。
```

## Day43 Conclusion

Day43 已通过。

当前已经完成：

```text
vLLM server 启动成功。
Qwen2.5-0.5B-Instruct 加载成功。
OpenAI-compatible /v1/models 可访问。
OpenAI-compatible /v1/chat/completions 可生成。
token usage 可返回。
启动日志中的关键阶段已完成初步理解。
```

下一步不要急着大压测。

建议先复盘 Day43，再进入：

```text
Day44：vLLM Python smoke client
```

Day44 的目标应该是：

```text
用 Python client 稳定调用本地 vLLM server。
记录 latency、prompt_tokens、completion_tokens、total_tokens。
为后续 streaming TTFT / TPOT benchmark 做准备。
```

