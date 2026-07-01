# Daily Log

Date: 2026-07-01

## Learned

- Day42 的目标是检查真实 vLLM 实验前的环境，而不是启动模型。
- 服务器的 GPU、PyTorch、CUDA 环境已经可用。
- 当前服务器还没有同步完整项目，所以脚本识别到的 `project_root` 是 `/`。
- 当前服务器缺少 `vllm`，因此还不能进入真实 vLLM serving。
- 真实 benchmark 前必须记录 Python、PyTorch、CUDA、GPU、vLLM、模型缓存目录和结果目录。

## Ran

服务器上临时创建 `test.py`，运行 Day42 readiness check 代码。

## Server Result

服务器 Python 环境：

```text
python: 3.10.20
python_executable: /home/hzk/.conda/envs/llm-sprint/bin/python
platform: Linux-5.15.0-179-generic-x86_64-with-glibc2.35
machine: x86_64
```

关键 Python 包：

```text
torch: installed, 2.12.0
transformers: installed, 5.8.1
vllm: missing
openai: installed, 2.36.0
requests: installed, 2.34.1
fastapi: installed, 0.136.1
uvicorn: installed, 0.46.0
```

CUDA / GPU：

```text
torch version: 2.12.0+cu130
torch CUDA version: 13.0
CUDA available: True
CUDA device count: 3
```

GPU 设备：

```text
GPU 0: NVIDIA GeForce RTX 4090, 23.52 GiB, compute capability 8.9
GPU 1: NVIDIA GeForce RTX 4090, 23.52 GiB, compute capability 8.9
GPU 2: NVIDIA GeForce RTX 4090, 23.52 GiB, compute capability 8.9
```

`nvidia-smi` 显示三张 4090 基本空闲：

```text
GPU 0 memory.used: 18 MiB
GPU 1 memory.used: 18 MiB
GPU 2 memory.used: 18 MiB
```

模型缓存环境变量：

```text
HF_HOME: not set
HUGGINGFACE_HUB_CACHE: not set
TRANSFORMERS_CACHE: not set
MODELSCOPE_CACHE: not set
```

项目目录检查：

```text
benchmarks: False
results: False
docs/performance_reports: False
logs/08_serving_vllm: False
```

这是因为这次是在服务器临时 `test.py` 中运行，不是在完整项目目录下运行。

## Readiness Summary

```text
local code/dev readiness: True
server CUDA readiness: True
server vLLM readiness: False
benchmark client readiness: True
```

这说明：

```text
服务器硬件环境 OK。
PyTorch + CUDA OK。
benchmark client 依赖基本 OK。
vLLM 缺失。
完整项目还没有同步到服务器。
```

## My Explanation

当前最重要的结论是：

```text
不是 GPU 问题，也不是 CUDA 不可用。
```

三张 4090 都能被 PyTorch 识别：

```text
CUDA device count = 3
```

而且每张卡显存约 24GB，当前几乎空闲。

这说明服务器适合做后面的 vLLM serving、并发 benchmark、量化、多卡 tensor parallel 实验。

现在不能直接进入 Day43 的原因有两个：

```text
1. 服务器还没有完整的 llm-infra-learning-lab 项目。
2. vLLM 还没有安装。
```

另外，模型缓存环境变量还没设置：

```text
HF_HOME
HUGGINGFACE_HUB_CACHE
TRANSFORMERS_CACHE
MODELSCOPE_CACHE
```

这不是致命问题，但后面下载 Qwen 模型时，最好明确设置模型缓存目录，避免模型下载到默认位置后不好管理。

## Next

下一步不要急着启动 vLLM。

先完成三件事：

```text
1. 把 llm-infra-learning-lab 同步到服务器。
2. 在服务器 llm-sprint 环境安装 vLLM。
3. 设置模型缓存目录和项目输出目录。
```

完成后重新运行：

```text
python exercises/07_kv_cache_paged_attention/day42_vllm_readiness_check.py
```

只有当服务器输出：

```text
server CUDA readiness: True
server vLLM readiness: True
```

再进入 Day43。

## Update After Installing vLLM

后续已经把完整项目同步到服务器，并在 `llm-sprint` 环境里安装了 vLLM。

重新运行 readiness check 后，服务器输出：

```text
project_root: /home/hzk/llm-infra-learning-lab
```

这说明脚本已经在完整项目目录下运行，不再是之前临时 `test.py` 的 `/` 根目录。

关键包版本：

```text
torch: installed, 2.11.0+cu130
transformers: installed, 5.8.1
vllm: installed, 0.24.0
openai: installed, 2.36.0
requests: installed, 2.34.1
fastapi: installed, 0.136.1
uvicorn: installed, 0.46.0
```

需要注意的是，安装 vLLM 后：

```text
torch version = 2.11.0+cu130
```

而之前 readiness check 时是：

```text
torch version = 2.12.0+cu130
```

这说明安装 vLLM 过程中 PyTorch 版本发生了变化。

这不是立即的问题，因为当前检查结果显示：

```text
CUDA available: True
CUDA device count: 3
```

但后续实验报告里必须记录这个版本变化，不能混用安装前后的 benchmark 数据。

GPU 状态：

```text
GPU 0: NVIDIA GeForce RTX 4090, 23.52 GiB
GPU 1: NVIDIA GeForce RTX 4090, 23.52 GiB
GPU 2: NVIDIA GeForce RTX 4090, 23.52 GiB
```

模型缓存目录：

```text
HF_HOME = /home/hzk/llm-sprint/model_cache/huggingface
HUGGINGFACE_HUB_CACHE = /home/hzk/llm-sprint/model_cache/huggingface/hub
```

项目输出目录全部存在：

```text
benchmarks: True
results: True
docs/performance_reports: True
logs/08_serving_vllm: True
```

最终 readiness summary：

```text
local code/dev readiness: True
server CUDA readiness: True
server vLLM readiness: True
benchmark client readiness: True
```

## Final Day42 Conclusion

Day42 已通过。

当前服务器已经满足进入 Day43 的条件：

```text
完整项目已同步。
3 张 RTX 4090 可用。
PyTorch CUDA 可用。
vLLM 已安装。
OpenAI / requests benchmark client 依赖可用。
模型缓存目录已设置。
benchmark/results/logs 输出目录已存在。
```

下一步可以进入：

```text
Day43：vLLM OpenAI-compatible server prep
```

