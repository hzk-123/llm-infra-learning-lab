# Day 42：vLLM Readiness Check

## 目标

今天开始从 toy serving 过渡到真实 vLLM。

这一天不启动模型，也不压测。

只做一件事：

```text
检查本机 / 服务器是否已经具备跑 vLLM 的基本条件。
```

## 为什么要先做 readiness check

真实 vLLM 实验会同时依赖：

```text
Python 环境
PyTorch + CUDA
GPU 显存
vLLM
Transformers
OpenAI-compatible client
模型缓存目录
结果输出目录
```

如果这些没有提前检查，后面启动服务时很容易把问题混在一起：

```text
到底是 CUDA 问题？
还是 vLLM 没装？
还是模型没下载？
还是显存不够？
还是 benchmark 输出目录没准备？
```

Day42 的作用就是先把这些拆开。

## 本机和服务器分工

你的本机是 4070 Laptop 8GB：

```text
适合写代码、读文档、跑轻量脚本。
不适合作为 vLLM 真实 benchmark 的主环境。
```

你的服务器是 3 张 4090 24GB：

```text
适合启动 vLLM、跑 Qwen 模型、做并发压测、量化和多卡实验。
```

所以 Day42 脚本可以在本机跑，但最终要在服务器跑一遍。

## 脚本检查什么

脚本会检查：

```text
Python 版本
操作系统
关键包是否安装
torch 版本
CUDA 是否可用
GPU 数量和显存
nvidia-smi 是否可用
模型缓存环境变量
项目 benchmark/results 目录
```

关键包包括：

```text
torch
transformers
vllm
openai
requests
fastapi
uvicorn
```

其中 `vllm` 在 Windows 本机缺失不奇怪。

但在服务器上，后面启动真实服务前最好安装好。

## 重点观察

运行后重点看：

```text
Readiness summary
Next action
```

如果服务器输出类似：

```text
CUDA available: True
GPU count: 3
vllm: installed
```

就可以进入 Day43，准备 vLLM OpenAI-compatible server 启动命令。

如果输出：

```text
vllm: missing
CUDA available: False
```

就先不要继续启动模型，要先修环境。

## 你应该掌握

学完 Day42 后，你应该能说清楚：

```text
本机和服务器分别承担什么任务。
vLLM 实验依赖哪些基础条件。
为什么真实 benchmark 前要记录 torch/CUDA/vLLM/GPU 信息。
为什么实验报告里必须写清楚环境版本和硬件配置。
```

