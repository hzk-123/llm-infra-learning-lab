# Day 43：vLLM OpenAI-compatible Server Prep

## 目标

Day42 已经确认服务器具备 vLLM readiness。

Day43 不做压测，只做启动前准备：

```text
准备 vLLM OpenAI-compatible server 的启动命令、模型缓存目录、日志目录和 smoke test 命令。
```

## 为什么先用小模型

第一次启动服务，目标不是追求性能数据，而是验证链路：

```text
模型能否下载
vLLM server 能否启动
端口是否正常监听
OpenAI-compatible API 能否访问
日志是否保存
GPU 显存是否被占用
```

所以默认 smoke model 使用：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

等 Day44 跑通后，再切换到：

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-7B-Instruct
```

## 推荐启动方式

服务器上使用 tmux：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab

tmux new -s vllm-day43
```

在 tmux 里启动：

```bash
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

默认脚本会设置：

```text
CUDA_VISIBLE_DEVICES=0
HF_HOME=~/llm-sprint/model_cache/huggingface
HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
SERVED_MODEL_NAME=qwen2.5-0.5b-instruct
PORT=8000
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.85
```

## 如何退出和回到 tmux

启动后按：

```text
Ctrl-b，然后按 d
```

可以 detach，不会停止 server。

重新进入：

```bash
tmux attach -t vllm-day43
```

停止 server：

```text
在 tmux 窗口里按 Ctrl-c
```

## 启动后检查

另开一个 SSH 终端：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
```

检查 GPU：

```bash
nvidia-smi
```

检查 API：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果能返回模型列表，说明 OpenAI-compatible server 已经可访问。

## 你应该掌握

完成 Day43 后，你应该能解释：

```text
为什么第一次用小模型做 smoke test。
为什么 vLLM 服务要放在 tmux 里运行。
为什么要固定 HF_HOME 和日志路径。
为什么 OpenAI-compatible API 的 /v1/models 是最小健康检查。
为什么启动命令里的 max_model_len、gpu_memory_utilization 会影响显存使用。
```

