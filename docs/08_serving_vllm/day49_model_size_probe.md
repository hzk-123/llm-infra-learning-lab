# Day49：0.5B vs 7B 模型大小对 vLLM Serving 的影响

## 目标

今天不继续增加新的 benchmark 维度，只做一个非常明确的对照实验：

```text
同一台服务器
同一张 4090
同一个 max_model_len
同一个 gpu_memory_utilization
只改变模型大小：0.5B -> 7B
```

要观察：

```text
1. nvidia-smi 里的显存占用
2. vLLM 启动日志里的 Available KV cache memory
3. vLLM 启动日志里的 GPU KV cache size
4. 单请求 TTFT / TPOT / total latency
5. 7B 是否能在单张 4090 上稳定启动
```

## 先理解一个关键点

`nvidia-smi` 看到的显存占用不是模型权重本身。

vLLM 进程显存大致包含：

```text
模型权重
KV cache pool
CUDA graph
运行时 buffer
attention / sampling 临时空间
```

所以 `0.5B` 模型也可能占用 20GB 左右，这是因为 vLLM 会尽量把你允许使用的显存预留给 KV cache。

对比逻辑是：

```text
0.5B:
  权重小
  剩余显存多
  KV cache pool 大

7B:
  权重大
  剩余显存少
  KV cache pool 小
```

## 运行前准备

如果之前的 vLLM server 还在 tmux 里运行，先进入：

```bash
tmux attach -t vllm-day43
```

停止服务：

```text
Ctrl-C
exit
```

如果只是想强制关闭这个 tmux session：

```bash
tmux kill-session -t vllm-day43
```

## 实验 A：启动 0.5B baseline

在服务器上：

```bash
cd ~/llm-infra-learning-lab
conda activate llm-sprint

tmux new -s vllm-05b
```

在 tmux 里启动：

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
SERVED_MODEL_NAME=qwen2.5-0.5b-instruct \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.85 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day49_05b_server.log
```

看到 API server 启动完成后，另开一个终端运行：

```bash
cd ~/llm-infra-learning-lab
conda activate llm-sprint

python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 05b \
  --model qwen2.5-0.5b-instruct \
  --server-log logs/08_serving_vllm/day49_05b_server.log
```

运行完后停止 0.5B server。

## 实验 B：启动 7B candidate

重新开 tmux：

```bash
tmux new -s vllm-7b
```

在 tmux 里启动：

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=qwen2.5-7b-instruct \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.85 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day49_7b_server.log
```

如果启动成功，另开终端运行：

```bash
cd ~/llm-infra-learning-lab
conda activate llm-sprint

python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 7b \
  --model qwen2.5-7b-instruct \
  --server-log logs/08_serving_vllm/day49_7b_server.log
```

## 如果 7B OOM

先不要慌，这也是有效实验结果。

可以按这个顺序尝试：

```bash
GPU_MEMORY_UTILIZATION=0.80
```

如果还是失败，再试：

```bash
VLLM_EXTRA_ARGS="--enforce-eager"
```

完整命令：

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=qwen2.5-7b-instruct \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.80 \
CUDA_VISIBLE_DEVICES=0 \
VLLM_EXTRA_ARGS="--enforce-eager" \
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day49_7b_server_eager.log
```

`--enforce-eager` 可能降低性能，但可以减少 CUDA graph 相关显存压力。

## 结果文件

probe 脚本会把每次结果追加到：

```text
results/08_serving_vllm/day49_model_size_probe.csv
```

这个 CSV 后面可以直接变成 README 里的实验表。

## 你需要重点看什么

对比 0.5B 和 7B 时，不要只看 latency。

更重要的是：

```text
模型变大后：
  nvidia-smi 显存占用怎么变？
  Available KV cache memory 怎么变？
  GPU KV cache size tokens 怎么变？
  TTFT 是否明显变大？
  TPOT 是否变慢？
```

如果 7B 成功启动，说明单张 4090 可以支持 7B 基础 serving。

如果 7B 启动失败，也要记录失败日志，因为这能帮助你理解：

```text
权重显存
KV cache 预留
CUDA graph 额外开销
max_model_len
gpu_memory_utilization
```

这些因素是怎么共同决定能不能跑起来的。

## 今日验收

完成后，你应该能回答：

```text
为什么 0.5B 也会占用很多显存？
为什么 7B 会挤压 KV cache 空间？
单卡 4090 跑 7B 时，max_model_len 和 gpu_memory_utilization 为什么重要？
如果 7B OOM，应该优先调整哪些参数？
```
