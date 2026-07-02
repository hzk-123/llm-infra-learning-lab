# LLM Infra Learning Lab

这是一个面向 **大模型推理服务 / AI Infra / vLLM 性能工程** 的增量学习项目。

项目原则很简单：

```text
学到哪里，代码和目录就长到哪里。
不提前堆空文件夹。
不一口气生成大工程。
每天只解决一个小问题，并留下可复盘产物。
```

## 当前目标

主线方向：

```text
大模型推理服务与性能工程实习
```

当前重点：

```text
PyTorch 基础
LLM 推理原理
KV Cache
PagedAttention 直觉
Continuous Batching
Prefix Cache
vLLM OpenAI-compatible Server
```

后续会逐步进入：

```text
vLLM Python client
真实 TTFT / TPOT / latency benchmark
并发压测
量化推理
多卡 Tensor Parallel
C++ / CUDA / Nsight
```

这些后续模块会在真正开始学习时再创建目录。

## 当前进度

已经完成到 Day43，正在进入 Day44：

```text
Day01-Day04：PyTorch next-token / Embedding / Bigram LM / Training Loop
Day05-Day07：字符 tokenizer / 文本训练 / MLP LM
Day08-Day13：位置编码 / Attention / QKV / Causal Mask / Multi-Head Attention
Day14-Day18：FFN / Residual / LayerNorm / Tiny Transformer LM
Day19-Day27：Generation / Prefill / Decode / KV Cache / 指标和 toy benchmark
Day28-Day32：RMSNorm / SwiGLU / RoPE / MHA-MQA-GQA / Tiny Llama Block
Day33-Day43：KV Cache Layout / Attention Scaling / PagedAttention Toy / Scheduler / Prefix Cache / vLLM Server
Day44：vLLM Python Smoke Client
```

Day43 已完成：

```text
服务器 3x4090 环境检查通过。
vLLM 0.24.0 已安装。
Qwen/Qwen2.5-0.5B-Instruct 已用 vLLM 启动。
/v1/models 可访问。
/v1/chat/completions 可正常生成。
```

## 当前目录结构

```text
docs/       每天的概念说明和阶段索引
exercises/  每天的最小可运行脚本
logs/       每天的学习日志和复盘
scripts/    可复用启动脚本，例如 vLLM server
```

核心文件：

```text
README.md                  项目说明
ROADMAP.md                 长线学习路线
docs/milestone_index.md    已完成里程碑索引
docs/learning_plan_v3.md   当前长期计划
scripts/start_vllm_server.sh
```

## 如何继续学习

本机开发：

```powershell
cd C:\Users\胡泽坤\Desktop\do\llm-infra-learning-lab
conda activate llm-sprint
```

服务器实验：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
```

同步流程：

```text
本机修改 -> git add/commit/push
服务器 -> git pull
服务器运行实验
```

本机提交：

```powershell
git add .
git commit -m "update learning files"
git push
```

服务器更新：

```bash
git pull
```

## vLLM Server

启动：

```bash
conda activate llm-sprint
cd ~/llm-infra-learning-lab
tmux new -s vllm-day43
bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log
```

健康检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

聊天接口：

```bash
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

关闭：

```bash
tmux attach -t vllm-day43
```

进入后按：

```text
Ctrl-c
exit
```

## 每天的产物

每个 Day 至少留下三个东西：

```text
docs/.../dayXX_*.md      概念说明
exercises/.../dayXX_*.py 最小实验脚本
logs/.../dayXX.md        学习复盘
```

如果当天是服务器实验，还要记录：

```text
启动命令
环境版本
GPU 信息
关键日志
运行结果
结论和问题
```

## 当前学习边界

现在不急着做：

```text
大规模 benchmark
多卡 tensor parallel
量化模型部署
CUDA kernel
Nsight 分析
Megatron / DeepSpeed
```

这些会在主线推进到对应阶段时再创建目录和代码。

当前下一步建议：

```text
启动或确认 vLLM server 仍在运行。
运行 Day44 Python smoke client。
确认 Python 程序能拿到 latency 和 token usage。
```
