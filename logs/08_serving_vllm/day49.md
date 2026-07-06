# Daily Log

Date: 2026-07-06

## Topic

Day49：0.5B vs 7B 模型大小对 vLLM Serving 的影响。

今天的目标不是继续横向堆 benchmark，而是做一个更贴近真实推理服务的问题：

```text
单张 RTX 4090 24GB 能不能跑 7B 模型？
为什么 0.5B 和 7B 在 nvidia-smi 里看起来都占 21GB 左右？
模型变大以后，vLLM 的 KV cache 空间和单请求性能会怎么变化？
```

## Setup

服务器环境：

```text
conda env: llm-sprint
GPU: NVIDIA GeForce RTX 4090 24GB
vLLM: 0.24.0
server: http://127.0.0.1:8000
CUDA_VISIBLE_DEVICES: 0
MAX_MODEL_LEN: 2048
GPU_MEMORY_UTILIZATION: 0.85
```

本次只改变模型大小：

```text
0.5B baseline: Qwen/Qwen2.5-0.5B-Instruct
7B candidate: Qwen/Qwen2.5-7B-Instruct
```

## Ran

0.5B probe：

```bash
python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 05b \
  --model qwen2.5-0.5b-instruct \
  --server-log logs/08_serving_vllm/day49_05b_server.log
```

7B probe：

```bash
python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 7b \
  --model qwen2.5-7b-instruct \
  --server-log logs/08_serving_vllm/day49_7b_server.log
```

## Server Startup Observation

7B 启动成功。

7B vLLM 日志里的关键启动信息：

```text
Resolved architecture: Qwen2ForCausalLM
dtype: torch.bfloat16
tensor_parallel_size: 1
pipeline_parallel_size: 1
quantization: None
Checkpoint size: 14.19 GiB
Loading weights took: 3.75 seconds
Model loading took: 14.29 GiB memory and 7.016438 seconds
torch.compile took: 3.07 seconds
Estimated CUDA graph memory: 0.53 GiB
Available KV cache memory: 4.45 GiB
GPU KV cache size: 83,376 tokens
Maximum concurrency for 2,048 tokens per request: 40.71x
init engine took: 14.64 seconds
```

这说明：

```text
单卡 4090 可以启动 Qwen2.5-7B-Instruct。
7B 权重本身已经占用约 14.29 GiB。
剩余可用于 KV cache 的空间只有 4.45 GiB。
```

## Result Comparison

| 指标 | 0.5B | 7B | 观察 |
| --- | ---: | ---: | --- |
| GPU0 memory used | 21085 MiB | 20939 MiB | 总占用接近，不代表模型一样大 |
| Available KV cache memory | 18.08 GiB | 4.45 GiB | 7B 权重大，KV cache 空间被压缩 |
| GPU KV cache size | 1,579,808 tokens | 83,376 tokens | 7B 可缓存 token 数大幅下降 |
| torch.compile time | 4.18s | 3.07s | 7B 这次命中了部分编译缓存 |
| engine init time | 8.77s | 14.64s | 7B 初始化更慢 |
| TTFT | 0.1748s | 0.2083s | 7B 首 token 更慢 |
| TPOT | 0.0021s/token | 0.0155s/token | 7B decode 每 token 明显更慢 |
| total latency | 0.3744s | 1.4791s | 7B 总延迟明显更高 |
| output tokens/s | 256.38 | 56.12 | 7B 单请求生成吞吐更低 |
| completion_tokens | 96 | 83 | 两次输出长度不同，latency 不能只看绝对值 |
| finish_reason | length | stop | 0.5B 被 max_tokens 截断，7B 自然结束 |

## Key Finding

这次最重要的发现是：

```text
nvidia-smi 里的总显存占用相近，不代表模型权重显存相近。
```

原因是 vLLM 会把允许使用的显存尽量规划起来。显存主要被分成：

```text
模型权重
KV cache pool
CUDA graph
运行时 buffer
attention / sampling 临时空间
```

所以：

```text
0.5B:
  模型权重小
  留给 KV cache 的空间大
  Available KV cache memory = 18.08 GiB

7B:
  模型权重大
  权重加载占 14.29 GiB
  留给 KV cache 的空间小
  Available KV cache memory = 4.45 GiB
```

也就是说，`0.5B` 和 `7B` 都能让 `nvidia-smi` 显示接近 21GB，但是内部结构完全不同。

## Performance Interpretation

7B 比 0.5B 慢是符合预期的。

主要原因：

```text
7B 参数更多。
每层 attention 和 FFN 计算量更大。
decode 每生成一个 token 都要跑一遍大模型。
所以 TPOT 从 0.0021s/token 增加到 0.0155s/token。
```

TTFT 的差距相对没 TPOT 大，是因为这次 prompt 只有 50 tokens，prefill 压力不大。后续如果用更长 prompt，7B 的 TTFT 差距可能会更明显。

## Answer Quality Observation

0.5B 的回答明显不准确：

```text
把 KV cache 说成普通键值对存储。
提到了训练速度和网络请求。
finish_reason = length，回答被截断。
```

7B 的回答好一些，但仍不够精确。

更准确的解释应该是：

```text
KV cache 存的是每一层 self-attention 中历史 token 的 Key / Value 张量。
decode 新 token 时，不需要重新计算旧 token 的 K/V，只需要用新 token 的 Query 去 attend 已缓存的 K/V。
它占显存，是因为要为 batch、层数、KV heads、上下文长度、head_dim 保存这些张量。
```

这个 bad case 提醒我：benchmark 时不能只看速度，也要检查输出质量。

## Bug Found

Day49 probe 脚本在解析 server log 时有一个 bug。

现象：

```text
SERVED_MODEL_NAME 被解析成了一大段 vLLM config。
```

原因：

```text
旧正则是 SERVED_MODEL_NAME=(.+)
它没有限定必须匹配启动脚本打印的配置行。
vLLM 后续日志里也有 served_model_name=...
所以误匹配到了 engine config 的长文本。
```

## Bug Fix

修复方式：

```text
把启动脚本配置项解析改成按行匹配：

^\s*SERVED_MODEL_NAME=(.+)$
^\s*MODEL_ID=(.+)$
^\s*MAX_MODEL_LEN=(.+)$
^\s*GPU_MEMORY_UTILIZATION=(.+)$
^\s*CUDA_VISIBLE_DEVICES=(.+)$
```

同时增加了更多 server log 字段解析：

```text
checkpoint_size
model_loading_memory
model_loading_time_s
max_concurrency_estimate
```

这些字段后续能直接进入 CSV，方便写 README 实验表。

## Need Rerun After Fix

修复后不新开 Day50，仍然作为 Day49 的重跑验证。

同步代码后，在当前 7B server 还运行时，可以先重跑：

```bash
python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 7b_rerun \
  --model qwen2.5-7b-instruct \
  --server-log logs/08_serving_vllm/day49_7b_server.log
```

如果要让 0.5B 也得到修复后的干净 CSV 行，需要重新启动 0.5B server 后再跑：

```bash
python exercises/08_serving_vllm/day49_model_size_probe.py \
  --run-label 05b_rerun \
  --model qwen2.5-0.5b-instruct \
  --server-log logs/08_serving_vllm/day49_05b_server.log
```

注意：同一时间端口 `8000` 只能对应一个 vLLM server。要重跑 0.5B，需要先停掉 7B server，再启动 0.5B server。

## Takeaway

今天真正学到的是：

```text
1. 单卡 4090 可以跑 Qwen2.5-7B-Instruct 的 vLLM serving。
2. vLLM 的显存占用不能只看 nvidia-smi 总量，要看权重、KV cache、CUDA graph 的拆分。
3. 7B 权重占用约 14.29 GiB，因此 KV cache 空间从 18.08 GiB 降到 4.45 GiB。
4. 7B 的 TPOT 明显高于 0.5B，说明 decode 阶段每 token 计算成本更高。
5. benchmark 结果要结合输出质量分析，不能只看 tokens/s。
6. 实验脚本本身也需要验证，日志 parser 的误匹配会污染结果表。
```

Day49 后，当前主线应该继续围绕 7B 做真实 benchmark，而不是继续使用 0.5B。
