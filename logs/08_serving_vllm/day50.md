# Daily Log

Date: 2026-07-06

## Topic

Day50：使用 7B 模型重跑 workload benchmark。

Day49 已经证明单张 RTX 4090 24GB 可以启动 `Qwen2.5-7B-Instruct`。今天不写新脚本，直接复用 Day47 和 Day48 的 benchmark 脚本，把正式实验模型从 `0.5B` 切到 `7B`。

目标：

```text
1. 用 7B 重跑 single-request workload。
2. 用 7B 重跑轻量 concurrency benchmark。
3. 观察 7B 的 TTFT、TPOT、latency、QPS、aggregate output tokens/s。
4. 和之前 0.5B 的结果做直觉对照。
```

## Setup

服务器环境：

```text
conda env: llm-sprint
server: vLLM on http://127.0.0.1:8000
model: qwen2.5-7b-instruct
base model: Qwen/Qwen2.5-7B-Instruct
GPU: RTX 4090 24GB
MAX_MODEL_LEN: 2048
GPU_MEMORY_UTILIZATION: 0.85
```

Day49 中该 7B server 的关键状态：

```text
Checkpoint size: 14.19 GiB
Model loading memory: 14.29 GiB
Available KV cache memory: 4.45 GiB
GPU KV cache size: 83,376 tokens
```

## Part 1：Single-request Workload

运行命令：

```bash
python exercises/08_serving_vllm/day47_workload_single_request_benchmark.py \
  --model qwen2.5-7b-instruct \
  --warmup-runs 1 \
  --runs 5 \
  --out-csv results/08_serving_vllm/day50_7b_single_workload.csv
```

### Workload Summary

| workload | avg prompt tok | avg completion tok | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | latency p50 | latency p95 | avg output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_prompt | 38.0 | 21.0 | 0.0291s | 0.0331s | 0.0157s/token | 0.0158s/token | 0.3447s | 0.3463s | 61.02 |
| long_prompt | 1673.0 | 22.0 | 0.0383s | 0.0412s | 0.0157s/token | 0.0158s/token | 0.3681s | 0.3700s | 59.76 |
| long_output | 49.0 | 124.0 | 0.0293s | 0.0323s | 0.0157s/token | 0.0157s/token | 1.9552s | 1.9577s | 63.41 |

### Single-request 观察

最稳定的指标是：

```text
TPOT 基本固定在 0.0157s/token 左右。
```

这说明在单请求场景下，7B decode 阶段每生成一个 token 的成本非常稳定。

`long_output` 的总延迟可以用下面的近似解释：

```text
total_latency ≈ TTFT + completion_tokens * TPOT
              ≈ 0.029s + 124 * 0.0157s
              ≈ 1.98s
```

实测：

```text
latency p50 = 1.9552s
```

这个结果说明：对 7B 来说，长输出请求的主要瓶颈是 decode token 数量，而不是 prompt 长度。

`long_prompt` 的 prompt tokens 是 `1673`，但 TTFT p50 只有 `0.0383s`，比 short_prompt 的 `0.0291s` 只高一点。原因可能包括：

```text
1. 7B + FlashAttention 的 prefill 对 1673 tokens 仍然不算太重。
2. benchmark 使用相同 prompt 重复请求，prefix cache 可能让后续请求受益。
3. 单请求场景没有并发排队压力。
```

所以不能简单说“long prompt 一定明显慢”，要看模型大小、prompt 长度、prefix cache、并发状态和 prefill kernel 效率。

## Part 2：Concurrent Workload

运行命令：

```bash
python exercises/08_serving_vllm/day48_concurrent_workload_benchmark.py \
  --model qwen2.5-7b-instruct \
  --concurrency-levels 1,2,4 \
  --requests-per-level 6 \
  --out-csv results/08_serving_vllm/day50_7b_concurrent_workload.csv
```

### Concurrent Summary

#### short_prompt

| concurrency | QPS | agg output tok/s | TTFT p95 | TPOT p95 | latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.89 | 60.59 | 0.0331s | 0.0157s/token | 0.3477s |
| 2 | 5.44 | 114.21 | 0.0493s | 0.0168s/token | 0.3812s |
| 4 | 7.97 | 167.33 | 0.0545s | 0.0161s/token | 0.3757s |

#### long_prompt

| concurrency | QPS | agg output tok/s | TTFT p95 | TPOT p95 | latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.72 | 59.95 | 0.0398s | 0.0158s/token | 0.3674s |
| 2 | 5.07 | 111.51 | 0.0659s | 0.0160s/token | 0.3998s |
| 4 | 7.55 | 166.19 | 0.0836s | 0.0163s/token | 0.4222s |

#### long_output

| concurrency | QPS | agg output tok/s | TTFT p95 | TPOT p95 | latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.51 | 63.26 | 0.0412s | 0.0157s/token | 1.9648s |
| 2 | 1.00 | 122.89 | 0.0628s | 0.0161s/token | 2.0199s |
| 4 | 1.50 | 182.90 | 0.0907s | 0.0162s/token | 2.0226s |

## Concurrent Interpretation

### 1. 吞吐随并发升高而上升

short_prompt：

```text
QPS: 2.89 -> 5.44 -> 7.97
agg output tokens/s: 60.59 -> 114.21 -> 167.33
```

long_prompt：

```text
QPS: 2.72 -> 5.07 -> 7.55
agg output tokens/s: 59.95 -> 111.51 -> 166.19
```

long_output：

```text
QPS: 0.51 -> 1.00 -> 1.50
agg output tokens/s: 63.26 -> 122.89 -> 182.90
```

这说明 vLLM 的 continuous batching 在 7B 上有效：并发请求可以让 GPU 更忙，整体吞吐更高。

### 2. 尾延迟和 TTFT 随并发上升

long_prompt 的 TTFT p95：

```text
0.0398s -> 0.0659s -> 0.0836s
```

long_output 的 TTFT p95：

```text
0.0412s -> 0.0628s -> 0.0907s
```

这说明并发升高后，请求可能需要等待调度、batch 组装、prefill/decode 资源，首 token 延迟会变差。

这是 serving 系统中的典型 tradeoff：

```text
提高吞吐，通常会牺牲一部分单请求尾延迟。
```

### 3. 7B 的 TPOT 很稳定

无论 single-request 还是 concurrency=4，TPOT 大致都在：

```text
0.0157s/token 到 0.0163s/token
```

这说明当前 concurrency 还不算太高，没有把 7B 服务压到严重退化。

如果继续提高到 concurrency=8 或更高，可能会看到：

```text
TPOT 上升
latency p95 上升
KV cache usage 上升
可能出现排队或 OOM
```

## Compared With 0.5B

之前 0.5B 的 Day47 单请求 TPOT 大概是：

```text
0.0027s/token 到 0.0041s/token
```

7B 的 Day50 单请求 TPOT 是：

```text
0.0157s/token
```

所以 7B decode 单 token 成本大约是 0.5B 的数倍。

之前 0.5B 的 Day48 并发 benchmark 中，short_prompt concurrency=8 可以达到：

```text
QPS = 64.53
agg output tokens/s = 1653.50
```

7B 这次 short_prompt concurrency=4：

```text
QPS = 7.97
agg output tokens/s = 167.33
```

这个差距是合理的：

```text
0.5B 参数少，decode 快，吞吐高。
7B 参数多，decode 慢，吞吐低，但模型能力更接近真实服务场景。
```

因此后续正式 benchmark 应以 7B 为主，0.5B 只作为快速 smoke test。

## Output Quality Notes

这次输出质量仍然需要注意。

`long_prompt` 的样例回答：

```text
KV cache 在 decode 阶段主要解决重复 prompt 的缓存问题，避免重复计算。
```

这句话不够准确。更准确说法应该是：

```text
KV cache 缓存的是历史 token 在每层 attention 中的 Key / Value 张量。
decode 新 token 时可以复用旧 token 的 K/V，避免每一步重新计算整个上下文的 K/V。
```

所以 benchmark 不只要记录速度，也要保留 bad case。速度数据可以支撑推理服务能力，bad case 可以支撑模型行为分析能力。

## Takeaway

今天真正学到的是：

```text
1. 7B 单请求 TPOT 稳定在 0.0157s/token 左右。
2. long_output 的 total latency 主要由 completion_tokens * TPOT 决定。
3. concurrency 从 1 提升到 4 后，QPS 和 aggregate output tokens/s 明显上升。
4. 并发升高后，TTFT p95 和 latency p95 会变差，这是吞吐和尾延迟的 tradeoff。
5. 当前 concurrency=4 还没有明显压垮 7B 服务，TPOT 仍然比较稳定。
6. 后续正式实验应默认使用 7B，0.5B 只用于快速验证脚本和服务链路。
```

## Next

下一步不建议立刻继续加模型或并发。

更好的下一步是做一个小而清楚的参数实验：

```text
固定 7B、固定 workload。
只改变 max_model_len 或 gpu_memory_utilization。
观察 Available KV cache memory、GPU KV cache size、TTFT、TPOT、p95 latency 的变化。
```

这会把 Day49 的显存理解和 Day50 的 benchmark 结果连接起来。
