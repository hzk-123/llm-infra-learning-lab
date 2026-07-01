# Daily Log

Date: 2026-07-01

## Learned

- Static batching 的 batch 是固定的，必须等当前 batch 里的所有请求都结束，下一批请求才能进入。
- Continuous batching 的 running batch 是动态的，请求完成后，新请求可以立刻补进来。
- 当请求输出长度差异很大时，static batching 容易产生 idle slot。
- Continuous batching 可以减少 idle slot，提高 decode slot 利用率。
- vLLM 的 serving 吞吐提升，很大一部分来自这种 decode-step 级别的动态调度。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day40_static_vs_continuous_batching.py
```

## My Explanation

这次 Day40 对比的是：

```text
static batching
vs
continuous batching
```

配置是：

```text
batch_size / max_running = 2
request_count = 4
total requested output tokens = 14
```

请求分别是：

```text
req_a: 1 token
req_b: 6 tokens
req_c: 1 token
req_d: 6 tokens
```

这个请求组合是故意设计的：短请求和长请求混在一起时，更容易看出 static batching 的浪费。

## Static Batching

Static batching 会把请求固定分成两批：

```text
batch_1 = [req_a, req_b]
batch_2 = [req_c, req_d]
```

第一批里：

```text
req_a 只需要 1 个 token
req_b 需要 6 个 token
```

第 1 步时，`req_a` 和 `req_b` 都在 decode：

```text
req_a: 1/1
req_b: 1/6
```

这一步结束后，`req_a` 已经完成。

但是 static batching 下，`req_c` 不能马上进入。

因为当前 batch 还没整体结束，必须等 `req_b` 也生成完：

```text
req_b: 6/6
```

所以 step 2 到 step 6，batch 里只有 `req_b` 在工作，另一个 slot 一直空着：

```text
idle_slots = 1
```

第二批 `[req_c, req_d]` 也一样：

```text
req_c 只需要 1 个 token
req_d 需要 6 个 token
```

`req_c` 很快完成，后面又留下一个空闲 slot。

所以 static batching 的总结是：

```text
total_steps = 12
capacity_slots = 24
useful_decode_slots = 14
idle_slots = 10
slot_utilization = 58.33%
```

这里：

```text
capacity_slots = total_steps * batch_size = 12 * 2 = 24
```

真实生成 token 的 slot 只有：

```text
useful_decode_slots = 14
```

浪费掉的 slot 是：

```text
idle_slots = 24 - 14 = 10
```

## Continuous Batching

Continuous batching 不固定 batch 边界。

初始状态是：

```text
running = [req_a, req_b]
waiting = [req_c, req_d]
```

step 1 后，`req_a` 完成：

```text
finished = [req_a]
```

这时 `req_c` 立刻补进 running：

```text
running = [req_b:1/6, req_c:0/1]
```

step 2 后，`req_c` 又完成，于是 `req_d` 继续补进来：

```text
running = [req_b:2/6, req_d:0/6]
```

这就是 continuous batching 的核心：

```text
请求不是一批一批固定等完，而是在每个 decode step 后动态进出。
```

所以它减少了短请求完成后的空闲时间。

Continuous batching 的总结是：

```text
total_steps = 8
capacity_slots = 16
useful_decode_slots = 14
idle_slots = 2
slot_utilization = 87.50%
```

对比 static batching：

```text
static batching:
  total_steps = 12
  idle_slots = 10
  slot_utilization = 58.33%

continuous batching:
  total_steps = 8
  idle_slots = 2
  slot_utilization = 87.50%
```

同样生成 14 个 token，continuous batching 用更少的 step，浪费更少的 slot。

## Key Takeaway

这次实验真正要理解的是：

```text
LLM serving 的 batch 不是训练里的固定 batch。
训练 batch 通常一次 forward/backward 后就结束。
推理 serving 的请求会持续多个 decode step，而且每个请求的输出长度不同。
```

因此，如果用 static batching：

```text
短请求完成后，会被长请求拖住，空出来的 slot 不能马上复用。
```

如果用 continuous batching：

```text
短请求完成后，新请求可以立刻补位，让 GPU decode slot 更忙。
```

这就是 vLLM continuous batching 的核心价值之一。

## Next

- 进入 Day41：prefix cache toy。
- 重点看多个请求共享相同 prompt prefix 时，为什么可以减少重复 prefill。

