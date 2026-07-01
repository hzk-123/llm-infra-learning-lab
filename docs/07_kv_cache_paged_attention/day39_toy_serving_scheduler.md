# Day 39：Toy Serving Scheduler

## 目标

把 request lifecycle、KV block allocation、decode step、block release 合并到一个小型 serving loop 里。

这一天不再单独讲状态机，而是直接模拟：

```text
waiting queue
-> allocate KV blocks
-> running requests
-> 每轮 decode 1 token
-> 请求完成
-> free KV blocks
```

## 为什么要这样学

真实 vLLM scheduler 不是只做一个状态转换。

它要同时考虑：

```text
哪些请求能进入 running
KV cache blocks 是否够
每轮 decode 哪些请求
请求完成后释放 blocks
新的请求是否能补进来
```

所以 Day39 直接看一个 serving loop，比单独学 `waiting/running/finished` 更有用。

## Toy 设定

脚本使用：

```text
total_blocks = 8
block_size = 4
max_running = 2
```

每个请求有：

```text
prompt_tokens
max_new_tokens
generated_tokens
physical_blocks
```

请求需要的 token capacity 是：

```text
prompt_tokens + max_new_tokens
```

需要的 blocks 是：

```text
ceil((prompt_tokens + max_new_tokens) / block_size)
```

这表示为了简单起见，我们在请求进入 running 时，一次性预留它最多可能需要的 KV blocks。

## 输出重点

脚本会打印一张表：

```text
step | waiting | running | finished | free_blocks | events
```

重点观察：

- 请求什么时候进入 running。
- 进入 running 时分配了哪些 physical blocks。
- 每轮 decode 后 `generated_tokens` 如何增加。
- 请求完成后 blocks 如何释放。
- blocks 释放后，新请求如何进入 running。
- scheduler 如何受到 `max_running` 和 `free_blocks` 双重限制。

## 你应该形成的理解

完成 Day39 后，你应该能说清楚：

```text
LLM serving 不是一次处理一个请求，而是维护 waiting/running/finished 请求集合。
请求进入 running 前需要申请 KV blocks。
每轮 decode 会让 running 请求各生成一个 token。
请求完成后释放 KV blocks。
释放出来的 blocks 可以让 waiting 请求继续进入 running。
这就是 continuous batching 和 vLLM scheduler 的前置直觉。
```

