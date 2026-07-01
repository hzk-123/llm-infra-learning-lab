# Daily Log

Date: 2026-07-01

## Learned

- Serving scheduler 同时管理 waiting、running、finished 三类请求。
- 请求进入 running 前，需要同时满足 running slot 和 KV blocks 都足够。
- 每个 running request 在一个 scheduler step 中 decode 1 个 token。
- 请求生成满 `max_new_tokens` 后进入 finished，并释放 KV blocks。
- 被释放的 KV blocks 可以立刻被 waiting queue 里的新请求复用。
- 这就是 continuous batching 的前置直觉：请求可以动态进入和退出 running batch。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day39_toy_serving_scheduler.py
```

## My Explanation

这次 Day39 学的是 toy serving scheduler。

配置是：

```text
total_blocks = 8
block_size = 4
max_running = 2
total_token_capacity = 32
```

`max_running=2` 表示最多同时有 2 个请求处于 running 状态。

请求列表是：

```text
req_a: prompt_tokens=5, max_new_tokens=3, reserved_tokens=8,  blocks_needed=2
req_b: prompt_tokens=8, max_new_tokens=4, reserved_tokens=12, blocks_needed=3
req_c: prompt_tokens=4, max_new_tokens=2, reserved_tokens=6,  blocks_needed=2
req_d: prompt_tokens=7, max_new_tokens=3, reserved_tokens=10, blocks_needed=3
```

这里为了简化，脚本在请求进入 running 时一次性预留：

```text
reserved_tokens = prompt_tokens + max_new_tokens
```

对应需要的 blocks 是：

```text
blocks_needed = ceil(reserved_tokens / block_size)
```

### Step 0：admit 初始请求

step 0 时，scheduler 把 `req_a` 和 `req_b` 放入 running：

```text
admit req_a blocks=[0, 1] reserved_tokens=8
admit req_b blocks=[2, 3, 4] reserved_tokens=12
```

此时：

```text
waiting = ['req_c', 'req_d']
running = ['req_a:0/3', 'req_b:0/4']
finished = []
free_blocks = [5, 6, 7]
```

虽然还有 free blocks，但 `max_running=2` 已经满了，所以 `req_c` 和 `req_d` 继续等待。

### Step 1-2：running 请求一起 decode

step 1：

```text
decode req_a -> 1
decode req_b -> 1
```

running 变成：

```text
req_a:1/3
req_b:1/4
```

step 2：

```text
decode req_a -> 2
decode req_b -> 2
```

running 变成：

```text
req_a:2/3
req_b:2/4
```

这表示每个 scheduler step 中，running batch 里的每个请求都生成一个 token。

### Step 3：req_a 完成，req_c 补进来

step 3：

```text
decode req_a -> 3
finish req_a, free [0, 1]
decode req_b -> 3
admit req_c blocks=[0, 1] reserved_tokens=6
```

`req_a` 已经生成满：

```text
3/3
```

所以它完成并释放 blocks：

```text
[0, 1]
```

释放后，`req_c` 立刻从 waiting 进入 running，并复用：

```text
blocks=[0, 1]
```

这一行是 Day39 的重点之一：请求完成释放 KV blocks 后，新请求可以马上补位。

### Step 4：req_b 完成，req_d 补进来

step 4：

```text
decode req_b -> 4
finish req_b, free [2, 3, 4]
decode req_c -> 1
admit req_d blocks=[2, 3, 4] reserved_tokens=10
```

`req_b` 完成：

```text
4/4
```

释放：

```text
[2, 3, 4]
```

然后 `req_d` 进入 running，并复用这些 blocks：

```text
blocks=[2, 3, 4]
```

此时 waiting 已经为空：

```text
waiting = []
```

### Step 5-7：剩余请求完成

step 5：

```text
decode req_c -> 2
finish req_c, free [0, 1]
decode req_d -> 1
```

`req_c` 完成并释放 `[0, 1]`。

step 6：

```text
decode req_d -> 2
```

step 7：

```text
decode req_d -> 3
finish req_d, free [2, 3, 4]
```

最后：

```text
waiting = []
running = []
finished = ['req_a', 'req_b', 'req_c', 'req_d']
free_blocks = [0, 1, 2, 3, 4, 5, 6, 7]
```

所有请求完成，所有 blocks 都回到 free list。

### Day39 核心结论

这次脚本把 serving loop 的核心过程串起来了：

```text
waiting queue
-> admit request if running slot and KV blocks are enough
-> running requests decode one token per step
-> finished request frees KV blocks
-> waiting request reuses freed KV blocks
```

这比单独讲 `waiting/running/finished` 更接近真实 serving scheduler。

从 vLLM / continuous batching 的角度看，关键不是“有几个状态”，而是：

```text
running batch 可以动态变化。
请求完成后可以退出。
新请求可以补进来。
KV blocks 是调度约束之一。
```

Day39 的核心理解是：

```text
continuous batching = 每轮 decode 时，running batch 不是固定不变的；
它会随着请求完成和新请求进入而动态变化。
```

## Next

- 进入 Day40：continuous batching step-level scheduler，更明确地区分 static batching 和 continuous batching 的差异。
