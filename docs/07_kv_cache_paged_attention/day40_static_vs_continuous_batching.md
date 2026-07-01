# Day 40：Static Batching vs Continuous Batching

## 目标

今天只看一个核心问题：

```text
为什么 continuous batching 比 static batching 更适合 LLM serving？
```

Day39 已经把 waiting、running、finished 和 KV block 释放串起来了。

Day40 不再继续扩展状态机，而是用同一组请求对比两种调度方式：

- static batching：一个 batch 固定不变，必须等整个 batch 都结束，下一批请求才能进来。
- continuous batching：每个 decode step 后，完成的请求退出，新请求可以立刻补进 running batch。

## Toy 设定

脚本使用 4 个请求：

```text
req_a: 1 token
req_b: 6 tokens
req_c: 1 token
req_d: 6 tokens
```

最大并发 / batch size 是：

```text
2
```

这个设计是故意的：

```text
短请求和长请求混在一起时，static batching 很容易出现空闲 slot。
```

## Static Batching 的问题

假设第一批是：

```text
[req_a, req_b]
```

`req_a` 只需要生成 1 个 token，很快完成。

但 static batching 下，第二批：

```text
[req_c, req_d]
```

不能立刻进来。

它必须等 `req_b` 也完成。

所以第一批后续很多 step 里，batch 里只有 `req_b` 在工作，另一个 slot 是空的。

## Continuous Batching 的优势

continuous batching 下：

```text
req_a 完成
-> req_c 立刻补进来
req_c 完成
-> req_d 立刻补进来
```

running batch 不再固定绑定一组请求。

它会在每个 decode step 后动态变化。

这就是 vLLM 提高吞吐的核心直觉之一：

```text
让 GPU 尽量少等短请求，让空出来的位置尽快被新请求填上。
```

## 重点观察

运行脚本后重点看 summary：

```text
total_steps
total_capacity_slots
useful_decode_slots
idle_slots
slot_utilization
```

含义是：

- `total_steps`：总共执行了多少个 decode step。
- `total_capacity_slots`：总 step 数乘以最大并发，代表理论可用 decode slot。
- `useful_decode_slots`：真实生成 token 的 slot 数。
- `idle_slots`：空转的 slot 数。
- `slot_utilization`：slot 利用率。

## 你应该掌握

学完 Day40 后，你应该能解释：

```text
static batching 的 batch 边界是固定的，短请求完成后可能产生空闲 slot。
continuous batching 的 running batch 是动态的，请求完成后新请求可以补位。
在输出长度差异很大的请求混合时，continuous batching 通常有更高的 slot 利用率。
vLLM 的 continuous batching 本质上就是围绕 decode step 做动态请求调度。
```

