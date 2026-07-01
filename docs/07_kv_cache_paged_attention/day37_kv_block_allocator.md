# Day 37：Toy KV Block Allocator

## 目标

理解 PagedAttention 的前置概念：KV cache 不一定要给每个请求分配一整块连续内存，可以按固定大小的 block/page 管理。

前面 Day33-Day36 学的是：

```text
KV cache shape
cache append
prefill scaling
decode scaling
CUDA memory measurement
```

Day37 开始进入系统层：

```text
KV cache 如何被分配、释放、复用？
```

## 为什么需要 block allocator

如果每个请求都要求连续 KV cache，那么服务端会遇到两个问题：

```text
1. 请求长度不同，容易浪费空间。
2. 请求动态进入/结束，容易产生碎片。
```

PagedAttention 的核心直觉之一是：

```text
把 KV cache 切成固定大小 block。
每个请求拿若干个 block。
请求结束后释放 block。
一个请求的 block 不一定物理连续。
```

## 本日 toy 设计

脚本中使用：

```text
total_blocks = 8
block_size = 4 tokens
```

每个请求需要的 block 数：

```text
required_blocks = ceil(token_count / block_size)
```

例如：

```text
token_count = 6
block_size = 4
required_blocks = 2
capacity = 8
waste = 2
```

这里的 waste 是 block 内部没有用完的 token slots，也叫 internal fragmentation。

## 运行时重点看什么

运行脚本时重点观察：

- 每个请求需要几个 blocks。
- 每个请求实际拿到了哪些 physical blocks。
- request 释放后，free blocks 如何回到 free list。
- 新请求是否可以拿到非连续 blocks。
- `requested_tokens`、`capacity_tokens`、`wasted_slots` 的关系。
- block 不够时 allocator 如何拒绝新请求。

## 你应该形成的理解

完成 Day37 后，你应该能说清楚：

```text
KV block allocator 管理的是固定大小 physical blocks。
请求按 token 数申请若干 blocks。
block 内可能有未使用 slots，形成 internal fragmentation。
请求结束后 blocks 可以释放并复用。
一个请求的 physical blocks 可以不连续。
这为后面的 PagedAttention page table 做准备。
```

## 和后续 Day38 的关系

Day37 只关心：

```text
request -> physical blocks
```

Day38 会进一步细化：

```text
logical token index -> logical block -> physical block + offset
```

也就是从 allocator 走到 page table。

