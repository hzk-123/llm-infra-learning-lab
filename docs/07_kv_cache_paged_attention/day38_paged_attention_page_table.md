# Day 38：Toy PagedAttention Page Table

## 目标

理解 PagedAttention 中 page table 的核心映射：

```text
logical token index
-> logical block + offset
-> physical block + offset
```

Day37 学的是：

```text
request -> physical blocks
```

Day38 更进一步，学习请求内部某个 token 如何找到真实的 KV cache 存储位置。

## 为什么需要 page table

在 block/page KV cache 中，一个请求的 physical blocks 可以不连续。

例如 Day37 里的 `req_e`：

```text
logical blocks:  [0, 1, 2, 3]
physical blocks: [0, 1, 6, 7]
```

从请求视角看，token index 是连续的：

```text
0, 1, 2, ..., 13
```

但从物理 KV cache 视角看，它们分布在不同 physical blocks 中。

所以需要一张 page table：

```text
logical_block -> physical_block
```

## Token 地址转换

给定：

```text
block_size = 4
token_index = 8
```

先算 logical block 和 offset：

```text
logical_block = token_index // block_size = 8 // 4 = 2
offset = token_index % block_size = 8 % 4 = 0
```

再查 page table：

```text
page_table[2] = physical_block 6
```

所以 token 8 的真实位置是：

```text
physical_block = 6
offset = 0
```

## 运行时重点看什么

运行脚本时重点观察：

- page table 的 logical block 到 physical block 映射。
- token index 如何拆成 logical block 和 offset。
- 同一个请求的 tokens 如何落在不连续 physical blocks 上。
- 最后一个 block 可能没有填满，形成 unused slots。
- invalid token index 如何被拒绝。

## 你应该形成的理解

完成 Day38 后，你应该能说清楚：

```text
请求内部 token index 是连续的。
底层 physical KV blocks 可以不连续。
page table 把 logical block 映射到 physical block。
token index 先变成 logical block + offset。
再通过 page table 找到 physical block + offset。
PagedAttention 的 kernel 需要根据类似映射读取正确的 K/V。
```

