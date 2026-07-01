# Daily Log

Date: 2026-06-30

## Learned

- KV block allocator 把 KV cache 切成固定大小的 physical blocks。
- 每个请求根据 token 数申请若干 blocks。
- `required_blocks = ceil(token_count / block_size)`。
- block 内部没有用完的位置会形成 internal fragmentation。
- 请求结束后，释放的 blocks 会回到 free list。
- 新请求可以使用不连续的 physical blocks。
- PagedAttention 的 page table 思想，就是在 logical block 和 physical block 之间建立映射。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day37_kv_block_allocator.py
```

## My Explanation

这次 Day37 学的是 toy KV block allocator。

配置是：

```text
total_blocks = 8
block_size = 4
total_token_capacity = 32
```

意思是 allocator 总共有 8 个 physical blocks，每个 block 可以放 4 个 token 的 KV cache，所以总容量是：

```text
8 * 4 = 32 tokens
```

初始状态下，所有 blocks 都是空闲的：

```text
free_blocks: [0, 1, 2, 3, 4, 5, 6, 7]
used_blocks: 0
```

### 初始分配

请求 `req_a` 有 6 个 tokens：

```text
token_count = 6
block_size = 4
required_blocks = ceil(6 / 4) = 2
```

它拿到：

```text
physical_blocks = [0, 1]
capacity = 8
wasted = 2
```

这里 wasted=2，是因为 2 个 blocks 总容量是 8，但实际只用了 6 个 token slots。

请求 `req_b` 有 10 个 tokens：

```text
required_blocks = ceil(10 / 4) = 3
physical_blocks = [2, 3, 4]
capacity = 12
wasted = 2
```

请求 `req_c` 有 4 个 tokens：

```text
required_blocks = 1
physical_blocks = [5]
capacity = 4
wasted = 0
```

请求 `req_d` 有 8 个 tokens：

```text
required_blocks = 2
physical_blocks = [6, 7]
capacity = 8
wasted = 0
```

分配完 `req_a/b/c/d` 后：

```text
free_blocks: []
used_blocks: 8
requested_tokens: 28
capacity_tokens: 32
wasted_slots: 4
```

这说明所有 physical blocks 都被用完了，但真正请求的 tokens 只有 28 个，剩下 4 个 slots 是 block 内部浪费。

### 释放 blocks

释放 `req_a`：

```text
released_blocks = [0, 1]
free_blocks = [0, 1]
```

释放 `req_d`：

```text
released_blocks = [6, 7]
free_blocks = [0, 1, 6, 7]
```

这时 free blocks 不是连续的一整段，而是：

```text
[0, 1, 6, 7]
```

中间的 `[2, 3, 4, 5]` 仍然被 `req_b` 和 `req_c` 占用。

### 非连续 block 分配

然后分配 `req_e`，它有 14 个 tokens：

```text
required_blocks = ceil(14 / 4) = 4
```

它拿到：

```text
physical_blocks = [0, 1, 6, 7]
capacity = 16
wasted = 2
```

这是今天最重要的现象：`req_e` 的 physical blocks 不是连续的。

但是从请求自己的视角看，它的 logical blocks 是连续的：

```text
logical_block 0 -> physical_block 0 -> token range [0, 4)
logical_block 1 -> physical_block 1 -> token range [4, 8)
logical_block 2 -> physical_block 6 -> token range [8, 12)
logical_block 3 -> physical_block 7 -> token range [12, 14)
```

也就是说，请求内部仍然可以认为自己的 token 是连续的；真正底层物理存储可以是不连续的。

这就是 page table 思想的雏形：

```text
logical block -> physical block
```

后面 PagedAttention 要解决的就是：attention kernel 怎么根据这个映射找到每个 token 对应的 KV block。

### 分配失败

最后分配 `req_f`：

```text
token_count = 1
needed_blocks = 1
free_blocks = 0
```

因为没有空闲 block，所以分配失败：

```text
Allocate req_f failed
```

这说明 block allocator 需要做资源管理：如果 free blocks 不够，新请求就不能进入 running 状态，可能要等待、排队或被拒绝。

### Day37 核心结论

Day37 的核心不是 attention 计算，而是 KV cache 的内存管理方式：

```text
连续 KV cache:
  每个请求需要一整块连续空间

block/page KV cache:
  每个请求拿多个固定大小 blocks
  physical blocks 可以不连续
  通过 logical -> physical 映射找到真实位置
```

PagedAttention 的关键直觉就是：

```text
像操作系统虚拟内存分页一样管理 KV cache。
```

这样可以让不同长度、动态进出的请求更灵活地共享 GPU KV cache 空间。

## Next

- 进入 Day38：toy PagedAttention page table，把 `logical token index` 映射到 `logical block + offset`，再映射到 `physical block + offset`。
