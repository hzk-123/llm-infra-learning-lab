# Daily Log

Date: 2026-06-30

## Learned

- Page table 用来把 logical block 映射到 physical block。
- 请求内部看到的是连续的 logical token sequence。
- 底层 physical KV blocks 可以不连续。
- 一个 token 的地址转换流程是：
  `token_index -> logical_block + offset -> physical_block + offset`。
- `logical_block = token_index // block_size`。
- `offset = token_index % block_size`。
- PagedAttention 需要依赖类似映射，从 block-based KV cache 中读取正确的 K/V。

## Ran

```text
python exercises\07_kv_cache_paged_attention\day38_paged_attention_page_table.py
```

## My Explanation

这次 Day38 学的是 toy PagedAttention page table。

配置是：

```text
request_id = req_e
token_count = 14
block_size = 4
total_blocks = 8
physical_blocks = [0, 1, 6, 7]
```

这延续了 Day37 的 `req_e` 场景：请求逻辑上有 14 个连续 tokens，但底层 physical blocks 不是连续的。

### Page table

Page table 是：

```text
logical_block 0 -> physical_block 0 -> token range [0, 4)
logical_block 1 -> physical_block 1 -> token range [4, 8)
logical_block 2 -> physical_block 6 -> token range [8, 12)
logical_block 3 -> physical_block 7 -> token range [12, 14)
```

这张表说明：

```text
请求视角:
  logical blocks = [0, 1, 2, 3]

物理存储:
  physical blocks = [0, 1, 6, 7]
```

请求内部 token 是连续的，但物理 block 可以不连续。

### Physical KV memory

物理内存里的内容是：

```text
block 0: req_e_tok_0  到 req_e_tok_3
block 1: req_e_tok_4  到 req_e_tok_7
block 6: req_e_tok_8  到 req_e_tok_11
block 7: req_e_tok_12 到 req_e_tok_13
```

中间的 blocks 2、3、4、5 是空的：

```text
block 2: [None, None, None, None]
block 3: [None, None, None, None]
block 4: [None, None, None, None]
block 5: [None, None, None, None]
```

这说明 `req_e` 的 KV cache 没有要求在物理上连续存储。

最后一个 block 7 是：

```text
['req_e_tok_12', 'req_e_tok_13', None, None]
```

因为 `req_e` 只有 14 个 tokens，block size 是 4，所以最后一个 block 没有填满，这就是 internal fragmentation。

### Token address lookup

查询 token 0：

```text
token_index = 0
logical_block = 0 // 4 = 0
offset = 0 % 4 = 0
physical_block = page_table[0] = 0
physical address = block 0, offset 0
```

查询 token 4：

```text
token_index = 4
logical_block = 4 // 4 = 1
offset = 4 % 4 = 0
physical_block = page_table[1] = 1
physical address = block 1, offset 0
```

查询 token 8：

```text
token_index = 8
logical_block = 8 // 4 = 2
offset = 8 % 4 = 0
physical_block = page_table[2] = 6
physical address = block 6, offset 0
```

这是今天最关键的例子。

虽然 token 8 是请求里的第 8 个 logical token，但它不在 physical block 2，而是在 physical block 6。

原因是 page table 中：

```text
logical_block 2 -> physical_block 6
```

查询 token 13：

```text
token_index = 13
logical_block = 13 // 4 = 3
offset = 13 % 4 = 1
physical_block = page_table[3] = 7
physical address = block 7, offset 1
```

这说明 page table 不只告诉我们 block 在哪里，也需要配合 offset 找到 block 内部的具体 slot。

### Invalid token

查询 token 14 失败：

```text
token_index 14 is outside request length 14
```

因为请求长度是 14，所以合法 token index 是：

```text
0 到 13
```

index 14 已经越界。

### Day38 核心结论

Day37 解决的是：

```text
request -> physical_blocks
```

Day38 解决的是：

```text
token_index -> logical_block + offset -> physical_block + offset
```

PagedAttention 的核心直觉可以总结为：

```text
请求看到连续 token。
底层存储可以不连续。
page table 负责把逻辑位置翻译成物理位置。
```

后面如果要理解 vLLM 的 PagedAttention，就要一直记住这个映射关系。

## Next

- 进入 Day39：request lifecycle，模拟请求从 waiting 到 running，再到 finished，并观察它何时申请和释放 KV blocks。
