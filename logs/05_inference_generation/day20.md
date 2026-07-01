# Daily Log

Date: 2026-06-15

## Learned

- Prefill 是第一次处理完整 prompt。
- Decode 是后续一个 token 一个 token 地生成。
- 当前脚本还没有 KV cache，所以 decode 阶段仍然会重新计算最近的完整 context。
- 每次生成出的 `next_id` 会被拼接到已有 `ids` 后面。
- 生成是自回归过程：生成一个 token，再基于更长的序列继续生成下一个 token。

## Ran

```text
python exercises\05_inference_generation\day20_prefill_decode.py
```

## My Explanation

这次输入 prompt 是：

```text
prompt_ids: [1, 2, 3, 4]
prompt_ids.shape: [1, 4]
```

Prefill 阶段处理完整 prompt：

```text
prefill_context: [1, 2, 3, 4]
prefill_context.shape: [1, 4]
prefill_logits.shape: [1, 4, 10]
```

这里 `10` 是 `vocab_size`，表示模型对每个位置都输出 10 个 token 的预测分数。

prefill 后采样得到第一个新 token：

```text
first_next_id: [1]
first_next_id.shape: [1, 1]
```

把它拼接到原序列后：

```text
ids: [1, 2, 3, 4, 1]
ids.shape: [1, 5]
```

接着进入 decode 阶段。因为当前还没有 KV cache，所以 decode 不是只处理新生成的 token，而是重新处理完整 context：

```text
decode_context: [1, 2, 3, 4, 1]
decode_context.shape: [1, 5]
decode_logits.shape: [1, 5, 10]
```

然后采样得到第二个新 token：

```text
second_next_id: [5]
second_next_id.shape: [1, 1]
```

再次拼接后：

```text
ids: [1, 2, 3, 4, 1, 5]
ids.shape: [1, 6]
```

这说明生成过程是：

```text
完整 prompt -> 生成 token1 -> 拼接 -> 重新处理更长 context -> 生成 token2 -> 拼接
```

没有 KV cache 时，旧 token 会被反复计算。后面引入 KV cache 后，prefill 会缓存旧 token 的 K/V，decode 阶段就可以只处理新 token，并复用历史 K/V。

## Next

- 学 KV cache 的直觉：缓存历史 Key/Value，避免 decode 阶段重复计算旧 token。
