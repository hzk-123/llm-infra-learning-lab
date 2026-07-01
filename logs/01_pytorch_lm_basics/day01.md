# Daily Log

Date:2026-05-19

## Learned

- LLM 训练数据可以通过 token 右移一位构造。
- `x = chunk[:-1]` 是模型输入。
- `y = chunk[1:]` 是 next-token target。
- batch 后 shape 是 `[batch_size, block_size]`。

## Ran

```
python exercises\day01_next_token_data.py
```

## My Explanation

- 模型看到 [10, 11, 12]，目标是预测 [11, 12, 13]。也就是每个位置都在预测下一个 token。

## Next

- 学 Embedding：token id 如何变成向量。
