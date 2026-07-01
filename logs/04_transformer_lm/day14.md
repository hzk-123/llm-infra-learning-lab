# Daily Log

Date: 2026-06-12

## Learned

- Feed Forward Network 是 Transformer block 里的第二个核心子层。
- Attention 负责在 token 之间混合信息。
- FFN 负责对每个 token 自己的表示做非线性变换。
- FFN 不改变 batch size，也不改变 sequence length。
- 最简单的 FFN 结构是 `n_embd -> 4 * n_embd -> n_embd`。
- 激活函数 ReLU 提供非线性表达能力。

## Ran

```text
python exercises\day14_feed_forward.py
```

## My Explanation

这次输入是：

```text
x.shape: [2, 4, 8]
```

表示 2 条样本，每条 4 个 token，每个 token 是 8 维表示。

经过 Feed Forward 后：

```text
out.shape: [2, 4, 8]
```

说明 FFN 保持 batch 维和 sequence 维不变，也保持最终 embedding 维度不变。

FFN 的结构是：

```text
Linear(8 -> 32)
ReLU()
Linear(32 -> 8)
```

对应参数形状：

```text
net.0.weight: [32, 8]
net.0.bias:   [32]
net.2.weight: [8, 32]
net.2.bias:   [8]
```

这里中间维度 `32 = 4 * n_embd`，先把 token 表示扩展到更高维空间，再通过 ReLU 加入非线性，最后投影回原来的 `n_embd = 8`。

单个 token 进入 FFN 前后都还是 8 维向量，但数值发生了变化：

```text
before FFN: [ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047]
after FFN:  [ 0.3873, -0.3486,  0.1011, -0.1938, -0.1850,  0.4244, -0.1804, -0.3889]
```

这说明 FFN 会对每个 token 的表示做变换，但它不会像 attention 那样跨 token 混合信息。

核心区别：

```text
Attention: token 与 token 之间交换信息
FFN: 每个 token 内部做非线性变换
```

## Next

- 把 Multi-Head Attention 和 Feed Forward 组合起来，形成最小 Transformer Block。
