import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch_size, block_size, n_embd = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        mask = self.mask[:block_size, :block_size]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")

        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=self.head_size,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        out = self.proj(concat)
        return out, head_outputs, concat


def main():
    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8
    num_heads = 2

    x = torch.randn(batch_size, block_size, n_embd)
    mha = MultiHeadAttention(n_embd=n_embd, num_heads=num_heads, block_size=block_size)

    out, head_outputs, concat = mha(x)

    print("x.shape:", x.shape)
    print("num_heads:", num_heads)
    print("head_size:", mha.head_size)

    print("\nEach head output:")
    for i, head_out in enumerate(head_outputs):
        print(f"head_outputs[{i}].shape:", head_out.shape)

    print("\nconcat.shape:", concat.shape)
    print("proj.weight.shape:", mha.proj.weight.shape)
    print("out.shape:", out.shape)

    print("\nNamed child modules:")
    for name, module in mha.named_children():
        print(name, "->", module.__class__.__name__)

    print("\nMeaning:")
    print("Each head runs causal self-attention in its own subspace.")
    print("Outputs are concatenated along the embedding dimension.")
    print("The final projection mixes information from all heads.")


if __name__ == "__main__":
    main()
