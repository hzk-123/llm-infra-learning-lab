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

        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(concat)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd=n_embd, num_heads=num_heads, block_size=block_size)
        self.ffn = FeedForward(n_embd=n_embd)

    def forward(self, x):
        attn_out = self.attn(x)
        x = x + attn_out

        ffn_out = self.ffn(x)
        x = x + ffn_out

        return x, attn_out, ffn_out


def main():
    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8
    num_heads = 2

    x = torch.randn(batch_size, block_size, n_embd)
    block = TransformerBlock(n_embd=n_embd, num_heads=num_heads, block_size=block_size)

    out, attn_out, ffn_out = block(x)

    print("x.shape:", x.shape)
    print("attn_out.shape:", attn_out.shape)
    print("ffn_out.shape:", ffn_out.shape)
    print("out.shape:", out.shape)

    print("\nResidual compatibility:")
    print("x and attn_out same shape:", x.shape == attn_out.shape)
    print("x and ffn_out same shape:", x.shape == ffn_out.shape)

    print("\nBlock modules:")
    for name, module in block.named_children():
        print(name, "->", module.__class__.__name__)

    print("\nMeaning:")
    print("The block first mixes information across tokens with attention.")
    print("Then it transforms each token with a feed-forward network.")
    print("Residual connections add each sublayer output back to the running representation.")


if __name__ == "__main__":
    main()
