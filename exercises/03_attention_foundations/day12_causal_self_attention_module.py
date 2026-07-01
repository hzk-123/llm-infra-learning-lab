import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
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

        return out, weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 6
    head_size = 3

    x = torch.randn(batch_size, block_size, n_embd)
    attn = CausalSelfAttention(n_embd=n_embd, head_size=head_size, block_size=block_size)

    out, weights = attn(x)

    print("x.shape:", x.shape)
    print("attn.mask.shape:", attn.mask.shape)
    print("q_proj.weight.shape:", attn.q_proj.weight.shape)
    print("k_proj.weight.shape:", attn.k_proj.weight.shape)
    print("v_proj.weight.shape:", attn.v_proj.weight.shape)

    print("\nout.shape:", out.shape)
    print("weights.shape:", weights.shape)

    print("\nweights[0]:")
    print(weights[0])
    print("row sums for weights[0]:")
    print(weights[0].sum(dim=-1))

    print("\nFuture-position weights:")
    print("weights[0, 0, 1:]:", weights[0, 0, 1:])
    print("weights[0, 1, 2:]:", weights[0, 1, 2:])

    print("\nNamed parameters:")
    for name, param in attn.named_parameters():
        print(name, param.shape)

    print("\nNamed buffers:")
    for name, buffer in attn.named_buffers():
        print(name, buffer.shape)

    print("\nMeaning:")
    print("CausalSelfAttention is now a reusable nn.Module.")
    print("Projection weights are trainable parameters.")
    print("The causal mask is a non-trainable buffer.")


if __name__ == "__main__":
    main()
