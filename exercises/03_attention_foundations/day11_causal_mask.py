import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    torch.manual_seed(42)

    batch_size = 1
    block_size = 4
    n_embd = 6
    head_size = 3

    x = torch.randn(batch_size, block_size, n_embd)

    q_proj = nn.Linear(n_embd, head_size, bias=False)
    k_proj = nn.Linear(n_embd, head_size, bias=False)
    v_proj = nn.Linear(n_embd, head_size, bias=False)

    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)

    mask = torch.tril(torch.ones(block_size, block_size))
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(masked_scores, dim=-1)
    out = weights @ v

    print("scores before mask:")
    print(scores)
    print("scores.shape:", scores.shape)

    print("\ncausal mask:")
    print(mask)
    print("mask.shape:", mask.shape)

    print("\nmasked_scores:")
    print(masked_scores)

    print("\nweights after softmax:")
    print(weights)
    print("weights.shape:", weights.shape)
    print("row sums:")
    print(weights.sum(dim=-1))

    print("\nout.shape:", out.shape)

    print("\nFuture-position weights:")
    print("weights[0, 0, 1:] should be 0 because token 0 cannot look ahead.")
    print(weights[0, 0, 1:])
    print("weights[0, 1, 2:] should be 0 because token 1 cannot look ahead.")
    print(weights[0, 1, 2:])

    print("\nMeaning:")
    print("Causal mask prevents each token from attending to future tokens.")
    print("This makes self-attention valid for next-token prediction.")


if __name__ == "__main__":
    main()
