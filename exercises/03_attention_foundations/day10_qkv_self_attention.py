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

    scores = q @ k.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(head_size)
    weights = F.softmax(scaled_scores, dim=-1)
    out = weights @ v

    print("x.shape:", x.shape)
    print("q_proj.weight.shape:", q_proj.weight.shape)
    print("k_proj.weight.shape:", k_proj.weight.shape)
    print("v_proj.weight.shape:", v_proj.weight.shape)

    print("\nq.shape:", q.shape)
    print("k.shape:", k.shape)
    print("v.shape:", v.shape)

    print("\nscores.shape:", scores.shape)
    print("scaled_scores.shape:", scaled_scores.shape)
    print("weights.shape:", weights.shape)
    print("row sums:")
    print(weights.sum(dim=-1))

    print("\nout.shape:", out.shape)

    print("\nOne score example:")
    print("scores[0, 1, 2] is token 1's query matched with token 2's key.")
    print("scores[0, 1, 2]:", scores[0, 1, 2].item())
    print("scaled_scores[0, 1, 2]:", scaled_scores[0, 1, 2].item())

    print("\nMeaning:")
    print("Q and K are used to compute attention weights.")
    print("V is the information that gets mixed by those weights.")
    print("The output is a weighted sum of V, not a weighted sum of raw x.")


if __name__ == "__main__":
    main()
