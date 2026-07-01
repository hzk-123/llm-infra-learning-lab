import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for this simple RoPE demo")

    position_ids = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(position_ids, inv_freq)

    cos = torch.repeat_interleave(torch.cos(freqs), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(freqs), repeats=2, dim=-1)
    return position_ids, inv_freq, cos, sin


def rotate_half_pairwise(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half_pairwise(x) * sin


def main():
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    n_embd = 12
    num_heads = 2
    head_dim = 6

    x = torch.randn(batch_size, seq_len, n_embd)
    q_proj = nn.Linear(n_embd, num_heads * head_dim, bias=False)
    k_proj = nn.Linear(n_embd, num_heads * head_dim, bias=False)

    q = q_proj(x)
    k = k_proj(x)

    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    position_ids, inv_freq, cos, sin = build_rope_cache(
        seq_len=seq_len,
        head_dim=head_dim,
    )

    q_rope = apply_rope(q, cos, sin)
    k_rope = apply_rope(k, cos, sin)

    print("x.shape:", x.shape)
    print("q_proj.weight.shape:", q_proj.weight.shape)
    print("k_proj.weight.shape:", k_proj.weight.shape)

    print("\nAfter projection and reshape:")
    print("q.shape:", q.shape)
    print("k.shape:", k.shape)
    print("q_rope.shape:", q_rope.shape)
    print("k_rope.shape:", k_rope.shape)

    print("\nRoPE cache:")
    print("position_ids:", position_ids)
    print("inv_freq:", inv_freq)
    print("cos.shape:", cos.shape)
    print("sin.shape:", sin.shape)
    print("cos[0]:", cos[0])
    print("sin[0]:", sin[0])
    print("cos[1]:", cos[1])
    print("sin[1]:", sin[1])

    print("\nPosition 0, head 0, q before and after RoPE:")
    print("before:", q[0, 0, 0])
    print("after: ", q_rope[0, 0, 0])
    print("allclose:", torch.allclose(q[0, 0, 0], q_rope[0, 0, 0], atol=1e-6))

    print("\nPosition 1, head 0, q before and after RoPE:")
    print("before:", q[0, 0, 1])
    print("after: ", q_rope[0, 0, 1])
    print("allclose:", torch.allclose(q[0, 0, 1], q_rope[0, 0, 1], atol=1e-6))

    before_norm = torch.linalg.vector_norm(q[0, 0, 1]).item()
    after_norm = torch.linalg.vector_norm(q_rope[0, 0, 1]).item()

    print("\nNorm check for position 1:")
    print("before norm:", before_norm)
    print("after norm:", after_norm)

    print("\nMeaning:")
    print("RoPE is applied to Q and K, not directly added to token embeddings.")
    print("RoPE keeps Q/K shapes unchanged.")
    print("Position 0 has zero rotation, so the vector stays the same.")
    print("Later positions rotate Q/K by different angles.")
    print("Rotation changes vector direction while keeping its norm approximately unchanged.")


if __name__ == "__main__":
    main()
