import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x, return_info=False):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden)

        if return_info:
            return out, {
                "gate": gate,
                "up": up,
                "ffn_hidden": hidden,
            }
        return out  


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
    return cos, sin


def rotate_half_pairwise(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half_pairwise(x) * sin


def repeat_kv_for_attention(x, num_q_heads):
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    repeats = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeats, dim=1)


class GQAAttentionWithRoPE(nn.Module):
    def __init__(self, n_embd, num_q_heads, num_kv_heads, head_dim, max_seq_len):
        super().__init__()
        if n_embd != num_q_heads * head_dim:
            raise ValueError("n_embd must equal num_q_heads * head_dim")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.n_embd = n_embd
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embd, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        cos, sin = build_rope_cache(max_seq_len, head_dim)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x, return_info=False):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        k_for_attention = repeat_kv_for_attention(k_rope, self.num_q_heads)
        v_for_attention = repeat_kv_for_attention(v, self.num_q_heads)

        scores = q_rope @ k_for_attention.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        attn = weights @ v_for_attention
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch_size, seq_len, self.n_embd)
        out = self.out_proj(attn)

        if return_info:
            return out, {
                "q": q,
                "k": k,
                "v": v,
                "q_rope": q_rope,
                "k_rope": k_rope,
                "k_for_attention": k_for_attention,
                "v_for_attention": v_for_attention,
                "scores": scores,
                "weights": weights,
                "attn_before_out_proj": attn,
            }
        return out


class TinyLlamaBlock(nn.Module):
    def __init__(self, n_embd, num_q_heads, num_kv_heads, head_dim, hidden_dim, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = GQAAttentionWithRoPE(
            n_embd=n_embd,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )
        self.norm2 = RMSNorm(n_embd)
        self.ffn = SwiGLUFeedForward(n_embd=n_embd, hidden_dim=hidden_dim)

    def forward(self, x, return_info=False):
        norm1_out = self.norm1(x)
        attn_out, attn_info = self.attn(norm1_out, return_info=True)
        after_attn_residual = x + attn_out

        norm2_out = self.norm2(after_attn_residual)
        ffn_out, ffn_info = self.ffn(norm2_out, return_info=True)
        out = after_attn_residual + ffn_out

        if return_info:
            info = {
                "norm1_out": norm1_out,
                "attn_out": attn_out,
                "after_attn_residual": after_attn_residual,
                "norm2_out": norm2_out,
                "ffn_out": ffn_out,
                "out": out,
            }
            info.update(attn_info)
            info.update(ffn_info)
            return out, info
        return out


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    n_embd = 16
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 4
    hidden_dim = 32
    max_seq_len = 8

    x = torch.randn(batch_size, seq_len, n_embd)
    block = TinyLlamaBlock(
        n_embd=n_embd,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
    )

    out, info = block(x, return_info=True)

    print("Config:")
    print("batch_size:", batch_size)
    print("seq_len:", seq_len)
    print("n_embd:", n_embd)
    print("num_q_heads:", num_q_heads)
    print("num_kv_heads:", num_kv_heads)
    print("head_dim:", head_dim)
    print("hidden_dim:", hidden_dim)
    print("parameter_count:", count_parameters(block))

    print("\nBlock-level shapes:")
    print("x.shape:", x.shape)
    print("norm1_out.shape:", info["norm1_out"].shape)
    print("attn_out.shape:", info["attn_out"].shape)
    print("after_attn_residual.shape:", info["after_attn_residual"].shape)
    print("norm2_out.shape:", info["norm2_out"].shape)
    print("ffn_out.shape:", info["ffn_out"].shape)
    print("out.shape:", out.shape)

    print("\nAttention internal shapes:")
    print("q.shape:", info["q"].shape)
    print("k.shape:", info["k"].shape)
    print("v.shape:", info["v"].shape)
    print("q_rope.shape:", info["q_rope"].shape)
    print("k_rope.shape:", info["k_rope"].shape)
    print("k_for_attention.shape:", info["k_for_attention"].shape)
    print("v_for_attention.shape:", info["v_for_attention"].shape)
    print("scores.shape:", info["scores"].shape)
    print("weights.shape:", info["weights"].shape)
    print("attn_before_out_proj.shape:", info["attn_before_out_proj"].shape)

    print("\nSwiGLU internal shapes:")
    print("gate.shape:", info["gate"].shape)
    print("up.shape:", info["up"].shape)
    print("ffn_hidden.shape:", info["ffn_hidden"].shape)

    print("\nNamed child modules:")
    for name, module in block.named_children():
        print(name, "->", module.__class__.__name__)

    print("\nShape checks:")
    print("Attention output can be added to x:", info["attn_out"].shape == x.shape)
    print("FFN output can be added after attention:", info["ffn_out"].shape == x.shape)
    print("Final output keeps input shape:", out.shape == x.shape)

    print("\nMeaning:")
    print("This block uses RMSNorm instead of LayerNorm.")
    print("RoPE is applied to Q and K inside attention.")
    print("GQA uses fewer K/V heads than Q heads to reduce KV cache size.")
    print("SwiGLU replaces the simple ReLU feed-forward network.")
    print("Residual connections require attention and FFN outputs to return to [B, T, n_embd].")


if __name__ == "__main__":
    main()
