from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AttentionConfig:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    batch_size: int = 1
    seq_len: int = 4
    num_layers: int = 32
    bytes_per_element: int = 2


def readable_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def kv_cache_bytes(cfg):
    return (
        2
        * cfg.batch_size
        * cfg.seq_len
        * cfg.num_layers
        * cfg.num_kv_heads
        * cfg.head_dim
        * cfg.bytes_per_element
    )


def project_qkv(x, cfg):
    n_embd = cfg.num_q_heads * cfg.head_dim

    q_proj = nn.Linear(n_embd, cfg.num_q_heads * cfg.head_dim, bias=False)
    k_proj = nn.Linear(n_embd, cfg.num_kv_heads * cfg.head_dim, bias=False)
    v_proj = nn.Linear(n_embd, cfg.num_kv_heads * cfg.head_dim, bias=False)

    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    q = q.view(cfg.batch_size, cfg.seq_len, cfg.num_q_heads, cfg.head_dim)
    k = k.view(cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)
    v = v.view(cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    return q, k, v, q_proj, k_proj, v_proj


def repeat_kv_for_attention(x, num_q_heads):
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    repeats = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeats, dim=1)


def print_config_result(cfg, x):
    q, k, v, q_proj, k_proj, v_proj = project_qkv(x, cfg)
    k_for_attention = repeat_kv_for_attention(k, cfg.num_q_heads)
    v_for_attention = repeat_kv_for_attention(v, cfg.num_q_heads)

    group_size = cfg.num_q_heads // cfg.num_kv_heads
    cache_bytes = kv_cache_bytes(cfg)

    print(f"\n=== {cfg.name} ===")
    print("num_q_heads:", cfg.num_q_heads)
    print("num_kv_heads:", cfg.num_kv_heads)
    print("query heads per KV head:", group_size)

    print("\nProjection weight shapes:")
    print("q_proj.weight.shape:", q_proj.weight.shape)
    print("k_proj.weight.shape:", k_proj.weight.shape)
    print("v_proj.weight.shape:", v_proj.weight.shape)

    print("\nProjected tensor shapes:")
    print("q.shape:", q.shape)
    print("k.shape:", k.shape)
    print("v.shape:", v.shape)

    print("\nLogical K/V used by attention after repeat:")
    print("k_for_attention.shape:", k_for_attention.shape)
    print("v_for_attention.shape:", v_for_attention.shape)

    print("\nKV cache memory estimate:")
    print("bytes:", cache_bytes)
    print("readable:", readable_bytes(cache_bytes))

    return cache_bytes


def main():
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    num_q_heads = 8
    head_dim = 6
    n_embd = num_q_heads * head_dim

    x = torch.randn(batch_size, seq_len, n_embd)

    configs = [
        AttentionConfig(
            name="MHA: Multi-Head Attention",
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
        AttentionConfig(
            name="GQA: Grouped-Query Attention",
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
        AttentionConfig(
            name="MQA: Multi-Query Attention",
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
    ]

    print("Input:")
    print("x.shape:", x.shape)
    print("n_embd:", n_embd)

    cache_by_name = {}
    for cfg in configs:
        cache_by_name[cfg.name] = print_config_result(cfg, x)

    mha_bytes = cache_by_name["MHA: Multi-Head Attention"]
    gqa_bytes = cache_by_name["GQA: Grouped-Query Attention"]
    mqa_bytes = cache_by_name["MQA: Multi-Query Attention"]

    print("\nKV cache comparison:")
    print("| type | num_q_heads | num_kv_heads | cache | relative to MHA |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for cfg in configs:
        cache = cache_by_name[cfg.name]
        print(
            f"| {cfg.name.split(':')[0]} | {cfg.num_q_heads} | {cfg.num_kv_heads} | "
            f"{readable_bytes(cache)} | {cache / mha_bytes:.2f}x |"
        )

    print("\nMemory saving:")
    print("GQA cache / MHA cache:", f"{gqa_bytes / mha_bytes:.2f}x")
    print("MQA cache / MHA cache:", f"{mqa_bytes / mha_bytes:.2f}x")

    print("\nMeaning:")
    print("MHA gives every query head its own K/V head.")
    print("MQA makes all query heads share one K/V head.")
    print("GQA is between MHA and MQA: each group of query heads shares one K/V head.")
    print("Q is temporary, but K/V are stored in KV cache during decoding.")
    print("Reducing num_kv_heads directly reduces KV cache memory.")


if __name__ == "__main__":
    main()
