from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CacheStats:
    step_name: str
    cache_position: int
    k_cache_shape: torch.Size
    v_cache_shape: torch.Size
    stored_elements: int
    repeated_elements: int


class MultiHeadKVCacheDemo(nn.Module):
    def __init__(self, n_embd, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        if n_embd != num_q_heads * head_dim:
            raise ValueError("n_embd must equal num_q_heads * head_dim")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.n_embd = n_embd
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, num_kv_heads * head_dim, bias=False)

    def project_q(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        return q.transpose(1, 2)

    def project_kv(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return k, v

    def prefill(self, x_prompt):
        k, v = self.project_kv(x_prompt)
        cache = {
            "k": k,
            "v": v,
            "seq_len": x_prompt.shape[1],
        }
        return cache

    def decode_append(self, x_new, cache):
        k_new, v_new = self.project_kv(x_new)

        k_all = torch.cat([cache["k"], k_new], dim=2)
        v_all = torch.cat([cache["v"], v_new], dim=2)

        new_cache = {
            "k": k_all,
            "v": v_all,
            "seq_len": cache["seq_len"] + 1,
        }
        return new_cache, k_new, v_new


def repeat_kv_for_attention(x, num_q_heads):
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    repeats = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeats, dim=1)


def count_elements(k, v):
    return k.numel() + v.numel()


def collect_stats(step_name, cache_position, cache, num_q_heads):
    k_for_attention = repeat_kv_for_attention(cache["k"], num_q_heads)
    v_for_attention = repeat_kv_for_attention(cache["v"], num_q_heads)

    return CacheStats(
        step_name=step_name,
        cache_position=cache_position,
        k_cache_shape=cache["k"].shape,
        v_cache_shape=cache["v"].shape,
        stored_elements=count_elements(cache["k"], cache["v"]),
        repeated_elements=count_elements(k_for_attention, v_for_attention),
    )


def print_stats_table(stats):
    print("\nCache growth:")
    print(
        "| step | cache_position | k_cache.shape | v_cache.shape | "
        "stored elems | repeated-for-attn elems |"
    )
    print("| --- | ---: | --- | --- | ---: | ---: |")
    for item in stats:
        print(
            f"| {item.step_name} | {item.cache_position} | "
            f"{list(item.k_cache_shape)} | {list(item.v_cache_shape)} | "
            f"{item.stored_elements} | {item.repeated_elements} |"
        )


def main():
    torch.manual_seed(42)

    batch_size = 2
    prompt_len = 5
    decode_steps = 3
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 4
    n_embd = num_q_heads * head_dim

    demo = MultiHeadKVCacheDemo(
        n_embd=n_embd,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    print("Config:")
    print("batch_size:", batch_size)
    print("prompt_len:", prompt_len)
    print("decode_steps:", decode_steps)
    print("num_q_heads:", num_q_heads)
    print("num_kv_heads:", num_kv_heads)
    print("query heads per KV head:", num_q_heads // num_kv_heads)
    print("head_dim:", head_dim)
    print("n_embd:", n_embd)

    x_prompt = torch.randn(batch_size, prompt_len, n_embd)
    q_prompt = demo.project_q(x_prompt)
    cache = demo.prefill(x_prompt)

    print("\nPrefill:")
    print("x_prompt.shape:", x_prompt.shape)
    print("q_prompt.shape:", q_prompt.shape)
    print("k_cache.shape:", cache["k"].shape)
    print("v_cache.shape:", cache["v"].shape)
    print("cache seq_len:", cache["seq_len"])

    k_for_attention = repeat_kv_for_attention(cache["k"], num_q_heads)
    v_for_attention = repeat_kv_for_attention(cache["v"], num_q_heads)

    print("\nFor attention after prefill:")
    print("k_for_attention.shape:", k_for_attention.shape)
    print("v_for_attention.shape:", v_for_attention.shape)
    print("stored cache elements:", count_elements(cache["k"], cache["v"]))
    print("repeated-for-attention elements:", count_elements(k_for_attention, v_for_attention))

    stats = [collect_stats("prefill", prompt_len - 1, cache, num_q_heads)]

    for step in range(decode_steps):
        cache_position = prompt_len + step
        x_new = torch.randn(batch_size, 1, n_embd)
        q_new = demo.project_q(x_new)
        cache, k_new, v_new = demo.decode_append(x_new, cache)

        print(f"\nDecode step {step + 1}:")
        print("cache_position:", cache_position)
        print("x_new.shape:", x_new.shape)
        print("q_new.shape:", q_new.shape)
        print("k_new.shape:", k_new.shape)
        print("v_new.shape:", v_new.shape)
        print("updated k_cache.shape:", cache["k"].shape)
        print("updated v_cache.shape:", cache["v"].shape)
        print("cache seq_len:", cache["seq_len"])

        stats.append(
            collect_stats(
                step_name=f"decode_{step + 1}",
                cache_position=cache_position,
                cache=cache,
                num_q_heads=num_q_heads,
            )
        )

    print_stats_table(stats)

    print("\nMeaning:")
    print("Realistic KV cache is stored as [B, H_kv, T, D].")
    print("Prefill writes all prompt K/V into the cache.")
    print("Each decode step appends one new K/V position to the cache.")
    print("In GQA, H_kv is smaller than H_q, so stored KV cache is smaller.")
    print("For attention, K/V can be logically repeated from H_kv to H_q.")
    print("The cache stores H_kv K/V heads, not the repeated H_q version.")


if __name__ == "__main__":
    main()

# 3(D)*["Ptf
