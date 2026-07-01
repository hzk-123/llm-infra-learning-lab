import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DecodeResult:
    cache_len: int
    q_shape: torch.Size
    k_cache_shape: torch.Size
    k_for_attention_shape: torch.Size
    scores_shape: torch.Size
    score_elements: int
    score_memory_bytes: int
    stored_kv_elements: int
    repeated_kv_elements: int
    relative_score_elements: float
    score_matmul_ms: float
    attention_output_ms: float


def readable_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def best_time_ms(fn, repeats=5):
    best_seconds = None
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if best_seconds is None or elapsed < best_seconds:
            best_seconds = elapsed

    return result, best_seconds * 1000


def repeat_kv_for_attention(x, num_q_heads):
    batch_size, num_kv_heads, cache_len, head_dim = x.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    repeats = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeats, dim=1)


def count_kv_elements(k, v):
    return k.numel() + v.numel()


def run_decode_case(
    batch_size,
    num_q_heads,
    num_kv_heads,
    cache_len,
    head_dim,
    baseline_score_elements,
):
    q_new = torch.randn(batch_size, num_q_heads, 1, head_dim)
    k_cache = torch.randn(batch_size, num_kv_heads, cache_len, head_dim)
    v_cache = torch.randn(batch_size, num_kv_heads, cache_len, head_dim)

    k_for_attention = repeat_kv_for_attention(k_cache, num_q_heads)
    v_for_attention = repeat_kv_for_attention(v_cache, num_q_heads)

    scores, score_matmul_ms = best_time_ms(
        lambda: (q_new @ k_for_attention.transpose(-2, -1)) / math.sqrt(head_dim)
    )

    def compute_attention_output():
        weights = F.softmax(scores, dim=-1)
        return weights @ v_for_attention

    _, attention_output_ms = best_time_ms(compute_attention_output)

    score_elements = scores.numel()
    score_memory_bytes = score_elements * scores.element_size()

    return DecodeResult(
        cache_len=cache_len,
        q_shape=q_new.shape,
        k_cache_shape=k_cache.shape,
        k_for_attention_shape=k_for_attention.shape,
        scores_shape=scores.shape,
        score_elements=score_elements,
        score_memory_bytes=score_memory_bytes,
        stored_kv_elements=count_kv_elements(k_cache, v_cache),
        repeated_kv_elements=count_kv_elements(k_for_attention, v_for_attention),
        relative_score_elements=score_elements / baseline_score_elements,
        score_matmul_ms=score_matmul_ms,
        attention_output_ms=attention_output_ms,
    )


def print_shape_examples(results):
    print("\nShape examples:")
    for item in results:
        print(f"\ncache_len = {item.cache_len}")
        print("q_new.shape:", item.q_shape)
        print("k_cache.shape:", item.k_cache_shape)
        print("k_for_attention.shape:", item.k_for_attention_shape)
        print("scores.shape:", item.scores_shape)


def print_result_table(results):
    print("\nDecode attention scaling:")
    print(
        "| cache_len | scores.shape | score elems | relative | score memory | "
        "stored KV elems | repeated KV elems | score matmul ms | attn output ms |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in results:
        print(
            f"| {item.cache_len} | {list(item.scores_shape)} | "
            f"{item.score_elements} | {item.relative_score_elements:.1f}x | "
            f"{readable_bytes(item.score_memory_bytes)} | "
            f"{item.stored_kv_elements} | {item.repeated_kv_elements} | "
            f"{item.score_matmul_ms:.3f} | {item.attention_output_ms:.3f} |"
        )


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    batch_size = 1
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 32
    cache_lens = [64, 128, 256, 512, 1024]

    baseline_score_elements = batch_size * num_q_heads * 1 * cache_lens[0]

    print("Config:")
    print("batch_size:", batch_size)
    print("num_q_heads:", num_q_heads)
    print("num_kv_heads:", num_kv_heads)
    print("query heads per KV head:", num_q_heads // num_kv_heads)
    print("head_dim:", head_dim)
    print("cache_lens:", cache_lens)
    print("dtype: fp32")

    with torch.inference_mode():
        results = [
            run_decode_case(
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                cache_len=cache_len,
                head_dim=head_dim,
                baseline_score_elements=baseline_score_elements,
            )
            for cache_len in cache_lens
        ]

    print_shape_examples(results)
    print_result_table(results)

    print("\nGrowth check:")
    for prev, curr in zip(results, results[1:]):
        cache_ratio = curr.cache_len / prev.cache_len
        score_ratio = curr.score_elements / prev.score_elements
        stored_ratio = curr.stored_kv_elements / prev.stored_kv_elements
        print(
            f"T {prev.cache_len} -> {curr.cache_len}: "
            f"cache_len {cache_ratio:.1f}x, "
            f"score elements {score_ratio:.1f}x, "
            f"stored KV elements {stored_ratio:.1f}x"
        )

    print("\nMeaning:")
    print("Decode processes one new token at a time.")
    print("The new token attends to all cached K/V positions.")
    print("Decode scores have shape [B, H_q, 1, T].")
    print("When cache length T doubles, decode score elements grow by about 2x.")
    print("This is linear in T for one decode step, unlike prefill's [B, H, T, T].")
    print("Long context increases per-token decode cost and can raise TPOT.")


if __name__ == "__main__":
    main()
