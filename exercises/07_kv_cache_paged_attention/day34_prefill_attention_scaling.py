import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PrefillResult:
    seq_len: int
    q_shape: torch.Size
    k_shape: torch.Size
    scores_shape: torch.Size
    score_elements: int
    score_memory_bytes: int
    relative_elements: float
    score_matmul_ms: float
    softmax_ms: float


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


def run_prefill_case(batch_size, num_heads, seq_len, head_dim, baseline_elements):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    scores, score_matmul_ms = best_time_ms(
        lambda: (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    )
    _, softmax_ms = best_time_ms(lambda: F.softmax(scores, dim=-1))

    score_elements = scores.numel()
    score_memory_bytes = score_elements * scores.element_size()

    return PrefillResult(
        seq_len=seq_len,
        q_shape=q.shape,
        k_shape=k.shape,
        scores_shape=scores.shape,
        score_elements=score_elements,
        score_memory_bytes=score_memory_bytes,
        relative_elements=score_elements / baseline_elements,
        score_matmul_ms=score_matmul_ms,
        softmax_ms=softmax_ms,
    )


def print_shape_examples(results):
    print("\nShape examples:")
    for item in results:
        print(f"\nseq_len = {item.seq_len}")
        print("q.shape:", item.q_shape)
        print("k.shape:", item.k_shape)
        print("scores.shape:", item.scores_shape)


def print_result_table(results):
    print("\nPrefill attention scaling:")
    print(
        "| seq_len | scores.shape | score elements | relative | "
        "score memory | score matmul ms | softmax ms |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in results:
        print(
            f"| {item.seq_len} | {list(item.scores_shape)} | "
            f"{item.score_elements} | {item.relative_elements:.1f}x | "
            f"{readable_bytes(item.score_memory_bytes)} | "
            f"{item.score_matmul_ms:.3f} | {item.softmax_ms:.3f} |"
        )


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    batch_size = 1
    num_heads = 4
    head_dim = 32
    seq_lens = [64, 128, 256, 512]

    baseline_elements = batch_size * num_heads * seq_lens[0] * seq_lens[0]

    print("Config:")
    print("batch_size:", batch_size)
    print("num_heads:", num_heads)
    print("head_dim:", head_dim)
    print("seq_lens:", seq_lens)
    print("dtype: fp32")

    with torch.inference_mode():
        results = [
            run_prefill_case(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                baseline_elements=baseline_elements,
            )
            for seq_len in seq_lens
        ]

    print_shape_examples(results)
    print_result_table(results)

    print("\nGrowth check:")
    for prev, curr in zip(results, results[1:]):
        seq_ratio = curr.seq_len / prev.seq_len
        element_ratio = curr.score_elements / prev.score_elements
        print(
            f"T {prev.seq_len} -> {curr.seq_len}: "
            f"seq_len {seq_ratio:.1f}x, score elements {element_ratio:.1f}x"
        )

    print("\nMeaning:")
    print("Prefill attention processes the whole prompt at once.")
    print("Attention scores have shape [B, H, T, T].")
    print("When T doubles, score elements grow by about 4x.")
    print("Long prompts increase prefill work, so TTFT usually becomes higher.")
    print("CPU timing can be noisy; score element growth is the key deterministic signal.")


if __name__ == "__main__":
    main()
