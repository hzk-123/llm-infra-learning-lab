import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class MemoryResult:
    experiment: str
    dtype_name: str
    batch_size: int
    seq_len: int
    tensor_shape: str
    theoretical_memory: str
    peak_allocated: str
    reserved_after: str
    status: str


def readable_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def tensor_bytes(shape, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * element_size


def clear_cuda():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def measure_prefill_scores(batch_size, num_heads, seq_len, head_dim, dtype_name, device):
    dtype = DTYPES[dtype_name]
    scores_shape = (batch_size, num_heads, seq_len, seq_len)
    theoretical = tensor_bytes(scores_shape, dtype)

    clear_cuda()
    try:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        _ = F.softmax(scores, dim=-1)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        status = "OOM"
    finally:
        del_vars = ["q", "k", "scores"]
        for name in del_vars:
            if name in locals():
                del locals()[name]
        clear_cuda()

    return MemoryResult(
        experiment="prefill_scores",
        dtype_name=dtype_name,
        batch_size=batch_size,
        seq_len=seq_len,
        tensor_shape=str(list(scores_shape)),
        theoretical_memory=readable_bytes(theoretical),
        peak_allocated=readable_bytes(peak),
        reserved_after=readable_bytes(reserved),
        status=status,
    )


def measure_kv_cache_one_layer(batch_size, num_kv_heads, seq_len, head_dim, dtype_name, device):
    dtype = DTYPES[dtype_name]
    k_shape = (batch_size, num_kv_heads, seq_len, head_dim)
    one_layer_bytes = 2 * tensor_bytes(k_shape, dtype)

    clear_cuda()
    try:
        k_cache = torch.empty(k_shape, device=device, dtype=dtype)
        v_cache = torch.empty(k_shape, device=device, dtype=dtype)
        k_cache.zero_()
        v_cache.zero_()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        status = "OOM"
    finally:
        del_vars = ["k_cache", "v_cache"]
        for name in del_vars:
            if name in locals():
                del locals()[name]
        clear_cuda()

    return MemoryResult(
        experiment="kv_cache_one_layer",
        dtype_name=dtype_name,
        batch_size=batch_size,
        seq_len=seq_len,
        tensor_shape=f"2 x {list(k_shape)}",
        theoretical_memory=readable_bytes(one_layer_bytes),
        peak_allocated=readable_bytes(peak),
        reserved_after=readable_bytes(reserved),
        status=status,
    )


def estimate_full_kv_cache(batch_size, seq_len, num_layers, num_kv_heads, head_dim, dtype_name):
    dtype = DTYPES[dtype_name]
    k_shape = (batch_size, num_kv_heads, seq_len, head_dim)
    total_bytes = num_layers * 2 * tensor_bytes(k_shape, dtype)
    return total_bytes


def print_results(results):
    print(
        "| experiment | dtype | B | T | tensor shape | theoretical | "
        "peak allocated | reserved after | status |"
    )
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for item in results:
        print(
            f"| {item.experiment} | {item.dtype_name} | {item.batch_size} | "
            f"{item.seq_len} | {item.tensor_shape} | {item.theoretical_memory} | "
            f"{item.peak_allocated} | {item.reserved_after} | {item.status} |"
        )


def print_full_kv_estimates(batch_sizes, seq_lens, dtype_names, num_layers, num_kv_heads, head_dim):
    print("\nFull-model KV cache estimates:")
    print(
        "| dtype | B | T | layers | H_kv | D | estimated KV cache |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dtype_name in dtype_names:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                total = estimate_full_kv_cache(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype_name=dtype_name,
                )
                print(
                    f"| {dtype_name} | {batch_size} | {seq_len} | {num_layers} | "
                    f"{num_kv_heads} | {head_dim} | {readable_bytes(total)} |"
                )


def main():
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nMeaning:")
        print("CUDA is not available in this environment.")
        print("Run this script on the 4090 server to measure real GPU memory.")
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print("device:", device)
    print("GPU name:", props.name)
    print("GPU total memory:", readable_bytes(props.total_memory))
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)

    batch_sizes = [1, 2]
    seq_lens = [512, 1024, 2048]
    dtype_names = ["fp16", "fp32"]

    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    num_layers = 32

    print("\nExperiment config:")
    print("batch_sizes:", batch_sizes)
    print("seq_lens:", seq_lens)
    print("dtype_names:", dtype_names)
    print("num_heads:", num_heads)
    print("num_kv_heads:", num_kv_heads)
    print("head_dim:", head_dim)
    print("num_layers for full KV estimate:", num_layers)

    results = []
    for dtype_name in dtype_names:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                results.append(
                    measure_prefill_scores(
                        batch_size=batch_size,
                        num_heads=num_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        dtype_name=dtype_name,
                        device=device,
                    )
                )
                results.append(
                    measure_kv_cache_one_layer(
                        batch_size=batch_size,
                        num_kv_heads=num_kv_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        dtype_name=dtype_name,
                        device=device,
                    )
                )

    print("\nMeasured CUDA memory:")
    print_results(results)

    print_full_kv_estimates(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        dtype_names=dtype_names,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    print("\nMeaning:")
    print("Prefill scores use shape [B, H, T, T], so seq_len has a quadratic effect.")
    print("KV cache uses shape [B, H_kv, T, D] per layer, so seq_len has a linear effect.")
    print("Full-model KV cache multiplies the one-layer cache by num_layers.")
    print("FP32 uses about 2x the memory of FP16/BF16 for these tensors.")
    print("Peak allocated can be larger than one tensor's theoretical memory because q/k/scores/softmax may coexist.")
    print("Record OOM cases as useful engineering data, not just failures.")


if __name__ == "__main__":
    main()
