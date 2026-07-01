from dataclasses import dataclass


DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


@dataclass
class KVCacheConfig:
    batch_size: int
    seq_len: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: str


def kv_cache_bytes(config):
    bytes_per_element = DTYPE_BYTES[config.dtype]
    return (
        2
        * config.batch_size
        * config.seq_len
        * config.num_layers
        * config.num_kv_heads
        * config.head_size
        * bytes_per_element
    )


def format_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def print_case(name, config):
    total_bytes = kv_cache_bytes(config)
    print(name)
    print("  config:", config)
    print("  bytes:", total_bytes)
    print("  readable:", format_bytes(total_bytes))


def main():
    toy = KVCacheConfig(
        batch_size=1,
        seq_len=5,
        num_layers=1,
        num_kv_heads=1,
        head_size=3,
        dtype="fp32",
    )

    base = KVCacheConfig(
        batch_size=1,
        seq_len=4096,
        num_layers=32,
        num_kv_heads=32,
        head_size=128,
        dtype="fp16",
    )

    longer_context = KVCacheConfig(
        batch_size=1,
        seq_len=8192,
        num_layers=32,
        num_kv_heads=32,
        head_size=128,
        dtype="fp16",
    )

    larger_batch = KVCacheConfig(
        batch_size=4,
        seq_len=4096,
        num_layers=32,
        num_kv_heads=32,
        head_size=128,
        dtype="fp16",
    )

    fp32_case = KVCacheConfig(
        batch_size=1,
        seq_len=4096,
        num_layers=32,
        num_kv_heads=32,
        head_size=128,
        dtype="fp32",
    )

    gqa_case = KVCacheConfig(
        batch_size=1,
        seq_len=4096,
        num_layers=32,
        num_kv_heads=8,
        head_size=128,
        dtype="fp16",
    )

    print("KV cache formula:")
    print("bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_size * bytes_per_element")
    print("\nThe leading 2 is for K and V.\n")

    print_case("Toy example", toy)
    print()
    print_case("Base fp16 example", base)
    print()
    print_case("Longer context doubles seq_len", longer_context)
    print()
    print_case("Larger batch quadruples batch_size", larger_batch)
    print()
    print_case("FP32 doubles fp16 memory", fp32_case)
    print()
    print_case("GQA-style fewer KV heads", gqa_case)

    print("\nMeaning:")
    print("KV cache memory grows linearly with batch_size and seq_len.")
    print("FP32 uses 2x the memory of FP16/BF16 for KV cache.")
    print("Using fewer KV heads, as in GQA/MQA, can significantly reduce KV cache memory.")


if __name__ == "__main__":
    main()
