from dataclasses import dataclass


@dataclass
class BenchmarkRequest:
    request_id: str
    workload_type: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    concurrency: int


def fake_prompt(token_count, token="hello"):
    return " ".join([token] * token_count)


def build_workload(concurrency):
    requests = []

    requests.append(
        BenchmarkRequest(
            request_id="short_prompt_001",
            workload_type="short_prompt",
            prompt=fake_prompt(32, "short"),
            prompt_tokens=32,
            max_new_tokens=32,
            concurrency=concurrency,
        )
    )

    requests.append(
        BenchmarkRequest(
            request_id="long_prompt_001",
            workload_type="long_prompt",
            prompt=fake_prompt(2048, "context"),
            prompt_tokens=2048,
            max_new_tokens=32,
            concurrency=concurrency,
        )
    )

    requests.append(
        BenchmarkRequest(
            request_id="long_output_001",
            workload_type="long_output",
            prompt=fake_prompt(32, "generate"),
            prompt_tokens=32,
            max_new_tokens=256,
            concurrency=concurrency,
        )
    )

    requests.append(
        BenchmarkRequest(
            request_id="mixed_001",
            workload_type="mixed",
            prompt=fake_prompt(512, "mixed"),
            prompt_tokens=512,
            max_new_tokens=128,
            concurrency=concurrency,
        )
    )

    return requests


def print_workload_table(requests):
    print("| request_id | workload_type | prompt_tokens | max_new_tokens | concurrency |")
    print("| --- | --- | ---: | ---: | ---: |")
    for req in requests:
        print(
            f"| {req.request_id} | {req.workload_type} | "
            f"{req.prompt_tokens} | {req.max_new_tokens} | {req.concurrency} |"
        )


def estimate_pressure(req):
    prefill_pressure = req.prompt_tokens
    decode_pressure = req.max_new_tokens
    kv_cache_pressure = req.prompt_tokens + req.max_new_tokens
    return prefill_pressure, decode_pressure, kv_cache_pressure


def main():
    concurrency_levels = [1, 4, 8]

    for concurrency in concurrency_levels:
        print(f"\nConcurrency = {concurrency}")
        requests = build_workload(concurrency)
        print_workload_table(requests)

        print("\nPressure estimate:")
        print("| request_id | prefill_pressure | decode_pressure | kv_cache_pressure |")
        print("| --- | ---: | ---: | ---: |")
        for req in requests:
            prefill, decode, kv_cache = estimate_pressure(req)
            print(f"| {req.request_id} | {prefill} | {decode} | {kv_cache} |")

    print("\nMeaning:")
    print("short_prompt checks basic latency.")
    print("long_prompt stresses prefill and TTFT.")
    print("long_output stresses decode, TPOT, and tokens/s.")
    print("mixed workload is closer to real serving traffic.")
    print("higher concurrency increases scheduling pressure and KV cache memory pressure.")


if __name__ == "__main__":
    main()
