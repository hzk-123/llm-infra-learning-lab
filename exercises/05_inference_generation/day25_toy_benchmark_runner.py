from dataclasses import dataclass


@dataclass
class BenchmarkRequest:
    request_id: str
    workload_type: str
    prompt_tokens: int
    max_new_tokens: int


@dataclass
class BenchmarkResult:
    request_id: str
    workload_type: str
    prompt_tokens: int
    output_tokens: int
    ttft: float
    tpot: float
    total_latency: float
    output_tokens_per_second: float


def build_workload():
    return [
        BenchmarkRequest(
            request_id="short_prompt_001",
            workload_type="short_prompt",
            prompt_tokens=32,
            max_new_tokens=32,
        ),
        BenchmarkRequest(
            request_id="long_prompt_001",
            workload_type="long_prompt",
            prompt_tokens=2048,
            max_new_tokens=32,
        ),
        BenchmarkRequest(
            request_id="long_output_001",
            workload_type="long_output",
            prompt_tokens=32,
            max_new_tokens=256,
        ),
        BenchmarkRequest(
            request_id="mixed_001",
            workload_type="mixed",
            prompt_tokens=512,
            max_new_tokens=128,
        ),
    ]


def simulate_request_metrics(req, concurrency):
    base_ttft = 0.08
    prefill_seconds_per_token = 0.00055
    base_tpot = 0.035

    concurrency_slowdown = 1.0 + 0.08 * (concurrency - 1)

    ttft = (base_ttft + req.prompt_tokens * prefill_seconds_per_token) * concurrency_slowdown
    tpot = base_tpot * concurrency_slowdown
    decode_time = req.max_new_tokens * tpot
    total_latency = ttft + decode_time
    output_tokens_per_second = req.max_new_tokens / decode_time

    return BenchmarkResult(
        request_id=req.request_id,
        workload_type=req.workload_type,
        prompt_tokens=req.prompt_tokens,
        output_tokens=req.max_new_tokens,
        ttft=ttft,
        tpot=tpot,
        total_latency=total_latency,
        output_tokens_per_second=output_tokens_per_second,
    )


def print_result_table(results):
    print(
        "| request_id | type | prompt_tokens | output_tokens | "
        "TTFT(s) | TPOT(s/token) | total_latency(s) | output tok/s |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.request_id} | {result.workload_type} | "
            f"{result.prompt_tokens} | {result.output_tokens} | "
            f"{result.ttft:.3f} | {result.tpot:.3f} | "
            f"{result.total_latency:.3f} | {result.output_tokens_per_second:.2f} |"
        )


def print_aggregate_metrics(results, concurrency):
    total_request_types = len(results)
    total_requests = total_request_types * concurrency
    total_output_tokens = sum(result.output_tokens for result in results) * concurrency
    wall_time = max(result.total_latency for result in results)

    qps = total_requests / wall_time
    aggregate_output_tokens_per_second = total_output_tokens / wall_time

    print("\nAggregate metrics:")
    print(f"  total_request_types: {total_request_types}")
    print(f"  concurrency: {concurrency}")
    print(f"  total_requests: {total_requests}")
    print(f"  total_output_tokens: {total_output_tokens}")
    print(f"  estimated_wall_time: {wall_time:.3f}s")
    print(f"  QPS: {qps:.2f}")
    print(f"  aggregate output tokens/s: {aggregate_output_tokens_per_second:.2f}")


def main():
    workload = build_workload()
    concurrency_levels = [1, 4, 8]

    for concurrency in concurrency_levels:
        print(f"\nConcurrency = {concurrency}")
        results = [simulate_request_metrics(req, concurrency) for req in workload]
        print_result_table(results)
        print_aggregate_metrics(results, concurrency)

    print("\nMeaning:")
    print("TTFT grows with prompt length because prefill processes the prompt.")
    print("TPOT describes decode speed for each generated token.")
    print("total latency is roughly TTFT plus output_tokens * TPOT.")
    print("QPS and aggregate tokens/s are workload-level metrics, not single-request metrics.")
    print("This is a toy benchmark; real numbers must come from an actual model server later.")


if __name__ == "__main__":
    main()
