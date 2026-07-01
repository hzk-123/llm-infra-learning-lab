import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class BenchmarkRequest:
    request_id: str
    workload_type: str
    prompt_tokens: int
    output_tokens: int


@dataclass
class TimingResult:
    request_id: str
    workload_type: str
    prompt_tokens: int
    output_tokens: int
    ttft: float
    tpot: float
    total_latency: float
    output_tokens_per_second: float


@dataclass
class AggregateResult:
    concurrency: int
    total_requests: int
    total_output_tokens: int
    wall_time: float
    qps: float
    aggregate_output_tokens_per_second: float
    avg_ttft: float
    avg_latency: float
    p95_latency: float


def build_requests():
    request_templates = [
        ("short_prompt", 32, 8),
        ("long_prompt", 512, 8),
        ("long_output", 32, 32),
        ("mixed", 256, 16),
    ]

    requests = []
    for round_id in range(3):
        for workload_type, prompt_tokens, output_tokens in request_templates:
            requests.append(
                BenchmarkRequest(
                    request_id=f"{workload_type}_{round_id + 1:03d}",
                    workload_type=workload_type,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
            )

    return requests


def simulate_streaming_request(req, concurrency):
    prefill_base_seconds = 0.020
    prefill_seconds_per_token = 0.00020
    decode_seconds_per_token = 0.015

    contention_slowdown = 1.0 + 0.06 * (concurrency - 1)

    request_start = time.perf_counter()

    prefill_time = (
        prefill_base_seconds + req.prompt_tokens * prefill_seconds_per_token
    ) * contention_slowdown
    time.sleep(prefill_time)
    prefill_done = time.perf_counter()

    first_token_time = None
    per_token_time = decode_seconds_per_token * contention_slowdown
    for token_index in range(req.output_tokens):
        time.sleep(per_token_time)
        if token_index == 0:
            first_token_time = time.perf_counter()

    request_end = time.perf_counter()

    decode_time = request_end - prefill_done
    ttft = first_token_time - request_start
    tpot = decode_time / req.output_tokens
    total_latency = request_end - request_start
    output_tokens_per_second = req.output_tokens / decode_time

    return TimingResult(
        request_id=req.request_id,
        workload_type=req.workload_type,
        prompt_tokens=req.prompt_tokens,
        output_tokens=req.output_tokens,
        ttft=ttft,
        tpot=tpot,
        total_latency=total_latency,
        output_tokens_per_second=output_tokens_per_second,
    )


def percentile(values, percent):
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * percent / 100) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def run_benchmark(requests, concurrency):
    benchmark_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(simulate_streaming_request, req, concurrency)
            for req in requests
        ]
        for future in as_completed(futures):
            results.append(future.result())

    benchmark_end = time.perf_counter()
    wall_time = benchmark_end - benchmark_start

    total_requests = len(results)
    total_output_tokens = sum(result.output_tokens for result in results)
    latencies = [result.total_latency for result in results]
    ttfts = [result.ttft for result in results]

    aggregate = AggregateResult(
        concurrency=concurrency,
        total_requests=total_requests,
        total_output_tokens=total_output_tokens,
        wall_time=wall_time,
        qps=total_requests / wall_time,
        aggregate_output_tokens_per_second=total_output_tokens / wall_time,
        avg_ttft=sum(ttfts) / len(ttfts),
        avg_latency=sum(latencies) / len(latencies),
        p95_latency=percentile(latencies, 95),
    )

    return results, aggregate


def print_aggregate_table(aggregates):
    print(
        "| concurrency | requests | wall_time(s) | QPS | output tok/s | "
        "avg TTFT(s) | avg latency(s) | p95 latency(s) |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in aggregates:
        print(
            f"| {item.concurrency} | {item.total_requests} | "
            f"{item.wall_time:.3f} | {item.qps:.2f} | "
            f"{item.aggregate_output_tokens_per_second:.2f} | "
            f"{item.avg_ttft:.3f} | {item.avg_latency:.3f} | "
            f"{item.p95_latency:.3f} |"
        )


def print_sample_results(results, limit=6):
    ordered = sorted(results, key=lambda item: item.request_id)

    print("\nSample request metrics:")
    print(
        "| request_id | type | TTFT(s) | TPOT(s/token) | "
        "latency(s) | output tok/s |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for result in ordered[:limit]:
        print(
            f"| {result.request_id} | {result.workload_type} | "
            f"{result.ttft:.3f} | {result.tpot:.3f} | "
            f"{result.total_latency:.3f} | {result.output_tokens_per_second:.2f} |"
        )


def main():
    requests = build_requests()
    concurrency_levels = [1, 2, 4]
    aggregates = []
    sample_results = None

    print(f"Total benchmark requests: {len(requests)}")

    for concurrency in concurrency_levels:
        results, aggregate = run_benchmark(requests, concurrency)
        aggregates.append(aggregate)
        if concurrency == 4:
            sample_results = results

    print("\nAggregate benchmark results:")
    print_aggregate_table(aggregates)

    print_sample_results(sample_results)

    print("\nMeaning:")
    print("The same request set is tested with different concurrency levels.")
    print("Higher concurrency can reduce wall time and increase QPS.")
    print("Higher concurrency can also increase per-request latency because of contention.")
    print("p95 latency is a tail-latency metric: it focuses on slow requests.")
    print("Real serving benchmarks use the same structure with actual HTTP requests.")


if __name__ == "__main__":
    main()
