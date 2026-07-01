import time
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
    request_start: float
    prefill_done: float
    first_token_time: float
    request_end: float


def build_workload():
    return [
        BenchmarkRequest(
            request_id="short_prompt_001",
            workload_type="short_prompt",
            prompt_tokens=32,
            output_tokens=8,
        ),
        BenchmarkRequest(
            request_id="long_prompt_001",
            workload_type="long_prompt",
            prompt_tokens=512,
            output_tokens=8,
        ),
        BenchmarkRequest(
            request_id="long_output_001",
            workload_type="long_output",
            prompt_tokens=32,
            output_tokens=32,
        ),
    ]


def simulate_streaming_request(req):
    prefill_base_seconds = 0.020
    prefill_seconds_per_token = 0.00020
    decode_seconds_per_token = 0.015

    request_start = time.perf_counter()

    prefill_time = prefill_base_seconds + req.prompt_tokens * prefill_seconds_per_token
    time.sleep(prefill_time)
    prefill_done = time.perf_counter()

    first_token_time = None
    for token_index in range(req.output_tokens):
        time.sleep(decode_seconds_per_token)
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
        request_start=request_start,
        prefill_done=prefill_done,
        first_token_time=first_token_time,
        request_end=request_end,
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


def print_timing_events(result):
    base = result.request_start

    print(f"\nTiming events for {result.request_id}:")
    print(f"  request_start:    {result.request_start - base:.3f}s")
    print(f"  prefill_done:     {result.prefill_done - base:.3f}s")
    print(f"  first_token_time: {result.first_token_time - base:.3f}s")
    print(f"  request_end:      {result.request_end - base:.3f}s")


def main():
    workload = build_workload()

    benchmark_start = time.perf_counter()
    results = [simulate_streaming_request(req) for req in workload]
    benchmark_end = time.perf_counter()

    print_result_table(results)
    print_timing_events(results[1])

    total_requests = len(results)
    total_output_tokens = sum(result.output_tokens for result in results)
    wall_time = benchmark_end - benchmark_start

    print("\nAggregate metrics:")
    print(f"  total_requests: {total_requests}")
    print(f"  total_output_tokens: {total_output_tokens}")
    print(f"  measured_wall_time: {wall_time:.3f}s")
    print(f"  QPS: {total_requests / wall_time:.2f}")
    print(f"  aggregate output tokens/s: {total_output_tokens / wall_time:.2f}")

    print("\nMeaning:")
    print("request_start is when the client sends the request.")
    print("first_token_time is when the first generated token arrives.")
    print("TTFT = first_token_time - request_start.")
    print("TPOT is the average decode time per output token.")
    print("total_latency = request_end - request_start.")
    print("Real streaming benchmarks use the same timing idea, but with actual HTTP chunks.")


if __name__ == "__main__":
    main()
