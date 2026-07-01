from dataclasses import dataclass


@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    start_time: float
    first_token_time: float
    end_time: float


def compute_metrics(trace):
    ttft = trace.first_token_time - trace.start_time
    total_latency = trace.end_time - trace.start_time
    decode_time = trace.end_time - trace.first_token_time
    tpot = decode_time / trace.output_tokens if trace.output_tokens > 0 else 0.0
    output_tokens_per_second = trace.output_tokens / decode_time if decode_time > 0 else 0.0
    return {
        "request_id": trace.request_id,
        "prompt_tokens": trace.prompt_tokens,
        "output_tokens": trace.output_tokens,
        "ttft": ttft,
        "decode_time": decode_time,
        "tpot": tpot,
        "output_tokens_per_second": output_tokens_per_second,
        "total_latency": total_latency,
    }


def print_metrics(metrics):
    print(metrics["request_id"])
    print("  prompt_tokens:", metrics["prompt_tokens"])
    print("  output_tokens:", metrics["output_tokens"])
    print("  TTFT:", f"{metrics['ttft']:.3f}s")
    print("  decode_time:", f"{metrics['decode_time']:.3f}s")
    print("  TPOT:", f"{metrics['tpot']:.3f}s/token")
    print("  output tokens/s:", f"{metrics['output_tokens_per_second']:.2f}")
    print("  total_latency:", f"{metrics['total_latency']:.3f}s")


def main():
    traces = [
        RequestTrace(
            request_id="short_prompt",
            prompt_tokens=64,
            output_tokens=20,
            start_time=0.0,
            first_token_time=0.20,
            end_time=1.20,
        ),
        RequestTrace(
            request_id="long_prompt",
            prompt_tokens=4096,
            output_tokens=20,
            start_time=0.0,
            first_token_time=1.80,
            end_time=2.80,
        ),
        RequestTrace(
            request_id="long_output",
            prompt_tokens=64,
            output_tokens=100,
            start_time=0.0,
            first_token_time=0.20,
            end_time=5.20,
        ),
    ]

    print("Per-request metrics:\n")
    all_metrics = []
    for trace in traces:
        metrics = compute_metrics(trace)
        all_metrics.append(metrics)
        print_metrics(metrics)
        print()

    total_requests = len(traces)
    wall_time = max(trace.end_time for trace in traces) - min(trace.start_time for trace in traces)
    qps = total_requests / wall_time
    total_output_tokens = sum(trace.output_tokens for trace in traces)
    aggregate_output_tokens_per_second = total_output_tokens / wall_time

    print("Aggregate throughput:")
    print("  total_requests:", total_requests)
    print("  wall_time:", f"{wall_time:.3f}s")
    print("  QPS:", f"{qps:.2f}")
    print("  aggregate output tokens/s:", f"{aggregate_output_tokens_per_second:.2f}")

    print("\nMeaning:")
    print("TTFT mostly reflects prefill cost and is sensitive to prompt length.")
    print("TPOT reflects decode speed per output token.")
    print("tokens/s measures token throughput.")
    print("QPS measures request throughput.")


if __name__ == "__main__":
    main()
