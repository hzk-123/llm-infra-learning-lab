from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day45 vLLM streaming TTFT / TPOT client.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument(
        "--prompt",
        default="在 LLM 推理服务性能指标中，请用两句话分别解释 TTFT 和 TPOT，不要展开其他概念。",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-usage", action="store_true", help="Do not request streaming usage.")
    return parser.parse_args()


def parse_sse_data_line(line: bytes) -> str | None:
    text = line.decode("utf-8").strip()
    if not text or text.startswith(":"):
        return None
    if not text.startswith("data:"):
        return None
    return text[len("data:") :].strip()


def stream_chat_completion(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if not args.no_usage:
        payload["stream_options"] = {"include_usage": True}

    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    request_start = time.perf_counter()
    first_content_time: float | None = None
    request_end: float | None = None
    chunks: list[str] = []
    raw_event_count = 0
    content_event_count = 0
    finish_reason = "-"
    usage: dict[str, Any] = {}
    response_id = "-"

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            data_text = parse_sse_data_line(raw_line)
            if data_text is None:
                continue
            if data_text == "[DONE]":
                request_end = time.perf_counter()
                break

            raw_event_count += 1
            event_time = time.perf_counter()
            event = json.loads(data_text)
            response_id = event.get("id", response_id)

            if event.get("usage") is not None:
                usage = event["usage"]

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if first_content_time is None:
                    first_content_time = event_time
                content_event_count += 1
                chunks.append(content)

    if request_end is None:
        request_end = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError("No assistant content chunk was received from the streaming response.")

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    fallback_output_units = content_event_count
    output_units = completion_tokens if isinstance(completion_tokens, int) else fallback_output_units

    ttft = first_content_time - request_start
    total_latency = request_end - request_start
    decode_after_first = max(request_end - first_content_time, 0.0)
    tpot_denominator = max(output_units - 1, 1)
    tpot = decode_after_first / tpot_denominator
    output_tokens_per_second = output_units / total_latency if total_latency > 0 else 0.0

    return {
        "response_id": response_id,
        "finish_reason": finish_reason,
        "ttft": ttft,
        "tpot": tpot,
        "total_latency": total_latency,
        "decode_after_first": decode_after_first,
        "output_tokens_per_second": output_tokens_per_second,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "-",
        "completion_tokens": completion_tokens if completion_tokens is not None else "-",
        "total_tokens": total_tokens if total_tokens is not None else "-",
        "raw_event_count": raw_event_count,
        "content_event_count": content_event_count,
        "assistant_content": "".join(chunks),
        "used_completion_tokens_for_tpot": isinstance(completion_tokens, int),
    }


def main() -> None:
    args = parse_args()

    print("Config:")
    print_table(
        ["name", "value"],
        [
            ["base_url", args.base_url.rstrip("/")],
            ["model", args.model],
            ["max_tokens", args.max_tokens],
            ["temperature", args.temperature],
            ["timeout", f"{args.timeout:.1f}s"],
            ["request_stream_usage", not args.no_usage],
        ],
    )

    try:
        metrics = stream_chat_completion(args)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        print("\nStatus: failed")
        print("Could not finish streaming chat completion.")
        print("Reason:", exc)
        print("\nChecklist:")
        print("1. Is vLLM server running?")
        print("2. Is /v1/chat/completions reachable?")
        print("3. Is the served model name correct?")
        print("4. If stream_options causes problems, retry with --no-usage.")
        raise SystemExit(1)

    print("\nStreaming metrics:")
    rows = [
        ["status", "ok"],
        ["response_id", metrics["response_id"]],
        ["finish_reason", metrics["finish_reason"]],
        ["TTFT", f"{metrics['ttft']:.3f}s"],
        ["TPOT", f"{metrics['tpot']:.4f}s/token"],
        ["total_latency", f"{metrics['total_latency']:.3f}s"],
        ["decode_after_first", f"{metrics['decode_after_first']:.3f}s"],
        ["output tokens/s", f"{metrics['output_tokens_per_second']:.2f}"],
        ["prompt_tokens", metrics["prompt_tokens"]],
        ["completion_tokens", metrics["completion_tokens"]],
        ["total_tokens", metrics["total_tokens"]],
        ["raw_stream_events", metrics["raw_event_count"]],
        ["content_chunks", metrics["content_event_count"]],
        ["TPOT uses completion_tokens", metrics["used_completion_tokens_for_tpot"]],
    ]
    print_table(["metric", "value"], rows)

    print("\nPrompt:")
    print(args.prompt)

    print("\nAssistant content:")
    print(metrics["assistant_content"])

    print("\nMeaning:")
    print("TTFT measures time from request start to the first assistant content chunk.")
    print("TPOT estimates average time per output token after the first content token.")
    print("This is a single streaming smoke test, not a stable benchmark yet.")
    print("Run warmup and repeated requests before drawing performance conclusions.")


if __name__ == "__main__":
    main()
