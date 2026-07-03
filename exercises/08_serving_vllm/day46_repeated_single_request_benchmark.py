from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day46 repeated single-request vLLM benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument(
        "--prompt",
        default="请用两句话解释 KV cache 在 LLM 推理中的作用，每句话不超过三十个字。",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between measured runs.")
    parser.add_argument("--no-usage", action="store_true", help="Do not request streaming usage.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def parse_sse_data_line(line: bytes) -> str | None:
    text = line.decode("utf-8").strip()
    if not text or text.startswith(":"):
        return None
    if not text.startswith("data:"):
        return None
    return text[len("data:") :].strip()


def request_json(method: str, url: str, timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method=method)
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body), time.perf_counter() - start


def stream_chat_completion(args: argparse.Namespace, phase: str, run_index: int) -> dict[str, Any]:
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
    output_units = completion_tokens if isinstance(completion_tokens, int) else content_event_count

    ttft = first_content_time - request_start
    total_latency = request_end - request_start
    decode_after_first = max(request_end - first_content_time, 0.0)
    tpot = decode_after_first / max(output_units - 1, 1)
    output_tokens_per_second = output_units / total_latency if total_latency > 0 else 0.0

    return {
        "phase": phase,
        "run_index": run_index,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "ttft": ttft,
        "tpot": tpot,
        "total_latency": total_latency,
        "decode_after_first": decode_after_first,
        "output_tokens_per_second": output_tokens_per_second,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "",
        "completion_tokens": completion_tokens if completion_tokens is not None else "",
        "total_tokens": total_tokens if total_tokens is not None else "",
        "raw_event_count": raw_event_count,
        "content_event_count": content_event_count,
        "assistant_content": "".join(chunks),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = math.ceil((percent / 100.0) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def summarize_metric(name: str, values: list[float], unit: str) -> list[object]:
    return [
        name,
        f"{min(values):.4f}{unit}",
        f"{statistics.mean(values):.4f}{unit}",
        f"{statistics.median(values):.4f}{unit}",
        f"{percentile(values, 95):.4f}{unit}",
        f"{max(values):.4f}{unit}",
    ]


def write_csv(path_text: str, rows: list[dict[str, Any]]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "run_index",
        "response_id",
        "finish_reason",
        "ttft",
        "tpot",
        "total_latency",
        "decode_after_first",
        "output_tokens_per_second",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "raw_event_count",
        "content_event_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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
            ["warmup_runs", args.warmup_runs],
            ["measured_runs", args.runs],
            ["sleep_between_runs", f"{args.sleep:.1f}s"],
            ["request_stream_usage", not args.no_usage],
            ["out_csv", args.out_csv or "-"],
        ],
    )

    try:
        models_json, models_latency = request_json(
            "GET",
            f"{args.base_url.rstrip('/')}/v1/models",
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("\nStatus: failed")
        print("Could not call /v1/models.")
        print("Reason:", exc)
        raise SystemExit(1)

    model_ids = [item.get("id", "") for item in models_json.get("data", [])]
    print("\nPreflight:")
    print_table(
        ["metric", "value"],
        [
            ["/v1/models latency", f"{models_latency:.3f}s"],
            ["requested_model_found", args.model in model_ids],
            ["model_ids", model_ids],
        ],
    )

    all_rows: list[dict[str, Any]] = []

    print("\nWarmup:")
    warmup_rows = []
    for i in range(1, args.warmup_runs + 1):
        row = stream_chat_completion(args, phase="warmup", run_index=i)
        all_rows.append(row)
        warmup_rows.append(
            [
                i,
                f"{row['ttft']:.3f}s",
                f"{row['tpot']:.4f}s/token",
                f"{row['total_latency']:.3f}s",
                row["completion_tokens"],
                row["finish_reason"],
            ]
        )
    print_table(["run", "TTFT", "TPOT", "total_latency", "completion_tokens", "finish_reason"], warmup_rows)

    print("\nMeasured runs:")
    measured_rows: list[dict[str, Any]] = []
    table_rows = []
    for i in range(1, args.runs + 1):
        row = stream_chat_completion(args, phase="measured", run_index=i)
        measured_rows.append(row)
        all_rows.append(row)
        table_rows.append(
            [
                i,
                f"{row['ttft']:.3f}s",
                f"{row['tpot']:.4f}s/token",
                f"{row['total_latency']:.3f}s",
                f"{row['output_tokens_per_second']:.2f}",
                row["completion_tokens"],
                row["finish_reason"],
            ]
        )
        if args.sleep > 0 and i != args.runs:
            time.sleep(args.sleep)

    print_table(
        ["run", "TTFT", "TPOT", "total_latency", "output tok/s", "completion_tokens", "finish_reason"],
        table_rows,
    )

    ttft_values = [float(row["ttft"]) for row in measured_rows]
    tpot_values = [float(row["tpot"]) for row in measured_rows]
    latency_values = [float(row["total_latency"]) for row in measured_rows]
    throughput_values = [float(row["output_tokens_per_second"]) for row in measured_rows]

    print("\nSummary for measured runs:")
    summary_rows = [
        summarize_metric("TTFT", ttft_values, "s"),
        summarize_metric("TPOT", tpot_values, "s/token"),
        summarize_metric("total_latency", latency_values, "s"),
        summarize_metric("output tokens/s", throughput_values, ""),
    ]
    print_table(["metric", "min", "avg", "p50", "p95", "max"], summary_rows)

    first_content = measured_rows[0]["assistant_content"] if measured_rows else ""
    print("\nFirst measured assistant content:")
    print(first_content)

    if args.out_csv:
        write_csv(args.out_csv, all_rows)
        print("\nCSV saved:")
        print(args.out_csv)

    print("\nMeaning:")
    print("Warmup runs are executed first and excluded from the summary.")
    print("Measured runs are repeated single-request streaming calls.")
    print("p50 is the typical latency; p95 highlights slower tail behavior.")
    print("This is still single-request benchmarking, not concurrency benchmarking.")


if __name__ == "__main__":
    main()

