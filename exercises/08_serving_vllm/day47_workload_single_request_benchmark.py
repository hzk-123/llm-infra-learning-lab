from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    prompt: str
    max_tokens: int
    description: str


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day47 single-request workload benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--no-usage", action="store_true", help="Do not request streaming usage.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def build_workloads() -> list[Workload]:
    long_context_facts = "\n".join(
        [
            f"- Fact {i:02d}: 在 LLM 推理服务中，请求 {i:02d} 的 prompt 会先进入 prefill 阶段，然后进入 decode 阶段。"
            for i in range(1, 41)
        ]
    )
    long_prompt = (
        "下面是一段用于压测长 prompt 的上下文，请先阅读，然后用一句话总结 KV cache 的作用。\n"
        f"{long_context_facts}\n"
        "问题：KV cache 在 decode 阶段主要解决什么问题？请只回答一句话。"
    )

    return [
        Workload(
            name="short_prompt",
            prompt="用一句话解释 vLLM 是什么。",
            max_tokens=48,
            description="short input, short output",
        ),
        Workload(
            name="long_prompt",
            prompt=long_prompt,
            max_tokens=48,
            description="long input, short output",
        ),
        Workload(
            name="long_output",
            prompt="请列出 8 条 LLM 推理服务性能优化建议，每条一句话。",
            max_tokens=192,
            description="short input, longer output",
        ),
    ]


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


def stream_workload(
    args: argparse.Namespace,
    workload: Workload,
    phase: str,
    run_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": workload.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if not args.no_usage:
        payload["stream_options"] = {"include_usage": True}

    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
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
        raise RuntimeError(f"No assistant content chunk received for workload {workload.name}.")

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
        "workload": workload.name,
        "description": workload.description,
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
        "max_tokens": workload.max_tokens,
        "assistant_content": "".join(chunks),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = math.ceil((percent / 100.0) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def write_csv(path_text: str, rows: list[dict[str, Any]]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "description",
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
        "max_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    workloads = build_workloads()

    print("Config:")
    print_table(
        ["name", "value"],
        [
            ["base_url", args.base_url.rstrip("/")],
            ["model", args.model],
            ["temperature", args.temperature],
            ["timeout", f"{args.timeout:.1f}s"],
            ["warmup_runs_per_workload", args.warmup_runs],
            ["measured_runs_per_workload", args.runs],
            ["sleep_between_runs", f"{args.sleep:.1f}s"],
            ["request_stream_usage", not args.no_usage],
            ["out_csv", args.out_csv or "-"],
        ],
    )

    print("\nWorkloads:")
    print_table(
        ["workload", "description", "max_tokens", "prompt_chars"],
        [[w.name, w.description, w.max_tokens, len(w.prompt)] for w in workloads],
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
    measured_by_workload: dict[str, list[dict[str, Any]]] = {}

    for workload in workloads:
        print(f"\nWarmup for {workload.name}:")
        warmup_table = []
        for i in range(1, args.warmup_runs + 1):
            row = stream_workload(args, workload, phase="warmup", run_index=i)
            all_rows.append(row)
            warmup_table.append(
                [
                    i,
                    f"{row['ttft']:.3f}s",
                    f"{row['tpot']:.4f}s/token",
                    f"{row['total_latency']:.3f}s",
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    row["finish_reason"],
                ]
            )
        print_table(
            ["run", "TTFT", "TPOT", "total_latency", "prompt_tokens", "completion_tokens", "finish_reason"],
            warmup_table,
        )

        print(f"\nMeasured runs for {workload.name}:")
        measured_rows = []
        measured_table = []
        for i in range(1, args.runs + 1):
            row = stream_workload(args, workload, phase="measured", run_index=i)
            all_rows.append(row)
            measured_rows.append(row)
            measured_table.append(
                [
                    i,
                    f"{row['ttft']:.3f}s",
                    f"{row['tpot']:.4f}s/token",
                    f"{row['total_latency']:.3f}s",
                    f"{row['output_tokens_per_second']:.2f}",
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    row["finish_reason"],
                ]
            )
            if args.sleep > 0 and i != args.runs:
                time.sleep(args.sleep)
        measured_by_workload[workload.name] = measured_rows
        print_table(
            [
                "run",
                "TTFT",
                "TPOT",
                "total_latency",
                "output tok/s",
                "prompt_tokens",
                "completion_tokens",
                "finish_reason",
            ],
            measured_table,
        )

    print("\nWorkload summary:")
    summary_rows = []
    for workload in workloads:
        rows = measured_by_workload[workload.name]
        ttft = metric_summary([float(row["ttft"]) for row in rows])
        tpot = metric_summary([float(row["tpot"]) for row in rows])
        latency = metric_summary([float(row["total_latency"]) for row in rows])
        output_tps = metric_summary([float(row["output_tokens_per_second"]) for row in rows])
        prompt_tokens = [row["prompt_tokens"] for row in rows]
        completion_tokens = [row["completion_tokens"] for row in rows]
        summary_rows.append(
            [
                workload.name,
                f"{statistics.mean([float(x) for x in prompt_tokens]):.1f}",
                f"{statistics.mean([float(x) for x in completion_tokens]):.1f}",
                f"{ttft['p50']:.4f}s",
                f"{ttft['p95']:.4f}s",
                f"{tpot['p50']:.4f}s/token",
                f"{tpot['p95']:.4f}s/token",
                f"{latency['p50']:.4f}s",
                f"{latency['p95']:.4f}s",
                f"{output_tps['avg']:.2f}",
            ]
        )
    print_table(
        [
            "workload",
            "avg prompt tok",
            "avg completion tok",
            "TTFT p50",
            "TTFT p95",
            "TPOT p50",
            "TPOT p95",
            "latency p50",
            "latency p95",
            "avg output tok/s",
        ],
        summary_rows,
    )

    print("\nSample outputs:")
    sample_rows = []
    for workload in workloads:
        first = measured_by_workload[workload.name][0]
        sample = str(first["assistant_content"]).replace("\n", " ")
        if len(sample) > 120:
            sample = sample[:117] + "..."
        sample_rows.append([workload.name, sample])
    print_table(["workload", "first measured assistant content"], sample_rows)

    if args.out_csv:
        write_csv(args.out_csv, all_rows)
        print("\nCSV saved:")
        print(args.out_csv)

    print("\nMeaning:")
    print("This benchmark compares workload types under single-request conditions.")
    print("Long prompt should mostly affect prefill and TTFT.")
    print("Long output should mostly affect total latency because more decode tokens are generated.")
    print("This is still not a concurrency benchmark.")


if __name__ == "__main__":
    main()

