from __future__ import annotations

import argparse
import concurrent.futures
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


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day48 concurrent vLLM workload benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency-levels", default="1,2,4,8")
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--warmup-runs-per-workload", type=int, default=1)
    parser.add_argument("--workloads", default="short_prompt,long_prompt,long_output")
    parser.add_argument("--no-usage", action="store_true", help="Do not request streaming usage.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    return parser.parse_args()


def build_workloads() -> dict[str, Workload]:
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

    items = [
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
    return {item.name: item for item in items}


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
    *,
    base_url: str,
    model: str,
    temperature: float,
    timeout: float,
    include_usage: bool,
    workload: Workload,
    phase: str,
    concurrency: int,
    run_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": workload.max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
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
    raw_event_count = 0
    content_event_count = 0
    finish_reason = "-"
    usage: dict[str, Any] = {}
    response_id = "-"
    error = ""

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
    except Exception as exc:
        request_end = time.perf_counter()
        error = repr(exc)

    if request_end is None:
        request_end = time.perf_counter()

    success = not error and first_content_time is not None
    prompt_tokens = usage.get("prompt_tokens", "")
    completion_tokens = usage.get("completion_tokens", "")
    total_tokens = usage.get("total_tokens", "")
    output_units = completion_tokens if isinstance(completion_tokens, int) else content_event_count

    ttft = (first_content_time - request_start) if first_content_time is not None else float("nan")
    total_latency = request_end - request_start
    decode_after_first = max(request_end - first_content_time, 0.0) if first_content_time is not None else float("nan")
    tpot = decode_after_first / max(output_units - 1, 1) if success else float("nan")
    output_tokens_per_second = output_units / total_latency if success and total_latency > 0 else 0.0

    return {
        "workload": workload.name,
        "description": workload.description,
        "phase": phase,
        "concurrency": concurrency,
        "run_index": run_index,
        "success": success,
        "error": error,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "ttft": ttft,
        "tpot": tpot,
        "total_latency": total_latency,
        "decode_after_first": decode_after_first,
        "output_tokens_per_second": output_tokens_per_second,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_event_count": raw_event_count,
        "content_event_count": content_event_count,
        "max_tokens": workload.max_tokens,
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = math.ceil((percent / 100.0) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def f4(value: float, suffix: str = "") -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}{suffix}"


def write_csv(path_text: str, rows: list[dict[str, Any]]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "description",
        "phase",
        "concurrency",
        "run_index",
        "success",
        "error",
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


def summarize_group(
    workload: Workload,
    concurrency: int,
    rows: list[dict[str, Any]],
    wall_time: float,
) -> list[object]:
    success_rows = [row for row in rows if row["success"]]
    success_count = len(success_rows)
    total_count = len(rows)
    success_rate = success_count / total_count if total_count else 0.0

    ttft = [float(row["ttft"]) for row in success_rows]
    tpot = [float(row["tpot"]) for row in success_rows]
    latency = [float(row["total_latency"]) for row in success_rows]
    output_tokens = [
        int(row["completion_tokens"])
        for row in success_rows
        if isinstance(row["completion_tokens"], int)
    ]
    prompt_tokens = [
        int(row["prompt_tokens"])
        for row in success_rows
        if isinstance(row["prompt_tokens"], int)
    ]

    qps = success_count / wall_time if wall_time > 0 else 0.0
    aggregate_output_tps = sum(output_tokens) / wall_time if wall_time > 0 else 0.0

    return [
        workload.name,
        concurrency,
        total_count,
        f"{success_rate:.0%}",
        f"{wall_time:.3f}s",
        f"{qps:.2f}",
        f"{aggregate_output_tps:.2f}",
        f"{statistics.mean(prompt_tokens):.1f}" if prompt_tokens else "-",
        f"{statistics.mean(output_tokens):.1f}" if output_tokens else "-",
        f4(statistics.median(ttft), "s") if ttft else "nan",
        f4(percentile(ttft, 95), "s") if ttft else "nan",
        f4(statistics.median(tpot), "s/token") if tpot else "nan",
        f4(percentile(tpot, 95), "s/token") if tpot else "nan",
        f4(statistics.median(latency), "s") if latency else "nan",
        f4(percentile(latency, 95), "s") if latency else "nan",
    ]


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    concurrency_levels = parse_int_list(args.concurrency_levels)
    workload_names = parse_str_list(args.workloads)
    all_workloads = build_workloads()
    workloads = [all_workloads[name] for name in workload_names]

    print("Config:")
    print_table(
        ["name", "value"],
        [
            ["base_url", base_url],
            ["model", args.model],
            ["temperature", args.temperature],
            ["timeout", f"{args.timeout:.1f}s"],
            ["concurrency_levels", concurrency_levels],
            ["requests_per_level", args.requests_per_level],
            ["warmup_runs_per_workload", args.warmup_runs_per_workload],
            ["workloads", workload_names],
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
        models_json, models_latency = request_json("GET", f"{base_url}/v1/models", timeout=args.timeout)
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

    include_usage = not args.no_usage
    all_rows: list[dict[str, Any]] = []

    print("\nWarmup:")
    warmup_rows = []
    for workload in workloads:
        for run_index in range(1, args.warmup_runs_per_workload + 1):
            row = stream_workload(
                base_url=base_url,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                include_usage=include_usage,
                workload=workload,
                phase="warmup",
                concurrency=1,
                run_index=run_index,
            )
            all_rows.append(row)
            warmup_rows.append(
                [
                    workload.name,
                    run_index,
                    row["success"],
                    f4(float(row["ttft"]), "s") if row["success"] else "nan",
                    f4(float(row["total_latency"]), "s") if row["success"] else "nan",
                    row["completion_tokens"],
                    row["finish_reason"],
                ]
            )
    print_table(["workload", "run", "success", "TTFT", "latency", "completion_tokens", "finish_reason"], warmup_rows)

    print("\nConcurrent benchmark summary:")
    summary_rows = []
    for workload in workloads:
        for concurrency in concurrency_levels:
            group_rows: list[dict[str, Any]] = []
            wall_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        stream_workload,
                        base_url=base_url,
                        model=args.model,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        include_usage=include_usage,
                        workload=workload,
                        phase="measured",
                        concurrency=concurrency,
                        run_index=run_index,
                    )
                    for run_index in range(1, args.requests_per_level + 1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    group_rows.append(row)
                    all_rows.append(row)
            wall_time = time.perf_counter() - wall_start
            summary_rows.append(summarize_group(workload, concurrency, group_rows, wall_time))

    print_table(
        [
            "workload",
            "conc",
            "reqs",
            "success",
            "wall",
            "QPS",
            "agg out tok/s",
            "avg prompt tok",
            "avg comp tok",
            "TTFT p50",
            "TTFT p95",
            "TPOT p50",
            "TPOT p95",
            "lat p50",
            "lat p95",
        ],
        summary_rows,
    )

    if args.out_csv:
        write_csv(args.out_csv, all_rows)
        print("\nCSV saved:")
        print(args.out_csv)

    print("\nMeaning:")
    print("QPS is successful requests divided by wall time.")
    print("aggregate output tokens/s is total completion tokens divided by wall time.")
    print("p95 latency shows slower tail behavior as concurrency rises.")
    print("Compare workloads separately; do not average short, long-prompt, and long-output requests together.")


if __name__ == "__main__":
    main()

