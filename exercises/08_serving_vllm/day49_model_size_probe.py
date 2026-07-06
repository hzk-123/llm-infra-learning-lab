from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day49 compare vLLM model size serving probe.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--run-label", default="", help="Example: 05b or 7b.")
    parser.add_argument(
        "--prompt",
        default="请用三句话解释 KV cache 在大模型推理中的作用，并说明它为什么会占用显存。",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--server-log", default="", help="Optional vLLM startup log path.")
    parser.add_argument(
        "--out-csv",
        default="results/08_serving_vllm/day49_model_size_probe.csv",
        help="Append one result row to this CSV. Use empty string to disable.",
    )
    parser.add_argument("--no-usage", action="store_true", help="Do not request streaming usage.")
    return parser.parse_args()


def request_json(url: str, timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body), time.perf_counter() - start


def query_models(base_url: str, timeout: float) -> dict[str, Any]:
    payload, latency = request_json(f"{base_url.rstrip('/')}/v1/models", timeout)
    models = payload.get("data") or []
    return {
        "latency": latency,
        "models": models,
        "model_ids": [item.get("id", "-") for item in models],
    }


def query_nvidia_smi() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    rows: list[dict[str, str]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": parts[2],
                "memory_used_mib": parts[3],
            }
        )
    return rows


def last_regex_value(text: str, pattern: str) -> str:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return "-"
    value = matches[-1]
    if isinstance(value, tuple):
        value = " ".join(item for item in value if item)
    return str(value).strip()


def parse_server_log(log_path: str) -> dict[str, str]:
    if not log_path:
        return {}
    path = Path(log_path)
    if not path.exists():
        return {"server_log": str(path), "server_log_found": "False"}

    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "server_log": str(path),
        "server_log_found": "True",
        "MODEL_ID": last_regex_value(text, r"MODEL_ID=(.+)"),
        "SERVED_MODEL_NAME": last_regex_value(text, r"SERVED_MODEL_NAME=(.+)"),
        "MAX_MODEL_LEN": last_regex_value(text, r"MAX_MODEL_LEN=(.+)"),
        "GPU_MEMORY_UTILIZATION": last_regex_value(text, r"GPU_MEMORY_UTILIZATION=(.+)"),
        "CUDA_VISIBLE_DEVICES": last_regex_value(text, r"CUDA_VISIBLE_DEVICES=(.+)"),
        "torch_compile_time_s": last_regex_value(text, r"torch\.compile took\s*([0-9.]+)\s*s"),
        "available_kv_cache_memory": last_regex_value(
            text, r"Available KV cache memory:\s*([0-9.]+\s+[A-Za-z]+)"
        ),
        "gpu_kv_cache_size_tokens": last_regex_value(
            text, r"GPU KV cache size:\s*([0-9,]+)\s+tokens"
        ),
        "engine_init_time_s": last_regex_value(
            text, r"init engine .* took\s*([0-9.]+)\s*s"
        ),
    }


def parse_sse_data_line(line: bytes) -> str | None:
    text = line.decode("utf-8").strip()
    if not text or text.startswith(":"):
        return None
    if not text.startswith("data:"):
        return None
    return text[len("data:") :].strip()


def stream_chat_completion(args: argparse.Namespace) -> dict[str, Any]:
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
        raise RuntimeError("No assistant content chunk was received.")

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
        "response_id": response_id,
        "finish_reason": finish_reason,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "total_latency_s": total_latency,
        "decode_after_first_s": decode_after_first,
        "output_tokens_per_second": output_tokens_per_second,
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "-",
        "completion_tokens": completion_tokens if completion_tokens is not None else "-",
        "total_tokens": total_tokens if total_tokens is not None else "-",
        "raw_event_count": raw_event_count,
        "content_event_count": content_event_count,
        "assistant_content": "".join(chunks),
    }


def append_csv(path_text: str, row: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_label = args.run_label or args.model

    print("Config:")
    print_table(
        ["name", "value"],
        [
            ["run_label", run_label],
            ["base_url", args.base_url.rstrip("/")],
            ["model", args.model],
            ["max_tokens", args.max_tokens],
            ["temperature", args.temperature],
            ["timeout", f"{args.timeout:.1f}s"],
            ["server_log", args.server_log or "-"],
            ["out_csv", args.out_csv or "-"],
        ],
    )

    try:
        models_info = query_models(args.base_url, args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("\nStatus: failed")
        print("Could not reach /v1/models.")
        print("Reason:", exc)
        raise SystemExit(1)

    server_model = next(
        (item for item in models_info["models"] if item.get("id") == args.model),
        {},
    )

    print("\nModels endpoint:")
    print_table(
        ["metric", "value"],
        [
            ["latency", f"{models_info['latency']:.3f}s"],
            ["model_ids", models_info["model_ids"]],
            ["requested_model_found", bool(server_model)],
            ["server_root", server_model.get("root", "-")],
            ["server_max_model_len", server_model.get("max_model_len", "-")],
        ],
    )

    gpu_rows = query_nvidia_smi()
    print("\nnvidia-smi snapshot:")
    if gpu_rows:
        print_table(
            ["index", "name", "memory.used", "memory.total"],
            [
                [
                    row["index"],
                    row["name"],
                    f"{row['memory_used_mib']} MiB",
                    f"{row['memory_total_mib']} MiB",
                ]
                for row in gpu_rows
            ],
        )
    else:
        print("nvidia-smi is not available.")

    log_info = parse_server_log(args.server_log)
    if log_info:
        print("\nParsed vLLM startup log:")
        print_table(["field", "value"], [[key, value] for key, value in log_info.items()])

    try:
        metrics = stream_chat_completion(args)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        print("\nStatus: failed")
        print("Could not finish streaming request.")
        print("Reason:", exc)
        raise SystemExit(1)

    print("\nStreaming probe metrics:")
    print_table(
        ["metric", "value"],
        [
            ["status", "ok"],
            ["response_id", metrics["response_id"]],
            ["finish_reason", metrics["finish_reason"]],
            ["TTFT", f"{metrics['ttft_s']:.4f}s"],
            ["TPOT", f"{metrics['tpot_s']:.4f}s/token"],
            ["total_latency", f"{metrics['total_latency_s']:.4f}s"],
            ["decode_after_first", f"{metrics['decode_after_first_s']:.4f}s"],
            ["output tokens/s", f"{metrics['output_tokens_per_second']:.2f}"],
            ["prompt_tokens", metrics["prompt_tokens"]],
            ["completion_tokens", metrics["completion_tokens"]],
            ["total_tokens", metrics["total_tokens"]],
            ["raw_stream_events", metrics["raw_event_count"]],
            ["content_chunks", metrics["content_event_count"]],
        ],
    )

    gpu0 = gpu_rows[0] if gpu_rows else {}
    csv_row: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_label": run_label,
        "model": args.model,
        "server_root": server_model.get("root", "-"),
        "server_max_model_len": server_model.get("max_model_len", "-"),
        "gpu0_memory_used_mib": gpu0.get("memory_used_mib", "-"),
        "gpu0_memory_total_mib": gpu0.get("memory_total_mib", "-"),
        "available_kv_cache_memory": log_info.get("available_kv_cache_memory", "-"),
        "gpu_kv_cache_size_tokens": log_info.get("gpu_kv_cache_size_tokens", "-"),
        "engine_init_time_s": log_info.get("engine_init_time_s", "-"),
        "torch_compile_time_s": log_info.get("torch_compile_time_s", "-"),
        "ttft_s": f"{metrics['ttft_s']:.6f}",
        "tpot_s": f"{metrics['tpot_s']:.6f}",
        "total_latency_s": f"{metrics['total_latency_s']:.6f}",
        "output_tokens_per_second": f"{metrics['output_tokens_per_second']:.4f}",
        "prompt_tokens": metrics["prompt_tokens"],
        "completion_tokens": metrics["completion_tokens"],
        "finish_reason": metrics["finish_reason"],
    }
    append_csv(args.out_csv, csv_row)

    if args.out_csv:
        print("\nCSV:")
        print(f"Appended one row to {args.out_csv}")

    print("\nPrompt:")
    print(args.prompt)

    print("\nAssistant content:")
    print(metrics["assistant_content"])

    print("\nMeaning:")
    print("This probe records one model-serving snapshot for model-size comparison.")
    print("nvidia-smi memory includes weights, KV cache pool, CUDA graphs, and runtime buffers.")
    print("A 7B model uses much more memory for weights, so available KV cache capacity should shrink.")
    print("Use the CSV after running both 0.5B and 7B to compare startup memory and request metrics.")


if __name__ == "__main__":
    main()
