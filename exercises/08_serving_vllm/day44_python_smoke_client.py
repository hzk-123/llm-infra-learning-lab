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


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], float]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    latency = time.perf_counter() - start

    return json.loads(body), latency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day44 vLLM Python smoke client.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--prompt", default="你好，用三句话解释 vLLM 是什么。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    print("Config:")
    config_rows = [
        ["base_url", base_url],
        ["model", args.model],
        ["max_tokens", args.max_tokens],
        ["temperature", args.temperature],
        ["timeout", f"{args.timeout:.1f}s"],
    ]
    print_table(["name", "value"], config_rows)

    try:
        models_json, models_latency = request_json(
            "GET",
            f"{base_url}/v1/models",
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("\nStatus: failed")
        print("Could not call /v1/models.")
        print("Reason:", exc)
        print("\nChecklist:")
        print("1. Is vLLM server running in tmux?")
        print("2. Is the port correct? Default is 8000.")
        print("3. Can curl http://127.0.0.1:8000/v1/models work on the server?")
        raise SystemExit(1)

    model_ids = [item.get("id", "") for item in models_json.get("data", [])]
    model_found = args.model in model_ids

    print("\nModels endpoint:")
    print_table(
        ["metric", "value"],
        [
            ["latency", f"{models_latency:.3f}s"],
            ["model_count", len(model_ids)],
            ["requested_model_found", model_found],
            ["model_ids", model_ids],
        ],
    )

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    try:
        completion_json, completion_latency = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout=args.timeout,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("\nStatus: failed")
        print("Could not call /v1/chat/completions.")
        print("Reason:", exc)
        print("\nChecklist:")
        print("1. Is the model name exactly the served model name?")
        print("2. Did /v1/models show the same model id?")
        print("3. Is max_tokens too large for the current server config?")
        raise SystemExit(1)

    choice = completion_json["choices"][0]
    message = choice.get("message", {})
    usage = completion_json.get("usage", {})
    content = message.get("content", "")

    print("\nChat completion:")
    completion_rows = [
        ["status", "ok"],
        ["request_id", completion_json.get("id", "-")],
        ["model", completion_json.get("model", "-")],
        ["finish_reason", choice.get("finish_reason", "-")],
        ["non_stream_latency", f"{completion_latency:.3f}s"],
        ["prompt_tokens", usage.get("prompt_tokens", "-")],
        ["completion_tokens", usage.get("completion_tokens", "-")],
        ["total_tokens", usage.get("total_tokens", "-")],
    ]
    print_table(["metric", "value"], completion_rows)

    print("\nPrompt:")
    print(args.prompt)

    print("\nAssistant content:")
    print(content)

    print("\nMeaning:")
    print("This is a non-streaming smoke test.")
    print("non_stream_latency is the full response latency, not TTFT.")
    print("A successful response proves the Python client can call the local vLLM server.")
    print("The usage fields are the basis for later benchmark scripts.")


if __name__ == "__main__":
    main()

