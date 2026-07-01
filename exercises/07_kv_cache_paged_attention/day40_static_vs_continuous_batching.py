from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    max_new_tokens: int
    generated_tokens: int = 0


def clone_requests(requests: list[Request]) -> list[Request]:
    return [Request(r.request_id, r.max_new_tokens) for r in requests]


def format_items(items: list[Request] | list[str]) -> str:
    if not items:
        return "[]"
    names = [item.request_id if isinstance(item, Request) else item for item in items]
    return "[" + ", ".join(names) + "]"


def format_running(requests: list[Request]) -> str:
    if not requests:
        return "[]"
    return "[" + ", ".join(f"{r.request_id}:{r.generated_tokens}/{r.max_new_tokens}" for r in requests) + "]"


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def run_static_batching(requests: list[Request], batch_size: int) -> tuple[list[list[object]], dict[str, float]]:
    requests = clone_requests(requests)
    rows: list[list[object]] = []
    total_steps = 0
    useful_decode_slots = 0
    idle_slots = 0

    batches = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

    for batch_id, batch in enumerate(batches, start=1):
        while any(r.generated_tokens < r.max_new_tokens for r in batch):
            total_steps += 1

            active_before = [r for r in batch if r.generated_tokens < r.max_new_tokens]
            idle_this_step = batch_size - len(active_before)
            decoded: list[str] = []
            finished_this_step: list[str] = []

            for request in active_before:
                request.generated_tokens += 1
                useful_decode_slots += 1
                decoded.append(request.request_id)
                if request.generated_tokens == request.max_new_tokens:
                    finished_this_step.append(request.request_id)

            idle_slots += idle_this_step

            rows.append(
                [
                    total_steps,
                    f"batch_{batch_id}",
                    format_running(batch),
                    format_items(decoded),
                    format_items(finished_this_step),
                    idle_this_step,
                ]
            )

    summary = {
        "total_steps": total_steps,
        "capacity_slots": total_steps * batch_size,
        "useful_decode_slots": useful_decode_slots,
        "idle_slots": idle_slots,
        "slot_utilization": useful_decode_slots / (total_steps * batch_size),
    }
    return rows, summary


def admit_requests(waiting: list[Request], running: list[Request], max_running: int) -> list[str]:
    admitted: list[str] = []
    while waiting and len(running) < max_running:
        request = waiting.pop(0)
        running.append(request)
        admitted.append(request.request_id)
    return admitted


def run_continuous_batching(requests: list[Request], max_running: int) -> tuple[list[list[object]], dict[str, float]]:
    waiting = clone_requests(requests)
    running: list[Request] = []
    finished: list[Request] = []
    rows: list[list[object]] = []
    total_steps = 0
    useful_decode_slots = 0
    idle_slots = 0

    admitted = admit_requests(waiting, running, max_running)
    rows.append(
        [
            0,
            format_items(waiting),
            format_running(running),
            format_items(admitted),
            "[]",
            0,
        ]
    )

    while running:
        total_steps += 1
        active_before = list(running)
        idle_this_step = max_running - len(active_before)
        decoded: list[str] = []
        finished_this_step: list[str] = []

        for request in active_before:
            request.generated_tokens += 1
            useful_decode_slots += 1
            decoded.append(request.request_id)
            if request.generated_tokens == request.max_new_tokens:
                running.remove(request)
                finished.append(request)
                finished_this_step.append(request.request_id)

        admitted = admit_requests(waiting, running, max_running)
        idle_slots += idle_this_step

        rows.append(
            [
                total_steps,
                format_items(waiting),
                format_running(running),
                format_items(admitted),
                format_items(finished_this_step),
                idle_this_step,
            ]
        )

    summary = {
        "total_steps": total_steps,
        "capacity_slots": total_steps * max_running,
        "useful_decode_slots": useful_decode_slots,
        "idle_slots": idle_slots,
        "slot_utilization": useful_decode_slots / (total_steps * max_running),
    }
    return rows, summary


def main() -> None:
    batch_size = 2
    requests = [
        Request("req_a", max_new_tokens=1),
        Request("req_b", max_new_tokens=6),
        Request("req_c", max_new_tokens=1),
        Request("req_d", max_new_tokens=6),
    ]

    print("Config:")
    print("batch_size / max_running:", batch_size)
    print("request_count:", len(requests))
    print("total requested output tokens:", sum(r.max_new_tokens for r in requests))

    print("\nRequests:")
    request_rows = [[r.request_id, r.max_new_tokens] for r in requests]
    print_table(["request_id", "max_new_tokens"], request_rows)

    static_rows, static_summary = run_static_batching(requests, batch_size)
    continuous_rows, continuous_summary = run_continuous_batching(requests, batch_size)

    print("\nStatic batching timeline:")
    print_table(
        ["step", "batch", "batch_state", "decoded", "finished", "idle_slots"],
        static_rows,
    )

    print("\nContinuous batching timeline:")
    print_table(
        ["step", "waiting", "running_after_step", "admitted_after_step", "finished", "idle_slots"],
        continuous_rows,
    )

    print("\nSummary:")
    summary_rows = [
        [
            "static batching",
            int(static_summary["total_steps"]),
            int(static_summary["capacity_slots"]),
            int(static_summary["useful_decode_slots"]),
            int(static_summary["idle_slots"]),
            f"{static_summary['slot_utilization']:.2%}",
        ],
        [
            "continuous batching",
            int(continuous_summary["total_steps"]),
            int(continuous_summary["capacity_slots"]),
            int(continuous_summary["useful_decode_slots"]),
            int(continuous_summary["idle_slots"]),
            f"{continuous_summary['slot_utilization']:.2%}",
        ],
    ]
    print_table(
        ["mode", "total_steps", "capacity_slots", "useful_decode_slots", "idle_slots", "slot_utilization"],
        summary_rows,
    )

    print("\nMeaning:")
    print("Static batching keeps the same batch until every request in that batch finishes.")
    print("If one request is short and another is long, the finished request leaves an idle slot.")
    print("Continuous batching lets new waiting requests enter after a running request finishes.")
    print("This improves slot utilization when requests have different output lengths.")
    print("vLLM uses this decode-step scheduling idea to keep the serving batch busy.")


if __name__ == "__main__":
    main()
