import math
from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    physical_blocks: list[int] = field(default_factory=list)

    @property
    def total_reserved_tokens(self):
        return self.prompt_tokens + self.max_new_tokens

    @property
    def is_finished(self):
        return self.generated_tokens >= self.max_new_tokens

    def progress(self):
        return f"{self.request_id}:{self.generated_tokens}/{self.max_new_tokens}"


class KVBlockAllocator:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))

    def blocks_needed(self, token_count):
        return math.ceil(token_count / self.block_size)

    def can_allocate(self, token_count):
        return self.blocks_needed(token_count) <= len(self.free_blocks)

    def allocate(self, token_count):
        needed = self.blocks_needed(token_count)
        if needed > len(self.free_blocks):
            return None
        blocks = self.free_blocks[:needed]
        self.free_blocks = self.free_blocks[needed:]
        return blocks

    def free(self, blocks):
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()


class ToyServingScheduler:
    def __init__(self, requests, allocator, max_running):
        self.waiting = list(requests)
        self.running = []
        self.finished = []
        self.allocator = allocator
        self.max_running = max_running
        self.step_id = 0

    def admit_requests(self):
        events = []

        while self.waiting and len(self.running) < self.max_running:
            request = self.waiting[0]
            token_capacity = request.total_reserved_tokens

            if not self.allocator.can_allocate(token_capacity):
                events.append(
                    f"cannot admit {request.request_id}: need "
                    f"{self.allocator.blocks_needed(token_capacity)} blocks, "
                    f"free {len(self.allocator.free_blocks)}"
                )
                break

            blocks = self.allocator.allocate(token_capacity)
            request.physical_blocks = blocks
            self.running.append(request)
            self.waiting.pop(0)

            events.append(
                f"admit {request.request_id} blocks={blocks} "
                f"reserved_tokens={token_capacity}"
            )

        return events

    def decode_one_round(self):
        events = []

        for request in list(self.running):
            request.generated_tokens += 1
            events.append(f"decode {request.request_id} -> {request.generated_tokens}")

            if request.is_finished:
                self.running.remove(request)
                self.finished.append(request)
                self.allocator.free(request.physical_blocks)
                events.append(
                    f"finish {request.request_id}, free {request.physical_blocks}"
                )

        return events

    def snapshot(self, events):
        waiting_ids = [request.request_id for request in self.waiting]
        running_progress = [request.progress() for request in self.running]
        finished_ids = [request.request_id for request in self.finished]

        return {
            "step": self.step_id,
            "waiting": waiting_ids,
            "running": running_progress,
            "finished": finished_ids,
            "free_blocks": list(self.allocator.free_blocks),
            "events": events,
        }

    def run(self):
        snapshots = []

        initial_events = self.admit_requests()
        snapshots.append(self.snapshot(initial_events))

        while self.waiting or self.running:
            self.step_id += 1
            events = []
            events.extend(self.decode_one_round())
            events.extend(self.admit_requests())
            snapshots.append(self.snapshot(events))

        return snapshots


def print_requests(requests, allocator):
    print("Requests:")
    print(
        "| request_id | prompt_tokens | max_new_tokens | reserved_tokens | blocks_needed |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for request in requests:
        reserved = request.total_reserved_tokens
        print(
            f"| {request.request_id} | {request.prompt_tokens} | "
            f"{request.max_new_tokens} | {reserved} | "
            f"{allocator.blocks_needed(reserved)} |"
        )


def print_snapshots(snapshots):
    print("\nServing loop:")
    print("| step | waiting | running | finished | free_blocks | events |")
    print("| ---: | --- | --- | --- | --- | --- |")
    for item in snapshots:
        print(
            f"| {item['step']} | {item['waiting']} | {item['running']} | "
            f"{item['finished']} | {item['free_blocks']} | "
            f"{'; '.join(item['events'])} |"
        )


def main():
    total_blocks = 8
    block_size = 4
    max_running = 2

    requests = [
        Request("req_a", prompt_tokens=5, max_new_tokens=3),
        Request("req_b", prompt_tokens=8, max_new_tokens=4),
        Request("req_c", prompt_tokens=4, max_new_tokens=2),
        Request("req_d", prompt_tokens=7, max_new_tokens=3),
    ]

    allocator = KVBlockAllocator(
        total_blocks=total_blocks,
        block_size=block_size,
    )
    scheduler = ToyServingScheduler(
        requests=requests,
        allocator=allocator,
        max_running=max_running,
    )

    print("Config:")
    print("total_blocks:", total_blocks)
    print("block_size:", block_size)
    print("max_running:", max_running)
    print("total_token_capacity:", total_blocks * block_size)

    print()
    print_requests(requests, allocator)

    snapshots = scheduler.run()
    print_snapshots(snapshots)

    print("\nFinal allocations:")
    print("| request_id | physical_blocks | generated_tokens | status |")
    print("| --- | --- | ---: | --- |")
    for request in scheduler.finished:
        print(
            f"| {request.request_id} | {request.physical_blocks} | "
            f"{request.generated_tokens} | finished |"
        )

    print("\nMeaning:")
    print("The scheduler admits waiting requests only when running slots and KV blocks are available.")
    print("Each running request decodes one token per scheduler step.")
    print("When a request finishes, its KV blocks are returned to the allocator.")
    print("Freed blocks allow later waiting requests to enter running.")
    print("This is the serving-loop intuition behind continuous batching.")


if __name__ == "__main__":
    main()
