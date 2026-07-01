from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass, field


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, value: str) -> None:
        if key in self.items:
            self.items.move_to_end(key)
        self.items[key] = value
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)


@dataclass(order=True)
class Request:
    priority: int
    request_id: str = field(compare=False)
    prompt_tokens: int = field(compare=False)
    max_new_tokens: int = field(compare=False)


class PriorityBatchScheduler:
    def __init__(self) -> None:
        self.heap: list[Request] = []

    def add(self, request: Request) -> None:
        heapq.heappush(self.heap, request)

    def next_batch(self, max_batch_size: int, max_prompt_tokens: int) -> list[Request]:
        batch: list[Request] = []
        used_prompt_tokens = 0
        skipped: list[Request] = []
        while self.heap and len(batch) < max_batch_size:
            req = heapq.heappop(self.heap)
            if used_prompt_tokens + req.prompt_tokens <= max_prompt_tokens:
                batch.append(req)
                used_prompt_tokens += req.prompt_tokens
            else:
                skipped.append(req)
        for req in skipped:
            heapq.heappush(self.heap, req)
        return batch


def demo_lru() -> None:
    cache = LRUCache(capacity=2)
    cache.put("prefix-a", "kv-block-a")
    cache.put("prefix-b", "kv-block-b")
    print("get prefix-a:", cache.get("prefix-a"))
    cache.put("prefix-c", "kv-block-c")
    print("get prefix-b after eviction:", cache.get("prefix-b"))


def demo_scheduler() -> None:
    scheduler = PriorityBatchScheduler()
    scheduler.add(Request(priority=2, request_id="req-slow", prompt_tokens=900, max_new_tokens=128))
    scheduler.add(Request(priority=1, request_id="req-fast", prompt_tokens=100, max_new_tokens=64))
    scheduler.add(Request(priority=3, request_id="req-medium", prompt_tokens=400, max_new_tokens=128))
    batch = scheduler.next_batch(max_batch_size=2, max_prompt_tokens=1000)
    print("batch:", [(req.request_id, req.prompt_tokens) for req in batch])


if __name__ == "__main__":
    demo_lru()
    demo_scheduler()
