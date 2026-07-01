import math
from dataclasses import dataclass


@dataclass
class RequestAllocation:
    request_id: str
    token_count: int
    block_size: int
    physical_blocks: list[int]

    @property
    def required_blocks(self):
        return len(self.physical_blocks)

    @property
    def capacity_tokens(self):
        return self.required_blocks * self.block_size

    @property
    def wasted_slots(self):
        return self.capacity_tokens - self.token_count


class KVBlockAllocator:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.allocations = {}

    def allocate(self, request_id, token_count):
        if request_id in self.allocations:
            raise ValueError(f"request_id already exists: {request_id}")

        required_blocks = math.ceil(token_count / self.block_size)
        if required_blocks > len(self.free_blocks):
            return None

        physical_blocks = self.free_blocks[:required_blocks]
        self.free_blocks = self.free_blocks[required_blocks:]

        allocation = RequestAllocation(
            request_id=request_id,
            token_count=token_count,
            block_size=self.block_size,
            physical_blocks=physical_blocks,
        )
        self.allocations[request_id] = allocation
        return allocation

    def free(self, request_id):
        allocation = self.allocations.pop(request_id)
        self.free_blocks.extend(allocation.physical_blocks)
        self.free_blocks.sort()
        return allocation

    def used_blocks(self):
        return self.total_blocks - len(self.free_blocks)

    def requested_tokens(self):
        return sum(item.token_count for item in self.allocations.values())

    def capacity_tokens(self):
        return sum(item.capacity_tokens for item in self.allocations.values())

    def wasted_slots(self):
        return sum(item.wasted_slots for item in self.allocations.values())


def print_allocator_state(title, allocator):
    print(f"\n{title}")
    print("free_blocks:", allocator.free_blocks)
    print("used_blocks:", allocator.used_blocks())
    print("free_block_count:", len(allocator.free_blocks))
    print("requested_tokens:", allocator.requested_tokens())
    print("capacity_tokens:", allocator.capacity_tokens())
    print("wasted_slots:", allocator.wasted_slots())

    print("\nAllocations:")
    print(
        "| request_id | tokens | blocks_needed | physical_blocks | "
        "capacity | wasted_slots |"
    )
    print("| --- | ---: | ---: | --- | ---: | ---: |")
    for allocation in allocator.allocations.values():
        print(
            f"| {allocation.request_id} | {allocation.token_count} | "
            f"{allocation.required_blocks} | {allocation.physical_blocks} | "
            f"{allocation.capacity_tokens} | {allocation.wasted_slots} |"
        )


def allocate_and_print(allocator, request_id, token_count):
    allocation = allocator.allocate(request_id, token_count)
    if allocation is None:
        print(
            f"\nAllocate {request_id} failed: token_count={token_count}, "
            f"needed_blocks={math.ceil(token_count / allocator.block_size)}, "
            f"free_blocks={len(allocator.free_blocks)}"
        )
        return None

    print(
        f"\nAllocate {request_id}: token_count={token_count}, "
        f"physical_blocks={allocation.physical_blocks}, "
        f"capacity={allocation.capacity_tokens}, wasted={allocation.wasted_slots}"
    )
    return allocation


def free_and_print(allocator, request_id):
    allocation = allocator.free(request_id)
    print(
        f"\nFree {request_id}: released_blocks={allocation.physical_blocks}, "
        f"free_blocks={allocator.free_blocks}"
    )


def main():
    total_blocks = 8
    block_size = 4

    allocator = KVBlockAllocator(
        total_blocks=total_blocks,
        block_size=block_size,
    )

    print("Config:")
    print("total_blocks:", total_blocks)
    print("block_size:", block_size)
    print("total_token_capacity:", total_blocks * block_size)

    print_allocator_state("Initial state", allocator)

    allocate_and_print(allocator, "req_a", 6)
    allocate_and_print(allocator, "req_b", 10)
    allocate_and_print(allocator, "req_c", 4)
    allocate_and_print(allocator, "req_d", 8)
    print_allocator_state("After allocating req_a, req_b, req_c, req_d", allocator)

    free_and_print(allocator, "req_a")
    free_and_print(allocator, "req_d")
    print_allocator_state("After freeing req_a and req_d", allocator)

    allocate_and_print(allocator, "req_e", 14)
    print_allocator_state("After allocating req_e with non-contiguous blocks", allocator)

    allocate_and_print(allocator, "req_f", 1)
    print_allocator_state("Final state", allocator)

    print("\nRequest req_e block mapping:")
    req_e = allocator.allocations["req_e"]
    print("| logical_block | physical_block | token_range_inside_request |")
    print("| ---: | ---: | --- |")
    for logical_block, physical_block in enumerate(req_e.physical_blocks):
        start = logical_block * block_size
        end = min(start + block_size, req_e.token_count)
        print(f"| {logical_block} | {physical_block} | [{start}, {end}) |")

    print("\nMeaning:")
    print("A KV block allocator manages fixed-size physical blocks.")
    print("A request gets ceil(token_count / block_size) blocks.")
    print("Unused slots inside the last block are internal fragmentation.")
    print("When requests finish, their blocks return to the free list.")
    print("A later request can use non-contiguous physical blocks.")
    print("PagedAttention uses this block/page idea to manage KV cache more flexibly.")


if __name__ == "__main__":
    main()
