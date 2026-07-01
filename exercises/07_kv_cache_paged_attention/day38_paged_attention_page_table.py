from dataclasses import dataclass


@dataclass
class TokenAddress:
    token_index: int
    logical_block: int
    offset: int
    physical_block: int
    physical_slot: str
    value: str


class PagedKVRequest:
    def __init__(self, request_id, token_count, block_size, physical_blocks):
        self.request_id = request_id
        self.token_count = token_count
        self.block_size = block_size
        self.physical_blocks = physical_blocks
        self.page_table = {
            logical_block: physical_block
            for logical_block, physical_block in enumerate(physical_blocks)
        }

    def locate_token(self, token_index, physical_memory):
        if token_index < 0 or token_index >= self.token_count:
            raise IndexError(
                f"token_index {token_index} is outside request length {self.token_count}"
            )

        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        physical_block = self.page_table[logical_block]
        value = physical_memory.read(physical_block, offset)

        return TokenAddress(
            token_index=token_index,
            logical_block=logical_block,
            offset=offset,
            physical_block=physical_block,
            physical_slot=f"block {physical_block}, offset {offset}",
            value=value,
        )


class ToyPhysicalKVMemory:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.blocks = {
            block_id: [None for _ in range(block_size)]
            for block_id in range(total_blocks)
        }

    def write(self, physical_block, offset, value):
        self.blocks[physical_block][offset] = value

    def read(self, physical_block, offset):
        return self.blocks[physical_block][offset]

    def write_request_tokens(self, request):
        for token_index in range(request.token_count):
            logical_block = token_index // request.block_size
            offset = token_index % request.block_size
            physical_block = request.page_table[logical_block]
            value = f"{request.request_id}_tok_{token_index}"
            self.write(physical_block, offset, value)


def print_page_table(request):
    print("\nPage table:")
    print("| logical_block | physical_block | logical_token_range |")
    print("| ---: | ---: | --- |")
    for logical_block, physical_block in request.page_table.items():
        start = logical_block * request.block_size
        end = min(start + request.block_size, request.token_count)
        print(f"| {logical_block} | {physical_block} | [{start}, {end}) |")


def print_physical_memory(memory):
    print("\nPhysical KV memory:")
    print("| physical_block | slots |")
    print("| ---: | --- |")
    for block_id, slots in memory.blocks.items():
        print(f"| {block_id} | {slots} |")


def print_token_lookup_table(addresses):
    print("\nToken address lookup:")
    print(
        "| token_index | logical_block | offset | physical_block | "
        "physical_slot | value |"
    )
    print("| ---: | ---: | ---: | ---: | --- | --- |")
    for address in addresses:
        print(
            f"| {address.token_index} | {address.logical_block} | "
            f"{address.offset} | {address.physical_block} | "
            f"{address.physical_slot} | {address.value} |"
        )


def main():
    total_blocks = 8
    block_size = 4

    request = PagedKVRequest(
        request_id="req_e",
        token_count=14,
        block_size=block_size,
        physical_blocks=[0, 1, 6, 7],
    )
    memory = ToyPhysicalKVMemory(
        total_blocks=total_blocks,
        block_size=block_size,
    )
    memory.write_request_tokens(request)

    print("Config:")
    print("request_id:", request.request_id)
    print("token_count:", request.token_count)
    print("block_size:", block_size)
    print("total_blocks:", total_blocks)
    print("physical_blocks:", request.physical_blocks)

    print_page_table(request)
    print_physical_memory(memory)

    query_token_indices = [0, 3, 4, 7, 8, 12, 13]
    addresses = [
        request.locate_token(token_index, memory)
        for token_index in query_token_indices
    ]
    print_token_lookup_table(addresses)

    print("\nInvalid token lookup:")
    try:
        request.locate_token(14, memory)
    except IndexError as exc:
        print("token_index 14 failed:", exc)

    print("\nAddress translation example:")
    token_index = 8
    logical_block = token_index // block_size
    offset = token_index % block_size
    physical_block = request.page_table[logical_block]
    print("token_index:", token_index)
    print("logical_block = token_index // block_size:", logical_block)
    print("offset = token_index % block_size:", offset)
    print("physical_block = page_table[logical_block]:", physical_block)
    print("physical address:", f"block {physical_block}, offset {offset}")

    print("\nMeaning:")
    print("The request sees a continuous logical token sequence.")
    print("The physical KV blocks do not need to be continuous.")
    print("The page table maps logical blocks to physical blocks.")
    print("A token address is logical_block plus offset, then physical_block plus offset.")
    print("PagedAttention uses this kind of mapping to read K/V from block-based cache.")


if __name__ == "__main__":
    main()
