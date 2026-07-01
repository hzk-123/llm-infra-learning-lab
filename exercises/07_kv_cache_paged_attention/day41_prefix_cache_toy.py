from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRequest:
    request_id: str
    prompt_token_ids: list[int]


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def format_blocks(tokens: list[int], block_size: int) -> str:
    blocks: list[str] = []
    for start in range(0, len(tokens), block_size):
        block = tokens[start : start + block_size]
        if len(block) == block_size:
            blocks.append(str(block))
        else:
            blocks.append(f"{block} partial")
    return " ".join(blocks)


def full_prefix_keys(tokens: list[int], block_size: int) -> list[tuple[int, ...]]:
    full_block_count = len(tokens) // block_size
    keys: list[tuple[int, ...]] = []
    for block_index in range(1, full_block_count + 1):
        end = block_index * block_size
        keys.append(tuple(tokens[:end]))
    return keys


def find_reused_prefix_tokens(
    tokens: list[int],
    block_size: int,
    prefix_cache: set[tuple[int, ...]],
) -> int:
    reused_blocks = 0
    for key in full_prefix_keys(tokens, block_size):
        if key not in prefix_cache:
            break
        reused_blocks += 1
    return reused_blocks * block_size


def main() -> None:
    block_size = 2
    requests = [
        PromptRequest("req_a", [10, 11, 12, 13, 201, 202]),
        PromptRequest("req_b", [10, 11, 12, 13, 301, 302]),
        PromptRequest("req_c", [10, 11, 999, 998]),
        PromptRequest("req_d", [700, 701, 702, 703]),
    ]

    print("Config:")
    print("block_size:", block_size)
    print("request_count:", len(requests))

    print("\nRequests:")
    request_rows = [
        [
            request.request_id,
            request.prompt_token_ids,
            len(request.prompt_token_ids),
            format_blocks(request.prompt_token_ids, block_size),
        ]
        for request in requests
    ]
    print_table(["request_id", "prompt_token_ids", "prompt_tokens", "blocks"], request_rows)

    prefix_cache: set[tuple[int, ...]] = set()
    rows: list[list[object]] = []
    no_cache_prefill_tokens = 0
    with_cache_prefill_tokens = 0

    for request in requests:
        prompt_tokens = len(request.prompt_token_ids)
        no_cache_prefill_tokens += prompt_tokens

        reused_prefix_tokens = find_reused_prefix_tokens(
            request.prompt_token_ids,
            block_size,
            prefix_cache,
        )
        new_prefill_tokens = prompt_tokens - reused_prefix_tokens
        with_cache_prefill_tokens += new_prefill_tokens

        prefix_keys = full_prefix_keys(request.prompt_token_ids, block_size)
        for key in prefix_keys:
            prefix_cache.add(key)

        rows.append(
            [
                request.request_id,
                prompt_tokens,
                reused_prefix_tokens,
                new_prefill_tokens,
                len(prefix_keys),
                len(prefix_cache),
            ]
        )

    print("\nPrefix cache simulation:")
    print_table(
        [
            "request_id",
            "prompt_tokens",
            "reused_prefix_tokens",
            "new_prefill_tokens",
            "full_prefix_blocks",
            "cache_entries_after",
        ],
        rows,
    )

    saved_tokens = no_cache_prefill_tokens - with_cache_prefill_tokens
    saved_ratio = saved_tokens / no_cache_prefill_tokens

    print("\nSummary:")
    summary_rows = [
        ["without prefix cache", no_cache_prefill_tokens],
        ["with prefix cache", with_cache_prefill_tokens],
        ["saved prefill tokens", saved_tokens],
        ["saved ratio", f"{saved_ratio:.2%}"],
    ]
    print_table(["metric", "value"], summary_rows)

    print("\nCached prefix entries:")
    cache_rows = [[index, list(key)] for index, key in enumerate(sorted(prefix_cache), start=1)]
    print_table(["entry", "cached_token_prefix"], cache_rows)

    print("\nMeaning:")
    print("Prefix cache reuses KV cache only when token prefixes are exactly the same.")
    print("It reduces repeated prefill work for shared prompt prefixes.")
    print("It does not help if prompts are only semantically similar but token-different.")
    print("Real serving systems usually cache and reuse prefixes at KV block/page granularity.")
    print("The main user-visible effect is lower TTFT for requests with cached prefixes.")


if __name__ == "__main__":
    main()

