# vLLM Source Reading Notes

Do not start here. Fill this after you can run vLLM and benchmark it.

## Request Lifecycle

TODO:

1. API receives request.
2. Request enters scheduler.
3. Scheduler groups requests.
4. Worker executes model forward.
5. KV cache blocks are allocated/reused.
6. Tokens are returned to the client.

## Files / Modules To Read Later

- scheduler
- worker
- model runner
- KV cache manager
- OpenAI API server

## Questions

- Where does continuous batching happen?
- Where are KV cache blocks allocated?
- How does prefix cache match reusable prefixes?
- How are finished requests removed?
