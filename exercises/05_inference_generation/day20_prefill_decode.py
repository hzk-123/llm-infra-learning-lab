import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids):
        batch_size, current_length = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(current_length, device=input_ids.device)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def prefill(self, prompt_ids):
        context = prompt_ids[:, -self.block_size :]
        logits = self(context)
        next_logits = logits[:, -1, :]
        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        return context, logits, next_id

    @torch.no_grad()
    def decode_without_cache(self, ids):
        context = ids[:, -self.block_size :]
        logits = self(context)
        next_logits = logits[:, -1, :]
        next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        return context, logits, next_id


def main():
    torch.manual_seed(42)

    vocab_size = 10
    block_size = 8
    n_embd = 6
    model = ToyLM(vocab_size, block_size, n_embd)

    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    print("Prompt:")
    print("prompt_ids:", prompt_ids)
    print("prompt_ids.shape:", prompt_ids.shape)

    prefill_context, prefill_logits, first_next_id = model.prefill(prompt_ids)

    print("\nPrefill:")
    print("prefill_context:", prefill_context)
    print("prefill_context.shape:", prefill_context.shape)
    print("prefill_logits.shape:", prefill_logits.shape)
    print("first_next_id:", first_next_id)
    print("first_next_id.shape:", first_next_id.shape)

    ids = torch.cat([prompt_ids, first_next_id], dim=1)

    print("\nAfter appending first generated token:")
    print("ids:", ids)
    print("ids.shape:", ids.shape)

    decode_context, decode_logits, second_next_id = model.decode_without_cache(ids)

    print("\nDecode without KV cache:")
    print("decode_context:", decode_context)
    print("decode_context.shape:", decode_context.shape)
    print("decode_logits.shape:", decode_logits.shape)
    print("second_next_id:", second_next_id)
    print("second_next_id.shape:", second_next_id.shape)

    ids = torch.cat([ids, second_next_id], dim=1)

    print("\nAfter appending second generated token:")
    print("ids:", ids)
    print("ids.shape:", ids.shape)

    print("\nMeaning:")
    print("Prefill processes the whole prompt.")
    print("Decode generates one new token at a time.")
    print("Without KV cache, decode still recomputes the recent full context.")
    print("With KV cache later, decode will reuse old K/V and process only the new token.")


if __name__ == "__main__":
    main()
