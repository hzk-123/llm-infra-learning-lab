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
    def generate_one_step(self, ids):
        context = ids[:, -self.block_size :]
        logits = self(context)
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        return context, logits, last_logits, next_id


def main():
    torch.manual_seed(42)

    vocab_size = 10
    block_size = 4
    n_embd = 6

    model = ToyLM(vocab_size, block_size, n_embd)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    context, logits, last_logits, next_id = model.generate_one_step(ids)

    print("ids:", ids)
    print("ids.shape:", ids.shape)

    print("\nblock_size:", block_size)
    print("context = ids[:, -block_size:]")
    print("context:", context)
    print("context.shape:", context.shape)

    print("\nlogits.shape:", logits.shape)
    print("last_logits = logits[:, -1, :]")
    print("last_logits.shape:", last_logits.shape)

    print("\nnext_id:")
    print(next_id)
    print("next_id.shape:", next_id.shape)

    new_ids = torch.cat([ids, next_id], dim=1)
    print("\nafter appending next_id:")
    print("new_ids:", new_ids)
    print("new_ids.shape:", new_ids.shape)

    print("\nMeaning:")
    print("During generation, only the last position predicts the next token.")
    print("If ids is longer than block_size, the model only receives the most recent block_size tokens.")
    print("Generation is autoregressive: one new token is appended at each step.")


if __name__ == "__main__":
    main()
