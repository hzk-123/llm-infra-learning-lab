import torch
import torch.nn as nn
import torch.nn.functional as F


def build_char_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, labels=None):
        logits = self.token_embedding_table(input_ids)

        loss = None
        if labels is not None:
            batch_size, block_size, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * block_size, vocab_size)
            labels_flat = labels.reshape(batch_size * block_size)
            loss = F.cross_entropy(logits_flat, labels_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens):
        ids = start_ids
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -1:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def get_batch(data, batch_size, block_size):
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def main():
    torch.manual_seed(42)

    text = (
        "hello llm\n"
        "hello infra\n"
        "llm infra learns\n"
        "hello model\n"
        "model learns tokens\n"
    )

    stoi, itos = build_char_tokenizer(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    vocab_size = len(stoi)

    batch_size = 4
    block_size = 8
    max_steps = 500

    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    print("vocab_size:", vocab_size)
    print("data length:", len(data))
    print("first 30 token ids:", data[:30].tolist())
    print("first 30 decoded chars:")
    print(repr(decode(data[:30].tolist(), itos)))

    start_id = torch.tensor([[stoi["h"]]], dtype=torch.long)
    before = model.generate(start_id, max_new_tokens=80)
    print("\nBefore training sample:")
    print(repr(decode(before[0].tolist(), itos)))

    for step in range(max_steps + 1):
        x, y = get_batch(data, batch_size, block_size)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    after = model.generate(start_id, max_new_tokens=80)
    print("\nAfter training sample:")
    print(repr(decode(after[0].tolist(), itos)))

    print("\nMeaning:")
    print("The full loop is now text -> token ids -> x/y batches -> training -> generated ids -> text.")
    print("Bigram LM can learn local next-character patterns, but it cannot understand long context.")


if __name__ == "__main__":
    main()
