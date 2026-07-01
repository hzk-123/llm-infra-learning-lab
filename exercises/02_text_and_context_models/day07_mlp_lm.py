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


class MLPLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, hidden_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(block_size * n_embd, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, input_ids, labels=None):
        emb = self.token_embedding(input_ids)
        batch_size = emb.shape[0]
        emb_flat = emb.reshape(batch_size, -1)
        logits = self.net(emb_flat)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens):
        ids = start_ids
        for _ in range(max_new_tokens):
            context = ids[:, -self.block_size :]
            logits, _ = self(context)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def get_batch(data, batch_size, block_size):
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + block_size] for i in starts])
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

    batch_size = 8
    block_size = 4
    n_embd = 8
    hidden_size = 64
    max_steps = 1000

    model = MLPLanguageModel(vocab_size, block_size, n_embd, hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    for name, param in model.named_parameters():
        print(name, param.shape)
    print("\n")

    print("vocab_size:", vocab_size)
    print("data length:", len(data))
    print("block_size:", block_size)

    x, y = get_batch(data, batch_size=2, block_size=block_size)
    logits, loss = model(x, y)
    print("\nOne batch:")
    print("x:", x)
    print("y:", y)
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("logits.shape:", logits.shape)
    print("initial loss:", loss.item())

    start_text = "hell"
    start_ids = torch.tensor([encode(start_text, stoi)], dtype=torch.long)
    before = model.generate(start_ids, max_new_tokens=80)
    print("\nBefore training sample:")
    print(repr(decode(before[0].tolist(), itos)))

    for step in range(max_steps + 1):
        x, y = get_batch(data, batch_size, block_size)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    after = model.generate(start_ids, max_new_tokens=80)
    print("\nAfter training sample:")
    print(repr(decode(after[0].tolist(), itos)))

    print("\nMeaning:")
    print("MLP LM predicts the next token from a fixed-size context window.")
    print("It uses more context than Bigram LM, but it still has no attention and cannot handle arbitrary long context.")


if __name__ == "__main__":
    main()
