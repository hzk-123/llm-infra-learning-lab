import math

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


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch_size, block_size, n_embd = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        mask = self.mask[:block_size, :block_size]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")

        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(concat)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, num_heads, block_size) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, labels=None):
        batch_size, block_size = input_ids.shape

        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(block_size, device=input_ids.device)
        pos_emb = self.position_embedding(position_ids)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
            context = ids[:, -self.block_size :]
            logits, _ = self(context)
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


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def main():
    torch.manual_seed(42)

    text = (
        "hello llm\n"
        "hello infra\n"
        "llm infra learns\n"
        "hello model\n"
        "model learns tokens\n"
        "attention mixes tokens\n"
        "transformer learns context\n"
        "infra serves models\n"
    )

    stoi, itos = build_char_tokenizer(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    vocab_size = len(stoi)

    batch_size = 8
    block_size = 8
    n_embd = 32
    num_heads = 4
    num_layers = 2
    max_steps = 1000

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    print("vocab_size:", vocab_size)
    print("data length:", len(data))
    print("block_size:", block_size)
    print("n_embd:", n_embd)
    print("num_heads:", num_heads)
    print("num_layers:", num_layers)
    print("parameter count:", count_parameters(model))

    x, y = get_batch(data, batch_size=2, block_size=block_size)
    logits, loss = model(x, y)
    print("\nOne batch:")
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("logits.shape:", logits.shape)
    print("initial loss:", loss.item())

    start_text = "hello"
    start_ids = torch.tensor([encode(start_text, stoi)], dtype=torch.long)
    before = model.generate(start_ids, max_new_tokens=100)
    print("\nBefore training sample:")
    print(repr(decode(before[0].tolist(), itos)))

    for step in range(max_steps + 1):
        x, y = get_batch(data, batch_size=batch_size, block_size=block_size)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    after = model.generate(start_ids, max_new_tokens=100)
    print("\nAfter training sample:")
    print(repr(decode(after[0].tolist(), itos)))

    print("\nMeaning:")
    print("This is the first trained Tiny Transformer LM in the project.")
    print("It uses token embeddings, positional embeddings, causal self-attention, FFN, LayerNorm, and an LM head.")


if __name__ == "__main__":
    main()
