import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, vocab_size, block_size, n_embd, num_heads):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.block = TransformerBlock(n_embd, num_heads, block_size)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, labels=None):
        batch_size, block_size = input_ids.shape

        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(block_size, device=input_ids.device)
        pos_emb = self.position_embedding(position_ids)

        x = token_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            batch_size, block_size, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * block_size, vocab_size)
            labels_flat = labels.reshape(batch_size * block_size)
            loss = F.cross_entropy(logits_flat, labels_flat)

        return logits, loss


def main():
    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    vocab_size = 16
    n_embd = 8
    num_heads = 2

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        num_heads=num_heads,
    )

    logits, loss = model(input_ids, labels)

    print("input_ids.shape:", input_ids.shape)
    print("labels.shape:", labels.shape)
    print("token_embedding.weight.shape:", model.token_embedding.weight.shape)
    print("position_embedding.weight.shape:", model.position_embedding.weight.shape)
    print("lm_head.weight.shape:", model.lm_head.weight.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())

    print("\nModel modules:")
    for name, module in model.named_children():
        print(name, "->", module.__class__.__name__)

    print("\nMeaning:")
    print("TinyTransformerLM maps token ids to logits over the vocabulary at every position.")
    print("It combines token embeddings, positional embeddings, a Transformer block, final LayerNorm, and an LM head.")


if __name__ == "__main__":
    main()
