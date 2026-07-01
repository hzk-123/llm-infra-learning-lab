import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, labels=None):
        logits = self.token_embedding_table(input_ids)

        loss = None
        if labels is not None:
            batch_size, block_size, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * block_size, vocab_size)
            labels_flat = labels.view(batch_size * block_size)
            loss = F.cross_entropy(logits_flat, labels_flat)

        return logits, loss


def main():
    torch.manual_seed(42)

    x = torch.tensor(
        [
            [10, 11, 12],
            [11, 12, 13],
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        [
            [11, 12, 13],
            [12, 13, 14],
        ],
        dtype=torch.long,
    )

    vocab_size = 20
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    print("Before training:")
    logits, loss = model(x, y)
    print("loss:", loss.item())
    print("predicted token at x[0, 0]:", torch.argmax(logits[0, 0]).item())
    print("target token at y[0, 0]:", y[0, 0].item())

    for step in range(101):
        # print(model.token_embedding_table.weight[0])
        # print("\n")
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    print("\nAfter training:")
    logits, loss = model(x, y)
    print("loss:", loss.item())
    print("predicted token at x[0, 0]:", torch.argmax(logits[0, 0]).item())
    print("target token at y[0, 0]:", y[0, 0].item())

    print("\nMeaning:")
    print("The optimizer updated embedding weights so the model assigns higher scores to the next-token targets.")


if __name__ == "__main__":
    main()
