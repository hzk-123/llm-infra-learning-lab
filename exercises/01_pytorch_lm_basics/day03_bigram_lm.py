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

    logits, loss = model(x, y)

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("embedding table shape:", model.token_embedding_table.weight.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())

    print("\nOne position example:")
    print("input token id:", x[0, 0].item())
    print("target token id:", y[0, 0].item())
    print("logits for this position:")
    print(logits[0, 0])
    print("predicted token id:", torch.argmax(logits[0, 0]).item())

    print("\nMeaning:")
    print("For each input token position, logits contains one score for every token in the vocabulary.")
    print("Cross entropy compares those scores with the next-token target.")


if __name__ == "__main__":
    main()
