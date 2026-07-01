import torch
import torch.nn as nn


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


def main():
    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8

    x = torch.randn(batch_size, block_size, n_embd)
    ffn = FeedForward(n_embd)
    out = ffn(x)

    print("x.shape:", x.shape)
    print("out.shape:", out.shape)

    print("\nFFN layers:")
    for i, layer in enumerate(ffn.net):
        print(i, layer)

    print("\nParameter shapes:")
    for name, param in ffn.named_parameters():
        print(name, param.shape)

    print("\nOne token before FFN:")
    print(x[0, 0])
    print("\nSame token after FFN:")
    print(out[0, 0])

    print("\nMeaning:")
    print("FeedForward keeps the same batch and sequence dimensions.")
    print("It transforms each token representation independently.")
    print("Attention mixes information across tokens; FFN transforms each token after mixing.")


if __name__ == "__main__":
    main()
