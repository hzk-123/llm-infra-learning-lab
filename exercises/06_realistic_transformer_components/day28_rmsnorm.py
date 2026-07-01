import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def token_stats(name, x):
    token = x[0, 0]
    mean = token.mean().item()
    std = token.std(unbiased=False).item()
    rms = torch.sqrt(torch.mean(token * token)).item()

    print(f"\n{name}, one token:")
    print("values:", token)
    print("mean:", mean)
    print("std:", std)
    print("rms:", rms)


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    n_embd = 8

    x = torch.randn(batch_size, seq_len, n_embd) * 5 + 10

    layer_norm = nn.LayerNorm(n_embd)
    rms_norm = RMSNorm(n_embd)

    x_layer_norm = layer_norm(x)
    x_rms_norm = rms_norm(x)

    print("x.shape:", x.shape)
    print("x_layer_norm.shape:", x_layer_norm.shape)
    print("x_rms_norm.shape:", x_rms_norm.shape)

    print("\nParameter shapes:")
    print("layer_norm.weight.shape:", layer_norm.weight.shape)
    print("layer_norm.bias.shape:", layer_norm.bias.shape)
    print("rms_norm.weight.shape:", rms_norm.weight.shape)

    token_stats("Before normalization", x)
    token_stats("After LayerNorm", x_layer_norm)
    token_stats("After RMSNorm", x_rms_norm)

    print("\nShape checks:")
    print("LayerNorm keeps shape:", x.shape == x_layer_norm.shape)
    print("RMSNorm keeps shape:", x.shape == x_rms_norm.shape)

    print("\nMeaning:")
    print("LayerNorm subtracts the mean and divides by standard deviation.")
    print("RMSNorm does not subtract the mean; it rescales by root mean square.")
    print("Both keep the same tensor shape.")
    print("RMSNorm is simpler and is commonly used in Llama/Qwen-style Transformer blocks.")


if __name__ == "__main__":
    main()
