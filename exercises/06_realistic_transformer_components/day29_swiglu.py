import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUFeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x, return_intermediates=False):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden)

        if return_intermediates:
            return out, gate, up, hidden
        return out


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    n_embd = 8
    hidden_dim = 32

    x = torch.randn(batch_size, seq_len, n_embd)

    relu_ffn = ReLUFeedForward(n_embd=n_embd, hidden_dim=hidden_dim)
    swiglu_ffn = SwiGLUFeedForward(n_embd=n_embd, hidden_dim=hidden_dim)

    relu_out = relu_ffn(x)
    swiglu_out, gate, up, hidden = swiglu_ffn(x, return_intermediates=True)

    print("x.shape:", x.shape)
    print("hidden_dim:", hidden_dim)

    print("\nReLU FFN:")
    print("relu_out.shape:", relu_out.shape)
    print("parameter_count:", count_parameters(relu_ffn))
    for name, param in relu_ffn.named_parameters():
        print(name, param.shape)

    print("\nSwiGLU FFN:")
    print("gate.shape:", gate.shape)
    print("up.shape:", up.shape)
    print("hidden = silu(gate) * up")
    print("hidden.shape:", hidden.shape)
    print("swiglu_out.shape:", swiglu_out.shape)
    print("parameter_count:", count_parameters(swiglu_ffn))
    for name, param in swiglu_ffn.named_parameters():
        print(name, param.shape)

    print("\nOne token intermediate values:")
    print("x[0, 0]:", x[0, 0])
    print("gate[0, 0, :8]:", gate[0, 0, :8])
    print("silu(gate)[0, 0, :8]:", F.silu(gate)[0, 0, :8])
    print("up[0, 0, :8]:", up[0, 0, :8])
    print("hidden[0, 0, :8]:", hidden[0, 0, :8])

    print("\nShape checks:")
    print("ReLU FFN keeps shape:", x.shape == relu_out.shape)
    print("SwiGLU FFN keeps shape:", x.shape == swiglu_out.shape)

    print("\nMeaning:")
    print("ReLU FFN uses one hidden branch: relu(W1 x).")
    print("SwiGLU FFN uses two hidden branches: gate_proj(x) and up_proj(x).")
    print("silu(gate) controls how much of up passes through.")
    print("down_proj maps the hidden dimension back to n_embd for residual addition.")
    print("SwiGLU is commonly used in Llama/Qwen-style Transformer blocks.")


if __name__ == "__main__":
    main()
