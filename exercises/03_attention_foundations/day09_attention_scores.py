import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(42)

    batch_size = 1
    block_size = 4
    n_embd = 3

    x = torch.randn(batch_size, block_size, n_embd)

    scores = x @ x.transpose(-2, -1)
    weights = F.softmax(scores, dim=-1)
    out = weights @ x

    print("x:")
    print(x)
    print("x.shape:", x.shape)

    print("\nscores:")
    print(scores)
    print("scores.shape:", scores.shape)

    print("\nweights after softmax:")
    print(weights)
    print("weights.shape:", weights.shape)
    print("row sums:")
    print(weights.sum(dim=-1))

    print("\nout:")
    print(out)
    print("out.shape:", out.shape)

    print("\nOne score example:")
    print("scores[0, 1, 2] means token 1's similarity score with token 2.")
    print("scores[0, 1, 2]:", scores[0, 1, 2].item())

    print("\nMeaning:")
    print("Attention scores measure token-to-token similarity.")
    print("Softmax turns scores into weights.")
    print("The output is a weighted sum of token representations.")


if __name__ == "__main__":
    main()
