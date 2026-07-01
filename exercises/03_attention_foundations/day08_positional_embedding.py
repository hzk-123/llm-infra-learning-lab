import torch
import torch.nn as nn


def main():
    torch.manual_seed(42)

    input_ids = torch.tensor(
        [
            [7, 3, 7, 1],
            [2, 7, 4, 7],
        ],
        dtype=torch.long,
    )

    vocab_size = 10
    block_size = 4
    n_embd = 6

    token_embedding = nn.Embedding(vocab_size, n_embd)
    position_embedding = nn.Embedding(block_size, n_embd)

    position_ids = torch.arange(block_size)

    token_emb = token_embedding(input_ids)
    pos_emb = position_embedding(position_ids)
    x = token_emb + pos_emb

    print("input_ids:")
    print(input_ids)
    print("input_ids.shape:", input_ids.shape)

    print("\ntoken_embedding.weight.shape:", token_embedding.weight.shape)
    print("position_embedding.weight.shape:", position_embedding.weight.shape)

    print("\nposition_ids:")
    print(position_ids)
    print("position_ids.shape:", position_ids.shape)

    print("\ntoken_emb.shape:", token_emb.shape)
    print("pos_emb.shape:", pos_emb.shape)
    print("x.shape:", x.shape)

    print("\nSame token id, token embedding only:")
    print("input_ids[0, 0] token id:", input_ids[0, 0].item())
    print("input_ids[0, 2] token id:", input_ids[0, 2].item())
    print("token_emb[0, 0] == token_emb[0, 2]:", torch.allclose(token_emb[0, 0], token_emb[0, 2]))

    print("\nSame token id, after adding positional embedding:")
    print("x[0, 0] == x[0, 2]:", torch.allclose(x[0, 0], x[0, 2]))

    print("\nMeaning:")
    print("Token embedding tells the model what token it is.")
    print("Positional embedding tells the model where the token is.")
    print("After adding both, the same token can have different representations at different positions.")


if __name__ == "__main__":
    main()
