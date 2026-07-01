import torch


def main():
    batch_x = torch.tensor(
        [
            [10, 11, 12],
            [11, 12, 13],
        ],
        dtype=torch.long,
    )

    vocab_size = 20
    n_embd = 4
    embedding = torch.nn.Embedding(vocab_size, n_embd)

    x = embedding(batch_x)

    print("batch_x:")
    print(batch_x)
    print("batch_x.shape:", batch_x.shape)

    print("\nembedding.weight.shape:", embedding.weight.shape)

    print("\nembedded x:")
    print(x)
    print("embedded x.shape:", x.shape)

    print("\nMeaning:")
    print("batch_x stores token ids.")
    print("embedding.weight stores one vector for each token id.")
    print("embedded x replaces each token id with its trainable vector.")


if __name__ == "__main__":
    main()
