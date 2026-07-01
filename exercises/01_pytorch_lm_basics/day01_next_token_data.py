import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def main():
    ids = [10, 11, 12, 13, 14, 15]
    block_size = 3

    dataset = NextTokenDataset(ids, block_size)

    print("Single sample:")
    x, y = dataset[0]
    print("x:", x)
    print("y:", y)
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)

    print("\nBatch sample:")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    print("batch_x:", batch_x)
    print("batch_y:", batch_y)
    print("batch_x.shape:", batch_x.shape)
    print("batch_y.shape:", batch_y.shape)

    print("\nMeaning:")
    print("Each row in x is the model input.")
    print("Each row in y is the next-token target.")


if __name__ == "__main__":
    main()
