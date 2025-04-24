import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GPTBinDataset(Dataset):

    def __init__(self, bin_path, block_size, device):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.length = len(self.data) - block_size - 1
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        if self.device == 'cuda':
            # x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            x, y = x.to(self.device), y.to(self.device)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


if __name__ == "__main__":

    # Example usage
    # dataset = GPTBinDataset("val.bin", 5, "cuda")
    # print(f"Dataset length: {len(dataset)}")
    # x, y = dataset[0]
    # print(f"x: {x}, y: {y}")

    # # Dataloader usage
    # print("\n\nUsing DataLoader...")
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # print(f"DataLoader batch size: {dataloader.batch_size}")
    # print(f"DataLoader number of batches: {len(dataloader)}")
    # for i, batch in enumerate(dataloader):
    #     x_batch, y_batch = batch
    #     print(f"x_batch: {x_batch}, y_batch: {y_batch}")
    #     if i == 2:
    #         break
    # print("DataLoader finished.")

    train_dataset = GPTBinDataset("train.bin", 256, "cuda")
    val_dataset = GPTBinDataset("val.bin", 256, "cuda")
    print(f"Train dataset length: {len(train_dataset)}\n\n")
    print(f"Val dataset length: {len(val_dataset)}\n\n")

