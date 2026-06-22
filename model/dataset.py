
import os
import array
import numpy as np
import torch
from model.tokenizer import Tokenizer


def build_dataset(corpus_dir="data/corpus", tokenizer_path="tokenizer/tinygpt.model", block_size=512, data_dir="data"):
    tokenizer = Tokenizer(tokenizer_path)

    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md"))
    all_tokens = array.array("H")

    for fname in files:
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        all_tokens.extend(tokenizer.encode(text))

    arr = np.array(all_tokens, dtype=np.uint16)
    n = len(arr)
    split = int(n * 0.9)
    os.makedirs(data_dir, exist_ok=True)

    np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="write", shape=(split,))[:] = arr[:split]
    np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="write", shape=(n - split,))[:] = arr[split:]

    print(f"Training tokens: {split:,}")
    print(f"Validation tokens: {n - split:,}")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1: idx + self.block_size + 1].astype(np.int64))
        return x, y


def get_batch(split: str, cfg, device: torch.device, train_data=None, val_data=None):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    xs = torch.stack([torch.from_numpy(data[i: i + cfg.block_size].astype(np.int64)) for i in ix])
    ys = torch.stack([torch.from_numpy(data[i + 1: i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return xs.to(device), ys.to(device)
