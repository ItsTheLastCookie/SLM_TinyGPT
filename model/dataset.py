import os
import array
import numpy as np
import torch
from model.tokenizer import Tokenizer


def build_dataset(
    corpus_dir: str = "data/corpus",
    tokenizer_path: str = "tokenizer/tinygpt.model",
    block_size: int = 1024,
    data_dir: str = "data",
    val_split: float = 0.1,
) -> tuple[int, int]:
    tokenizer = Tokenizer(tokenizer_path)

    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {corpus_dir}")

    all_tokens = array.array("H")

    for fname in files:
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        all_tokens.extend(tokenizer.encode(text))

    arr = np.array(all_tokens, dtype=np.uint16)
    n = len(arr)
    split = int(n * (1 - val_split))
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(split,))[:] = arr[:split]
    np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n - split,))[:] = arr[split:]

    print(f"Training tokens: {split:,}")
    print(f"Validation tokens: {n - split:,}")
    return split, n - split

from dataclasses import dataclass
from typing import Literal


@dataclass
class BatchConfig:
    batch_size: int
    block_size: int


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1: idx + self.block_size + 1].astype(np.int64))
        return x, y


def get_batch(
    split: Literal["train", "val"],
    cfg: BatchConfig,
    device: torch.device,
    train_data: np.memmap | None = None,
    val_data: np.memmap | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    if data is None:
        raise ValueError(f"{split}_data is None")

    max_start = len(data) - cfg.block_size - 1
    if max_start <= 0:
        raise ValueError(f"Data too short for block_size={cfg.block_size}")

    ix = torch.randint(max_start, (cfg.batch_size,), device=device)
    xs = torch.from_numpy(np.stack([data[i: i + cfg.block_size] for i in ix.cpu().numpy()])).to(device, dtype=torch.long)
    ys = torch.from_numpy(np.stack([data[i + 1: i + 1 + cfg.block_size] for i in ix.cpu().numpy()])).to(device, dtype=torch.long)
    return xs, ys
