import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
from model.config import GPTConfig


@dataclass
class LRConfig:
    learning_rate: float
    min_lr: float
    warmup_iters: int
    max_iters: int


class CosineLRSchedule:
    def __init__(self, cfg: LRConfig):
        self.cfg = cfg

    def __call__(self, step: int) -> float:
        if step < self.cfg.warmup_iters:
            return self.cfg.learning_rate * step / self.cfg.warmup_iters
        progress = (step - self.cfg.warmup_iters) / (self.cfg.max_iters - self.cfg.warmup_iters)
        return self.cfg.min_lr + 0.5 * (self.cfg.learning_rate - self.cfg.min_lr) * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self) -> list[float]:
        return [self.__call__(0)]


def configure_optimizer(model: nn.Module, cfg: GPTConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    get_batch_fn: Callable,
    cfg: GPTConfig,
    device: torch.device,
    ctx: torch.autocast | None,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total = 0.0
        data = train_data if split == "train" else val_data
        for _ in range(cfg.eval_iters):
            x, y = get_batch_fn(split, cfg, device, train_data, val_data)
            with ctx:
                _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / cfg.eval_iters
    model.train()
    return losses


def get_autocast_ctx(device: torch.device, dtype: torch.dtype) -> torch.autocast | None:
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return None


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_cuda() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")