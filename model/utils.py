import math
import torch
from model.config import GPTConfig


def get_lr(step: int, cfg: GPTConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


def configure_optimizer(model: torch.nn.Module, cfg: GPTConfig):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_loss(model, get_batch_fn, cfg, device, ctx, train_data, val_data):
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
