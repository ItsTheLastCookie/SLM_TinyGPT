from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_embd: int = 768
    n_head: int = 12
    head_dim: int = 64
    n_layer: int = 13
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = False

    max_iters: int = 15000
    batch_size: int = 12
    grad_accum: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 20

    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: bool = True
    rope_theta: float = 10000.0

    ckpt_dir: str = "ckpt"
    data_dir: str = "data"
    tokenizer_path: str = "tokenizer/tinygpt.model"

    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.head_dim * self.n_head == self.n_embd, "head_dim * n_head must equal n_embd"
        assert self.dtype in ("float32", "bfloat16", "float16")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class SFTConfig:
    learning_rate: float = 5e-5
    max_iters: int = 200
    batch_size: int = 4
    grad_accum: int = 4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50