from dataclasses import dataclass


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
    dtype: str = "bfloat16"
    compile: bool = True
    rope_theta: float = 10000.0
