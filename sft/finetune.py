import os
import sys
import json
import time
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.config import GPTConfig
from model.tokenizer import Tokenizer
from model.transformer import GPT
from model.utils import configure_optimizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(4)
        torch.cuda.empty_cache()

    cfg = GPTConfig()
    tokenizer = Tokenizer(os.path.join(ROOT, "tokenizer/tinygpt.model"))
    cfg.vocab_size = tokenizer.vocab_size

    model = GPT(cfg)
    model.to(device)

    ckpt_path = os.path.join(ROOT, "ckpt/best.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError("No pretrained checkpoint found at ckpt/best.pt. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded pretrained weights from ckpt/best.pt (step {ckpt.get('step', '?')})")

    data_path = os.path.join(ROOT, "sft/sft_data.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(examples)} SFT examples")

    prefix = "User: "
    separator = "\nAssistant: "
    suffix = "\n\n"

    tokenized = []
    for ex in examples:
        full_text = prefix + ex["user"] + separator + ex["assistant"] + suffix
        tokens = tokenizer.encode(full_text)
        prompt_text = prefix + ex["user"] + separator
        prompt_tokens = tokenizer.encode(prompt_text)
        tokenized.append((tokens, len(prompt_tokens)))

    sft_lr = 5e-5
    sft_iters = 200
    sft_batch = 4
    sft_grad_accum = 4

    optimizer = configure_optimizer(model, cfg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = sft_lr

    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.dtype == "bfloat16" else torch.amp.autocast("cuda")

    model.train()
    print(f"Starting SFT: {sft_iters} iters, lr={sft_lr}, batch={sft_batch}, grad_accum={sft_grad_accum}")
    print(f"Effective batch: {sft_batch * sft_grad_accum}")

    train_start = time.time()

    for step in range(sft_iters):
        t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(sft_grad_accum):
            indices = np.random.randint(0, len(tokenized), size=sft_batch).tolist()
            batch_x = []
            batch_y = []
            for idx in indices:
                tokens, prompt_len = tokenized[idx]
                if len(tokens) > cfg.block_size:
                    tokens = tokens[:cfg.block_size]
                    prompt_len = min(prompt_len, cfg.block_size)
                x = tokens
                y = [-100] * prompt_len + tokens[prompt_len:]
                y = y[1:] + [-100]
                if len(x) < cfg.block_size:
                    pad_len = cfg.block_size - len(x)
                    x = x + [tokenizer.pad_id] * pad_len
                    y = y + [-100] * pad_len
                batch_x.append(x)
                batch_y.append(y)
            x = torch.tensor(batch_x, dtype=torch.long).to(device)
            y = torch.tensor(batch_y, dtype=torch.long).to(device)
            with ctx:
                _, loss = model(x, y)
            loss = loss / sft_grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            dt = time.time() - t0
            elapsed = time.time() - train_start
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            pct = 100 * step / sft_iters
            print(f"[{pct:5.1f}%] sft step {step}/{sft_iters} | loss {accum_loss:.4f} | {dt*1000:.0f}ms | elapsed {elapsed_m}m{elapsed_s:02d}s", flush=True)

        if step % 50 == 0 and step > 0:
            torch.cuda.empty_cache()

    os.makedirs(os.path.join(ROOT, "ckpt"), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ROOT, "ckpt/sft_final.pt"))
    print("SFT complete. Saved to ckpt/sft_final.pt")


if __name__ == "__main__":
    main()
