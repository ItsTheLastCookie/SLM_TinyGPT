import os
import sys
import time
import math
from contextlib import nullcontext
import torch
import numpy as np
from model.config import GPTConfig
from model.tokenizer import Tokenizer
from model.dataset import build_dataset, get_batch
from model.transformer import GPT
from model.utils import get_lr, configure_optimizer, estimate_loss


def main():
    cfg = GPTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(4)
        torch.cuda.empty_cache()

    if not os.path.exists("tokenizer/tinygpt.model"):
        print("Tokenizer not found. Training tokenizer...")
        from tokenizer.train_tokenizer import train_tokenizer
        train_tokenizer()

    tokenizer = Tokenizer("tokenizer/tinygpt.model")
    cfg.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {cfg.vocab_size}")

    if not os.path.exists("data/train.bin"):
        print("Building dataset...")
        build_dataset()

    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    model = GPT(cfg)
    model.to(device)

    use_compile = cfg.compile and device.type == "cuda"
    if use_compile:
        try:
            import triton
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile not available ({e}), running in eager mode")
            use_compile = False

    optimizer = configure_optimizer(model, cfg)
    if device.type == "cuda":
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.dtype == "bfloat16" else torch.amp.autocast("cuda")
    else:
        ctx = nullcontext()

    start_step = 0
    best_val_loss = float("inf")
    no_improve_count = 0

    if os.path.exists("ckpt/best.pt"):
        print("Resuming from checkpoint ckpt/best.pt")
        ckpt = torch.load("ckpt/best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed at step {start_step}, best val loss {best_val_loss:.4f}")

    os.makedirs("ckpt", exist_ok=True)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    mem_gb = param_count * 2 * 3 / 1e9
    print(f"Estimated VRAM for weights+optimizer: {mem_gb:.1f} GB (bf16 train)")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum}")

    model.train()
    train_start = time.time()
    for step in range(start_step, cfg.max_iters):
        t0 = time.time()
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch("train", cfg, device, train_data, val_data)
            with ctx:
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if step % 50 == 0:
            t1 = time.time()
            dt = t1 - t0
            elapsed = t1 - train_start
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            pct = 100 * step / cfg.max_iters
            print(f"[{pct:5.1f}%] step {step}/{cfg.max_iters} | loss {accum_loss:.4f} | {dt*1000:.0f}ms/step | elapsed {elapsed_m}m{elapsed_s:02d}s", flush=True)

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            torch.save(ckpt, f"ckpt/step_{step}.pt")
            print(f"  -> saved checkpoint step_{step}.pt", flush=True)

        if step % 10 == 0:
            torch.cuda.empty_cache()

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, get_batch, cfg, device, ctx, train_data, val_data)
            elapsed = time.time() - train_start
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            pct = 100 * step / cfg.max_iters
            print(f"[{pct:5.1f}%] step {step}/{cfg.max_iters} | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f} | lr {lr:.2e} | elapsed {elapsed_m}m{elapsed_s:02d}s", flush=True)

            if losses["train"] < losses["val"] - 0.5:
                print("WARNING: train_loss significantly lower than val_loss — possible overfitting")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                no_improve_count = 0
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                }
                torch.save(ckpt, "ckpt/best.pt")
                print(f"  -> saved best checkpoint (val_loss={best_val_loss:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= 5:
                    print(f"WARNING: val loss has not improved for 5 consecutive evaluations. Stopping training.")
                    break

    print("Training complete.")


if __name__ == "__main__":
    main()
