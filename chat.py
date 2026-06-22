import os
import argparse
import torch
from model.config import GPTConfig
from model.tokenizer import Tokenizer
from model.transformer import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Tokenizer("tokenizer/tinygpt.model")
    cfg = GPTConfig()
    cfg.vocab_size = tokenizer.vocab_size

    ckpt_path = None
    for path in ["ckpt/sft_final.pt", "ckpt/best.pt"]:
        if os.path.exists(path):
            ckpt_path = path
            print(f"Loading {path}")
            break

    if ckpt_path is None:
        print("No checkpoint found. Train the model first.")
        return

    model = GPT(cfg)
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}\n")

    def generate(prompt: str) -> str:
        full_prompt = f"User: {prompt}\nAssistant: "
        input_ids = tokenizer.encode(full_prompt)
        idx = torch.tensor([input_ids], dtype=torch.long).to(device)

        generated = []
        prev_text = ""
        for _ in range(args.max_tokens):
            with torch.no_grad():
                logits, _ = model(idx[:, -cfg.block_size:])
            logits = logits[:, -1, :] / args.temp
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            if token_id == tokenizer.pad_id:
                break
            idx = torch.cat((idx, next_token), dim=1)
            generated.append(token_id)
            full_text = tokenizer.decode(generated)
            delta = full_text[len(prev_text):]
            print(delta, end="", flush=True)
            prev_text = full_text
        print()
        return prev_text

    if args.prompt:
        print(f"You: {args.prompt}")
        print(f"GPT: ", end="", flush=True)
        generate(args.prompt)
        return

    print("TinyGPT Chat (type 'quit' to exit)\n")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit"):
            break
        print("GPT: ", end="", flush=True)
        generate(prompt)


if __name__ == "__main__":
    main()
