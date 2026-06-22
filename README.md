# TinyGPT

A template for training a small GPT language model from scratch on Wikipedia text, with LLaMA-style architecture (RMSNorm + SwiGLU + RoPE).

## Architecture

- **129M parameters**
- 13 layers, 768 embed dim, 12 heads, 64 head dim
- RMSNorm, SwiGLU, Rotary Position Embeddings
- SentencePiece tokenizer (8192 vocab)
- Context length: 1024

## Quick Start

```bash
pip install -r requirements.txt

# Download training data (Wikitext-103)
python download_datasets.py

# Scrape additional Wikipedia articles (optional, 10k articles)
python scrape_corpus.py

# Train tokenizer
python tokenizer/train_tokenizer.py

# Pretrain (~7 hours on GPU)
python train.py

# SFT fine-tune for chat
python sft/finetune.py

# Chat with the model
python chat.py

# Export to GGUF for llama.cpp / LM Studio
python export_gguf.py ckpt/sft_final.pt tinygpt.gguf
```

## Project Structure

```
tinygpt/
├── train.py                 # Pretraining loop
├── chat.py                  # Interactive chat
├── download_datasets.py     # Download Wikitext-103
├── scrape_corpus.py         # Scrape Wikipedia articles
├── export_gguf.py           # Export to GGUF format
├── model/
│   ├── config.py            # Model hyperparameters
│   ├── transformer.py       # RMSNorm + RoPE + SwiGLU transformer
│   ├── tokenizer.py         # SentencePiece wrapper
│   ├── dataset.py           # Data loading and batching
│   └── utils.py             # LR scheduler, optimizer, eval
├── sft/
│   ├── finetune.py          # Supervised fine-tuning
│   └── sft_data.jsonl       # Instruction-following examples
└── tokenizer/
    └── train_tokenizer.py   # Train SentencePiece tokenizer
```

## Requirements

- Python 3.12+
- PyTorch 2.0+ with CUDA
- sentencepiece
- numpy
- requests
- tqdm

## License

MIT
