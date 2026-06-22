import io
import os
import struct
import torch
import numpy as np
from gguf import GGUFWriter
from model.config import GPTConfig
from model.tokenizer import Tokenizer
from model.transformer import GPT


def convert(model_path, output_path, tokenizer_path):
    device = torch.device("cpu")
    cfg = GPTConfig()
    tokenizer = Tokenizer(tokenizer_path)
    cfg.vocab_size = tokenizer.vocab_size

    model = GPT(cfg)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_embd = cfg.n_embd
    n_head = cfg.n_head
    n_layer = cfg.n_layer
    block_size = cfg.block_size
    vocab_size = cfg.vocab_size
    head_dim = cfg.head_dim
    n_kv_head = n_head

    extra_token_count = 1
    gguf_vocab_size = vocab_size + extra_token_count

    print(f"Model: {n_layer} layers, {n_embd} embed, {n_head} heads, head_dim={head_dim}")

    writer = GGUFWriter(output_path, "llama")

    writer.add_context_length(block_size)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_embd * 4)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_vocab_size(gguf_vocab_size)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_name("tinygpt")
    writer.add_causal_attention(True)
    writer.add_string("tokenizer.ggml.model", "llama")

    sp = tokenizer.sp
    tokens = []
    scores = []
    token_types = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        tokens.append(piece)
        scores.append(sp.get_score(i))
        if i == sp.unk_id():
            token_types.append(2)
        elif i == sp.bos_id() or i == sp.eos_id() or i == sp.pad_id():
            token_types.append(3)
        else:
            token_types.append(1)

    tokens.append("\n")
    scores.append(-1000.0)
    token_types.append(1)

    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_pad_token_id(sp.pad_id())
    writer.add_precompiled_charsmap(b"")

    writer.add_string("tokenizer.ggml.pre", "default")

    def to_f16(t):
        return t.cpu().numpy().astype(np.float16)

    wte = model.wte.weight.data.clone()
    pad_row = torch.zeros(1, n_embd)
    wte_expanded = torch.cat([wte, pad_row], dim=0)
    writer.add_tensor("token_embd.weight", to_f16(wte_expanded))
    writer.add_tensor("output_norm.weight", model.ln_f.weight.data.cpu().numpy().astype(np.float32))

    for i in range(n_layer):
        block = model.blocks[i]

        writer.add_tensor(f"blk.{i}.attn_norm.weight", block.attn_norm.weight.data.cpu().numpy().astype(np.float32))

        q = block.attn.q_proj.weight.data.T.reshape(n_embd, n_head, head_dim).permute(1, 2, 0).reshape(n_head * head_dim, n_embd).contiguous()
        k = block.attn.k_proj.weight.data.T.reshape(n_embd, n_kv_head, head_dim).permute(1, 2, 0).reshape(n_kv_head * head_dim, n_embd).contiguous()
        v = block.attn.v_proj.weight.data.T.reshape(n_embd, n_kv_head, head_dim).permute(1, 2, 0).reshape(n_kv_head * head_dim, n_embd).contiguous()

        writer.add_tensor(f"blk.{i}.attn_q.weight", to_f16(q))
        writer.add_tensor(f"blk.{i}.attn_k.weight", to_f16(k))
        writer.add_tensor(f"blk.{i}.attn_v.weight", to_f16(v))
        writer.add_tensor(f"blk.{i}.attn_output.weight", to_f16(block.attn.o_proj.weight.data))

        writer.add_tensor(f"blk.{i}.ffn_norm.weight", block.ffn_norm.weight.data.cpu().numpy().astype(np.float32))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", to_f16(block.mlp.w1.weight.data))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", to_f16(block.mlp.w3.weight.data))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", to_f16(block.mlp.w2.weight.data))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"GGUF saved to {output_path} ({file_size / 1e6:.1f} MB)")
    print(f"  Architecture: llama (RMSNorm + SwiGLU + RoPE)")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/sft_final.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tinygpt.gguf"
    tokenizer_path = "tokenizer/tinygpt.model"
    convert(model_path, output_path, tokenizer_path)
