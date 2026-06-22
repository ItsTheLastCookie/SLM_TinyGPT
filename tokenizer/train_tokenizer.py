import os
import glob
import sentencepiece as spm


def train_tokenizer():
    corpus_dir = "data/corpus"
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.md")))
    input_file = "data/corpus.txt"
    with open(input_file, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                out.write(fp.read())
                out.write("\n")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix="tokenizer/tinygpt",
        vocab_size=8192,
        character_coverage=1.0,
        model_type="bpe",
        num_threads=4,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    os.remove(input_file)


if __name__ == "__main__":
    train_tokenizer()
