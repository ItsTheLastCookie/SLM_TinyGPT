import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path="tokenizer/tinygpt.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)
