import sentencepiece as spm
from typing import Optional


class Tokenizer:
    def __init__(self, model_path: str = "tokenizer/tinygpt.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self._pad_id = self.sp.pad_id()
        self._unk_id = self.sp.unk_id()
        self._bos_id = self.sp.bos_id()
        self._eos_id = self.sp.eos_id()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text)
        if add_bos and self._bos_id >= 0:
            ids = [self._bos_id] + ids
        if add_eos and self._eos_id >= 0:
            ids = ids + [self._eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def decode_stream(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self.sp.id_to_piece(id)

    @classmethod
    def train(
        cls,
        input_file: str,
        model_prefix: str = "tokenizer/tinygpt",
        vocab_size: int = 8192,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
        num_threads: int = 4,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> None:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            num_threads=num_threads,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )