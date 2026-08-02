import re
import string

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

REF_TOKEN_PATTERN = re.compile(r"<REF_\d+>")


class PRATokenizer:
    """Tiny educational tokenizer that preserves <REF_n> as atomic tokens."""

    def __init__(self, texts: list[str] | None = None, extra_tokens: list[str] | None = None):
        texts = texts or []
        extra_tokens = extra_tokens or ["<REF_0>"]
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self._build(texts, extra_tokens)

    @property
    def vocab_size(self) -> int:
        """Number of currently registered character and reference tokens."""
        return len(self.stoi)

    @classmethod
    def from_vocab(cls, stoi: dict[str, int]):
        """Recreate a tokenizer from a saved string-to-id vocabulary."""
        tok = cls([])
        tok.stoi = {str(token): int(idx) for token, idx in stoi.items()}
        tok.itos = {idx: token for token, idx in tok.stoi.items()}
        return tok

    def _build(self, texts: list[str], extra_tokens: list[str]) -> None:
        joined = "".join(texts + extra_tokens)
        ref_tokens = sorted(set(REF_TOKEN_PATTERN.findall(joined)))
        chars = sorted(set(REF_TOKEN_PATTERN.sub("", joined) + string.printable))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        for token in ref_tokens:
            self.register_reference_token(token)
        self.itos = {i: token for token, i in self.stoi.items()}

    def register_reference_token(self, token: str) -> int:
        """Register ``<REF_n>`` as one indivisible vocabulary item."""
        if not REF_TOKEN_PATTERN.fullmatch(token):
            raise ValueError(f"Invalid reference token: {token}")
        if token not in self.stoi:
            self.stoi[token] = len(self.stoi)
            self.itos[self.stoi[token]] = token
        return self.stoi[token]

    def encode(self, text: str) -> list[int]:
        """Encode text while preserving reference handles as single ids."""
        ids = []
        pos = 0
        for match in REF_TOKEN_PATTERN.finditer(text):
            for ch in text[pos : match.start()]:
                ids.append(self.stoi.get(ch, self.stoi["?"]))
            token = match.group(0)
            if token not in self.stoi:
                self.register_reference_token(token)
            ids.append(self.stoi[token])
            pos = match.end()
        for ch in text[pos:]:
            ids.append(self.stoi.get(ch, self.stoi["?"]))
        return ids

    def decode(self, ids) -> str:
        """Decode ids back to text using the current vocabulary."""
        return "".join(self.itos[int(i)] for i in ids)


class BPETokenizer:
    """Adapter around Hugging Face Tokenizers BPE with the PRA tokenizer interface."""

    DEFAULT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._refresh_vocab()

    @classmethod
    def train(
        cls,
        texts,
        *,
        vocab_size: int = 2_000,
        reference_tokens: list[str] | None = None,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = BPEDecoder(suffix="</w>")
        special_tokens = [*cls.DEFAULT_SPECIAL_TOKENS, *(reference_tokens or [])]
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=special_tokens,
            end_of_word_suffix="</w>",
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_json(cls, value: str) -> "BPETokenizer":
        return cls(Tokenizer.from_str(value))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        token_id = self.tokenizer.token_to_id("[PAD]")
        return 0 if token_id is None else token_id

    def _refresh_vocab(self) -> None:
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {index: token for token, index in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode([int(index) for index in ids], skip_special_tokens=False)

    def to_json(self) -> str:
        return self.tokenizer.to_str()
