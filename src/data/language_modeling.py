"""BPE token-block datasets and datamodules for language-model experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import load_wikitext_splits, wikitext_documents
from .tokenizer import BPETokenizer


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        usable = max((len(token_ids) - 1) // seq_len, 0) * seq_len
        self.tokens = torch.tensor(token_ids[: usable + 1], dtype=torch.long)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return max((len(self.tokens) - 1) // self.seq_len, 0)

    def __getitem__(self, index: int) -> dict:
        start = index * self.seq_len
        window = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": window[:-1],
            "labels": window[1:],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


class WikiTextDataModule:
    """Prepare WikiText splits as fixed BPE token blocks."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        dataset_name: str = "wikitext-2-raw-v1",
        vocab_size: int = 2_000,
        seq_len: int = 128,
        batch_size: int = 4,
        max_train_documents: int | None = 512,
        max_eval_documents: int | None = 128,
        max_train_blocks: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer: BPETokenizer | None = None,
        reference_tokens: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.max_train_documents = max_train_documents
        self.max_eval_documents = max_eval_documents
        self.max_train_blocks = max_train_blocks
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.tokenizer = tokenizer
        self.reference_tokens = reference_tokens or [f"<REF_{index}>" for index in range(1, 6)]
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def load(self):
        splits = load_wikitext_splits(
            self.dataset_name, cache_dir=self.data_dir / ".hf_cache"
        )
        train_texts = wikitext_documents(
            splits["train"], max_documents=self.max_train_documents
        )
        val_texts = wikitext_documents(
            splits["validation"], max_documents=self.max_eval_documents
        )
        test_texts = wikitext_documents(
            splits["test"], max_documents=self.max_eval_documents
        )
        if self.tokenizer is None:
            self.tokenizer = BPETokenizer.train(
                train_texts,
                vocab_size=self.vocab_size,
                reference_tokens=self.reference_tokens,
            )
        self.train_dataset = self._dataset(train_texts, self.max_train_blocks)
        self.val_dataset = self._dataset(val_texts)
        self.test_dataset = self._dataset(test_texts)
        return self

    def _dataset(self, texts: list[str], limit: int | None = None) -> TokenBlockDataset:
        token_ids = []
        eos = self.tokenizer.stoi["[EOS]"]
        for text in texts:
            token_ids.extend(self.tokenizer.encode(text))
            token_ids.append(eos)
        dataset = TokenBlockDataset(token_ids, self.max_seq_len)
        if limit is not None and len(dataset) > limit:
            dataset.tokens = dataset.tokens[: limit * self.max_seq_len + 1]
        return dataset

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_loader(self):
        return self._loader(self.train_dataset, True)

    def val_loader(self):
        return self._loader(self.val_dataset, False)

    def test_loader(self):
        return self._loader(self.test_dataset, False)
