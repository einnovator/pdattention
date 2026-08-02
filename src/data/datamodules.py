import torch
from torch.utils.data import DataLoader, random_split

from .collators import PRACollator
from .datasets import PRADataset, dataset_class_for_stage
from .tokenizer import PRATokenizer


class PRADataModule:
    """Small Lightning-style data module for PRA experiments."""

    def __init__(
        self,
        dataset_stage: str,
        data_dir: str = "data",
        max_examples: int | None = None,
        batch_size: int = 8,
        max_seq_len: int = 96,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        tokenizer: PRATokenizer | None = None,
        split_seed: int = 0,
    ):
        self.dataset_stage = dataset_stage
        self.data_dir = data_dir
        self.max_examples = max_examples
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.tokenizer = tokenizer
        self.split_seed = int(split_seed)
        self.dataset: PRADataset | None = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collator: PRACollator | None = None

    def load(self):
        """Instantiate the dataset, tokenizer, collator, and data splits."""
        dataset_cls = dataset_class_for_stage(self.dataset_stage)
        self.dataset = dataset_cls(self.data_dir, max_examples=self.max_examples)
        if self.tokenizer is None:
            self.tokenizer = PRATokenizer(self._corpus(self.dataset))
        self.collator = PRACollator(self.tokenizer, max_seq_len=self.max_seq_len)
        self.train_dataset, self.val_dataset, self.test_dataset = self._split(
            self.dataset, seed=self.split_seed
        )
        return self

    def train_loader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._loader(self.train_dataset, shuffle=self.shuffle)

    def val_loader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_loader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self._loader(self.test_dataset, shuffle=False)

    def build_reference_tables(self):
        """Build one reference table per sample in the loaded dataset."""
        if self.dataset is None:
            self.load()
        return [self.dataset.build_reference_table(sample) for sample in self.dataset]

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("PRADataModule.load() must be called before requesting loaders.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    @staticmethod
    def _corpus(dataset: PRADataset) -> list[str]:
        texts = []
        for sample in dataset:
            texts.append(sample.question + " " + sample.answer)
            for ref in sample.references:
                texts.append(ref.summary or "")
                texts.append(str(ref.metadata.get("text", "")))
        return texts

    @staticmethod
    def _split(dataset, *, seed: int = 0):
        n = len(dataset)
        if n < 3:
            return dataset, dataset, dataset
        train_len = max(1, int(n * 0.8))
        val_len = max(1, int(n * 0.1))
        test_len = n - train_len - val_len
        if test_len <= 0:
            test_len = 1
            train_len = n - val_len - test_len
        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, [train_len, val_len, test_len], generator=generator)
