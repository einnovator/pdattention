"""Compatibility exports for the new research-grade data package."""

import torch

from data.datamodules import PRADataModule
from data.schemas import QuestionSample
from data.tokenizer import PRATokenizer

CharTokenizer = PRATokenizer


def build_training_corpus(examples: list[QuestionSample]) -> list[str]:
    texts = []
    for ex in examples:
        if hasattr(ex, "question"):
            texts.append(ex.question + " " + ex.answer)
            for ref in ex.references:
                texts.append(ref.summary or "")
                texts.append(str(ref.metadata.get("text", "")))
        else:
            texts.append(ex.prompt + ex.target)
            texts.extend(ex.refs.values())
            texts.extend(ex.summaries.values())
    return texts


def batch_from_examples(examples, tokenizer, batch_size, max_seq_len, device):
    # Legacy helper retained for notebooks/tests that have not moved to DataLoader.
    from data.collators import PRACollator

    collator = PRACollator(tokenizer, max_seq_len=max_seq_len)
    batch = collator(list(examples)[:batch_size])
    return batch["input_ids"].to(device), batch["labels"].to(device), batch["metadata"]
