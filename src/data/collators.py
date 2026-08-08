from dataclasses import asdict

import torch

from pra_core.references import ReferenceTable
from .schemas import QuestionSample
from .tokenizer import PRATokenizer


class PRACollator:
    """Convert ``QuestionSample`` objects into model-ready tensor batches."""

    def __init__(self, tokenizer: PRATokenizer, max_seq_len: int, pad_token_id: int = 0):
        """Create a collator with tokenizer, sequence length, and pad id."""
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[QuestionSample]) -> dict:
        """Tokenize samples, pad tensors, and preserve reference metadata."""
        input_rows, label_rows, attention_rows = [], [], []
        reference_tables = []
        metadata = []

        for sample in samples:
            prompt_ids = self.tokenizer.encode(sample.question)
            answer_ids = self.tokenizer.encode(" " + sample.answer.strip() + "\n")
            full_ids = prompt_ids + answer_ids
            full_labels = full_ids[1:] + [self.pad_token_id]
            full_answer_start = max(len(prompt_ids) - 1, 0)
            full_labels[:full_answer_start] = [self.pad_token_id] * full_answer_start
            ids = full_ids[: self.max_seq_len]
            labels = ids[1:] + [self.pad_token_id]
            labels = labels[: len(ids)]
            answer_prediction_start = max(min(len(prompt_ids), len(ids)) - 1, 0)
            labels[:answer_prediction_start] = [self.pad_token_id] * answer_prediction_start

            input_rows.append(torch.tensor(ids, dtype=torch.long))
            label_rows.append(torch.tensor(labels, dtype=torch.long))
            attention_rows.append(torch.ones(len(ids), dtype=torch.long))
            reference_tables.append(self._build_reference_table(sample))
            metadata.append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "answer": sample.answer,
                    "references": sample.references,
                    "target_reference_ids": sample.target_reference_ids,
                    "target_reference_uris": sample.target_reference_uris
                    or sample.metadata.get("target_reference_uris", []),
                    "target_chunk_ids": sample.target_chunk_ids
                    or sample.metadata.get("target_chunk_ids", []),
                    "target_chunk_spans": sample.target_chunk_spans
                    or sample.metadata.get("target_chunk_spans", []),
                    "expected_anchors": sample.metadata.get("expected_anchors", []),
                    # Preserve exact overflow tokens for optional implicit prompt memory.
                    "full_input_ids": tuple(full_ids),
                    "full_labels": tuple(full_labels),
                    "prompt_token_count": len(prompt_ids),
                    "sample": sample,
                    "asdict": asdict(sample),
                }
            )

        return {
            "input_ids": self._pad(input_rows, self.pad_token_id),
            "labels": self._pad(label_rows, self.pad_token_id),
            "attention_mask": self._pad(attention_rows, 0),
            "reference_tables": reference_tables,
            "metadata": metadata,
        }

    def _build_reference_table(self, sample: QuestionSample) -> ReferenceTable:
        """Build the runtime reference table for one sample."""
        table = ReferenceTable()
        for ref in sample.references:
            table.register(
                uri=ref.uri,
                summary=ref.summary,
                metadata=ref.metadata,
                id=ref.id,
                token=f"<REF_{ref.id}>",
            )
        return table

    @staticmethod
    def _pad(rows: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Pad a list of one-dimensional tensors to a dense batch tensor."""
        max_len = max(int(row.numel()) for row in rows)
        out = torch.full((len(rows), max_len), pad_value, dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            out[i, : row.numel()] = row
        return out
