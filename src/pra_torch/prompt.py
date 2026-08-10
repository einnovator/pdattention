"""Long-prompt preparation through request-local implicit PRA references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .memory import PRAMemoryCache, PRASimpleMemoryCache


IMPLICIT_PROMPT_HEAD_URI = "pra://implicit/prompt/head"
IMPLICIT_PROMPT_HEAD_NAME = "#__head"


@dataclass(frozen=True)
class PreparedPrompt:
    """One exact token split between direct context and implicit prompt memory."""

    direct_ids: tuple[int, ...]
    implicit_ids: tuple[int, ...]
    total_tokens: int
    direct_limit: int
    overflow_mode: str

    @property
    def overflowed(self) -> bool:
        """Return whether any source token exceeded the direct window."""
        return bool(self.implicit_ids) or self.total_tokens > self.direct_limit


@dataclass(frozen=True)
class PromptPreparationStats:
    """Per-row counters used by training and evaluation reports."""

    total_tokens: int
    direct_tokens: int
    implicit_tokens: int
    implicit_chunks: int
    implicit_gists: int

    def as_metrics(self) -> dict[str, float]:
        """Return stable metric names expected by experiment reports."""
        return {
            "prompt_total_tokens": float(self.total_tokens),
            "prompt_direct_tokens": float(self.direct_tokens),
            "prompt_implicit_tokens": float(self.implicit_tokens),
            "prompt_implicit_chunks": float(self.implicit_chunks),
            "prompt_implicit_gists": float(self.implicit_gists),
        }


@dataclass
class PreparedPromptBatch:
    """Rectangular direct tails plus row-local caches and split diagnostics."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    position_offsets: torch.Tensor
    caches: list[PRAMemoryCache]
    splits: list[PreparedPrompt]
    stats: list[PromptPreparationStats]


def prepare_prompt_for_pra(input_ids: Sequence[int] | torch.Tensor, config) -> PreparedPrompt:
    """Split one unpadded prompt on token IDs according to overflow policy."""
    ids = tuple(int(token_id) for token_id in input_ids)
    direct_limit = int(config.effective_prompt_direct_tokens)
    if len(ids) <= direct_limit:
        return PreparedPrompt(ids, (), len(ids), direct_limit, config.prompt_overflow_mode)
    if config.prompt_overflow_mode == "error":
        raise ValueError(
            f"Prompt has {len(ids)} tokens, exceeding direct limit {direct_limit}."
        )
    tail = ids[-direct_limit:]
    if config.prompt_overflow_mode == "truncate":
        return PreparedPrompt(tail, (), len(ids), direct_limit, "truncate")
    if config.prompt_overflow_mode != "implicit_reference":
        raise ValueError(f"Unsupported prompt_overflow_mode: {config.prompt_overflow_mode}")
    return PreparedPrompt(tail, ids[:-direct_limit], len(ids), direct_limit, "implicit_reference")


def _unpadded_row(values: torch.Tensor, mask: torch.Tensor | None) -> list[int]:
    """Extract one row's valid IDs for either left- or right-padded tensors."""
    if mask is None:
        return [int(value) for value in values]
    return [int(value) for value, keep in zip(values, mask) if bool(keep)]


def _source_rows(input_ids, attention_mask, labels, metadata):
    """Prefer collator-preserved full rows, falling back to the provided tensors."""
    id_rows = []
    label_rows = [] if labels is not None else None
    for row_index in range(input_ids.shape[0]):
        item = metadata[row_index] if metadata is not None else {}
        ids = item.get("full_input_ids")
        if ids is None:
            mask = attention_mask[row_index] if attention_mask is not None else None
            ids = _unpadded_row(input_ids[row_index], mask)
        id_rows.append([int(value) for value in ids])
        if label_rows is not None:
            row_labels = item.get("full_labels")
            if row_labels is None:
                mask = attention_mask[row_index] if attention_mask is not None else None
                row_labels = _unpadded_row(labels[row_index], mask)
            label_rows.append([int(value) for value in row_labels])
    return id_rows, label_rows


def _pad_rows(rows, *, value, device, dtype):
    """Right-pad non-empty integer rows without changing token order."""
    if not rows or any(not row for row in rows):
        raise ValueError("Prompt preparation requires at least one token per batch row.")
    width = max(len(row) for row in rows)
    result = torch.full((len(rows), width), value, dtype=dtype, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, dtype=dtype, device=device)
        mask[index, : len(row)] = 1
    return result, mask


def prepare_prompt_batch_for_pra(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    metadata: list[dict] | None = None,
    caches: list[PRAMemoryCache] | None = None,
    pad_token_id: int = 0,
) -> PreparedPromptBatch:
    """Prepare mixed-length direct tails and populate each row's implicit head cache."""
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids [batch,tokens], got {tuple(input_ids.shape)}.")
    batch_size = int(input_ids.shape[0])
    if metadata is not None and len(metadata) != batch_size:
        raise ValueError("Prompt metadata must align with input batch rows.")
    if caches is None:
        caches = [PRASimpleMemoryCache() for _ in range(batch_size)]
    if len(caches) != batch_size:
        raise ValueError("Prompt caches must align with input batch rows.")

    source_ids, source_labels = _source_rows(input_ids, attention_mask, labels, metadata)
    splits = [prepare_prompt_for_pra(row, model.cfg) for row in source_ids]
    direct_rows = [list(split.direct_ids) for split in splits]
    direct_labels = (
        [row[-len(split.direct_ids) :] for row, split in zip(source_labels, splits)]
        if source_labels is not None
        else None
    )
    direct_ids, direct_mask = _pad_rows(
        direct_rows,
        value=pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    padded_labels = None
    if direct_labels is not None:
        padded_labels, _ = _pad_rows(
            direct_labels,
            value=pad_token_id,
            device=labels.device,
            dtype=labels.dtype,
        )

    stats = []
    for row_index, (split, cache) in enumerate(zip(splits, caches)):
        implicit_chunks = 0
        implicit_gists = 0
        if hasattr(cache, "invalidate"):
            cache.invalidate(IMPLICIT_PROMPT_HEAD_URI)
        if split.implicit_ids:
            entry = model.encode_reference_tokens_to_cache(
                IMPLICIT_PROMPT_HEAD_URI,
                split.implicit_ids,
                tokenizer,
                input_ids.device,
                metadata={
                    "implicit": True,
                    "source": "prompt",
                    "kind": "head",
                    "display_name": IMPLICIT_PROMPT_HEAD_NAME,
                    "prompt_row": row_index,
                    "prompt_total_tokens": split.total_tokens,
                    "prompt_direct_tokens": len(split.direct_ids),
                    "prompt_implicit_tokens": len(split.implicit_ids),
                    "max_prompt_gists": model.cfg.max_prompt_gists,
                },
                max_chunks=model.cfg.max_prompt_gists,
                use_configured_max_chunks=False,
                max_chunk_tokens=model.cfg.effective_model_max_context_tokens,
                historical_encoding=(model.cfg.prompt_position_mode == "historical"),
            )
            cache.put(entry)
            implicit_chunks = int(entry.metadata.get("chunk_count", 0))
            if entry.layer_memory:
                first_layer = entry.layer_memory[min(entry.layer_memory)]
                implicit_gists = sum(
                    int(chunk.routing_gist.k.shape[0]) for chunk in first_layer.chunks
                )
        stats.append(
            PromptPreparationStats(
                total_tokens=split.total_tokens,
                direct_tokens=len(split.direct_ids),
                implicit_tokens=len(split.implicit_ids),
                implicit_chunks=implicit_chunks,
                implicit_gists=implicit_gists,
            )
        )
    return PreparedPromptBatch(
        input_ids=direct_ids,
        attention_mask=direct_mask,
        labels=padded_labels,
        position_offsets=torch.tensor(
            [
                model.cfg.prompt_tail_position_offset(
                    len(split.implicit_ids), len(split.direct_ids)
                )
                for split in splits
            ],
            dtype=torch.long,
            device=input_ids.device,
        ),
        caches=caches,
        splits=splits,
        stats=stats,
    )


__all__ = [
    "IMPLICIT_PROMPT_HEAD_NAME",
    "IMPLICIT_PROMPT_HEAD_URI",
    "PreparedPrompt",
    "PreparedPromptBatch",
    "PromptPreparationStats",
    "prepare_prompt_batch_for_pra",
    "prepare_prompt_for_pra",
]
