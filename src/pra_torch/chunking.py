"""Deterministic, provenance-preserving partitioning for PRA references."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Protocol


@dataclass(frozen=True)
class ReferenceChunk:
    """Tokenizer-ready source span from which every PRA layer builds memory."""

    chunk_id: str  # Stable URI-qualified identity used by labels and traces.
    source_uri: str  # Canonical reference URI that owns the span.
    text: str  # Human-readable span supplied to the tokenizer.
    token_ids: tuple[int, ...]  # Exact token sequence encoded into layer K/V.
    token_start: int  # Inclusive offset in the full reference tokenization.
    token_end: int  # Exclusive offset in the full reference tokenization.
    char_start: int | None = None  # Optional inclusive source-character offset.
    char_end: int | None = None  # Optional exclusive source-character offset.
    marker_type: str | None = None  # Boundary rule that created the chunk.
    metadata: dict = field(default_factory=dict)  # Dataset and partition provenance.


class ReferencePartitioner(Protocol):
    """Contract for deterministic character-span partitioners."""

    def partition(self, uri: str, text: str, metadata: dict) -> list[tuple[int, int, str | None]]:
        """Return non-empty character spans and optional marker types."""


class SemanticChunker(Protocol):
    """Plugin contract for model- or domain-aware token chunking."""

    def partition(
        self,
        uri: str,
        text: str,
        token_ids: list[int],
        metadata: dict,
        max_chunks: int,
    ) -> list[ReferenceChunk]:
        """Return semantic chunks supplied by an explicit plugin."""


class ExplicitMarkerPartitioner:
    """Split text at configured literal markers while preserving source offsets."""

    def __init__(self, markers: tuple[str, ...], retain_markers: bool = True):
        """Store non-empty markers and whether each starts the following span."""
        self.markers = tuple(marker for marker in markers if marker)
        self.retain_markers = retain_markers

    def partition(self, uri: str, text: str, metadata: dict) -> list[tuple[int, int, str | None]]:
        """Return non-empty spans separated by any configured literal marker."""
        del uri, metadata
        if not self.markers:
            return [(0, len(text), None)] if text else []
        pattern = re.compile("|".join(re.escape(marker) for marker in self.markers))
        matches = list(pattern.finditer(text))
        if not matches:
            return [(0, len(text), None)] if text else []
        boundaries = [0]
        for match in matches:
            boundary = match.start() if self.retain_markers else match.end()
            if boundary > boundaries[-1]:
                boundaries.append(boundary)
        boundaries.append(len(text))
        spans = []
        for start, end in zip(boundaries, boundaries[1:]):
            if text[start:end].strip():
                spans.append((start, end, "explicit"))
        return spans


class MarkdownHeadingPartitioner:
    """Treat Markdown headings as stable, human-readable chunk boundaries."""

    _heading = re.compile(r"(?m)^#{1,6}\s+.+$")

    def partition(self, uri: str, text: str, metadata: dict) -> list[tuple[int, int, str | None]]:
        """Return preamble/section spans beginning at Markdown headings."""
        del uri, metadata
        matches = list(self._heading.finditer(text))
        if not matches:
            return [(0, len(text), None)] if text else []
        starts = [0] if text[: matches[0].start()].strip() else []
        starts.extend(match.start() for match in matches)
        starts = sorted(set(starts))
        spans = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            if text[start:end].strip():
                spans.append((start, end, "markdown_heading"))
        return spans


def _chunk_id(uri: str, index: int) -> str:
    """Build the stable local identity shared across layer-specific encodings."""
    return f"{uri}#chunk={index}"


def _char_chunks(uri, text, tokenizer, spans, metadata) -> list[ReferenceChunk]:
    """Tokenize character spans and translate them into full-document offsets."""
    chunks = []
    for index, (char_start, char_end, marker_type) in enumerate(spans):
        chunk_text = text[char_start:char_end]
        token_ids = tuple(tokenizer.encode(chunk_text))
        if not token_ids:
            continue
        token_start = len(tokenizer.encode(text[:char_start]))
        chunks.append(
            ReferenceChunk(
                chunk_id=_chunk_id(uri, index),
                source_uri=uri,
                text=chunk_text,
                token_ids=token_ids,
                token_start=token_start,
                token_end=token_start + len(token_ids),
                char_start=char_start,
                char_end=char_end,
                marker_type=marker_type,
                metadata=dict(metadata),
            )
        )
    return chunks


def _apply_overflow(chunks, max_chunks, policy, all_token_ids, text, tokenizer):
    """Apply the configured cap without silently losing overflow provenance."""
    if len(chunks) <= max_chunks:
        return chunks, 0
    discarded = len(chunks) - max_chunks
    if policy == "error":
        raise ValueError(
            f"Reference produced {len(chunks)} chunks, exceeding max_gists_per_reference={max_chunks}."
        )
    if policy == "truncate":
        return chunks[:max_chunks], discarded
    if policy != "merge_tail":
        raise ValueError(f"Unsupported gist overflow policy: {policy}")
    if max_chunks == 1:
        start = 0
        prefix = []
    else:
        prefix = chunks[: max_chunks - 1]
        start = chunks[max_chunks - 1].token_start
    tail_ids = tuple(all_token_ids[start:])
    tail = replace(
        chunks[max_chunks - 1],
        chunk_id=_chunk_id(chunks[0].source_uri, max_chunks - 1),
        text=tokenizer.decode(tail_ids) if tail_ids else text,
        token_ids=tail_ids,
        token_start=start,
        token_end=len(all_token_ids),
        char_end=len(text),
        metadata={**chunks[max_chunks - 1].metadata, "merged_tail_chunks": discarded + 1},
    )
    return [*prefix, tail], discarded


def partition_reference(uri: str, text: str, tokenizer, config, metadata=None) -> list[ReferenceChunk]:
    """Partition one resolved URI into the chunks independently routed by PRA.

    ``none`` keeps one document chunk, ``fixed`` creates optional overlapping
    token windows, ``markers`` preserves explicit/Markdown structure, and
    ``semantic`` delegates to a configured plugin. The returned source offsets
    let evaluation distinguish retrieval quality from answer quality.
    """
    metadata = dict(metadata or {})
    token_ids = list(tokenizer.encode(text))
    if not token_ids:
        return []

    # Construct candidate chunks according to one mutually exclusive mode.
    if config.chunking_mode == "none":
        chunks = [
            ReferenceChunk(
                chunk_id=_chunk_id(uri, 0),
                source_uri=uri,
                text=text,
                token_ids=tuple(token_ids),
                token_start=0,
                token_end=len(token_ids),
                char_start=0,
                char_end=len(text),
                metadata=metadata,
            )
        ]
    elif config.chunking_mode == "fixed":
        chunks = []
        step = config.fixed_chunk_tokens - config.fixed_chunk_overlap_tokens
        for index, start in enumerate(range(0, len(token_ids), step)):
            end = min(start + config.fixed_chunk_tokens, len(token_ids))
            ids = tuple(token_ids[start:end])
            if ids:
                chunks.append(
                    ReferenceChunk(
                        chunk_id=_chunk_id(uri, index),
                        source_uri=uri,
                        text=tokenizer.decode(ids),
                        token_ids=ids,
                        token_start=start,
                        token_end=end,
                        metadata={**metadata, "overlap_tokens": config.fixed_chunk_overlap_tokens},
                    )
                )
            if end == len(token_ids):
                break
    elif config.chunking_mode == "markers":
        partitioner = metadata.get("partitioner")
        if partitioner is None:
            partitioner = (
                MarkdownHeadingPartitioner()
                if metadata.get("format") == "markdown"
                else ExplicitMarkerPartitioner(config.marker_rules)
            )
        chunks = _char_chunks(uri, text, tokenizer, partitioner.partition(uri, text, metadata), metadata)
    elif config.chunking_mode == "semantic":
        if config.semantic_chunker is None:
            raise NotImplementedError(
                "chunking_mode='semantic' requires an explicit semantic_chunker implementation."
            )
        chunks = list(
            config.semantic_chunker.partition(
                uri, text, token_ids, metadata, config.max_gists_per_reference
            )
        )
    else:
        raise ValueError(f"Unsupported chunking_mode: {config.chunking_mode}")

    # Enforce the routing-gist budget after partitioning so every mode behaves alike.
    chunks, discarded = _apply_overflow(
        chunks,
        config.max_gists_per_reference,
        config.gist_overflow_policy,
        token_ids,
        text,
        tokenizer,
    )
    if discarded:
        chunks = [
            replace(chunk, metadata={**chunk.metadata, "discarded_chunk_count": discarded})
            for chunk in chunks
        ]
    return chunks
