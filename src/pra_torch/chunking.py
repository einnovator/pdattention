"""Deterministic, provenance-preserving partitioning for PRA references."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Protocol


@dataclass(frozen=True)
class ChunkingConfig:
    """Reusable source-partition policy for encoding blocks or routing chunks."""

    mode: str = "fixed"
    chunk_tokens: int | None = None
    overlap_fraction: float = 0.0
    overlap_tokens: int = 0
    markers: tuple[str, ...] = ("<PRA_CHUNK>",)
    semantic_chunker: object | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "fixed", "markers", "semantic"}:
            raise ValueError(f"Unsupported chunking mode: {self.mode}")
        if self.chunk_tokens is not None and int(self.chunk_tokens) <= 0:
            raise ValueError("chunk_tokens must be positive or None.")
        if int(self.overlap_tokens) < 0:
            raise ValueError("overlap_tokens must be non-negative.")
        if not 0.0 <= float(self.overlap_fraction) < 1.0:
            raise ValueError("overlap_fraction must satisfy 0 <= value < 1.")
        if self.overlap_tokens and self.overlap_fraction:
            raise ValueError("Configure overlap_tokens or overlap_fraction, not both.")
        if self.mode == "semantic" and self.semantic_chunker is None:
            raise NotImplementedError(
                "mode='semantic' requires an explicit semantic_chunker implementation."
            )

    @classmethod
    def from_value(cls, value) -> "ChunkingConfig":
        """Normalize YAML dictionaries and already-constructed policies."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            normalized = dict(value)
            if "markers" in normalized:
                normalized["markers"] = tuple(normalized["markers"] or ())
            return cls(**normalized)
        raise TypeError("Chunking configuration must be a mapping or ChunkingConfig.")

    def resolved_overlap_tokens(self, chunk_tokens: int | None = None) -> int:
        """Resolve token/fraction overlap for a concrete partition size."""
        if self.overlap_tokens:
            return int(self.overlap_tokens)
        size = int(chunk_tokens or self.chunk_tokens or 0)
        if self.overlap_fraction <= 0.0 or size <= 0:
            return 0
        return max(1, int(size * self.overlap_fraction))


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
    logical_start: int = -1  # Inclusive coordinate within this continuous source.
    logical_end: int | None = None  # Exclusive coordinate; start + length when contiguous.

    def __post_init__(self) -> None:
        """Default logical coordinates to source-token offsets and validate continuity."""
        if self.token_start < 0 or self.token_end < self.token_start:
            raise ValueError("Chunk token offsets must satisfy 0 <= token_start <= token_end.")
        logical_start = self.token_start if self.logical_start < 0 else int(self.logical_start)
        logical_end = (
            logical_start + len(self.token_ids)
            if self.logical_end is None
            else int(self.logical_end)
        )
        if logical_start < 0 or logical_end - logical_start != len(self.token_ids):
            raise ValueError("Logical chunk offsets must cover exactly the contiguous token IDs.")
        object.__setattr__(self, "logical_start", logical_start)
        object.__setattr__(self, "logical_end", logical_end)


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
    if max_chunks is None:
        return chunks, 0
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
        logical_start=start,
        logical_end=len(all_token_ids),
        char_end=len(text),
        metadata={**chunks[max_chunks - 1].metadata, "merged_tail_chunks": discarded + 1},
    )
    return [*prefix, tail], discarded


def _fixed_chunks(uri, token_ids, tokenizer, metadata, *, size, overlap=0):
    """Create exact token windows without decoding and re-tokenizing their contents."""
    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, len(token_ids), step)):
        end = min(start + size, len(token_ids))
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
                    metadata={
                        **metadata,
                        "overlap_tokens": overlap,
                        "unique_source_tokens": len(token_ids),
                    },
                )
            )
        if end == len(token_ids):
            break
    return chunks


def _split_oversized_chunks(chunks, tokenizer, max_chunk_tokens):
    """Bound encoded chunk length while preserving every token and source offset."""
    if max_chunk_tokens is None:
        return chunks
    bounded = []
    for chunk in chunks:
        if len(chunk.token_ids) <= max_chunk_tokens:
            bounded.append(replace(chunk, chunk_id=_chunk_id(chunk.source_uri, len(bounded))))
            continue
        for local_start in range(0, len(chunk.token_ids), max_chunk_tokens):
            ids = tuple(chunk.token_ids[local_start : local_start + max_chunk_tokens])
            bounded.append(
                replace(
                    chunk,
                    chunk_id=_chunk_id(chunk.source_uri, len(bounded)),
                    text=tokenizer.decode(ids),
                    token_ids=ids,
                    token_start=chunk.token_start + local_start,
                    token_end=chunk.token_start + local_start + len(ids),
                    logical_start=chunk.logical_start + local_start,
                    logical_end=chunk.logical_start + local_start + len(ids),
                    char_start=None,
                    char_end=None,
                    metadata={**chunk.metadata, "bounded_token_split": True},
                )
            )
    return bounded


def partition_reference_tokens(
    uri: str,
    token_ids,
    tokenizer,
    config,
    metadata=None,
    *,
    text: str | None = None,
    max_chunks: int | None = None,
    use_configured_max_chunks: bool = True,
    max_chunk_tokens: int | None = None,
) -> list[ReferenceChunk]:
    """Partition exact source token IDs with optional source-specific limits.

    ``use_configured_max_chunks=False`` makes ``max_chunks=None`` mean unlimited,
    which is used by implicit prompt history. Text-origin references retain the
    ordinary ``max_gists_per_reference`` cap through the default arguments.
    """
    routing = getattr(config, "routing_chunking_config", None)
    if routing is None:
        routing = ChunkingConfig(
            mode=config.chunking_mode,
            chunk_tokens=config.fixed_chunk_tokens,
            overlap_fraction=config.chunk_overlap_fraction,
            overlap_tokens=config.fixed_chunk_overlap_tokens,
            markers=tuple(config.marker_rules),
            semantic_chunker=config.semantic_chunker,
        )
    return partition_source(
        uri,
        token_ids,
        tokenizer,
        routing,
        metadata,
        text=text,
        max_chunks=(
            config.max_gists_per_reference
            if use_configured_max_chunks
            else max_chunks
        ),
        overflow_policy=config.gist_overflow_policy,
        max_chunk_tokens=max_chunk_tokens,
    )


def partition_source(
    uri: str,
    token_ids,
    tokenizer,
    chunking: ChunkingConfig | dict,
    metadata=None,
    *,
    text: str | None = None,
    max_chunks: int | None = None,
    overflow_policy: str = "truncate",
    max_chunk_tokens: int | None = None,
) -> list[ReferenceChunk]:
    """Partition exact IDs with one policy shared by encoding and routing stages."""
    chunking = ChunkingConfig.from_value(chunking)
    metadata = dict(metadata or {})
    token_ids = list(int(token_id) for token_id in token_ids)
    if not token_ids:
        return []
    text = tokenizer.decode(token_ids) if text is None else text

    # Construct candidate chunks according to one mutually exclusive mode.
    if chunking.mode == "none":
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
    elif chunking.mode == "fixed":
        size = int(chunking.chunk_tokens or max_chunk_tokens or len(token_ids))
        chunks = _fixed_chunks(
            uri,
            token_ids,
            tokenizer,
            metadata,
            size=size,
            overlap=chunking.resolved_overlap_tokens(size),
        )
    elif chunking.mode == "markers":
        partitioner = metadata.get("partitioner")
        if partitioner is None:
            partitioner = (
                MarkdownHeadingPartitioner()
                if metadata.get("format") == "markdown"
                else ExplicitMarkerPartitioner(chunking.markers)
            )
        chunks = _char_chunks(
            uri,
            text,
            tokenizer,
            partitioner.partition(uri, text, metadata),
            metadata,
        )
        reconstructed = tuple(chain.from_iterable(chunk.token_ids for chunk in chunks))
        if reconstructed != tuple(token_ids):
            fallback_size = max_chunk_tokens or chunking.chunk_tokens or len(token_ids)
            chunks = _fixed_chunks(
                uri,
                token_ids,
                tokenizer,
                {**metadata, "token_exact_marker_fallback": True},
                size=fallback_size,
            )
    elif chunking.mode == "semantic":
        plugin_limit = max_chunks if max_chunks is not None else len(token_ids)
        chunks = list(
            chunking.semantic_chunker.partition(uri, text, token_ids, metadata, plugin_limit)
        )
    else:
        raise ValueError(f"Unsupported chunking mode: {chunking.mode}")

    chunks = _split_oversized_chunks(chunks, tokenizer, max_chunk_tokens)
    chunks, discarded = _apply_overflow(
        chunks,
        max_chunks,
        overflow_policy,
        token_ids,
        text,
        tokenizer,
    )
    if discarded:
        chunks = [
            replace(chunk, metadata={**chunk.metadata, "discarded_chunk_count": discarded})
            for chunk in chunks
        ]
    encoded_tokens = sum(len(chunk.token_ids) for chunk in chunks)
    unique_tokens = len(
        {
            position
            for chunk in chunks
            for position in range(chunk.token_start, chunk.token_end)
        }
    )
    chunks = [
        replace(
            chunk,
            metadata={
                **chunk.metadata,
                "chunk_overlap_fraction": chunking.overlap_fraction,
                "resolved_chunk_overlap_tokens": chunking.resolved_overlap_tokens(
                    chunking.chunk_tokens
                ),
                "encoded_tokens_including_overlap": encoded_tokens,
                "stored_kv_tokens_including_overlap": encoded_tokens,
                "covered_unique_source_tokens": unique_tokens,
                "duplication_factor": encoded_tokens / max(unique_tokens, 1),
            },
        )
        for chunk in chunks
    ]
    return chunks


def partition_reference(uri: str, text: str, tokenizer, config, metadata=None) -> list[ReferenceChunk]:
    """Partition one resolved URI into the chunks independently routed by PRA.

    ``none`` keeps one document chunk, ``fixed`` creates optional overlapping
    token windows, ``markers`` preserves explicit/Markdown structure, and
    ``semantic`` delegates to a configured plugin. The returned source offsets
    let evaluation distinguish retrieval quality from answer quality.
    """
    token_ids = list(tokenizer.encode(text))
    return partition_reference_tokens(
        uri,
        token_ids,
        tokenizer,
        config,
        metadata,
        text=text,
    )
