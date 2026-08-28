"""Record-bounded native materialization geometry for frozen PRA selections.

Routing anchors remain small and immutable.  This module expands their logical
source spans, merges overlap within each reference, and maps the resulting
interval union to layer-native K/V without crossing a reference boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence

from pra_torch.memory import (
    LayerKV,
    PRACacheEntry,
    ReferenceChunkMemory,
    SelectedChunk,
)


@dataclass(frozen=True)
class FrozenNativeAnchor:
    """One immutable routing hit expressed in source-local token coordinates."""

    reference_uri: str
    chunk_id: str
    logical_start: int
    logical_end: int
    reference_score: float = 0.0
    chunk_score: float = 0.0
    reference_rank: int = 1
    rank_within_reference: int = 1

    def __post_init__(self) -> None:
        if not self.reference_uri or not self.chunk_id:
            raise ValueError("A frozen anchor requires reference and chunk identities.")
        if self.logical_start < 0 or self.logical_end <= self.logical_start:
            raise ValueError("A frozen anchor requires a non-empty logical interval.")


@dataclass(frozen=True, order=True)
class NativeInterval:
    """Half-open source-token interval belonging to exactly one reference."""

    reference_uri: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.reference_uri or self.start < 0 or self.end <= self.start:
            raise ValueError("Native intervals must be non-empty and source-local.")

    @property
    def tokens(self) -> int:
        return self.end - self.start


class NativeMaterializationMode(str, Enum):
    """How frozen routing anchors become record-local native K/V intervals."""

    SELECTED_CHUNK = "selected_chunk"
    EXPANDED_WINDOW = "expanded_window"
    FULL_SELECTED_RECORD = "full_selected_record"
    FULL_SCOPE = "full_scope"


@dataclass(frozen=True)
class NativeMaterializationProfile:
    """Named, ordinary configuration for reproducible materialization geometry."""

    name: str
    routing_chunk_tokens: int
    routing_chunk_overlap_tokens: int
    mode: NativeMaterializationMode
    left_context_tokens: int = 0
    right_context_tokens: int = 0


MATERIALIZATION_PROFILES: Mapping[str, NativeMaterializationProfile] = {
    # Frozen Paper 3 used 32-token, non-overlapping parent chunks and a
    # zero-radius evidence-centered materialization. Consumer layers remain a
    # dataset/model choice and are therefore intentionally not hidden here.
    "paper3_default": NativeMaterializationProfile(
        "paper3_default", 32, 0, NativeMaterializationMode.EXPANDED_WINDOW
    ),
    "paper7_selected_detail": NativeMaterializationProfile(
        "paper7_selected_detail", 32, 0, NativeMaterializationMode.SELECTED_CHUNK
    ),
    "paper8_full_record_diagnostic": NativeMaterializationProfile(
        "paper8_full_record_diagnostic",
        32,
        0,
        NativeMaterializationMode.FULL_SELECTED_RECORD,
    ),
}


def materialization_profile(name: str) -> NativeMaterializationProfile:
    """Resolve a canonical research profile without embedding magic constants."""

    try:
        return MATERIALIZATION_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown native materialization profile {name!r}; "
            f"choose one of {sorted(MATERIALIZATION_PROFILES)}."
        ) from error


@dataclass(frozen=True)
class FrozenNativeSelection:
    """Replayable routing decision whose anchors do not change across a sweep."""

    anchors: tuple[FrozenNativeAnchor, ...]

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("A frozen native selection cannot be empty.")

    @property
    def source_identity(self) -> tuple[tuple[str, str, int, int], ...]:
        return tuple(
            (row.reference_uri, row.chunk_id, row.logical_start, row.logical_end)
            for row in self.anchors
        )


@dataclass(frozen=True)
class NativeMaterializationPlan:
    """Merged logical intervals and their exact layer-native K/V selections."""

    frozen: FrozenNativeSelection
    intervals: tuple[NativeInterval, ...]
    selections_by_layer: Mapping[int, tuple[SelectedChunk, ...]]
    record_token_counts: Mapping[str, int]
    target_span_tokens: int | None
    full_selected_record: bool
    raw_interval_count: int
    raw_native_tokens: int

    @property
    def unique_native_tokens(self) -> int:
        return sum(interval.tokens for interval in self.intervals)

    @property
    def consumption_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.selections_by_layer))

    @property
    def query_position_offset(self) -> int:
        """Place query positions after the longest independent source record."""

        return max(self.record_token_counts.values(), default=0)

    @property
    def overlap_removed_tokens(self) -> int:
        """Return duplicate source positions removed by interval normalization."""

        return self.raw_native_tokens - self.unique_native_tokens

    @property
    def duplication_ratio(self) -> float:
        """Return raw selected positions divided by the unique source union."""

        return self.raw_native_tokens / max(self.unique_native_tokens, 1)


@dataclass(frozen=True)
class EvidenceTokenIntervals:
    """Answer and minimally interpretable evidence spans in one source record."""

    answer: tuple[int, int]
    semantic: tuple[int, int]
    full_record: tuple[int, int]


def _merge_intervals(rows: Iterable[NativeInterval]) -> tuple[NativeInterval, ...]:
    merged: list[NativeInterval] = []
    for row in sorted(rows):
        if merged and merged[-1].reference_uri == row.reference_uri and row.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = NativeInterval(previous.reference_uri, previous.start, max(previous.end, row.end))
        else:
            merged.append(row)
    return tuple(merged)


def expand_frozen_intervals(
    frozen: FrozenNativeSelection,
    record_token_counts: Mapping[str, int],
    *,
    target_span_tokens: int | None = None,
    full_selected_record: bool = False,
    left_context_tokens: int | None = None,
    right_context_tokens: int | None = None,
) -> tuple[NativeInterval, ...]:
    """Expand each anchor symmetrically, clamp to its record, then merge overlap."""

    if target_span_tokens is not None and target_span_tokens <= 0:
        raise ValueError("target_span_tokens must be positive or None.")
    if full_selected_record and target_span_tokens is not None:
        raise ValueError("Full-record and fixed-width materialization are mutually exclusive.")
    if left_context_tokens is not None and left_context_tokens < 0:
        raise ValueError("left_context_tokens must be non-negative or None.")
    if right_context_tokens is not None and right_context_tokens < 0:
        raise ValueError("right_context_tokens must be non-negative or None.")
    expanded = []
    for anchor in frozen.anchors:
        record_end = int(record_token_counts[anchor.reference_uri])
        if record_end < anchor.logical_end:
            raise ValueError("Frozen anchor extends beyond its owning record.")
        if full_selected_record:
            start, end = 0, record_end
        elif left_context_tokens is not None or right_context_tokens is not None:
            start = max(0, anchor.logical_start - int(left_context_tokens or 0))
            end = min(record_end, anchor.logical_end + int(right_context_tokens or 0))
        elif target_span_tokens is None or target_span_tokens <= anchor.logical_end - anchor.logical_start:
            start, end = anchor.logical_start, anchor.logical_end
        else:
            extra = target_span_tokens - (anchor.logical_end - anchor.logical_start)
            left = extra // 2
            right = extra - left
            start = max(0, anchor.logical_start - left)
            end = min(record_end, anchor.logical_end + right)
            missing = target_span_tokens - (end - start)
            if missing > 0 and start == 0:
                end = min(record_end, end + missing)
            elif missing > 0 and end == record_end:
                start = max(0, start - missing)
        expanded.append(NativeInterval(anchor.reference_uri, start, end))
    return _merge_intervals(expanded)


def _slice_layer_kv(kv: LayerKV, start: int, end: int) -> LayerKV:
    positions = kv.position_ids
    if positions is not None:
        positions = positions[..., start:end]
    return LayerKV(
        kv.k[:, :, start:end, :],
        kv.v[:, :, start:end, :],
        position_ids=positions,
        position_state=kv.position_state,
    )


def _anchor_for_interval(
    anchors: Sequence[FrozenNativeAnchor], interval: NativeInterval
) -> FrozenNativeAnchor:
    candidates = [row for row in anchors if row.reference_uri == interval.reference_uri]
    return max(candidates, key=lambda row: (row.chunk_score, -row.reference_rank, -row.rank_within_reference))


def _slice_interval(
    entry: PRACacheEntry,
    layer_id: int,
    interval: NativeInterval,
    anchor: FrozenNativeAnchor,
) -> list[SelectedChunk]:
    memory = entry.layer_memory.get(layer_id)
    if memory is None:
        raise ValueError(f"Reference {entry.uri} has no K/V at layer {layer_id}.")
    chunks = sorted(memory.chunks, key=lambda row: (row.logical_start, row.logical_end, row.chunk_id))
    selected: list[SelectedChunk] = []
    cursor = interval.start
    while cursor < interval.end:
        covering = [
            chunk for chunk in chunks
            if chunk.logical_start <= cursor < int(chunk.logical_end)
        ]
        if not covering:
            raise ValueError(
                f"Layer {layer_id} has a K/V gap at token {cursor} for {entry.uri}."
            )
        source = max(covering, key=lambda row: (int(row.logical_end), -row.logical_start))
        if source.token_kv is None:
            raise ValueError(
                f"Layer {layer_id} has routing addresses but no detail K/V for {entry.uri}."
            )
        take_end = min(interval.end, int(source.logical_end))
        local_start = cursor - source.logical_start
        local_end = take_end - source.logical_start
        sliced = ReferenceChunkMemory(
            chunk_id=f"{entry.uri}#materialized={cursor}:{take_end}:layer={layer_id}",
            source_uri=source.source_uri,
            token_start=cursor,
            token_end=take_end,
            token_kv=_slice_layer_kv(source.token_kv, local_start, local_end),
            routing_gist=source.routing_gist,
            char_start=source.char_start,
            char_end=source.char_end,
            metadata={
                **source.metadata,
                "frozen_anchor_chunk_id": anchor.chunk_id,
                "materialization_geometry": "record_bounded_interval",
            },
            logical_start=cursor,
            logical_end=take_end,
        )
        selected.append(SelectedChunk(
            entry=entry,
            chunk=sliced,
            reference_score=anchor.reference_score,
            chunk_score=anchor.chunk_score,
            layer_id=layer_id,
            reference_rank=anchor.reference_rank,
            rank_within_reference=anchor.rank_within_reference,
            metadata={"frozen_routing": True, "source_chunk_id": source.chunk_id},
        ))
        cursor = take_end
    return selected


def build_native_materialization_plan(
    entries: Sequence[PRACacheEntry],
    frozen: FrozenNativeSelection,
    *,
    consumption_layers: Sequence[int],
    target_span_tokens: int | None = None,
    full_selected_record: bool = False,
    left_context_tokens: int | None = None,
    right_context_tokens: int | None = None,
) -> NativeMaterializationPlan:
    """Construct unique record-local K/V slices for each requested consumer layer."""

    by_uri = {entry.uri: entry for entry in entries}
    missing = {row.reference_uri for row in frozen.anchors}.difference(by_uri)
    if missing:
        raise ValueError(f"Frozen references are absent from the active cache: {sorted(missing)}")
    counts = {
        uri: max(
            int(chunk.logical_end)
            for memory in entry.layer_memory.values()
            for chunk in memory.chunks
        )
        for uri, entry in by_uri.items()
        if uri in {row.reference_uri for row in frozen.anchors}
    }
    raw_intervals = tuple(
        interval
        for anchor in frozen.anchors
        for interval in expand_frozen_intervals(
            FrozenNativeSelection((anchor,)),
            counts,
            target_span_tokens=target_span_tokens,
            full_selected_record=full_selected_record,
            left_context_tokens=left_context_tokens,
            right_context_tokens=right_context_tokens,
        )
    )
    intervals = _merge_intervals(raw_intervals)
    layers = tuple(sorted({int(layer) for layer in consumption_layers}))
    if not layers:
        raise ValueError("At least one consumption layer is required.")
    selections: dict[int, tuple[SelectedChunk, ...]] = {}
    for layer_id in layers:
        rows: list[SelectedChunk] = []
        for interval in intervals:
            anchor = _anchor_for_interval(frozen.anchors, interval)
            rows.extend(_slice_interval(by_uri[interval.reference_uri], layer_id, interval, anchor))
        selections[layer_id] = tuple(rows)
    return NativeMaterializationPlan(
        frozen,
        intervals,
        selections,
        counts,
        target_span_tokens,
        full_selected_record,
        len(raw_intervals),
        sum(row.tokens for row in raw_intervals),
    )


def intervals_cover(
    intervals: Sequence[NativeInterval], reference_uri: str, target: tuple[int, int]
) -> bool:
    """Return whether a merged materialization union fully covers one target span."""

    start, end = target
    return any(row.reference_uri == reference_uri and row.start <= start and row.end >= end for row in intervals)


def _token_offset(tokenizer, text: str, character_offset: int) -> int:
    return len(tokenizer(text[:character_offset], add_special_tokens=False).input_ids)


def evidence_token_intervals(
    tokenizer,
    text: str,
    *,
    answer: str,
    semantic_anchors: Sequence[str],
) -> EvidenceTokenIntervals:
    """Derive deterministic answer and semantic source intervals from annotations."""

    lowered = text.lower()
    answer_start = lowered.find(answer.lower())
    if answer_start < 0:
        raise ValueError(f"Answer annotation {answer!r} is absent from the source record.")
    spans = [(answer_start, answer_start + len(answer))]
    for value in semantic_anchors:
        start = lowered.find(value.lower())
        if start < 0:
            raise ValueError(f"Semantic anchor {value!r} is absent from the source record.")
        spans.append((start, start + len(value)))
    answer_tokens = (
        _token_offset(tokenizer, text, spans[0][0]),
        _token_offset(tokenizer, text, spans[0][1]),
    )
    semantic_chars = (min(row[0] for row in spans), max(row[1] for row in spans))
    semantic_tokens = (
        _token_offset(tokenizer, text, semantic_chars[0]),
        _token_offset(tokenizer, text, semantic_chars[1]),
    )
    total = len(tokenizer(text, add_special_tokens=False).input_ids)
    return EvidenceTokenIntervals(answer_tokens, semantic_tokens, (0, total))
