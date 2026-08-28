"""Independent cheap indexes and bounded fusion for large typed records.

Full-body native Q/K is intentionally optional.  Typed postings, BM25, and a
small deterministic embedding index remain usable when the native size gate
skips or defers model-derived indexing.  Every hit carries an exact selector
back into the authorized backing record, which lets the runtime encode only a
selected region natively after discovery.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .agent_resources import hashed_semantic_vector
from .channel_geometry import reciprocal_rank_fusion


_TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]+")
_COLLECTIONS = ("rows", "items", "events", "results", "nodes", "edges", "chunks")


class LargeRecordChannel(str, Enum):
    """Retrieval representations that can address one retained record."""

    TYPED = "typed"
    BM25 = "bm25"
    EMBEDDING = "embedding"
    NATIVE_QK = "native_qk"


class LargeRecordSearchPolicy(str, Enum):
    """Public policy for resolving available large-record search channels."""

    AUTO = "AUTO"
    HYBRID = "HYBRID"
    LEXICAL_ONLY = "LEXICAL_ONLY"
    EMBEDDING_ONLY = "EMBEDDING_ONLY"
    NATIVE_ONLY = "NATIVE_ONLY"
    EXPLICIT = "EXPLICIT"


class IndexLifecycleState(str, Enum):
    """Lifecycle state tracked independently for each record representation."""

    NOT_BUILT = "NOT_BUILT"
    BUILT = "BUILT"
    SKIPPED_SIZE_LIMIT = "SKIPPED_SIZE_LIMIT"
    DEFERRED = "DEFERRED"
    LAZY = "LAZY"


@dataclass(frozen=True)
class IndexComponentState:
    """Build state and measured cost for one index component."""

    state: IndexLifecycleState
    build_latency_ms: float = 0.0
    index_bytes: int = 0
    units: int = 0


@dataclass(frozen=True)
class RecordIndexLifecycle:
    """Independent lifecycle of prompt, address, native, and detail views."""

    typed_index: IndexComponentState
    bm25_index: IndexComponentState
    embedding_index: IndexComponentState
    summary_view: IndexComponentState
    native_qk_index: IndexComponentState
    detail_kv: IndexComponentState


@dataclass(frozen=True)
class LargeRecordUnit:
    """Smallest independently recoverable unit and its exact selector."""

    unit_id: str
    selector: Mapping[str, object]
    text: str
    type_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class LargeRecordHit:
    """A fused recoverable unit; scores are diagnostic, not prompt content."""

    unit_id: str
    selector: Mapping[str, object]
    text: str
    fused_score: float
    channel_ranks: Mapping[str, int]


@dataclass(frozen=True)
class LargeRecordSearchTrace:
    """Auditable channel resolution, fusion, and search-cost accounting."""

    requested_policy: LargeRecordSearchPolicy
    resolved_channels: tuple[LargeRecordChannel, ...]
    fusion: str
    candidates_scored: Mapping[str, int]
    query_latency_ms: float
    index_bytes: Mapping[str, int]


@dataclass(frozen=True)
class LargeRecordSearchResult:
    """Ranked exact selectors plus the trace that produced them."""

    hits: tuple[LargeRecordHit, ...]
    trace: LargeRecordSearchTrace


def _terms(value: object) -> tuple[str, ...]:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return tuple(token.casefold() for token in _TOKEN.findall(text))


def _unit_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str
    )


def extract_large_record_units(payload: object) -> tuple[LargeRecordUnit, ...]:
    """Split payload into bounded units while preserving exact selectors."""

    units: list[LargeRecordUnit] = []
    if isinstance(payload, Mapping):
        for name in _COLLECTIONS:
            values = payload.get(name)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for index, value in enumerate(values):
                    typed = tuple(str(key).casefold() for key in value) if isinstance(value, Mapping) else ()
                    units.append(LargeRecordUnit(
                        f"{name}:{index}", {"collection": name, "range": [index, index + 1]},
                        _unit_text(value), typed,
                    ))
        for name, value in payload.items():
            if name in _COLLECTIONS:
                continue
            if isinstance(value, str) and "\n" in value:
                for index, line in enumerate(value.splitlines()):
                    units.append(LargeRecordUnit(
                        f"{name}:line:{index}", {"field": str(name), "lines": [index, index + 1]},
                        line, (str(name).casefold(),),
                    ))
            else:
                units.append(LargeRecordUnit(
                    f"field:{name}", {"fields": [str(name)]},
                    f"{name}: {_unit_text(value)}", (str(name).casefold(),),
                ))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            units.append(LargeRecordUnit(
                f"item:{index}", {"items": [index, index + 1]}, _unit_text(value)
            ))
    else:
        lines = _unit_text(payload).splitlines() or [""]
        units.extend(
            LargeRecordUnit(f"line:{index}", {"lines": [index, index + 1]}, line)
            for index, line in enumerate(lines)
        )
    return tuple(units)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class LargeRecordIndex:
    """Per-record typed/BM25/embedding index with bounded RRF fusion."""

    def __init__(self, payload: object, *, embedding_dimensions: int = 128) -> None:
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive.")
        started = time.perf_counter()
        self.units = extract_large_record_units(payload)
        self.by_id = {unit.unit_id: unit for unit in self.units}
        self.embedding_dimensions = embedding_dimensions
        self.term_postings: dict[str, set[str]] = defaultdict(set)
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        self.embeddings: dict[str, tuple[float, ...]] = {}
        for unit in self.units:
            frequencies = Counter(_terms(unit.text))
            self.term_frequencies[unit.unit_id] = frequencies
            self.document_frequency.update(frequencies)
            for term in set(frequencies).union(unit.type_terms):
                self.term_postings[term].add(unit.unit_id)
            self.embeddings[unit.unit_id] = tuple(
                hashed_semantic_vector(unit.text, dimensions=embedding_dimensions)
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        typed_bytes = sum(len(term.encode()) + 8 * len(ids) for term, ids in self.term_postings.items())
        bm25_bytes = sum(len(freq) * 16 for freq in self.term_frequencies.values())
        embedding_bytes = len(self.embeddings) * embedding_dimensions * 4
        per = elapsed / 3.0
        self.component_states = {
            LargeRecordChannel.TYPED: IndexComponentState(IndexLifecycleState.BUILT, per, typed_bytes, len(self.units)),
            LargeRecordChannel.BM25: IndexComponentState(IndexLifecycleState.BUILT, per, bm25_bytes, len(self.units)),
            LargeRecordChannel.EMBEDDING: IndexComponentState(IndexLifecycleState.BUILT, per, embedding_bytes, len(self.units)),
        }

    @property
    def index_bytes(self) -> Mapping[str, int]:
        return {channel.value: state.index_bytes for channel, state in self.component_states.items()}

    def _typed(self, query: str) -> dict[str, float]:
        scores: Counter[str] = Counter()
        query_terms = set(_terms(query))
        for term in query_terms:
            for unit_id in self.term_postings.get(term, ()):
                scores[unit_id] += 1.0
        for unit in self.units:
            overlap = len(query_terms.intersection(unit.type_terms))
            if overlap:
                scores[unit.unit_id] += 0.35 * overlap
        return {unit_id: score for unit_id, score in scores.items() if score > 0}

    def _bm25(self, query: str) -> dict[str, float]:
        query_terms = set(_terms(query))
        count = max(len(self.units), 1)
        lengths = {unit_id: sum(freq.values()) for unit_id, freq in self.term_frequencies.items()}
        average = sum(lengths.values()) / count if lengths else 1.0
        scores: dict[str, float] = {}
        for unit_id, frequencies in self.term_frequencies.items():
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * lengths[unit_id] / max(average, 1.0))
                score += idf * tf * 2.2 / denominator
            if score:
                scores[unit_id] = score
        return scores

    def _embedding(self, query: str) -> dict[str, float]:
        query_vector = hashed_semantic_vector(query, dimensions=self.embedding_dimensions)
        return {
            unit_id: score
            for unit_id, vector in self.embeddings.items()
            for score in [_cosine(query_vector, vector)]
            if score > 0.0
        }

    def search(
        self,
        query: str,
        *,
        policy: LargeRecordSearchPolicy | str = LargeRecordSearchPolicy.AUTO,
        channels: Sequence[LargeRecordChannel | str] | None = None,
        top_k: int = 4,
        candidate_limit: int = 64,
        native_ranking: Mapping[str, int] | None = None,
    ) -> LargeRecordSearchResult:
        """Search available channels and fuse ranks without raw-score mixing."""

        if top_k <= 0 or candidate_limit <= 0:
            raise ValueError("top_k and candidate_limit must be positive.")
        started = time.perf_counter()
        policy = LargeRecordSearchPolicy(policy)
        available = [LargeRecordChannel.TYPED, LargeRecordChannel.BM25, LargeRecordChannel.EMBEDDING]
        if native_ranking is not None:
            available.append(LargeRecordChannel.NATIVE_QK)
        if policy is LargeRecordSearchPolicy.EXPLICIT:
            if not channels:
                raise ValueError("EXPLICIT search requires channels.")
            resolved = tuple(LargeRecordChannel(value) for value in channels)
        elif policy is LargeRecordSearchPolicy.LEXICAL_ONLY:
            resolved = (LargeRecordChannel.TYPED, LargeRecordChannel.BM25)
        elif policy is LargeRecordSearchPolicy.EMBEDDING_ONLY:
            resolved = (LargeRecordChannel.EMBEDDING,)
        elif policy is LargeRecordSearchPolicy.NATIVE_ONLY:
            resolved = (LargeRecordChannel.NATIVE_QK,)
        else:
            resolved = tuple(available)
        missing = set(resolved).difference(available)
        if missing:
            raise RuntimeError(f"Requested indexes are unavailable: {sorted(value.value for value in missing)}")
        score_maps: dict[LargeRecordChannel, Mapping[str, float]] = {}
        for channel in resolved:
            if channel is LargeRecordChannel.TYPED:
                score_maps[channel] = self._typed(query)
            elif channel is LargeRecordChannel.BM25:
                score_maps[channel] = self._bm25(query)
            elif channel is LargeRecordChannel.EMBEDDING:
                score_maps[channel] = self._embedding(query)
            else:
                score_maps[channel] = {
                    unit_id: 1.0 / max(rank, 1) for unit_id, rank in (native_ranking or {}).items()
                }
        rankings: dict[str, dict[str, int]] = {}
        for channel, scores in score_maps.items():
            ordered = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))[:candidate_limit]
            rankings[channel.value] = {unit_id: rank for rank, unit_id in enumerate(ordered, 1)}
        fused = reciprocal_rank_fusion(rankings)
        chosen = sorted(fused, key=lambda unit_id: (-fused[unit_id], unit_id))[:top_k]
        hits = tuple(
            LargeRecordHit(
                unit_id,
                dict(self.by_id[unit_id].selector),
                self.by_id[unit_id].text,
                fused[unit_id],
                {channel: ranking[unit_id] for channel, ranking in rankings.items() if unit_id in ranking},
            )
            for unit_id in chosen if unit_id in self.by_id
        )
        trace = LargeRecordSearchTrace(
            policy,
            resolved,
            "rrf" if len(resolved) > 1 else "single_channel_rank",
            {channel.value: len(score_maps[channel]) for channel in resolved},
            (time.perf_counter() - started) * 1000.0,
            self.index_bytes,
        )
        return LargeRecordSearchResult(hits, trace)
