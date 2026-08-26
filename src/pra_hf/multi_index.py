"""Fixed-budget routing over typed address views of one native-memory store.

Every channel scores the same immutable source-parent identities.  The helpers
in this module combine only those address scores; selected integer indices are
resolved to original native K/V by the caller.  No sidecar text is returned as
answer evidence.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from pra_hf.summary_index import lexical_terms, minmax, stable_topk


_ENTITY = re.compile(
    r"\b(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)+|[A-Z]{2,}(?:[-_][A-Z0-9]+)*|"
    r"[A-Za-z]+[-_][A-Za-z0-9_-]+)\b"
)
_NUMBER_DATE = re.compile(
    r"\b(?:\d{1,4}(?:[-/.]\d{1,2}){1,2}|(?:19|20)\d{2}|\d+(?:\.\d+)?%?)\b"
)
_RELATION_TERMS = frozenset(
    {
        "authored",
        "born",
        "capital",
        "caused",
        "contains",
        "created",
        "directed",
        "founded",
        "located",
        "married",
        "member",
        "owned",
        "part",
        "published",
        "replaced",
        "served",
        "succeeded",
        "won",
        "wrote",
    }
)
_STOP_TERMS = frozenset(
    {
        "about",
        "after",
        "also",
        "among",
        "been",
        "before",
        "being",
        "between",
        "could",
        "from",
        "have",
        "into",
        "more",
        "most",
        "other",
        "over",
        "such",
        "than",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "through",
        "under",
        "using",
        "were",
        "which",
        "with",
        "would",
    }
)


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return tuple(output)


@dataclass(frozen=True)
class TypedAddressSidecar:
    """Lossless-looking terms extracted into explicit routing fields.

    Extraction is deterministic but not semantically lossless: regex entities
    and relation terms can be missed.  The source URI remains authoritative;
    this record is only a compact, inspectable address view.
    """

    entities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    numbers_dates: tuple[str, ...] = ()
    rare_terms: tuple[str, ...] = ()
    high_idf_ngrams: tuple[str, ...] = ()
    relation_terms: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """Serialize typed fields for lexical scoring without source prose."""

        fields = (
            ("entity", self.entities),
            ("alias", self.aliases),
            ("number_date", self.numbers_dates),
            ("rare", self.rare_terms),
            ("ngram", self.high_idf_ngrams),
            ("relation", self.relation_terms),
        )
        return " ; ".join(
            f"{label}: {' '.join(values)}" for label, values in fields if values
        ) or "empty typed address"

    @property
    def text_bytes(self) -> int:
        """Return UTF-8 persistent bytes, excluding shared source/native K/V."""

        return len(self.text.encode("utf-8"))


def extract_typed_sidecars(
    chunks: Sequence[str],
    *,
    aliases_by_chunk: Mapping[int, Sequence[str]] | None = None,
    max_terms_per_field: int = 12,
) -> tuple[TypedAddressSidecar, ...]:
    """Build entity/rare-term sidecars for aligned source chunks.

    Corpus rarity is computed only within the source associated with one query.
    Original spellings are retained for inspectability while downstream BM25
    applies the same Unicode/case normalization as other lexical channels.
    """

    if max_terms_per_field <= 0:
        raise ValueError("max_terms_per_field must be positive.")
    aliases_by_chunk = aliases_by_chunk or {}
    token_rows = [lexical_terms(chunk) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for terms in token_rows:
        document_frequency.update(set(terms))

    output = []
    for index, (chunk, terms) in enumerate(zip(chunks, token_rows)):
        entities = _stable_unique(_ENTITY.findall(chunk))[:max_terms_per_field]
        aliases = _stable_unique(aliases_by_chunk.get(index, ()))[:max_terms_per_field]
        numbers = _stable_unique(_NUMBER_DATE.findall(chunk))[:max_terms_per_field]
        rare = _stable_unique(
            term
            for term in terms
            if len(term) >= 4
            and term not in _STOP_TERMS
            and document_frequency[term] <= 1
        )[:max_terms_per_field]
        rare_set = set(rare)
        ngrams = _stable_unique(
            f"{left} {right}"
            for left, right in zip(terms, terms[1:])
            if left in rare_set or right in rare_set
        )[:max_terms_per_field]
        relations = _stable_unique(
            term
            for term in terms
            if term in _RELATION_TERMS
            or (len(term) >= 6 and term.endswith(("ed", "ing")))
        )[:max_terms_per_field]
        output.append(
            TypedAddressSidecar(
                entities=entities,
                aliases=aliases,
                numbers_dates=numbers,
                rare_terms=rare,
                high_idf_ngrams=ngrams,
                relation_terms=relations,
            )
        )
    return tuple(output)


def _score_arrays(
    channel_scores: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if not channel_scores:
        raise ValueError("At least one address channel is required.")
    arrays = {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in channel_scores.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Address channels must have the same non-zero candidate count.")
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Address channel scores must be finite one-dimensional arrays.")
    return arrays


def channel_rankings(
    channel_scores: Mapping[str, Sequence[float]],
) -> dict[str, tuple[int, ...]]:
    """Return complete deterministic rankings for every address channel."""

    arrays = _score_arrays(channel_scores)
    return {name: stable_topk(values, len(values)) for name, values in arrays.items()}


def candidate_provenance(
    channel_scores: Mapping[str, Sequence[float]],
) -> tuple[dict[str, dict[str, float | int]], ...]:
    """Retain each candidate's score and one-based rank in every channel."""

    arrays = _score_arrays(channel_scores)
    rankings = channel_rankings(arrays)
    inverse = {
        name: {candidate: rank for rank, candidate in enumerate(order, start=1)}
        for name, order in rankings.items()
    }
    count = len(next(iter(arrays.values())))
    return tuple(
        {
            name: {"score": float(values[candidate]), "rank": inverse[name][candidate]}
            for name, values in arrays.items()
        }
        for candidate in range(count)
    )


def rank_round_robin(
    channel_scores: Mapping[str, Sequence[float]],
    k: int,
) -> tuple[int, ...]:
    """Take the highest unseen address from each channel until ``k`` fills."""

    rankings = channel_rankings(channel_scores)
    selected: list[int] = []
    depth = 0
    while len(selected) < min(k, len(next(iter(rankings.values())))):
        advanced = False
        for ranking in rankings.values():
            if depth < len(ranking):
                candidate = ranking[depth]
                if candidate not in selected:
                    selected.append(candidate)
                    if len(selected) == k:
                        break
                advanced = True
        if not advanced:
            break
        depth += 1
    return tuple(selected)


def reserved_slot_union(
    channel_scores: Mapping[str, Sequence[float]],
    allocations: Mapping[str, int],
    *,
    k: int,
) -> tuple[int, ...]:
    """Reserve interpretable per-channel slots, then fill remaining slots fairly."""

    rankings = channel_rankings(channel_scores)
    unknown = set(allocations) - set(rankings)
    if unknown:
        raise ValueError(f"Reserved allocations reference unknown channels: {sorted(unknown)}")
    if any(int(value) < 0 for value in allocations.values()):
        raise ValueError("Reserved slot counts cannot be negative.")
    if sum(int(value) for value in allocations.values()) > k:
        raise ValueError("Reserved slots cannot exceed the final materialization budget.")

    selected: list[int] = []
    for name, count in allocations.items():
        added = 0
        for candidate in rankings[name]:
            if candidate not in selected:
                selected.append(candidate)
                added += 1
                if added >= count:
                    break
    if len(selected) < k:
        remainder = rank_round_robin(channel_scores, len(next(iter(rankings.values()))))
        selected.extend(candidate for candidate in remainder if candidate not in selected)
    return tuple(selected[:k])


def agreement_priority_union(
    channel_scores: Mapping[str, Sequence[float]],
    k: int,
    *,
    candidate_pool: int | None = None,
) -> tuple[int, ...]:
    """Prioritize candidates appearing near the top of multiple channels."""

    if k <= 0:
        return ()
    rankings = channel_rankings(channel_scores)
    count = len(next(iter(rankings.values())))
    pool = min(candidate_pool or k, count)
    inverse = {
        name: {candidate: rank for rank, candidate in enumerate(order, start=1)}
        for name, order in rankings.items()
    }
    candidates = set().union(*(set(order[:pool]) for order in rankings.values()))
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -sum(candidate in order[:pool] for order in rankings.values()),
            sum(inverse[name][candidate] for name in rankings),
            min(inverse[name][candidate] for name in rankings),
            candidate,
        ),
    )
    if len(ordered) < min(k, count):
        fallback = rank_round_robin(channel_scores, count)
        ordered.extend(candidate for candidate in fallback if candidate not in ordered)
    return tuple(ordered[:k])


def reciprocal_rank_fusion_scores(
    channel_scores: Mapping[str, Sequence[float]],
    *,
    constant: float = 60.0,
) -> np.ndarray:
    """Fuse incompatible channel scales using reciprocal ranks only."""

    if constant <= 0:
        raise ValueError("RRF constant must be positive.")
    rankings = channel_rankings(channel_scores)
    count = len(next(iter(rankings.values())))
    fused = np.zeros(count, dtype=np.float64)
    for ranking in rankings.values():
        for rank, candidate in enumerate(ranking, start=1):
            fused[candidate] += 1.0 / (constant + rank)
    return fused


def normalized_score_fusion(
    channel_scores: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Fuse min-max normalized channels with explicit non-negative weights."""

    arrays = _score_arrays(channel_scores)
    if set(weights) != set(arrays):
        raise ValueError("Fusion weights must name every and only active channel.")
    total = sum(float(value) for value in weights.values())
    if total <= 0 or any(float(value) < 0 or not math.isfinite(float(value)) for value in weights.values()):
        raise ValueError("Fusion weights must be finite, non-negative, and sum above zero.")
    fused = np.zeros(len(next(iter(arrays.values()))), dtype=np.float64)
    for name, values in arrays.items():
        fused += (float(weights[name]) / total) * minmax(values)
    return fused
