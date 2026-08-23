"""Token-native and hybrid candidate scoring for bounded PRA discovery.

The index in this module is a sidecar to :class:`pra_hf.iterative.GistIndex`.
It preserves exactly the same URI/chunk ordering and contains no native K/V.
The iterative router remains responsible for budgets, traversal, and graph
construction; this module only describes how each candidate is scored.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

import torch


DISCOVERY_MODES = {
    "gist_only",
    "bm25",
    "token_exact",
    "token_weighted",
    "token_approx",
    "union",
    "token_semantic_rerank",
    "semantic_token_rerank",
    "cascade",
    "iterative_hybrid",
}


def _normalize_piece(piece: str) -> str:
    """Normalize a tokenizer piece without retokenizing source text."""
    piece = piece.removeprefix("##").lstrip("\u0120\u2581")
    piece = unicodedata.normalize("NFKC", piece).casefold()
    return re.sub(r"[^\w]+", "", piece, flags=re.UNICODE)


def _word_terms(text: str) -> tuple[str, ...]:
    """Return decoded word terms used only by the classical BM25 baseline."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _longest_common_span(left: Sequence[Any], right: Sequence[Any]) -> tuple[int, int, int]:
    """Return ``(length, left_start, right_start)`` for the longest exact span."""
    if not left or not right:
        return 0, -1, -1
    previous = [0] * (len(right) + 1)
    best = (0, -1, -1)
    for left_index, left_value in enumerate(left):
        current = [0] * (len(right) + 1)
        for right_index, right_value in enumerate(right):
            if left_value == right_value:
                length = previous[right_index] + 1
                current[right_index + 1] = length
                candidate = (length, left_index - length + 1, right_index - length + 1)
                if candidate[0] > best[0]:
                    best = candidate
        previous = current
    return best


def _ordered_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """Compute normalized longest-common-subsequence overlap for token order."""
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for right_index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(previous[right_index], current[-1]))
        previous = current
    return previous[-1] / max(1, min(len(left), len(right)))


@dataclass(frozen=True)
class TokenChunkRecord:
    """Tokenizer-native lookup data for one gist-index chunk identity."""

    reference_uri: str
    chunk_id: str
    layer_id: int
    token_ids: tuple[int, ...]
    normalized_tokens: tuple[str, ...]
    bm25_terms: tuple[str, ...]
    aliases: tuple[str, ...]
    token_start: int
    token_end: int


@dataclass(frozen=True)
class DiscoveryCandidate:
    """Auditable score record emitted before bounded candidate admission."""

    reference_uri: str
    chunk_id: str
    layer_id: int
    semantic_score: float
    exact_span_score: float
    normalized_exact_score: float
    weighted_overlap_score: float
    ordered_score: float
    approximate_score: float
    embedding_score: float | None
    bm25_score: float
    entity_name_score: float
    explicit_score: float
    associative_score: float
    hop: int
    parent_id: str
    raw_exact_span: tuple[int, int] | None
    normalized_exact_span: tuple[int, int] | None
    selected_score: float
    selected_channel: str
    confidence: float
    provenance: tuple[str, ...]
    rank: int | None = None
    confidence_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible candidate provenance for retrieval graphs."""
        return asdict(self)


@dataclass(frozen=True)
class HybridDiscoveryPolicy:
    """Choose how token-native and semantic evidence compete at each hop."""

    mode: str = "iterative_hybrid"
    semantic_weight: float = 0.65
    token_weight: float = 0.35
    later_semantic_weight: float = 0.25
    later_token_weight: float = 0.75
    exact_min_tokens: int = 2
    cascade_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in DISCOVERY_MODES:
            raise ValueError(f"Unsupported hybrid discovery mode: {self.mode}")
        for name in (
            "semantic_weight",
            "token_weight",
            "later_semantic_weight",
            "later_token_weight",
            "cascade_threshold",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.exact_min_tokens <= 0:
            raise ValueError("exact_min_tokens must be positive.")


class TokenNativeIndex:
    """Lexical index whose rows are identity-aligned with a semantic gist index."""

    def __init__(
        self,
        records: Iterable[TokenChunkRecord],
        *,
        special_token_ids: Iterable[int] = (),
    ) -> None:
        self.records = tuple(records)
        self.special_token_ids = frozenset(int(value) for value in special_token_ids)
        document_frequency: Counter[str] = Counter()
        for record in self.records:
            document_frequency.update(set(record.normalized_tokens))
        count = max(len(self.records), 1)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        bm25_document_frequency: Counter[str] = Counter()
        for record in self.records:
            bm25_document_frequency.update(set(record.bm25_terms))
        self.bm25_idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in bm25_document_frequency.items()
        }
        lengths = [len(record.bm25_terms) for record in self.records]
        self.average_length = sum(lengths) / max(len(lengths), 1)

    @classmethod
    def from_gist_index(
        cls,
        gist_index,
        tokenizer,
        *,
        aliases: dict[str, Iterable[str]] | None = None,
    ) -> "TokenNativeIndex":
        """Tokenize each source once, then slice rows using cache token offsets."""
        aliases = aliases or {}
        source_ids: dict[str, tuple[int, ...]] = {}
        records: list[TokenChunkRecord] = []
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        for entry, chunk in gist_index.records:
            if entry.uri not in source_ids:
                encoded = tokenizer(entry.text, add_special_tokens=False)
                values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
                if isinstance(values, torch.Tensor):
                    values = values.reshape(-1).tolist()
                elif values and isinstance(values[0], (list, tuple)):
                    values = values[0]
                source_ids[entry.uri] = tuple(int(value) for value in values)
            token_ids = source_ids[entry.uri][chunk.token_start : chunk.token_end]
            pieces = tokenizer.convert_ids_to_tokens(list(token_ids))
            normalized = tuple(
                value
                for value in (_normalize_piece(str(piece)) for piece in pieces)
                if value
            )
            metadata_aliases = entry.metadata.get("aliases", ())
            if isinstance(metadata_aliases, str):
                metadata_aliases = (metadata_aliases,)
            provided_aliases = aliases.get(entry.uri, ())
            if isinstance(provided_aliases, str):
                provided_aliases = (provided_aliases,)
            title = entry.metadata.get("title")
            names = [*provided_aliases, *metadata_aliases]
            if title:
                names.append(str(title))
            names.append(entry.uri.rsplit("/", 1)[-1])
            records.append(
                TokenChunkRecord(
                    reference_uri=entry.uri,
                    chunk_id=chunk.chunk_id,
                    layer_id=gist_index.layer_id,
                    token_ids=tuple(value for value in token_ids if value not in special_ids),
                    normalized_tokens=normalized,
                    bm25_terms=_word_terms(tokenizer.decode(list(token_ids))),
                    aliases=tuple(dict.fromkeys(str(value).casefold() for value in names if value)),
                    token_start=int(chunk.token_start),
                    token_end=int(chunk.token_end),
                )
            )
        return cls(records, special_token_ids=special_ids)

    def validate_alignment(self, gist_index) -> None:
        """Reject stale sidecars before their scores can select the wrong K/V identity."""
        expected = tuple(
            (entry.uri, chunk.chunk_id, gist_index.layer_id)
            for entry, chunk in gist_index.records
        )
        actual = tuple(
            (record.reference_uri, record.chunk_id, record.layer_id)
            for record in self.records
        )
        if actual != expected:
            raise ValueError("TokenNativeIndex rows do not align with GistIndex records.")

    def _query_tokens(
        self, token_ids: Iterable[int], tokenizer
    ) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
        raw = tuple(int(value) for value in token_ids if int(value) not in self.special_token_ids)
        pieces = tokenizer.convert_ids_to_tokens(list(raw))
        normalized = tuple(
            value for value in (_normalize_piece(str(piece)) for piece in pieces) if value
        )
        return raw, normalized, _word_terms(tokenizer.decode(list(raw)))

    def _bm25(self, query: Sequence[str], record: TokenChunkRecord) -> float:
        frequencies = Counter(record.bm25_terms)
        length = len(record.bm25_terms)
        score = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if not frequency:
                continue
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / max(self.average_length, 1.0)
            )
            score += self.bm25_idf.get(token, 0.0) * frequency * 2.2 / denominator
        return score

    def score(
        self,
        query_token_ids: Iterable[int],
        semantic_scores: torch.Tensor | Sequence[float],
        tokenizer,
        policy: HybridDiscoveryPolicy,
        *,
        hop: int,
        parent_id: str,
        explicit_reference_uris: set[str] | None = None,
    ) -> list[DiscoveryCandidate]:
        """Score every aligned row while retaining channel-level provenance."""
        raw_query, normalized_query, bm25_query = self._query_tokens(
            query_token_ids, tokenizer
        )
        semantic_values = [float(value) for value in semantic_scores]
        if len(semantic_values) != len(self.records):
            raise ValueError("semantic_scores must contain one value per token-index row.")
        raw_rows: list[dict[str, Any]] = []
        bm25_values = [self._bm25(bm25_query, record) for record in self.records]
        bm25_scale = max(max(bm25_values, default=0.0), 1e-12)
        explicit_reference_uris = explicit_reference_uris or set()
        query_text = " ".join(normalized_query)
        for index, record in enumerate(self.records):
            raw_length, _, raw_start = _longest_common_span(raw_query, record.token_ids)
            norm_length, _, norm_start = _longest_common_span(
                normalized_query, record.normalized_tokens
            )
            denominator = max(1, min(8, len(record.normalized_tokens)))
            raw_exact = (
                min(1.0, raw_length / denominator)
                if raw_length >= policy.exact_min_tokens
                else 0.0
            )
            normalized_exact = (
                min(1.0, norm_length / denominator)
                if norm_length >= policy.exact_min_tokens
                else 0.0
            )
            intersection = set(normalized_query).intersection(record.normalized_tokens)
            weighted_denominator = sum(
                self.idf.get(token, 0.0) for token in set(record.normalized_tokens)
            )
            weighted = sum(self.idf.get(token, 0.0) for token in intersection) / max(
                weighted_denominator, 1e-12
            )
            ordered = _ordered_overlap(normalized_query, record.normalized_tokens)
            approximate = SequenceMatcher(
                None, normalized_query, record.normalized_tokens, autojunk=False
            ).ratio()
            entity = float(any(alias and alias in query_text for alias in record.aliases))
            explicit = float(record.reference_uri in explicit_reference_uris)
            semantic = max(0.0, min(1.0, (semantic_values[index] + 1.0) / 2.0))
            bm25 = bm25_values[index] / bm25_scale
            token = max(
                raw_exact,
                normalized_exact,
                0.55 * weighted + 0.25 * ordered + 0.20 * entity,
            )
            raw_rows.append(
                {
                    "record": record,
                    "semantic": semantic,
                    "raw_exact": raw_exact,
                    "normalized_exact": normalized_exact,
                    "weighted": weighted,
                    "ordered": ordered,
                    "approximate": approximate,
                    "bm25": bm25,
                    "entity": entity,
                    "explicit": explicit,
                    "token": token,
                    "raw_span": (raw_start, raw_start + raw_length) if raw_length else None,
                    "normalized_span": (
                        (norm_start, norm_start + norm_length) if norm_length else None
                    ),
                }
            )

        cascade_channel = "semantic"
        for channel in (
            "explicit",
            "entity",
            "raw_exact",
            "normalized_exact",
            "weighted",
            "approximate",
        ):
            if max((row[channel] for row in raw_rows), default=0.0) >= policy.cascade_threshold:
                cascade_channel = channel
                break

        candidates = []
        for row in raw_rows:
            mode = policy.mode
            token = row["token"]
            if mode == "gist_only":
                selected, channel = row["semantic"], "semantic"
            elif mode == "bm25":
                selected, channel = row["bm25"], "bm25"
            elif mode == "token_exact":
                selected = max(
                    row["raw_exact"],
                    row["normalized_exact"],
                    row["entity"],
                    row["explicit"],
                )
                channel = "token_exact"
            elif mode == "token_weighted":
                selected = max(token, row["explicit"])
                channel = "token_weighted"
            elif mode == "token_approx":
                selected = max(0.45 * token + 0.55 * row["approximate"], row["explicit"])
                channel = "token_approximate"
            elif mode == "union":
                selected = max(row["semantic"], token, row["explicit"])
                channel = "union"
            elif mode == "token_semantic_rerank":
                selected = 0.70 * token + 0.30 * row["semantic"]
                channel = "token_then_semantic"
            elif mode == "semantic_token_rerank":
                selected = 0.70 * row["semantic"] + 0.30 * token
                channel = "semantic_then_token"
            elif mode == "cascade":
                selected = row[cascade_channel]
                channel = f"cascade:{cascade_channel}"
            else:
                semantic_weight = (
                    policy.semantic_weight if hop == 1 else policy.later_semantic_weight
                )
                token_weight = policy.token_weight if hop == 1 else policy.later_token_weight
                selected = (
                    semantic_weight * row["semantic"]
                    + token_weight * max(token, row["explicit"])
                ) / max(semantic_weight + token_weight, 1e-12)
                channel = "hybrid_entry" if hop == 1 else "hybrid_associative"
            record = row["record"]
            candidates.append(
                DiscoveryCandidate(
                    reference_uri=record.reference_uri,
                    chunk_id=record.chunk_id,
                    layer_id=record.layer_id,
                    semantic_score=row["semantic"],
                    exact_span_score=row["raw_exact"],
                    normalized_exact_score=row["normalized_exact"],
                    weighted_overlap_score=row["weighted"],
                    ordered_score=row["ordered"],
                    approximate_score=row["approximate"],
                    embedding_score=None,
                    bm25_score=row["bm25"],
                    entity_name_score=row["entity"],
                    explicit_score=row["explicit"],
                    associative_score=token if hop > 1 else 0.0,
                    hop=hop,
                    parent_id=parent_id,
                    raw_exact_span=row["raw_span"],
                    normalized_exact_span=row["normalized_span"],
                    selected_score=max(0.0, min(1.0, selected)),
                    selected_channel=channel,
                    confidence=max(0.0, min(1.0, selected)),
                    provenance=tuple(
                        name
                        for name, value in (
                            ("semantic", row["semantic"]),
                            ("exact", row["raw_exact"]),
                            ("normalized_exact", row["normalized_exact"]),
                            ("weighted_overlap", row["weighted"]),
                            ("ordered", row["ordered"]),
                            ("approximate", row["approximate"]),
                            ("bm25", row["bm25"]),
                            ("entity_name", row["entity"]),
                            ("explicit", row["explicit"]),
                            ("associative", token if hop > 1 else 0.0),
                        )
                        if value > 0.0
                    ),
                )
            )
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-candidates[index].selected_score, index),
        )
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        return [replace(candidate, rank=ranks[index]) for index, candidate in enumerate(candidates)]
