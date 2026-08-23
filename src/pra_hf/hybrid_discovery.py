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
from collections import Counter, defaultdict
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
    "token_ngram",
    "token_edit",
    "token_embedding",
    "union",
    "token_semantic_rerank",
    "semantic_token_rerank",
    "cascade",
    "iterative_hybrid",
}

_FIXED_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


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


def _ngrams(values: Sequence[Any], sizes: Sequence[int]) -> tuple[tuple[Any, ...], ...]:
    """Return stable contiguous n-grams for posting-list lookup and scoring."""
    return tuple(
        tuple(values[start : start + size])
        for size in sizes
        if size > 0 and len(values) >= size
        for start in range(len(values) - size + 1)
    )


def _bounded_edit_similarity(
    left: Sequence[str], right: Sequence[str], max_distance: int
) -> float:
    """Return normalized token edit similarity, or zero outside the bound."""
    if not left or not right or abs(len(left) - len(right)) > max_distance:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
            row_minimum = min(row_minimum, current[-1])
        if row_minimum > max_distance:
            return 0.0
        previous = current
    distance = previous[-1]
    if distance > max_distance:
        return 0.0
    return 1.0 - distance / max(len(left), len(right), 1)


def _best_ngram_edit_similarity(
    left: Sequence[str],
    right: Sequence[str],
    sizes: Sequence[int],
    max_distance: int,
) -> float:
    """Find a bounded fuzzy match between short contiguous token spans."""
    left_ngrams = _ngrams(left, sizes)
    right_ngrams = _ngrams(right, sizes)
    return max(
        (
            _bounded_edit_similarity(left_ngram, right_ngram, max_distance)
            for left_ngram in left_ngrams
            for right_ngram in right_ngrams
            if abs(len(left_ngram) - len(right_ngram)) <= max_distance
        ),
        default=0.0,
    )


def _automatic_aliases(text: str) -> tuple[str, ...]:
    """Extract conservative name-like spans without consulting task labels."""
    normalized = unicodedata.normalize("NFKC", text)
    phrases = re.findall(
        r"\b(?:[A-Z][\w'-]*|[A-Z]{2,})(?:\s+(?:[A-Z][\w'-]*|[A-Z]{2,})){0,4}\b",
        normalized,
        flags=re.UNICODE,
    )
    identifiers = re.findall(r"\b[\w]+(?:[-_/.:][\w]+)+\b", normalized, flags=re.UNICODE)
    return tuple(
        dict.fromkeys(
            value.casefold().strip()
            for value in (*phrases, *identifiers)
            if len(value.strip()) >= 2
        )
    )


def _character_ngrams(value: str, size: int = 3) -> tuple[str, ...]:
    """Return boundary-aware character n-grams for fuzzy candidate narrowing."""
    padded = f"^{value}$"
    return tuple(
        padded[start : start + size]
        for start in range(max(0, len(padded) - size + 1))
    )


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
    token_ngrams: tuple[tuple[int, ...], ...] = ()
    normalized_ngrams: tuple[tuple[str, ...], ...] = ()


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
    ngram_score: float
    edit_score: float
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
    indexed: bool = False
    candidate_pool_size: int = 64
    ngram_sizes: tuple[int, ...] = (2, 3)
    approximate_max_distance: int = 2
    stop_token_strategy: str = "idf"
    automatic_aliases: bool = False
    enable_extended_channels: bool = False
    embedding_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.mode not in DISCOVERY_MODES:
            raise ValueError(f"Unsupported hybrid discovery mode: {self.mode}")
        for name in (
            "semantic_weight",
            "token_weight",
            "later_semantic_weight",
            "later_token_weight",
            "cascade_threshold",
            "embedding_weight",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.exact_min_tokens <= 0:
            raise ValueError("exact_min_tokens must be positive.")
        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive.")
        if not self.ngram_sizes or any(size <= 0 for size in self.ngram_sizes):
            raise ValueError("ngram_sizes must contain positive values.")
        if self.approximate_max_distance < 0:
            raise ValueError("approximate_max_distance cannot be negative.")
        if self.stop_token_strategy not in {"none", "fixed", "idf"}:
            raise ValueError("stop_token_strategy must be 'none', 'fixed', or 'idf'.")


class TokenNativeIndex:
    """Lexical index whose rows are identity-aligned with a semantic gist index."""

    def __init__(
        self,
        records: Iterable[TokenChunkRecord],
        *,
        special_token_ids: Iterable[int] = (),
        ngram_sizes: Sequence[int] = (2, 3),
        token_embedding_weight: torch.Tensor | None = None,
    ) -> None:
        self.records = tuple(records)
        self.special_token_ids = frozenset(int(value) for value in special_token_ids)
        self.ngram_sizes = tuple(sorted(set(int(value) for value in ngram_sizes)))
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
        raw_postings: dict[int, set[int]] = defaultdict(set)
        raw_ngram_postings: dict[tuple[int, ...], set[int]] = defaultdict(set)
        normalized_postings: dict[str, set[int]] = defaultdict(set)
        ngram_postings: dict[tuple[str, ...], set[int]] = defaultdict(set)
        approximate_postings: dict[str, set[int]] = defaultdict(set)
        bm25_postings: dict[str, set[int]] = defaultdict(set)
        uri_postings: dict[str, set[int]] = defaultdict(set)
        for index, record in enumerate(self.records):
            for token in set(record.token_ids):
                raw_postings[token].add(index)
            raw_ngrams = record.token_ngrams or _ngrams(
                record.token_ids, self.ngram_sizes
            )
            for ngram in set(raw_ngrams):
                raw_ngram_postings[ngram].add(index)
            for token in set(record.normalized_tokens):
                normalized_postings[token].add(index)
            normalized_ngrams = record.normalized_ngrams or _ngrams(
                record.normalized_tokens, self.ngram_sizes
            )
            for ngram in set(normalized_ngrams):
                ngram_postings[ngram].add(index)
            for value in {*record.normalized_tokens, *record.aliases}:
                for ngram in _character_ngrams(value):
                    approximate_postings[ngram].add(index)
            for term in set(record.bm25_terms):
                bm25_postings[term].add(index)
            uri_postings[record.reference_uri].add(index)
        self.raw_postings = {key: tuple(sorted(value)) for key, value in raw_postings.items()}
        self.raw_ngram_postings = {
            key: tuple(sorted(value)) for key, value in raw_ngram_postings.items()
        }
        self.normalized_postings = {
            key: tuple(sorted(value)) for key, value in normalized_postings.items()
        }
        self.ngram_postings = {
            key: tuple(sorted(value)) for key, value in ngram_postings.items()
        }
        self.approximate_postings = {
            key: tuple(sorted(value)) for key, value in approximate_postings.items()
        }
        self.bm25_postings = {key: tuple(sorted(value)) for key, value in bm25_postings.items()}
        self.uri_postings = {key: tuple(sorted(value)) for key, value in uri_postings.items()}
        self.chunk_embeddings: torch.Tensor | None = None
        if token_embedding_weight is not None:
            weight = token_embedding_weight.detach()
            vectors = []
            width = int(weight.shape[1])
            for record in self.records:
                ids = [value for value in record.token_ids if 0 <= value < weight.shape[0]]
                if ids:
                    vector = weight[torch.tensor(ids, device=weight.device)].float().mean(dim=0)
                else:
                    vector = torch.zeros(width, device=weight.device)
                vectors.append(vector.cpu())
            if vectors:
                self.chunk_embeddings = torch.nn.functional.normalize(
                    torch.stack(vectors), dim=-1, eps=1e-12
                )
        self.last_search_stats: dict[str, int | float | bool] = {}

    @classmethod
    def from_gist_index(
        cls,
        gist_index,
        tokenizer,
        *,
        aliases: dict[str, Iterable[str]] | None = None,
        automatic_aliases: bool = False,
        ngram_sizes: Sequence[int] = (2, 3),
        token_embedding_weight: torch.Tensor | None = None,
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
            decoded = tokenizer.decode(list(token_ids))
            if automatic_aliases:
                names.extend(_automatic_aliases(decoded))
            records.append(
                TokenChunkRecord(
                    reference_uri=entry.uri,
                    chunk_id=chunk.chunk_id,
                    layer_id=gist_index.layer_id,
                    token_ids=tuple(value for value in token_ids if value not in special_ids),
                    normalized_tokens=normalized,
                    bm25_terms=_word_terms(decoded),
                    aliases=tuple(dict.fromkeys(str(value).casefold() for value in names if value)),
                    token_start=int(chunk.token_start),
                    token_end=int(chunk.token_end),
                    token_ngrams=_ngrams(token_ids, ngram_sizes),
                    normalized_ngrams=_ngrams(normalized, ngram_sizes),
                )
            )
        return cls(
            records,
            special_token_ids=special_ids,
            ngram_sizes=ngram_sizes,
            token_embedding_weight=token_embedding_weight,
        )

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

    def _embedding_scores(
        self,
        raw_query: Sequence[int],
        token_embedding_weight: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compare mean input-token embeddings without retaining the model matrix."""
        if (
            self.chunk_embeddings is None
            or token_embedding_weight is None
            or not raw_query
        ):
            return None
        weight = token_embedding_weight.detach()
        ids = [value for value in raw_query if 0 <= value < weight.shape[0]]
        if not ids:
            return None
        query = weight[torch.tensor(ids, device=weight.device)].float().mean(dim=0).cpu()
        query = torch.nn.functional.normalize(query, dim=0, eps=1e-12)
        cosine = self.chunk_embeddings @ query
        return (cosine + 1.0).mul(0.5).clamp(0.0, 1.0)

    def _candidate_indices(
        self,
        raw_query: Sequence[int],
        normalized_query: Sequence[str],
        bm25_query: Sequence[str],
        semantic_values: Sequence[float],
        policy: HybridDiscoveryPolicy,
        explicit_reference_uris: set[str],
    ) -> tuple[set[int], dict[str, int | float | bool]]:
        """Use postings plus semantic top-k to bound expensive candidate scoring."""
        if not policy.indexed or len(self.records) <= policy.candidate_pool_size:
            rows = set(range(len(self.records)))
            return rows, {
                "indexed": bool(policy.indexed),
                "candidate_rows": len(rows),
                "total_rows": len(self.records),
                "posting_hits": len(rows),
            }
        votes: Counter[int] = Counter()
        explicit_rows: set[int] = set()
        for uri in explicit_reference_uris:
            explicit_rows.update(self.uri_postings.get(uri, ()))
        for token in set(raw_query):
            for row in self.raw_postings.get(token, ()):
                votes[row] += 4
        for ngram in set(_ngrams(raw_query, policy.ngram_sizes)):
            for row in self.raw_ngram_postings.get(ngram, ()):
                votes[row] += 8 * len(ngram)
        for token in set(normalized_query):
            if policy.stop_token_strategy == "fixed" and token in _FIXED_STOP_TOKENS:
                continue
            weight = max(1, round(3 * self.idf.get(token, 0.0)))
            for row in self.normalized_postings.get(token, ()):
                votes[row] += weight
        for ngram in set(_ngrams(normalized_query, policy.ngram_sizes)):
            for row in self.ngram_postings.get(ngram, ()):
                votes[row] += 6 * len(ngram)
        for term in set(bm25_query):
            if policy.stop_token_strategy == "fixed" and term in _FIXED_STOP_TOKENS:
                continue
            weight = max(1, round(2 * self.bm25_idf.get(term, 0.0)))
            for row in self.bm25_postings.get(term, ()):
                votes[row] += weight
        for value in set(normalized_query):
            for ngram in _character_ngrams(value):
                for row in self.approximate_postings.get(ngram, ()):
                    votes[row] += 1
        lexical_limit = max(1, policy.candidate_pool_size // 2)
        lexical = [
            row
            for row, _ in sorted(votes.items(), key=lambda item: (-item[1], item[0]))[
                :lexical_limit
            ]
        ]
        semantic_limit = max(1, policy.candidate_pool_size - len(lexical))
        semantic = sorted(
            range(len(self.records)), key=lambda row: (-semantic_values[row], row)
        )[:semantic_limit]
        rows = set(lexical) | set(semantic) | explicit_rows
        return rows, {
            "indexed": True,
            "candidate_rows": len(rows),
            "total_rows": len(self.records),
            "posting_hits": len(votes),
            "explicit_rows": len(explicit_rows),
        }

    def memory_bytes(self) -> int:
        """Return a conservative payload-byte estimate for index comparisons."""
        token_bytes = sum(
            8 * (len(record.token_ids) + len(record.normalized_tokens))
            + sum(len(value.encode("utf-8")) for value in record.bm25_terms)
            + sum(len(value.encode("utf-8")) for value in record.aliases)
            for record in self.records
        )
        posting_bytes = 8 * sum(
            len(rows)
            for table in (
                self.raw_postings,
                self.raw_ngram_postings,
                self.normalized_postings,
                self.ngram_postings,
                self.approximate_postings,
                self.bm25_postings,
                self.uri_postings,
            )
            for rows in table.values()
        )
        embedding_bytes = (
            self.chunk_embeddings.numel() * self.chunk_embeddings.element_size()
            if self.chunk_embeddings is not None
            else 0
        )
        return int(token_bytes + posting_bytes + embedding_bytes)

    def extended(
        self,
        records: Iterable[TokenChunkRecord],
        *,
        token_embedding_weight: torch.Tensor | None = None,
    ) -> "TokenNativeIndex":
        """Build an atomically swappable index with an additional row batch.

        IDF and BM25 statistics depend on the complete corpus. This explicit
        baseline therefore recomputes them instead of retaining stale weights;
        callers can measure the update cost without assuming mutable indexing.
        """
        return type(self)(
            (*self.records, *tuple(records)),
            special_token_ids=self.special_token_ids,
            ngram_sizes=self.ngram_sizes,
            token_embedding_weight=token_embedding_weight,
        )

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
        token_embedding_weight: torch.Tensor | None = None,
        sparse: bool = False,
    ) -> list[DiscoveryCandidate] | dict[int, DiscoveryCandidate]:
        """Score rows, returning an index-keyed sparse map for production routing."""
        raw_query, normalized_query, bm25_query = self._query_tokens(
            query_token_ids, tokenizer
        )
        semantic_values = [float(value) for value in semantic_scores]
        if len(semantic_values) != len(self.records):
            raise ValueError("semantic_scores must contain one value per token-index row.")
        explicit_reference_uris = explicit_reference_uris or set()
        candidate_rows, search_stats = self._candidate_indices(
            raw_query,
            normalized_query,
            bm25_query,
            semantic_values,
            policy,
            explicit_reference_uris,
        )
        evaluated_rows = (
            sorted(candidate_rows) if sparse else list(range(len(self.records)))
        )
        bm25_values = {
            index: self._bm25(bm25_query, self.records[index])
            for index in evaluated_rows
            if index in candidate_rows
        }
        bm25_scale = max(max(bm25_values.values(), default=0.0), 1e-12)
        embedding_scores = self._embedding_scores(raw_query, token_embedding_weight)
        query_ngrams = set(_ngrams(normalized_query, policy.ngram_sizes))
        query_text = " ".join(normalized_query)
        raw_rows: list[tuple[int, dict[str, Any]]] = []
        for index in evaluated_rows:
            record = self.records[index]
            evaluated = index in candidate_rows
            raw_length, _, raw_start = (
                _longest_common_span(raw_query, record.token_ids)
                if evaluated
                else (0, -1, -1)
            )
            norm_length, _, norm_start = (
                _longest_common_span(normalized_query, record.normalized_tokens)
                if evaluated
                else (0, -1, -1)
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
            query_content = set(normalized_query)
            record_content = set(record.normalized_tokens)
            if policy.stop_token_strategy == "fixed":
                query_content -= _FIXED_STOP_TOKENS
                record_content -= _FIXED_STOP_TOKENS
            intersection = query_content & record_content if evaluated else set()
            if policy.stop_token_strategy == "idf":
                weighted_denominator = sum(
                    self.idf.get(token, 0.0) for token in record_content
                )
                weighted = sum(
                    self.idf.get(token, 0.0) for token in intersection
                ) / max(weighted_denominator, 1e-12)
            else:
                weighted = len(intersection) / max(len(record_content), 1)
            ordered = (
                _ordered_overlap(normalized_query, record.normalized_tokens)
                if evaluated
                else 0.0
            )
            record_ngrams = set(
                record.normalized_ngrams
                if record.normalized_ngrams
                and tuple(policy.ngram_sizes) == self.ngram_sizes
                else _ngrams(record.normalized_tokens, policy.ngram_sizes)
            )
            ngram = (
                len(query_ngrams & record_ngrams) / max(len(record_ngrams), 1)
                if evaluated
                else 0.0
            )
            edit = (
                _best_ngram_edit_similarity(
                    normalized_query,
                    record.normalized_tokens,
                    policy.ngram_sizes,
                    policy.approximate_max_distance,
                )
                if evaluated
                and (policy.enable_extended_channels or policy.mode == "token_edit")
                else 0.0
            )
            sequence = (
                SequenceMatcher(
                    None, normalized_query, record.normalized_tokens, autojunk=False
                ).ratio()
                if evaluated
                else 0.0
            )
            approximate = max(sequence, edit)
            entity = float(
                evaluated
                and any(alias and alias in query_text for alias in record.aliases)
            )
            explicit = float(record.reference_uri in explicit_reference_uris)
            semantic = max(0.0, min(1.0, (semantic_values[index] + 1.0) / 2.0))
            bm25 = bm25_values.get(index, 0.0) / bm25_scale
            embedding = (
                float(embedding_scores[index]) if embedding_scores is not None else 0.0
            )
            token = max(
                raw_exact,
                normalized_exact,
                0.55 * weighted + 0.25 * ordered + 0.20 * entity,
            )
            if policy.enable_extended_channels:
                token = max(token, ngram, edit, policy.embedding_weight * embedding)
            raw_rows.append(
                (index, {
                    "record": record,
                    "semantic": semantic,
                    "raw_exact": raw_exact,
                    "normalized_exact": normalized_exact,
                    "weighted": weighted,
                    "ordered": ordered,
                    "ngram": ngram,
                    "edit": edit,
                    "approximate": approximate,
                    "embedding": embedding,
                    "bm25": bm25,
                    "entity": entity,
                    "explicit": explicit,
                    "token": token,
                    "raw_span": (raw_start, raw_start + raw_length) if raw_length else None,
                    "normalized_span": (
                        (norm_start, norm_start + norm_length) if norm_length else None
                    ),
                })
            )

        cascade_channel = "semantic"
        for channel in (
            "explicit",
            "entity",
            "raw_exact",
            "normalized_exact",
            "weighted",
            "ngram",
            "edit",
            "approximate",
            "embedding",
        ):
            if max((row[channel] for _, row in raw_rows), default=0.0) >= policy.cascade_threshold:
                cascade_channel = channel
                break

        candidates: dict[int, DiscoveryCandidate] = {}
        for index, row in raw_rows:
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
            elif mode == "token_ngram":
                selected, channel = max(row["ngram"], row["explicit"]), "token_ngram"
            elif mode == "token_edit":
                selected, channel = max(row["edit"], row["explicit"]), "token_edit"
            elif mode == "token_embedding":
                selected = max(row["embedding"], row["explicit"])
                channel = "token_embedding"
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
            if row["explicit"]:
                selected = 1.0
                channel = "explicit_reference"
            record = row["record"]
            candidates[index] = DiscoveryCandidate(
                    reference_uri=record.reference_uri,
                    chunk_id=record.chunk_id,
                    layer_id=record.layer_id,
                    semantic_score=row["semantic"],
                    exact_span_score=row["raw_exact"],
                    normalized_exact_score=row["normalized_exact"],
                    weighted_overlap_score=row["weighted"],
                    ordered_score=row["ordered"],
                    ngram_score=row["ngram"],
                    edit_score=row["edit"],
                    approximate_score=row["approximate"],
                    embedding_score=(row["embedding"] if embedding_scores is not None else None),
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
                            ("ngram", row["ngram"]),
                            ("edit", row["edit"]),
                            ("approximate", row["approximate"]),
                            ("embedding", row["embedding"]),
                            ("bm25", row["bm25"]),
                            ("entity_name", row["entity"]),
                            ("explicit", row["explicit"]),
                            ("associative", token if hop > 1 else 0.0),
                        )
                        if value > 0.0
                    ),
            )
        order = sorted(
            candidates,
            key=lambda index: (-candidates[index].selected_score, index),
        )
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        self.last_search_stats = {
            **search_stats,
            "expensive_comparisons": len(candidate_rows),
            "returned_rows": len(candidates),
            "candidate_fraction": len(candidate_rows) / max(len(self.records), 1),
            "index_bytes": self.memory_bytes(),
        }
        ranked = {
            index: replace(candidate, rank=ranks[index])
            for index, candidate in candidates.items()
        }
        if sparse:
            return ranked
        return [ranked[index] for index in range(len(self.records))]
