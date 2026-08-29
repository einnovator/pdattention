"""Shared contract for frozen-selection E0 selected-text versus E2 native K/V.

The contract separates retrieval from representation. A selector produces one
immutable identity containing the candidate set, selected IDs, selected source
intervals, and selected-source digest. Both execution conditions then consume
that identity under the same query variant and workload regime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "2.0"
CONDITIONS = ("e0_selected_text", "e2_native_kv")
REGIMES = (
    "cold_one_shot",
    "warm_repeated",
    "multi_query_same_resource",
    "concurrent_shared_resource",
)
METRIC_FAMILIES = (
    "quality",
    "input",
    "pra",
    "ingestion",
    "serving",
    "reuse",
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenSourceInterval:
    """One selector-owned character interval inside a candidate document."""

    document_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Frozen source intervals must be non-empty and ordered.")

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "coordinate_space": "unicode_codepoint",
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class FrozenSelectionIdentity:
    """Selector output reused verbatim by E0 and E2 execution modes."""

    dataset: str
    example_id: str
    candidate_document_ids: tuple[str, ...]
    selected_document_ids: tuple[str, ...]
    selected_intervals: tuple[FrozenSourceInterval, ...]
    selected_source_sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_document_ids:
            raise ValueError("A frozen selection requires a non-empty candidate set.")
        if not self.selected_document_ids:
            raise ValueError("A frozen selection requires at least one selected document.")
        if len(set(self.candidate_document_ids)) != len(self.candidate_document_ids):
            raise ValueError("Candidate document IDs must be unique.")
        if len(set(self.selected_document_ids)) != len(self.selected_document_ids):
            raise ValueError("Selected document IDs must be unique.")
        if not set(self.selected_document_ids).issubset(self.candidate_document_ids):
            raise ValueError("Selected documents must belong to the frozen candidate set.")
        interval_documents = tuple(interval.document_id for interval in self.selected_intervals)
        if interval_documents != self.selected_document_ids:
            raise ValueError(
                "Frozen intervals must follow the selected-document routing order."
            )
        if len(self.selected_source_sha256) != 64:
            raise ValueError("Selected-source identity must be a SHA-256 digest.")

    @property
    def candidate_set_sha256(self) -> str:
        return _digest(self.candidate_document_ids)

    @property
    def selection_id(self) -> str:
        return _digest(self.to_dict(include_selection_id=False))

    def to_dict(self, *, include_selection_id: bool = True) -> dict[str, object]:
        value = {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "candidate_document_ids": list(self.candidate_document_ids),
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_document_ids": list(self.selected_document_ids),
            "selected_intervals": [interval.to_dict() for interval in self.selected_intervals],
            "selected_source_sha256": self.selected_source_sha256,
        }
        if include_selection_id:
            value["selection_id"] = self.selection_id
        return value


@dataclass(frozen=True)
class QueryVariant:
    """Deterministic suffix used to probe reuse of one frozen resource."""

    variant_id: str
    text: str


@dataclass(frozen=True)
class BenchmarkRequest:
    """One logical request in a shared E0/E2 workload schedule."""

    regime: str
    request_ordinal: int
    query: QueryVariant
    concurrency_group: str | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"Unknown matched E0/E2 regime: {self.regime}")
        if self.request_ordinal < 0:
            raise ValueError("Request ordinals must be non-negative.")
        if self.regime == "concurrent_shared_resource" and not self.concurrency_group:
            raise ValueError("Concurrent requests require a concurrency-group identity.")


def query_variants(question: str) -> tuple[QueryVariant, ...]:
    """Return stable query suffixes that preserve the same short-answer target."""

    clean = " ".join(question.split())
    return (
        QueryVariant(
            "direct",
            "Answer the question using the available evidence. Give only the short "
            f"answer.\nQuestion: {clean}\nAnswer:",
        ),
        QueryVariant(
            "evidence_focused",
            "Use only the supplied evidence. Respond with the shortest supported "
            f"answer.\nQuestion: {clean}\nAnswer:",
        ),
        QueryVariant(
            "verification",
            "Verify the answer against the supplied evidence, then output only the "
            f"short answer.\nQuestion: {clean}\nAnswer:",
        ),
    )


def regime_schedule(
    question: str,
    *,
    warm_repeats: int = 2,
    multi_query_count: int = 3,
    concurrency: int = 8,
) -> tuple[BenchmarkRequest, ...]:
    """Build all four workloads around one frozen selected resource."""

    if warm_repeats < 1:
        raise ValueError("Warm repeats must be positive.")
    variants = query_variants(question)
    if not 1 <= multi_query_count <= len(variants):
        raise ValueError("Multi-query count exceeds the deterministic variant set.")
    if concurrency < 1:
        raise ValueError("Concurrency must be positive.")
    requests = [BenchmarkRequest("cold_one_shot", 0, variants[0])]
    requests.extend(
        BenchmarkRequest("warm_repeated", ordinal, variants[0])
        for ordinal in range(warm_repeats)
    )
    requests.extend(
        BenchmarkRequest("multi_query_same_resource", ordinal, variant)
        for ordinal, variant in enumerate(variants[:multi_query_count])
    )
    group = _digest((question, concurrency))[:16]
    requests.extend(
        BenchmarkRequest(
            "concurrent_shared_resource",
            ordinal,
            variants[ordinal % len(variants)],
            concurrency_group=group,
        )
        for ordinal in range(concurrency)
    )
    return tuple(requests)


_REQUIRED_METRICS = {
    "quality": (
        "exact_match",
        "token_f1",
        "task_score",
        "gold_answer_logprob",
        "evidence_recall",
    ),
    "input": (
        "candidate_tokens",
        "selected_source_tokens",
        "visible_prompt_tokens",
    ),
    "pra": (
        "selected_native_kv_tokens",
        "active_detail_bytes",
        "retained_detail_bytes",
    ),
    "ingestion": (
        "text_preparation_ms",
        "kv_encode_ms",
        "index_construction_ms",
        "time_to_usable_context_ms",
    ),
    "serving": (
        "ttft_ms",
        "itl_ms",
        "tpot_ms",
        "total_latency_ms",
        "generated_tokens",
        "tokens_per_second",
        "requests_per_second",
    ),
    "reuse": (
        "ordinary_prefix_cache_hit_tokens",
        "pra_hot_hit",
        "pra_warm_hit",
        "bytes_read",
        "bytes_promoted",
        "bytes_avoided",
        "duplicate_physical_kv_avoided_bytes",
    ),
}


def metric_family(name: str, values: Mapping[str, object]) -> dict[str, object]:
    """Build one complete metric family, preserving explicit unavailable values."""

    if name not in _REQUIRED_METRICS:
        raise ValueError(f"Unknown matched E0/E2 metric family: {name}")
    missing = set(_REQUIRED_METRICS[name]) - set(values)
    if missing:
        raise ValueError(f"Metric family {name!r} is missing {sorted(missing)}.")
    return {key: values[key] for key in _REQUIRED_METRICS[name]}


def benchmark_metrics(
    *,
    exact_match: float,
    token_f1: float,
    gold_answer_logprob: float | None,
    evidence_recall: float,
    candidate_tokens: int,
    selected_source_tokens: int,
    visible_prompt_tokens: int,
    selected_native_kv_tokens: int,
    active_detail_bytes: int,
    retained_detail_bytes: int,
    text_preparation_ms: float,
    kv_encode_ms: float | None,
    index_construction_ms: float | None,
    time_to_usable_context_ms: float,
    ttft_ms: float | None,
    itl_ms: float | None,
    total_latency_ms: float,
    generated_tokens: int,
    ordinary_prefix_cache_hit_tokens: int | None,
    pra_hot_hit: bool,
    pra_warm_hit: bool,
    bytes_read: int | None,
    bytes_promoted: int | None,
    bytes_avoided: int,
    duplicate_physical_kv_avoided_bytes: int,
    requests_per_second: float | None = None,
) -> dict[str, dict[str, object]]:
    """Build the six disjoint metric families used by every engine runner."""

    tokens_per_second = (
        generated_tokens / (total_latency_ms / 1000.0)
        if total_latency_ms > 0
        else None
    )
    return {
        "quality": metric_family(
            "quality",
            {
                "exact_match": exact_match,
                "token_f1": token_f1,
                "task_score": token_f1,
                "gold_answer_logprob": gold_answer_logprob,
                "evidence_recall": evidence_recall,
            },
        ),
        "input": metric_family(
            "input",
            {
                "candidate_tokens": candidate_tokens,
                "selected_source_tokens": selected_source_tokens,
                "visible_prompt_tokens": visible_prompt_tokens,
            },
        ),
        "pra": metric_family(
            "pra",
            {
                "selected_native_kv_tokens": selected_native_kv_tokens,
                "active_detail_bytes": active_detail_bytes,
                "retained_detail_bytes": retained_detail_bytes,
            },
        ),
        "ingestion": metric_family(
            "ingestion",
            {
                "text_preparation_ms": text_preparation_ms,
                "kv_encode_ms": kv_encode_ms,
                "index_construction_ms": index_construction_ms,
                "time_to_usable_context_ms": time_to_usable_context_ms,
            },
        ),
        "serving": metric_family(
            "serving",
            {
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "tpot_ms": itl_ms,
                "total_latency_ms": total_latency_ms,
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_second,
                "requests_per_second": requests_per_second,
            },
        ),
        "reuse": metric_family(
            "reuse",
            {
                "ordinary_prefix_cache_hit_tokens": ordinary_prefix_cache_hit_tokens,
                "pra_hot_hit": pra_hot_hit,
                "pra_warm_hit": pra_warm_hit,
                "bytes_read": bytes_read,
                "bytes_promoted": bytes_promoted,
                "bytes_avoided": bytes_avoided,
                "duplicate_physical_kv_avoided_bytes": (
                    duplicate_physical_kv_avoided_bytes
                ),
            },
        ),
    }


def benchmark_row(
    *,
    condition: str,
    selection: FrozenSelectionIdentity,
    request: BenchmarkRequest,
    output: str,
    metrics: Mapping[str, Mapping[str, object]],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create and validate one condition row in the shared schema."""

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown E0/E2 execution condition: {condition}")
    missing_families = set(METRIC_FAMILIES) - set(metrics)
    if missing_families:
        raise ValueError(f"Benchmark row is missing {sorted(missing_families)}.")
    row = {
        "condition": condition,
        "regime": request.regime,
        "request_ordinal": request.request_ordinal,
        "query_variant": request.query.variant_id,
        "query_sha256": hashlib.sha256(request.query.text.encode("utf-8")).hexdigest(),
        "concurrency_group": request.concurrency_group,
        "selection": selection.to_dict(),
        "output": output,
        "metrics": {
            family: metric_family(family, metrics[family])
            for family in METRIC_FAMILIES
        },
    }
    if extra:
        row["extra"] = dict(extra)
    return row


def validate_payload(payload: Mapping[str, object]) -> None:
    """Require matched E0/E2 rows for every request and frozen selection."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Matched E0/E2 payload has an unsupported schema version.")
    rows = list(payload.get("rows", ()))
    if not rows:
        raise ValueError("Matched E0/E2 payload contains no rows.")
    pairs: dict[tuple[str, str, int, str], set[str]] = {}
    observed_regimes = set()
    for row in rows:
        metrics = row.get("metrics", {})
        if set(metrics) != set(METRIC_FAMILIES):
            raise ValueError("Matched E0/E2 rows must keep metric families disjoint.")
        selection = row["selection"]
        key = (
            str(selection["selection_id"]),
            str(row["regime"]),
            int(row["request_ordinal"]),
            str(row["query_sha256"]),
        )
        pairs.setdefault(key, set()).add(str(row["condition"]))
        observed_regimes.add(str(row["regime"]))
    unmatched = [key for key, conditions in pairs.items() if conditions != set(CONDITIONS)]
    if unmatched:
        raise ValueError(f"Matched E0/E2 payload has unmatched requests: {unmatched[:3]}")
    missing_regimes = set(REGIMES) - observed_regimes
    if missing_regimes:
        raise ValueError(f"Matched E0/E2 payload is missing {sorted(missing_regimes)}.")


def candidate_digest(document_ids: Iterable[str]) -> str:
    """Public helper used by manifests and tests for candidate-set identity."""

    return _digest(tuple(map(str, document_ids)))
