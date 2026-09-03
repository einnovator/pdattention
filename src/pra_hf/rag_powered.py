"""Normalization, qualification, and reporting for powered RAG decomposition.

The module keeps retrieval, selection, execution representation, and bundle
effects separate. Experiment runners emit one row per condition and regime;
these helpers validate pair identities before computing deltas or public gates.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .rag_evaluation import ContextCondition


POWERED_CONDITIONS = (
    ContextCondition.NO_PRA_STANDARD_RAG.value,
    ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value,
    ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value,
    ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE.value,
    ContextCondition.PRA_NATIVE_MEMORY_BUNDLE.value,
)


def official_multihop_rag_score(prediction: str, answers: Sequence[str]) -> float:
    """Reproduce the upstream QA evaluator's lowercased token intersection.

    MultiHop-RAG's official one-answer-per-question precision, recall, F1, and
    accuracy all reduce to this binary score. We report normalized EM and token
    F1 separately because the official score is intentionally permissive.
    """

    predicted = set(prediction.lower().split())
    return float(any(predicted.intersection(answer.lower().split()) for answer in answers))


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` for no samples."""

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_selector_frozen_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Require SC/NM counterparts to share the exact selection receipt."""

    pairs: dict[tuple[object, ...], dict[str, str]] = {}
    paired_conditions = {
        ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value: "selected",
        ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value: "native",
        ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE.value: "selected",
        ContextCondition.PRA_NATIVE_MEMORY_BUNDLE.value: "native",
    }
    for row in rows:
        condition = str(row["condition"])
        side = paired_conditions.get(condition)
        if side is None or row.get("status") == "NO_QUALIFIED_ADAPTER":
            continue
        key = (
            row["example_id"],
            row["candidate_count"],
            row["token_budget"],
            row["regime"],
            row["selector_profile"],
            bool(row.get("bundle_id")),
        )
        pairs.setdefault(key, {})[side] = str(row["selection_receipt_id"])
    for key, pair in pairs.items():
        if set(pair) != {"selected", "native"}:
            raise ValueError(f"selector-frozen pair is incomplete for {key!r}")
        if pair["selected"] != pair["native"]:
            raise ValueError(f"selector receipt mismatch for {key!r}")


def _metric(row: Mapping[str, object], name: str) -> float | None:
    if name in row and row[name] is not None:
        return float(row[name])
    for group in ("retrieval_context_metrics", "serving_metrics", "resource_metrics"):
        nested = row.get(group)
        if isinstance(nested, Mapping) and nested.get(name) is not None:
            return float(nested[name])
    return None


SUMMARY_METRICS = (
    "exact_match",
    "token_f1",
    "official_multihop_rag_score",
    "answer_string_availability",
    "document_recall_at_candidate_k",
    "supporting_document_coverage",
    "supporting_span_coverage",
    "gold_chunk_recall",
    "false_selected_document_fraction",
    "logical_candidate_tokens",
    "physical_context_tokens",
    "visible_prompt_tokens",
    "selected_native_kv_tokens",
    "newly_materialized_tokens",
    "materialization_avoidance",
    "visible_reuse",
    "native_reuse",
    "ttft_ms",
    "itl_ms",
    "prefill_ms",
    "completion_latency_ms",
    "total_latency_ms",
    "tokens_per_second",
    "output_tokens_per_second",
    "requests_per_second",
    "ingestion_ms",
    "active_detail_bytes",
    "retained_detail_bytes",
    "kv_bytes",
    "peak_memory_bytes",
    "temporary_allocation_bytes",
)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate powered rows while preserving condition, selector, and regime."""

    validate_selector_frozen_rows(rows)
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row["condition"],
            row["selector_profile"],
            row["candidate_count"],
            row["token_budget"],
            row["regime"],
            row.get("status", "MEASURED"),
        )
        groups.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        condition, selector, candidate_count, token_budget, regime, status = key
        summary: dict[str, object] = {
            "condition": condition,
            "selector_profile": selector,
            "candidate_count": candidate_count,
            "token_budget": token_budget,
            "regime": regime,
            "status": status,
            "examples": len(values),
        }
        for metric in SUMMARY_METRICS:
            samples = [value for row in values if (value := _metric(row, metric)) is not None]
            summary[metric] = statistics.fmean(samples) if samples else None
            if metric == "ttft_ms":
                summary["ttft_p50_ms"] = percentile(samples, 0.50)
                summary["ttft_p95_ms"] = percentile(samples, 0.95)
                summary["ttft_p99_ms"] = percentile(samples, 0.99)
            elif metric == "itl_ms":
                summary["itl_p50_ms"] = percentile(samples, 0.50)
                summary["itl_p95_ms"] = percentile(samples, 0.95)
        failures = Counter(str(row.get("failure_class", "UNKNOWN")) for row in values)
        summary["successful_examples"] = failures.get("SUCCESS", 0)
        summary["failure_counts"] = dict(sorted(failures.items()))
        result.append(summary)
    return result


def paired_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    left_condition: str,
    right_condition: str,
    metric: str,
    selector_profile: str,
    regime: str,
) -> dict[str, object]:
    """Compute a matched right-minus-left delta without mixing selectors."""

    index: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        if row.get("selector_profile") != selector_profile or row.get("regime") != regime:
            continue
        condition = str(row["condition"])
        if condition not in {left_condition, right_condition}:
            continue
        key = (row["example_id"], row["candidate_count"], row["token_budget"])
        index.setdefault(key, {})[condition] = row
    deltas = []
    for pair in index.values():
        if left_condition not in pair or right_condition not in pair:
            continue
        left = _metric(pair[left_condition], metric)
        right = _metric(pair[right_condition], metric)
        if left is not None and right is not None:
            deltas.append(right - left)
    return {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "selector_profile": selector_profile,
        "regime": regime,
        "metric": metric,
        "paired_examples": len(deltas),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
    }


def qualification_gates(
    summaries: Sequence[Mapping[str, object]],
    *,
    minimum_examples: int = 50,
) -> dict[str, object]:
    """Apply conservative selection, native, economic, and card gates."""

    measured = [row for row in summaries if row.get("status") == "MEASURED"]
    powered = [row for row in measured if int(row.get("examples", 0)) >= minimum_examples]
    selected = [
        row
        for row in powered
        if row["condition"] == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value
        and row["regime"] == "COLD"
    ]
    native = [
        row
        for row in powered
        if row["condition"] == ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value
        and row["regime"] == "COLD"
    ]
    bundle = [
        row
        for row in powered
        if row["condition"] in {
            ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE.value,
            ContextCondition.PRA_NATIVE_MEMORY_BUNDLE.value,
        }
    ]
    selection_pass = any(
        float(row.get("token_f1") or 0.0) > 0.0
        and float(row.get("materialization_avoidance") or 0.0) >= 0.25
        for row in selected
    )
    native_pass = any(
        float(row.get("token_f1") or 0.0) > 0.0
        and float(row.get("total_latency_ms") or math.inf) < math.inf
        for row in native
    )
    economic_pass = any(
        float(row.get("native_reuse") or 0.0) > 0.0
        and float(row.get("total_latency_ms") or math.inf) < math.inf
        for row in native
    )
    bundle_pass = bool(bundle) and all(
        row.get("status") == "MEASURED" for row in bundle
    )
    return {
        "minimum_examples": minimum_examples,
        "selection_gate": "PASS" if selection_pass else "FAIL",
        "native_memory_gate": "PASS" if native_pass else "FAIL",
        "economic_gate": "PASS" if economic_pass else "FAIL",
        "bundle_gate": "PASS" if bundle_pass else "NO_QUALIFIED_ADAPTER",
        "card_gate": (
            "PASS"
            if selection_pass and native_pass and economic_pass and bundle_pass
            else "FAILED_OR_CANDIDATE_ONLY"
        ),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write mappings with stable columns and JSON-encoded nested values."""

    values = list(rows)
    columns = sorted({key for row in values for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_results(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write normalized condition rows as deterministic compressed JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
