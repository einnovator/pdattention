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
import random
import re
import statistics
import string
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


def normalize_answer(text: str) -> str:
    """Apply the benchmark's explicit EM/F1 normalization contract."""

    value = text.casefold()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_metrics(prediction: str, answers: Sequence[str]) -> tuple[float, float]:
    """Return normalized exact match and best token F1 over accepted answers."""

    predicted = normalize_answer(prediction).split()
    exact = 0.0
    best_f1 = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        exact = max(exact, float(predicted == gold))
        common = sum(
            min(predicted.count(token), gold.count(token)) for token in set(predicted)
        )
        if not predicted or not gold:
            score = float(predicted == gold)
        else:
            precision = common / len(predicted)
            recall = common / len(gold)
            score = (
                0.0
                if not precision + recall
                else 2 * precision * recall / (precision + recall)
            )
        best_f1 = max(best_f1, score)
    return exact, best_f1


def answer_normalization_diagnostics(
    prediction: str, answers: Sequence[str]
) -> dict[str, object]:
    """Expose normalization and format effects instead of hiding them in EM."""

    normalized_prediction = normalize_answer(prediction)
    normalized_answers = tuple(normalize_answer(answer) for answer in answers)
    exact, token_f1 = answer_metrics(prediction, answers)
    answer_format_ok = bool(prediction.strip()) and len(prediction.split()) <= 64
    if not normalized_prediction:
        match_kind = "EMPTY"
    elif exact:
        match_kind = "NORMALIZED_EXACT"
    elif token_f1 > 0.0:
        match_kind = "TOKEN_OVERLAP"
    else:
        match_kind = "NO_OVERLAP"
    return {
        "normalized_prediction": normalized_prediction,
        "normalized_gold_answers": list(normalized_answers),
        "prediction_token_count": len(normalized_prediction.split()),
        "normalization_changed_prediction": (
            normalized_prediction != prediction.casefold().strip()
        ),
        "answer_format_ok": answer_format_ok,
        "match_kind": match_kind,
        "normalized_exact_match": exact,
        "normalized_token_f1": token_f1,
    }


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


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 11, samples: int = 2000
) -> tuple[float | None, float | None]:
    """Return a deterministic percentile-bootstrap 95% interval for a mean."""

    if not values:
        return None, None
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    values = tuple(float(value) for value in values)
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


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


def validate_strong_reranker_parity(rows: Sequence[Mapping[str, object]]) -> None:
    """Verify the shared strong-reranker visible-text plumbing control."""

    pairs: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    profiles = {
        "strong_conventional_reranker": "standard",
        "pra_strong_reranker": "pra_selected",
    }
    for row in rows:
        side = profiles.get(str(row.get("selector_profile")))
        if side is None or row.get("status") != "MEASURED":
            continue
        if side == "pra_selected" and row.get("condition") != ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value:
            continue
        key = (
            row["example_id"],
            row["candidate_count"],
            row["token_budget"],
            row["regime"],
        )
        pairs.setdefault(key, {})[side] = row
    for key, pair in pairs.items():
        if set(pair) != {"standard", "pra_selected"}:
            raise ValueError(f"strong reranker plumbing pair is incomplete for {key!r}")
        standard = pair["standard"]
        selected = pair["pra_selected"]
        if standard.get("selection_receipt_id") != selected.get("selection_receipt_id"):
            raise ValueError(f"strong reranker receipt mismatch for {key!r}")
        if standard.get("prediction") != selected.get("prediction"):
            raise ValueError(f"strong reranker visible-text output mismatch for {key!r}")


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
    validate_strong_reranker_parity(rows)
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
            "answer_quality_publishable": all(
                row.get("answer_quality_publishable", True) is not False
                for row in values
            ),
        }
        for metric in SUMMARY_METRICS:
            samples = [value for row in values if (value := _metric(row, metric)) is not None]
            summary[metric] = statistics.fmean(samples) if samples else None
            if metric in {
                "token_f1",
                "official_multihop_rag_score",
                "ttft_ms",
                "total_latency_ms",
            }:
                low, high = bootstrap_mean_ci(samples)
                summary[f"{metric}_ci95_low"] = low
                summary[f"{metric}_ci95_high"] = high
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
    right_selector_profile: str | None = None,
    regime: str,
) -> dict[str, object]:
    """Compute a matched right-minus-left delta with explicit selector arms."""

    left_selector_profile = selector_profile
    right_selector_profile = right_selector_profile or selector_profile
    index: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        if row.get("regime") != regime or row.get("status") != "MEASURED":
            continue
        condition = str(row["condition"])
        profile = str(row.get("selector_profile"))
        if condition == left_condition and profile == left_selector_profile:
            side = "left"
        elif condition == right_condition and profile == right_selector_profile:
            side = "right"
        else:
            continue
        key = (row["example_id"], row["candidate_count"], row["token_budget"])
        index.setdefault(key, {})[side] = row
    deltas = []
    for pair in index.values():
        if set(pair) != {"left", "right"}:
            continue
        left = _metric(pair["left"], metric)
        right = _metric(pair["right"], metric)
        if left is not None and right is not None:
            deltas.append(right - left)
    return {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "selector_profile": left_selector_profile,
        "left_selector_profile": left_selector_profile,
        "right_selector_profile": right_selector_profile,
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
    powered = [
        row
        for row in measured
        if int(row.get("examples", 0)) >= minimum_examples
        and row.get("answer_quality_publishable", True) is not False
    ]
    standard_baselines = [
        row
        for row in powered
        if row["condition"] == ContextCondition.NO_PRA_STANDARD_RAG.value
        and row["selector_profile"] == "standard_bm25"
        and row["regime"] == "COLD"
    ]
    strong_baselines = [
        row
        for row in powered
        if row["condition"] == ContextCondition.NO_PRA_STANDARD_RAG.value
        and row["selector_profile"] == "strong_conventional_reranker"
        and row["regime"] == "COLD"
    ]
    generic_selected = [
        row
        for row in powered
        if row["condition"] == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value
        and row["selector_profile"] == "pra_generic"
        and row["regime"] == "COLD"
    ]
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
    warm_native = [
        row
        for row in powered
        if row["condition"] == ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value
        and row["regime"] == "WARM"
    ]
    warm_selected = [
        row
        for row in powered
        if row["condition"] == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value
        and row["regime"] == "WARM"
    ]
    bundle = [
        row
        for row in powered
        if row["condition"] in {
            ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE.value,
            ContextCondition.PRA_NATIVE_MEMORY_BUNDLE.value,
        }
    ]
    def by_config(
        values: Sequence[Mapping[str, object]], *, include_selector: bool = False
    ) -> dict[tuple[object, ...], Mapping[str, object]]:
        if include_selector:
            return {
                (row["candidate_count"], row["token_budget"], row["selector_profile"]): row
                for row in values
            }
        return {(row["candidate_count"], row["token_budget"]): row for row in values}

    standard_by_config = by_config(standard_baselines)
    strong_by_config = by_config(strong_baselines)
    generic_selected_by_config = by_config(generic_selected)
    strongest_baseline_by_config = {
        **standard_by_config,
        **strong_by_config,
    }
    selected_by_config = by_config(selected, include_selector=True)
    native_by_config = by_config(native, include_selector=True)
    warm_selected_by_config = by_config(warm_selected, include_selector=True)
    warm_native_by_config = by_config(warm_native, include_selector=True)
    selection_pass = any(
        (
            float(selected_row.get("token_f1") or 0.0)
            >= float(strongest_baseline_by_config[key].get("token_f1") or 0.0) - 0.02
            and (
                float(selected_row.get("physical_context_tokens") or math.inf)
                <= 0.8
                * float(
                    strongest_baseline_by_config[key].get("physical_context_tokens")
                    or 0.0
                )
                or float(selected_row.get("token_f1") or 0.0)
                > float(strongest_baseline_by_config[key].get("token_f1") or 0.0)
            )
        )
        for key, selected_row in generic_selected_by_config.items()
        if key in strongest_baseline_by_config
    )
    native_pass = any(
        float(native_row.get("token_f1") or 0.0)
        >= float(selected_by_config[key].get("token_f1") or 0.0) - 0.02
        and float(native_row.get("official_multihop_rag_score") or 0.0)
        >= float(selected_by_config[key].get("official_multihop_rag_score") or 0.0) - 0.02
        for key, native_row in native_by_config.items()
        if key in selected_by_config
    )
    economic_pass = any(
        float(native_row.get("native_reuse") or 0.0) > 0.0
        and float(native_row.get("total_latency_ms") or math.inf)
        <= float(warm_selected_by_config[key].get("total_latency_ms") or 0.0)
        for key, native_row in warm_native_by_config.items()
        if key in warm_selected_by_config
    )
    bundle_pass = bool(bundle) and all(
        row.get("status") == "MEASURED" for row in bundle
    )
    model_backed = bool(powered)
    powered_regimes = {str(row.get("regime")) for row in powered}
    persistent_only = model_backed and powered_regimes == {"PERSISTENT_CORPUS"}
    decomposition_status = (
        "NOT_APPLICABLE_PERSISTENT_CORPUS"
        if persistent_only
        else None
    )
    return {
        "minimum_examples": minimum_examples,
        "selection_gate": (
            decomposition_status or ("PASS" if selection_pass else "FAIL")
        ) if model_backed else "NOT_APPLICABLE_NON_MODEL",
        "selection_comparator": (
            (
                decomposition_status
                or (
                    "strong_conventional_reranker"
                    if strong_by_config
                    else "standard_bm25"
                )
            )
            if model_backed
            else "NOT_APPLICABLE_NON_MODEL"
        ),
        "native_memory_gate": (
            decomposition_status or ("PASS" if native_pass else "FAIL")
        ) if model_backed else "NOT_APPLICABLE_NON_MODEL",
        "economic_gate": (
            decomposition_status or ("PASS" if economic_pass else "FAIL")
        ) if model_backed else "NOT_APPLICABLE_NON_MODEL",
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
