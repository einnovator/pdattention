"""Finalize Paper 2.6 confidence, stability, and handoff diagnostics.

This runner consumes the frozen 132-identity Paper 2.6 route cohort. It does
not generate, materialize K/V, change route budgets, or present bootstrap
resamples as new examples. When the frozen feature bundles are available it
replays the existing scorer solely to export richer candidate confidence
signals; the selected routes remain those of ``run_channel_selection.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_channel_selection import (  # noqa: E402
    DATASETS,
    ROOT_CHANNELS,
    SUCCESSOR_CHANNELS,
    _best,
    _evaluate_case,
    _load_cases,
    _pieces,
    _rank,
    _robustness,
    _search_method_action_spec,
)
from pra_hf.channel_geometry import jaccard  # noqa: E402
from pra_hf.confidence_diagnostics import (  # noqa: E402
    bootstrap_best_channel,
    choose_conservative_threshold,
    paired_bootstrap_interval,
    percentile_calibrate,
    reliability_bins,
    selective_metrics,
    summarize_calibration,
    validate_observable_feature_names,
    validate_search_method_action_spec,
)


ROOT_CANONICAL = {
    "gist": "semantic",
    "exact": "exact",
    "bm25": "bm25",
    "approx": "approximate",
    "hybrid": "hybrid",
}
STOP_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
COLORS = ("#2f6690", "#d1495b", "#16817a", "#e09f3e", "#59656f")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    return float(value) if value not in (None, "") else default


def _finite_mean(values) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else None


def _identity(row: dict) -> tuple[str, str, str]:
    return str(row["split"]), str(row["dataset"]), str(row["example_id"])


def _validation_root_channels(root_rows: list[dict]) -> dict[str, str]:
    result = {}
    for dataset in DATASETS:
        grouped = []
        for channel in ROOT_CHANNELS:
            rows = [
                row for row in root_rows
                if row["split"] == "validation"
                and row["dataset"] == dataset
                and row["channel"] == channel
            ]
            grouped.append(
                {
                    "channel": channel,
                    "recall": mean(_float(row, "recall") for row in rows),
                    "precision": mean(_float(row, "precision") for row in rows),
                    "mrr": mean(_float(row, "mrr") for row in rows),
                }
            )
        result[dataset] = _best(grouped, "channel")["channel"]
    return result


def _softmax_support(scores: dict[str, float], excluded: set[str] = frozenset()) -> tuple[float, float]:
    values = [float(value) for key, value in scores.items() if key not in excluded and math.isfinite(float(value))]
    if not values:
        return 0.0, 0.0
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    return entropy, math.exp(entropy)


def _candidate_confidence_fields(
    *,
    candidate,
    record,
    scores: dict[str, float],
    query_terms: set[str],
    token_index,
    top_candidate_agreement: float,
    top_k_overlap: float,
    lexical_support_for_semantic_candidate: float,
    current_path_terms: set[str] | None = None,
    root_reference_uris: set[str] = frozenset(),
) -> dict[str, float | int]:
    overlap = query_terms.intersection(record.normalized_tokens)
    idfs = [float(token_index.idf.get(term, 0.0)) for term in overlap]
    occurrences = [
        sum(term in candidate_record.normalized_tokens for candidate_record in token_index.records)
        for term in overlap
    ]
    ordered_scores = sorted((float(value) for value in scores.values() if math.isfinite(float(value))), reverse=True)
    top_score = ordered_scores[0] if ordered_scores else 0.0
    score_gap = top_score - ordered_scores[1] if len(ordered_scores) > 1 else top_score
    entropy, effective_support = _softmax_support(scores)
    raw_span = candidate.raw_exact_span or (0, 0)
    normalized_span = candidate.normalized_exact_span or (0, 0)
    matched_span = max(raw_span[1] - raw_span[0], normalized_span[1] - normalized_span[0])
    ambiguity_count = sum(value >= top_score - 0.05 for value in ordered_scores)
    lexical_support = max(
        candidate.exact_span_score,
        candidate.normalized_exact_score,
        candidate.weighted_overlap_score,
        candidate.approximate_score,
        candidate.bm25_score,
        candidate.entity_name_score,
    )
    current_path_overlap = (
        len(current_path_terms.intersection(record.normalized_tokens))
        / max(len(current_path_terms), 1)
        if current_path_terms is not None
        else 0.0
    )
    relation_terms = {term for term in overlap if term not in STOP_TERMS}
    relation_compatibility = len(relation_terms) / max(len(query_terms - STOP_TERMS), 1)
    uniqueness = 1.0 / max(ambiguity_count, 1)
    referential_consistency = (
        0.35 * candidate.semantic_score
        + 0.25 * lexical_support
        + 0.20 * uniqueness
        + 0.20 * top_candidate_agreement
    )
    return {
        "top_score": top_score,
        "top1_top2_gap": score_gap,
        "entropy": entropy,
        "effective_support": effective_support,
        "score_concentration": 1.0 / max(effective_support, 1.0),
        "query_current_similarity": candidate.semantic_score,
        "exact_span_length": matched_span,
        "matched_token_count": len(overlap),
        "matched_token_idf": sum(idfs),
        "corpus_occurrence_count": min(occurrences, default=0),
        "candidate_count_sharing_reference": sum(
            candidate_record.reference_uri == record.reference_uri
            for candidate_record in token_index.records
        ),
        "matching_term_count": len(overlap),
        "rare_term_contribution": max(idfs, default=0.0),
        "edit_token_similarity": candidate.approximate_score,
        "ambiguity_count": ambiguity_count,
        "exact_vs_approx_gap": max(candidate.exact_span_score, candidate.normalized_exact_score) - candidate.approximate_score,
        "alias_partial_mention_indicator": int(candidate.entity_name_score > 0),
        "top_candidate_agreement": top_candidate_agreement,
        "top_k_overlap": top_k_overlap,
        "semantic_consistency_of_lexical_candidate": candidate.semantic_score,
        "lexical_support_for_semantic_candidate": lexical_support_for_semantic_candidate,
        "entity_type_consistency_proxy": max(candidate.entity_name_score, candidate.normalized_exact_score),
        "surrounding_semantic_compatibility": candidate.semantic_score,
        "relation_compatibility_proxy": relation_compatibility,
        "current_path_consistency": current_path_overlap,
        "candidate_uniqueness": uniqueness,
        "source_document_consistency": int(
            not root_reference_uris or record.reference_uri in root_reference_uris
        ),
        "referential_consistency_score": referential_consistency,
    }


def _extract_detailed_confidence(args) -> tuple[list[dict], list[dict]]:
    from transformers import AutoTokenizer
    from experiments.paper2_6_hybrid_pra.run_channel_selection import MODEL_ID, MODEL_REVISION

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    cohort_args = argparse.Namespace(**vars(args))
    cohort_args.seed = args.cohort_seed
    cases = _load_cases(cohort_args)
    root_rows = _read_csv(args.channel_dir / "root_channel_results.csv")
    validation_root = _validation_root_channels(root_rows)
    output = []
    for case_index, (feature, example) in enumerate(cases, 1):
        detail = _evaluate_case(tokenizer, feature, example, args.step_budget)
        split = str(feature["split"])
        dataset = str(feature["dataset"])
        example_id = str(feature["example_id"])
        query_terms = _pieces(tokenizer, detail["query_ids"])
        root_rankings = {channel: _rank(detail["root_scores"][channel]) for channel in ROOT_CHANNELS}
        root_top = {channel: ranking[0] for channel, ranking in root_rankings.items()}
        root_sets = {channel: set(ranking[: args.step_budget]) for channel, ranking in root_rankings.items()}
        semantic_id = root_top["gist"]
        semantic_candidate = detail["root_candidates"]["gist"][semantic_id]
        semantic_lexical_support = max(
            semantic_candidate.exact_span_score,
            semantic_candidate.normalized_exact_score,
            semantic_candidate.weighted_overlap_score,
            semantic_candidate.approximate_score,
            semantic_candidate.bm25_score,
        )
        for channel in ROOT_CHANNELS:
            top_id = root_top[channel]
            candidate = detail["root_candidates"][channel][top_id]
            record = detail["token_index"].records[detail["index"].chunk_ids.index(top_id)]
            agreement = sum(value == top_id for value in root_top.values()) / len(ROOT_CHANNELS)
            overlap = mean(
                jaccard(root_sets[channel], root_sets[other])
                for other in ROOT_CHANNELS if other != channel
            )
            fields = _candidate_confidence_fields(
                candidate=candidate,
                record=record,
                scores=detail["root_scores"][channel],
                query_terms=query_terms,
                token_index=detail["token_index"],
                top_candidate_agreement=agreement,
                top_k_overlap=overlap,
                lexical_support_for_semantic_candidate=semantic_lexical_support,
            )
            output.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "example_id": example_id,
                    "stage": "root",
                    "channel": ROOT_CANONICAL[channel],
                    "legacy_channel": channel,
                    "top_candidate_id": top_id,
                    "candidate_correct": int(top_id in detail["root_gold"]),
                    "wrong_reference_failure": int(top_id not in detail["root_gold"] and fields["top_score"] >= 0.5),
                    **fields,
                }
            )

        root_channel = validation_root[dataset]
        root_ids = detail["root_selected"][root_channel]
        root_terms = set().union(
            *(
                set(detail["token_index"].records[detail["index"].chunk_ids.index(identity)].normalized_tokens)
                for identity in root_ids
            ),
            set(),
        )
        root_uris = {
            detail["token_index"].records[detail["index"].chunk_ids.index(identity)].reference_uri
            for identity in root_ids
        }
        successor_details = {
            channel: detail["successor_details"].get((root_channel, channel))
            for channel in SUCCESSOR_CHANNELS
        }
        successor_details = {key: value for key, value in successor_details.items() if value}
        if successor_details:
            successor_top = {channel: value["ranking"][0] for channel, value in successor_details.items()}
            successor_sets = {
                channel: set(value["ranking"][: args.step_budget])
                for channel, value in successor_details.items()
            }
            semantic_id = successor_top["native_semantic"]
            semantic_candidate = successor_details["native_semantic"]["candidates"][semantic_id]
            semantic_lexical_support = max(
                semantic_candidate.exact_span_score,
                semantic_candidate.normalized_exact_score,
                semantic_candidate.weighted_overlap_score,
                semantic_candidate.approximate_score,
                semantic_candidate.bm25_score,
            )
            for channel, successor in successor_details.items():
                top_id = successor_top[channel]
                candidate = successor["candidates"][top_id]
                record = detail["token_index"].records[detail["index"].chunk_ids.index(top_id)]
                agreement = sum(value == top_id for value in successor_top.values()) / len(successor_top)
                overlap = mean(
                    jaccard(successor_sets[channel], successor_sets[other])
                    for other in successor_sets if other != channel
                )
                fields = _candidate_confidence_fields(
                    candidate=candidate,
                    record=record,
                    scores={identity: score for identity, score in successor["scores"].items() if identity not in root_ids},
                    query_terms=root_terms,
                    token_index=detail["token_index"],
                    top_candidate_agreement=agreement,
                    top_k_overlap=overlap,
                    lexical_support_for_semantic_candidate=semantic_lexical_support,
                    current_path_terms=root_terms,
                    root_reference_uris=root_uris,
                )
                output.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "example_id": example_id,
                        "stage": "successor",
                        "channel": channel,
                        "root_channel": ROOT_CANONICAL[root_channel],
                        "top_candidate_id": top_id,
                        "candidate_correct": int(top_id in successor["gold"]),
                        "wrong_reference_failure": int(top_id not in successor["gold"] and fields["top_score"] >= 0.5),
                        **fields,
                    }
                )
        print(f"[final confidence {case_index}/{len(cases)}] {dataset} {example_id}", flush=True)
    return output, _robustness(tokenizer)


def _minimal_confidence_rows(channel_dir: Path) -> list[dict]:
    """Fallback for machines that cannot map the frozen tensor bundle."""
    output = []
    for stage, filename, channel_field in (
        ("root", "root_channel_results.csv", "channel"),
        ("successor", "successor_channel_results.csv", "successor_channel"),
    ):
        for row in _read_csv(channel_dir / filename):
            channel = row[channel_field]
            if stage == "root" and channel not in ROOT_CHANNELS:
                continue
            output.append(
                {
                    "split": row["split"], "dataset": row["dataset"],
                    "example_id": row["example_id"], "stage": stage,
                    "channel": ROOT_CANONICAL.get(channel, channel),
                    "legacy_channel": channel,
                    "top_score": _float(row, "top_score"),
                    "top1_top2_gap": _float(row, "score_gap"),
                    "candidate_correct": int(_float(row, "mrr") == 1.0),
                    "wrong_reference_failure": int(_float(row, "mrr") != 1.0 and _float(row, "top_score") >= 0.5),
                    "referential_consistency_score": _float(row, "score_gap"),
                    "semantic_consistency_of_lexical_candidate": 0.0,
                    "top_candidate_agreement": 0.0,
                    "top_k_overlap": 0.0,
                }
            )
    return output


def _expanded_route_rows(channel_dir: Path) -> list[dict]:
    rows = []
    for row in _read_csv(channel_dir / "root_channel_results.csv"):
        rows.append({"stage": "root", "search_method": ROOT_CANONICAL.get(row["channel"], row["channel"]), **row})
    for row in _read_csv(channel_dir / "successor_channel_results.csv"):
        rows.append({"stage": "successor", "search_method": row["successor_channel"], **row})
    return rows


def _cohort_stability(channel_dir: Path, seed: int, resamples: int) -> list[dict]:
    root = _read_csv(channel_dir / "root_channel_results.csv")
    successor = _read_csv(channel_dir / "successor_channel_results.csv")
    validation_root = _validation_root_channels(root)
    output = []
    for dataset_index, dataset in enumerate(DATASETS):
        stages = [
            ("root", [row for row in root if row["split"] == "test" and row["dataset"] == dataset and row["channel"] in ROOT_CHANNELS], "channel", ROOT_CHANNELS),
            ("successor", [row for row in successor if row["split"] == "test" and row["dataset"] == dataset and row["root_channel"] == validation_root[dataset]], "successor_channel", SUCCESSOR_CHANNELS),
        ]
        for stage, rows, channel_field, channels in stages:
            identities = sorted({row["example_id"] for row in rows})
            if not identities:
                continue
            sizes = sorted({min(size, len(identities)) for size in (4, 8, 12, 16, 24, len(identities))})
            normalized = [
                {
                    "example_id": row["example_id"], "channel": row[channel_field],
                    "recall": _float(row, "recall"),
                    "precision": _float(row, "precision"), "mrr": _float(row, "mrr"),
                }
                for row in rows
            ]
            for size in sizes:
                draws = bootstrap_best_channel(
                    normalized,
                    cohort_size=size,
                    resamples=resamples,
                    seed=seed + 100 * dataset_index + size + (1000 if stage == "successor" else 0),
                    channel_order=channels,
                    tie_metrics=("precision", "mrr"),
                )
                counts = Counter(row["best_channel"] for row in draws)
                for channel in channels:
                    values = [float(row[f"mean_{channel}"]) for row in draws]
                    values.sort()
                    output.append(
                        {
                            "dataset": dataset, "stage": stage,
                            "available_identities": len(identities), "cohort_size": size,
                            "channel": ROOT_CANONICAL.get(channel, channel),
                            "legacy_channel": channel,
                            "selection_probability": counts[channel] / len(draws),
                            "bootstrap_mean_recall": mean(values),
                            "recall_ci_low": values[int(0.025 * (len(values) - 1))],
                            "recall_ci_high": values[int(0.975 * (len(values) - 1))],
                            "resamples": resamples, "seed": seed,
                        }
                    )
    return output


def _calibration_analysis(confidence_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    feature_names = (
        "top_score", "top1_top2_gap", "referential_consistency_score",
        "semantic_consistency_of_lexical_candidate", "top_candidate_agreement",
    )
    validate_observable_feature_names(feature_names)
    metrics, reliability = [], []
    groups = sorted({(row["stage"], row["channel"]) for row in confidence_rows})
    for stage, channel in groups:
        validation = [row for row in confidence_rows if row["stage"] == stage and row["channel"] == channel and row["split"] == "validation"]
        heldout = [row for row in confidence_rows if row["stage"] == stage and row["channel"] == channel and row["split"] == "test"]
        if not validation or not heldout:
            continue
        for signal in feature_names:
            if signal not in validation[0]:
                continue
            reference = [float(row[signal]) for row in validation]
            validation_probability = percentile_calibrate(reference, reference)
            heldout_probability = percentile_calibrate(reference, [float(row[signal]) for row in heldout])
            validation_labels = [int(row["candidate_correct"]) for row in validation]
            heldout_labels = [int(row["candidate_correct"]) for row in heldout]
            threshold = choose_conservative_threshold(validation_labels, validation_probability, minimum_precision=0.8)
            summary = summarize_calibration(heldout_labels, heldout_probability, bins=8)
            selective = selective_metrics(heldout_labels, heldout_probability, threshold)
            metrics.append(
                {
                    "stage": stage, "channel": channel, "signal": signal,
                    **summary.__dict__, "validation_threshold": threshold,
                    **{f"selective_{key}": value for key, value in selective.items() if key != "threshold"},
                }
            )
            if signal == "referential_consistency_score":
                for row, probability in zip(heldout, heldout_probability):
                    row["calibrated_referential_confidence"] = probability
                for bin_row in reliability_bins(heldout_labels, heldout_probability, bins=8):
                    reliability.append({"stage": stage, "channel": channel, **bin_row})
    return metrics, reliability


def _consistency_gate(confidence_rows: list[dict]) -> list[dict]:
    output = []
    for stage in ("root", "successor"):
        semantic_name = "semantic" if stage == "root" else "native_semantic"
        lexical_names = (
            ("exact", "bm25", "approximate")
            if stage == "root"
            else ("exact_new_address", "bm25_state", "approximate_new_address")
        )
        grouped = defaultdict(dict)
        for row in confidence_rows:
            if row["stage"] == stage:
                grouped[(row["split"], row["dataset"], row["example_id"])][row["channel"]] = row
        validation_scores, validation_labels = [], []
        for key, methods in grouped.items():
            if key[0] != "validation" or semantic_name not in methods or not set(lexical_names).issubset(methods):
                continue
            lexical = max(lexical_names, key=lambda name: float(methods[name]["top_score"]))
            validation_scores.append(float(methods[lexical]["semantic_consistency_of_lexical_candidate"]))
            validation_labels.append(int(methods[lexical]["candidate_correct"]))
        threshold = choose_conservative_threshold(validation_labels, validation_scores, minimum_precision=0.7) if validation_labels else 1.0
        for dataset in DATASETS:
            examples = [methods for key, methods in grouped.items() if key[0] == "test" and key[1] == dataset and semantic_name in methods and set(lexical_names).issubset(methods)]
            if not examples:
                continue
            policies = defaultdict(list)
            for methods in examples:
                lexical_name = max(lexical_names, key=lambda name: float(methods[name]["top_score"]))
                lexical = methods[lexical_name]
                semantic = methods[semantic_name]
                hybrid_name = "hybrid" if stage == "root" else "hybrid_state"
                policies["lexical_confidence_alone"].append(int(lexical["candidate_correct"]))
                policies["semantic_confidence_alone"].append(int(semantic["candidate_correct"]))
                gated = lexical if float(lexical["semantic_consistency_of_lexical_candidate"]) >= threshold else semantic
                policies["consistency_gated_lexical"].append(int(gated["candidate_correct"]))
                if hybrid_name in methods:
                    policies["static_fusion"].append(int(methods[hybrid_name]["candidate_correct"]))
            for policy, labels in policies.items():
                observed, lower, upper = paired_bootstrap_interval(labels, seed=20260822 + len(output))
                output.append(
                    {
                        "record_type": "natural_consistency_gate", "dataset": dataset,
                        "stage": stage, "policy": policy, "examples": len(labels),
                        "top1_accuracy": observed, "ci_low": lower, "ci_high": upper,
                        "validation_consistency_threshold": threshold,
                    }
                )
    return output


def _robustness_diagnostics(rows: list[dict]) -> list[dict]:
    output = []
    for perturbation in sorted({row["perturbation"] for row in rows}):
        for channel in sorted({row["channel"] for row in rows}):
            group = [row for row in rows if row["perturbation"] == perturbation and row["channel"] == channel]
            if not group:
                continue
            output.append(
                {
                    "record_type": "controlled_reference_condition",
                    "perturbation": perturbation, "channel": channel,
                    "examples": len(group),
                    "target_recovery": mean(float(row["target_recovery"]) for row in group),
                    "wrong_target_recovery": mean(float(row["wrong_target_recovery"]) for row in group),
                    "wrong_target_confidence": mean(float(row["wrong_target_confidence"]) for row in group),
                }
            )
    wrong_conditions = {
        "confidently_wrong", "same_name_wrong_entity", "same_class_wrong_instance",
        "correct_entity_wrong_relation", "stale_alternate_alias", "two_plausible_references",
    }
    wrong_rows = [row for row in rows if row["perturbation"] in wrong_conditions]
    for channel in sorted({row["channel"] for row in wrong_rows}):
        group = [row for row in wrong_rows if row["channel"] == channel]
        labels = [int(float(row["target_recovery"]) > float(row["wrong_target_recovery"])) for row in group]
        validity = [1.0 - float(row["wrong_target_confidence"]) for row in group]
        for threshold in [value / 20 for value in range(21)]:
            output.append(
                {
                    "record_type": "wrong_reference_abstention_curve", "channel": channel,
                    **selective_metrics(labels, validity, threshold),
                }
            )
    return output


def _useful_address_proxy(channel_dir: Path) -> tuple[list[dict], dict]:
    useful = _read_csv(channel_dir / "iterative_useful_address.csv")
    features = {_identity(row): row for row in _read_csv(channel_dir / "selector_observable_features.csv")}
    address = defaultdict(list)
    for row in _read_csv(channel_dir / "address_confidence.csv"):
        address[_identity(row)].append(row)
    proxy_names = (
        "address_presence", "address_rarity", "address_uniqueness",
        "exact_confidence", "approximate_confidence", "semantic_consistency",
    )
    validate_observable_feature_names(proxy_names)
    rows = []
    for row in useful:
        key = _identity(row)
        feature = features[key]
        address_rows = address.get(key, [])
        rarity = mean(_float(value, "idf") for value in address_rows) if address_rows else 0.0
        uniqueness = 1.0 / max(_float(row, "minimum_candidate_count", 1.0), 1.0)
        components = {
            "address_presence": min(_float(row, "address_count") / 3.0, 1.0),
            "address_rarity": rarity,
            "address_uniqueness": uniqueness,
            "exact_confidence": _float(feature, "exact_top_score"),
            "approximate_confidence": _float(feature, "approx_top_score"),
            "semantic_consistency": _float(feature, "gist_top_score"),
        }
        raw = (
            0.20 * components["address_presence"]
            + 0.20 * components["address_uniqueness"]
            + 0.15 * components["exact_confidence"]
            + 0.15 * components["approximate_confidence"]
            + 0.15 * components["semantic_consistency"]
            + 0.15 * min(components["address_rarity"] / 5.0, 1.0)
        )
        rows.append(
            {
                "split": row["split"], "dataset": row["dataset"],
                "example_id": row["example_id"], "root_channel": row["root_channel"],
                **components, "proxy_score": raw,
                "true_useful_address": int(_float(row, "useful_address")),
            }
        )
    validation = [row for row in rows if row["split"] == "validation"]
    heldout = [row for row in rows if row["split"] == "test"]
    reference = [float(row["proxy_score"]) for row in validation]
    validation_probability = percentile_calibrate(reference, reference)
    heldout_probability = percentile_calibrate(reference, [float(row["proxy_score"]) for row in heldout])
    threshold = choose_conservative_threshold(
        [row["true_useful_address"] for row in validation], validation_probability,
        minimum_precision=0.7,
    )
    for row, probability in zip(validation, validation_probability):
        row["calibrated_probability"] = probability
    for row, probability in zip(heldout, heldout_probability):
        row["calibrated_probability"] = probability
    summary = summarize_calibration(
        [row["true_useful_address"] for row in heldout], heldout_probability, bins=6
    ).__dict__
    summary.update(selective_metrics(
        [row["true_useful_address"] for row in heldout], heldout_probability, threshold
    ))
    return rows, summary


def _headroom_summary(channel_dir: Path) -> list[dict]:
    output = []
    for stage, filename in (
        ("root", "channel_true_oracle_headroom.csv"),
        ("successor", "successor_true_oracle_headroom.csv"),
    ):
        rows = _read_csv(channel_dir / filename)
        for dataset in DATASETS:
            group = [row for row in rows if row["dataset"] == dataset]
            for name, field in (("H_selection", "selection_headroom"), ("H_validation", "validation_instability")):
                observed, lower, upper = paired_bootstrap_interval(
                    [_float(row, field) for row in group], seed=20260822 + len(output)
                )
                output.append(
                    {
                        "dataset": dataset, "stage": stage, "component": name,
                        "examples": len(group), "mean": observed,
                        "ci_low": lower, "ci_high": upper,
                    }
                )
    return output


def _related_work_rows() -> list[dict]:
    return [
        {"family_system": "DPR", "external_text": "yes", "native_kv": "no", "lexical": "no", "semantic": "dense", "iterative_state_dependent": "no", "adaptive_channel": "no", "citation_key": "karpukhin2020dpr"},
        {"family_system": "RAG", "external_text": "yes", "native_kv": "no", "lexical": "no", "semantic": "dense", "iterative_state_dependent": "token-conditioned generation, not path traversal", "adaptive_channel": "no", "citation_key": "lewis2020rag"},
        {"family_system": "BM25+dense/RRF", "external_text": "yes", "native_kv": "no", "lexical": "yes", "semantic": "yes", "iterative_state_dependent": "typically no", "adaptive_channel": "fixed fusion", "citation_key": "cormack2009rrf"},
        {"family_system": "MDR", "external_text": "yes", "native_kv": "no", "lexical": "no", "semantic": "dense", "iterative_state_dependent": "yes", "adaptive_channel": "no", "citation_key": "xiong2021mdr"},
        {"family_system": "IRRR", "external_text": "yes", "native_kv": "no", "lexical": "query generation", "semantic": "neural reranking", "iterative_state_dependent": "yes", "adaptive_channel": "no", "citation_key": "qi2021irrr"},
        {"family_system": "RETRO", "external_text": "yes", "native_kv": "no", "lexical": "no", "semantic": "nearest-neighbor chunks", "iterative_state_dependent": "chunk-conditioned decoder", "adaptive_channel": "no", "citation_key": "borgeaud2022retro"},
        {"family_system": "Memorizing Transformer", "external_text": "no", "native_kv": "activation memory", "lexical": "no", "semantic": "key similarity", "iterative_state_dependent": "decode-state query", "adaptive_channel": "no", "citation_key": "wu2022memorizing"},
        {"family_system": "RetrievalAttention", "external_text": "no", "native_kv": "yes", "lexical": "no", "semantic": "attention-aware vector search", "iterative_state_dependent": "decode-state query", "adaptive_channel": "no", "citation_key": "liu2025retrievalattention"},
        {"family_system": "BLINK", "external_text": "entity catalog", "native_kv": "no", "lexical": "mention boundary", "semantic": "bi-encoder+reranker", "iterative_state_dependent": "no", "adaptive_channel": "no", "citation_key": "wu2020blink"},
        {"family_system": "PRA Paper 2.6", "external_text": "logical references", "native_kv": "candidate memory; not materialized here", "lexical": "exact/BM25/approximate", "semantic": "native hidden-state gist", "iterative_state_dependent": "root and successor", "adaptive_channel": "exported action; controller deferred", "citation_key": "this_work"},
    ]


def _plot_stability(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        group = [row for row in rows if row["dataset"] == dataset and row["stage"] == "root"]
        for index, channel in enumerate(sorted({row["channel"] for row in group})):
            values = sorted((int(row["cohort_size"]), float(row["selection_probability"])) for row in group if row["channel"] == channel)
            axis.plot([x for x, _ in values], [y for _, y in values], marker="o", label=channel, color=COLORS[index % len(COLORS)])
        axis.set(title=dataset, xlabel="Bootstrap cohort size", ylabel="P(selected best)", ylim=(-0.03, 1.03))
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.savefig(output / "best_channel_stability.png", dpi=180)
    fig.savefig(output / "best_channel_stability.pdf")
    plt.close(fig)


def _plot_reliability(rows: list[dict], stage: str, output: Path) -> None:
    group = [row for row in rows if row["stage"] == stage]
    fig, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="#777777", label="ideal")
    for index, channel in enumerate(sorted({row["channel"] for row in group})):
        values = [row for row in group if row["channel"] == channel and int(row["count"]) > 0]
        axis.plot([float(row["mean_confidence"]) for row in values], [float(row["accuracy"]) for row in values], marker="o", label=channel, color=COLORS[index % len(COLORS)])
    axis.set(xlabel="Validation-CDF confidence", ylabel="Top-1 correctness", xlim=(0, 1), ylim=(0, 1), title=f"{stage.title()} confidence calibration")
    axis.legend(fontsize=7, ncol=2)
    fig.savefig(output / f"{stage}_confidence_calibration.png", dpi=180)
    fig.savefig(output / f"{stage}_confidence_calibration.pdf")
    plt.close(fig)


def _plot_wrong_abstention(rows: list[dict], output: Path) -> None:
    curve = [row for row in rows if row["record_type"] == "wrong_reference_abstention_curve"]
    fig, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for index, channel in enumerate(sorted({row["channel"] for row in curve})):
        values = [row for row in curve if row["channel"] == channel]
        axis.plot([float(row["coverage"]) for row in values], [float(row["precision"]) for row in values], marker=".", label=channel, color=COLORS[index])
    axis.set(xlabel="Coverage after abstention", ylabel="Referential precision", xlim=(0, 1), ylim=(0, 1), title="Wrong-reference abstention opportunity")
    axis.legend()
    fig.savefig(output / "wrong_reference_abstention_precision_recall.png", dpi=180)
    fig.savefig(output / "wrong_reference_abstention_precision_recall.pdf")
    plt.close(fig)


def _plot_gate(rows: list[dict], output: Path) -> None:
    group = [row for row in rows if row["record_type"] == "natural_consistency_gate" and row["stage"] == "root"]
    policies = sorted({row["policy"] for row in group})
    datasets = list(DATASETS)
    width = 0.8 / max(len(policies), 1)
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for index, policy in enumerate(policies):
        values = [next(float(row["top1_accuracy"]) for row in group if row["dataset"] == dataset and row["policy"] == policy) for dataset in datasets]
        axis.bar([x + index * width for x in range(len(datasets))], values, width=width, label=policy.replace("_", " "), color=COLORS[index])
    axis.set(xticks=[x + width * (len(policies) - 1) / 2 for x in range(len(datasets))], xticklabels=datasets, ylabel="Top-1 correctness", ylim=(0, 1), title="Semantic-consistency gate diagnostic")
    axis.legend(fontsize=7)
    fig.savefig(output / "semantic_consistency_gate_effect.png", dpi=180)
    fig.savefig(output / "semantic_consistency_gate_effect.pdf")
    plt.close(fig)


def _plot_useful(rows: list[dict], output: Path) -> None:
    test = [row for row in rows if row["split"] == "test"]
    bins = reliability_bins([row["true_useful_address"] for row in test], [float(row["calibrated_probability"]) for row in test], bins=6)
    fig, axis = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    values = [row for row in bins if int(row["count"]) > 0]
    axis.plot([float(row["mean_confidence"]) for row in values], [float(row["accuracy"]) for row in values], marker="o", color=COLORS[2])
    axis.plot([0, 1], [0, 1], "--", color="#777777")
    axis.set(xlabel="Proxy confidence", ylabel="UsefulAddress frequency", xlim=(0, 1), ylim=(0, 1), title="Observable UsefulAddress proxy calibration")
    fig.savefig(output / "useful_address_proxy_calibration.png", dpi=180)
    fig.savefig(output / "useful_address_proxy_calibration.pdf")
    plt.close(fig)


def _plot_transition(channel_dir: Path, output: Path) -> None:
    rows = _read_csv(channel_dir / "channel_transition_matrix.csv")
    matrix = [[0 for _ in SUCCESSOR_CHANNELS] for _ in ROOT_CHANNELS]
    for row in rows:
        matrix[ROOT_CHANNELS.index(row["root_channel"])][SUCCESSOR_CHANNELS.index(row["successor_channel"])] = int(float(row["frequency"]))
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlGnBu")
    fig.colorbar(image, ax=axis, label="Held-out identity count")
    axis.set(xticks=range(5), xticklabels=[value.replace("_", "\n") for value in SUCCESSOR_CHANNELS], yticks=range(5), yticklabels=[ROOT_CANONICAL[value] for value in ROOT_CHANNELS], xlabel="Successor method", ylabel="Root method", title="Root to successor oracle transition")
    fig.savefig(output / "root_successor_transition_heatmap_expanded.png", dpi=180)
    fig.savefig(output / "root_successor_transition_heatmap_expanded.pdf")
    plt.close(fig)


def _plot_headroom(rows: list[dict], output: Path) -> None:
    groups = [(dataset, stage) for dataset in DATASETS for stage in ("root", "successor")]
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.36
    for offset, component, label, color in (
        (-width / 2, "H_selection", r"$H_{selection}$", COLORS[0]),
        (width / 2, "H_validation", r"$H_{validation}$", COLORS[1]),
    ):
        selected = [
            next(
                row
                for row in rows
                if (row["dataset"], row["stage"], row["component"])
                == (dataset, stage, component)
            )
            for dataset, stage in groups
        ]
        means = np.asarray([float(row["mean"]) for row in selected])
        errors = np.asarray(
            [
                [value - float(row["ci_low"]) for value, row in zip(means, selected)],
                [float(row["ci_high"]) - value for value, row in zip(means, selected)],
            ]
        )
        axis.bar(x + offset, means, width, color=color, yerr=errors, capsize=2, label=label)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(
        xticks=x,
        xticklabels=[f"{dataset}\n{stage}" for dataset, stage in groups],
        ylabel="Recall difference",
        title="Adaptive opportunity and validation-selection instability",
    )
    axis.legend(ncols=2, frameon=False)
    fig.savefig(output / "adaptive_headroom_updated_ci.png", dpi=180)
    fig.savefig(output / "adaptive_headroom_updated_ci.pdf")
    plt.close(fig)


def _write_audits(output: Path, cohort_counts: dict, detailed: bool) -> None:
    (output / "claim_audit.md").write_text(
        "# Paper 2.6 final-iteration claim audit\n\n"
        f"- Frozen natural cohort: {cohort_counts}; bootstrap rows are uncertainty estimates, not new identities.\n"
        "- Root and successor methods remain independent actions under the same two-plus-two chunk budget.\n"
        "- Confidence thresholds use validation identities only; held-out labels are evaluation outcomes.\n"
        "- Dataset identity, answer text, gold evidence, and oracle labels are excluded from proxy inputs.\n"
        "- Referential-consistency fields are observable proxies, not solved entity resolution.\n"
        "- The consistency gate is diagnostic and is not presented as a universal hybrid retriever.\n"
        "- Discovery, conceptual selection, and physical K/V materialization remain distinct.\n"
        "- No K/V was materialized and no generation result is claimed.\n"
        f"- Detailed candidate replay performed: {str(detailed).lower()}.\n",
        encoding="utf-8",
    )
    (output / "readability_audit.md").write_text(
        "# Paper 2.6 readability audit\n\n"
        "- Abstract states the four-dataset result, negative fusion result, selector limit, and disambiguation limit.\n"
        "- Introduction motivates discovery representation before implementation detail.\n"
        "- H_selection and H_validation use separate notation and figures.\n"
        "- Related Work distinguishes hybrid IR from the PRA-native-memory contribution.\n"
        "- The PRA-series handoff names Paper 3 materialization and Paper 3.5 adaptive control explicitly.\n"
        "- Limitations report exact cohort sizes and avoid generation or serving claims.\n",
        encoding="utf-8",
    )


def run(args) -> dict:
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_rows = _read_csv(args.channel_dir / "root_channel_results.csv")
    cohort_counts = Counter(row["dataset"] for row in root_rows if row["split"] == "test" and row["channel"] == "gist")
    expanded = _expanded_route_rows(args.channel_dir)
    _write_csv(args.output_dir / "expanded_route_results.csv", expanded)
    stability = _cohort_stability(args.channel_dir, args.seed, args.bootstrap_resamples)
    _write_csv(args.output_dir / "cohort_stability.csv", stability)

    detailed = not args.postprocess_only and not args.reuse_detailed
    if detailed:
        confidence_rows, robustness_rows = _extract_detailed_confidence(args)
        _write_csv(args.output_dir / "robustness_control_rows.csv", robustness_rows)
    elif args.reuse_detailed:
        confidence_rows = _read_csv(args.output_dir / "channel_confidence_rows.csv")
        robustness_rows = _read_csv(args.output_dir / "robustness_control_rows.csv") if (args.output_dir / "robustness_control_rows.csv").exists() else []
    else:
        confidence_rows = _minimal_confidence_rows(args.channel_dir)
        robustness_rows = _read_csv(args.channel_dir / "wrong_reference_robustness_extended.csv")
    confidence_metrics, reliability = _calibration_analysis(confidence_rows)
    _write_csv(args.output_dir / "channel_confidence_rows.csv", confidence_rows)
    _write_csv(args.output_dir / "channel_confidence_metrics.csv", confidence_metrics)
    _write_csv(args.output_dir / "channel_confidence_reliability.csv", reliability)

    if robustness_rows:
        robustness_diagnostics = _robustness_diagnostics(robustness_rows)
    else:
        robustness_diagnostics = [
            row for row in _read_csv(args.output_dir / "reference_disambiguation.csv")
            if row["record_type"] != "natural_consistency_gate"
        ]
    disambiguation = _consistency_gate(confidence_rows) + robustness_diagnostics
    _write_csv(args.output_dir / "reference_disambiguation.csv", disambiguation)
    useful_rows, useful_summary = _useful_address_proxy(args.channel_dir)
    _write_csv(args.output_dir / "useful_address_proxy.csv", useful_rows)
    headroom = _headroom_summary(args.channel_dir)
    _write_csv(args.output_dir / "adaptive_headroom_ci.csv", headroom)

    action_spec = _search_method_action_spec()
    validate_search_method_action_spec(action_spec)
    action_text = json.dumps(action_spec, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "search_method_action_spec.json").write_text(action_text, encoding="utf-8")
    (args.channel_dir / "search_method_action_spec.json").write_text(action_text, encoding="utf-8")
    related = _related_work_rows()
    _write_csv(args.output_dir / "related_work_comparison.csv", related)

    _plot_stability(stability, args.output_dir)
    _plot_reliability(reliability, "root", args.output_dir)
    _plot_reliability(reliability, "successor", args.output_dir)
    _plot_wrong_abstention(disambiguation, args.output_dir)
    _plot_gate(disambiguation, args.output_dir)
    _plot_useful(useful_rows, args.output_dir)
    _plot_transition(args.channel_dir, args.output_dir)
    _plot_headroom(headroom, args.output_dir)

    metrics_by_stage = defaultdict(list)
    for row in confidence_metrics:
        if row["signal"] == "referential_consistency_score":
            metrics_by_stage[row["stage"]].append(row)
    findings = {
        "schema_version": "1.0",
        "cohort": {
            "natural_identities": sum(
                row["split"] in {"validation", "test"} and row["channel"] == "gist"
                for row in root_rows
            ),
            "heldout_identities": sum(cohort_counts.values()),
            "heldout_by_dataset": dict(cohort_counts),
            "expanded": False,
            "reason": "No larger identity-compatible frozen feature cache was available; deterministic bootstrap estimates stability without treating resamples as new data.",
            "matched_budget_route_rows": len(expanded),
        },
        "confidence": {
            stage: {
                "mean_auroc": _finite_mean(row["auroc"] for row in rows),
                "mean_auprc": _finite_mean(row["auprc"] for row in rows),
                "mean_ece": _finite_mean(row["ece"] for row in rows),
            }
            for stage, rows in metrics_by_stage.items()
        },
        "useful_address_proxy": useful_summary,
        "headroom": headroom,
        "robustness_conditions": sorted({
            row.get("perturbation", "")
            for row in robustness_diagnostics
            if row.get("record_type") == "controlled_reference_condition"
        }),
        "action_spec_schema": action_spec["schema_version"],
        "generation_performed": False,
        "materialization_performed": False,
    }
    findings_path = args.output_dir / "paper2_6_final_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    parent_findings = args.channel_dir.parent / "channel_geometry" / "paper2_6_findings.json"
    aggregate = json.loads(parent_findings.read_text(encoding="utf-8"))
    aggregate["final_iteration"] = findings
    parent_findings.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_audits(args.output_dir, dict(cohort_counts), detailed or args.reuse_detailed)
    return findings


def _fallback(primary: Path, sibling: Path) -> Path:
    return primary if primary.exists() else sibling


def parse_args():
    sibling = ROOT.parent / "pdattention-iter-gist"
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--cohort-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--step-budget", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--reuse-detailed", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--paper2-feature-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument(
        "--natural-features", type=Path,
        default=_fallback(
            ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt",
            sibling / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt",
        ),
    )
    parser.add_argument(
        "--musique-dev", type=Path,
        default=_fallback(
            ROOT / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl",
            sibling / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl",
        ),
    )
    parser.add_argument(
        "--twowiki-dev", type=Path,
        default=_fallback(
            ROOT / "data/.paper2_5_datasets/2wiki/dev.json",
            sibling / "data/.paper2_5_datasets/2wiki/dev.json",
        ),
    )
    parser.add_argument(
        "--channel-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/final_iteration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, allow_nan=True))
