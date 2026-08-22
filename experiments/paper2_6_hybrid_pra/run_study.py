"""Run the Paper 2.6 token-native and hybrid discovery study.

The natural benchmark reuses Paper 2's frozen Qwen3-0.6B routing features.
No model weights are loaded and no K/V is materialized. Validation examples
select the hybrid weights and confidence bins; the identity-disjoint test split
is evaluated once with those choices frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex
from pra_hf.iterative import GistIndex, IterativeGistRouter, IterativeRoutingConfig
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


CONDITIONS = {
    "B0_gist": "gist_only",
    "B1_bm25": "bm25",
    "B2_exact": "token_exact",
    "B3_weighted": "token_weighted",
    "B4_approx": "token_approx",
    "H1_union": "union",
    "H2_token_semantic": "token_semantic_rerank",
    "H3_semantic_token": "semantic_token_rerank",
    "H4_cascade": "cascade",
    "H5_iterative_hybrid": "iterative_hybrid",
}
SPLITS = {"validation": (0, 8), "test": (8, 16)}


def _records(feature: dict, example: dict) -> tuple[GistIndex, set[str]]:
    """Reconstruct cache identities around frozen feature tensors."""
    uri = f"benchmark://{feature['dataset']}/{feature['example_id']}"
    chunks = []
    positive_ids = set()
    for index, ((start, end), gist, positive) in enumerate(
        zip(feature["chunk_spans"], feature["memory_gists"], feature["positive_mask"])
    ):
        chunk_id = f"{uri}#chunk={index}"
        dummy = torch.zeros((1, 1, int(end) - int(start), 1), dtype=torch.float32)
        chunks.append(
            ReferenceChunkMemory(
                chunk_id=chunk_id,
                source_uri=uri,
                token_start=int(start),
                token_end=int(end),
                token_kv=LayerKV(dummy, dummy),
                routing_gist=ChunkRoutingGist(k=gist.float().unsqueeze(0)),
                logical_start=int(start),
                logical_end=int(end),
            )
        )
        if bool(positive):
            positive_ids.add(chunk_id)
    entry = PRACacheEntry(
        uri=uri,
        text=example["source"],
        layer_memory={27: LayerReferenceMemory(chunks)},
    )
    return GistIndex.from_entries([entry], 27), positive_ids


def _metrics(selected: list[str], positive: set[str]) -> dict[str, float]:
    hits = [identity in positive for identity in selected]
    true_positive = sum(hits)
    recall = true_positive / max(len(positive), 1)
    precision = true_positive / max(len(selected), 1)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(positive), len(selected)) + 1)
    )
    return {
        "evidence_recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
        "ndcg": dcg / max(ideal, 1e-12),
        "any_hit": float(true_positive > 0),
        "path_completion": float(positive.issubset(selected)),
        "missed_evidence": 1.0 - recall,
    }


def _route_case(
    tokenizer,
    feature: dict,
    example: dict,
    condition: str,
    mode: str,
    weights: tuple[float, float],
    budget: int,
) -> tuple[dict, list[tuple[float, int, str]]]:
    gist_build_started = time.perf_counter()
    index, positives = _records(feature, example)
    gist_index_build_ms = (time.perf_counter() - gist_build_started) * 1000.0
    token_index = None
    token_index_build_ms = 0.0
    if mode not in {"gist_only", "oracle"}:
        token_build_started = time.perf_counter()
        token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
        token_index_build_ms = (time.perf_counter() - token_build_started) * 1000.0
    query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
    query = feature["queries"]["question_exp_h2.0"].float()
    started = time.perf_counter()
    if condition == "O1_oracle":
        selected = [identity for identity in index.chunk_ids if identity in positives][:budget]
        confidence_rows = [(1.0, 1, "oracle") for _ in selected]
        costs = {"semantic_gist_comparisons": 0, "token_index_comparisons": 0}
    else:
        route_budget = 8 if condition == "A1_broad_semantic" else budget
        iterative = condition == "H5_iterative_hybrid"
        policy = HybridDiscoveryPolicy(
            mode=mode,
            semantic_weight=weights[0],
            token_weight=1.0 - weights[0],
            later_semantic_weight=weights[1],
            later_token_weight=1.0 - weights[1],
        )
        config = IterativeRoutingConfig(
            depth=2 if iterative else 1,
            branch_top_k=max(1, budget // 2) if iterative else route_budget,
            beam_size=budget if iterative else route_budget,
            max_unique_chunks=route_budget,
            root_anchor_alpha=0.0 if iterative else 1.0,
            path_score_mode="last" if iterative else "direct",
        )
        route_arguments = {
            "example_id": f"{feature['dataset']}:{feature['example_id']}",
            "evidence_chunk_ids": positives,
        }
        if token_index is not None:
            route_arguments.update(
                token_index=token_index,
                root_token_ids=query_ids,
                tokenizer=tokenizer,
                discovery_policy=policy,
            )
        result = IterativeGistRouter(index).route(query, config, **route_arguments)
        selected = [index.chunk_ids[row] for row in result.selected_indices]
        confidence_rows = [
            (
                float(
                    node.confidence
                    if node.confidence is not None
                    else (node.direct_query_score + 1.0) / 2.0
                ),
                int(node.node_id in positives),
                str(node.discovery_channels.get("selected_channel", "semantic")),
            )
            for node in result.graph.nodes
        ]
        costs = result.graph.costs
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "split": feature["split"],
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "condition": condition,
        **_metrics(selected, positives),
        "logical_chunks": len(index.records),
        "requested_chunks": len(selected),
        "resident_chunks": len(index.records),
        "materialized_chunks": 0,
        "materialized_tokens": 0,
        "active_fraction": 0.0,
        "distractor_requests": sum(identity not in positives for identity in selected),
        "routing_ms": elapsed_ms,
        "gist_index_build_ms": gist_index_build_ms,
        "token_index_build_ms": token_index_build_ms,
        "cold_discovery_ms": gist_index_build_ms + token_index_build_ms + elapsed_ms,
        "semantic_comparisons": costs.get("semantic_gist_comparisons", 0),
        "token_comparisons": costs.get("token_index_comparisons", 0),
        "selected_chunk_ids": "|".join(selected),
        "positive_chunk_ids": "|".join(sorted(positives)),
        "selected_hops": "|".join(
            str(getattr(node, "hop", 0))
            for node in (result.graph.nodes if condition != "O1_oracle" else ())
            if node.node_id in selected
        ),
        "selected_hop_pairs": "|".join(
            f"{node.node_id}@{getattr(node, 'hop', 0)}"
            for node in (result.graph.nodes if condition != "O1_oracle" else ())
            if node.node_id in selected
        ),
    }
    return row, confidence_rows


def _fit_bins(rows: list[tuple[float, int, str]], bins: int = 10) -> list[dict]:
    """Fit a validation-only monotonic histogram calibrator."""
    ordered = sorted(rows, key=lambda row: row[0])
    fitted = []
    for start in range(0, len(ordered), max(1, math.ceil(len(ordered) / bins))):
        group = ordered[start : start + max(1, math.ceil(len(ordered) / bins))]
        fitted.append(
            {
                "low": group[0][0],
                "high": group[-1][0],
                "probability": sum(row[1] for row in group) / len(group),
                "count": len(group),
            }
        )
    for index in range(1, len(fitted)):
        fitted[index]["probability"] = max(
            fitted[index - 1]["probability"], fitted[index]["probability"]
        )
    return fitted


def _calibrate(value: float, bins: list[dict]) -> float:
    if not bins:
        return value
    return min(bins, key=lambda row: abs(value - (row["low"] + row["high"]) / 2))["probability"]


def _calibration_metrics(rows, bins) -> dict[str, float]:
    values = [(_calibrate(score, bins), label) for score, label, _ in rows]
    brier = sum((score - label) ** 2 for score, label in values) / max(len(values), 1)
    ece = 0.0
    for lower in [index / 10 for index in range(10)]:
        group = [(score, label) for score, label in values if lower <= score < lower + 0.1]
        if group:
            confidence = sum(score for score, _ in group) / len(group)
            accuracy = sum(label for _, label in group) / len(group)
            ece += len(group) / len(values) * abs(confidence - accuracy)
    high = [(score, label) for score, label in values if score >= 0.5]
    return {
        "brier": brier,
        "ece": ece,
        "coverage_at_0.5": len(high) / max(len(values), 1),
        "selective_precision_at_0.5": (
            sum(label for _, label in high) / len(high) if high else 0.0
        ),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "evidence_recall",
        "precision",
        "mrr",
        "ndcg",
        "any_hit",
        "path_completion",
        "missed_evidence",
        "requested_chunks",
        "distractor_requests",
        "routing_ms",
        "gist_index_build_ms",
        "token_index_build_ms",
        "cold_discovery_ms",
        "semantic_comparisons",
        "token_comparisons",
    )
    groups = {}
    for row in rows:
        groups.setdefault((row["split"], row["dataset"], row["condition"]), []).append(row)
    output = []
    for (split, dataset, condition), group in sorted(groups.items()):
        output.append(
            {
                "split": split,
                "dataset": dataset,
                "condition": condition,
                "examples": len(group),
                **{metric: sum(row[metric] for row in group) / len(group) for metric in metrics},
            }
        )
    return output


def _paired_effects(rows: list[dict], seed: int, samples: int = 10_000) -> list[dict]:
    """Bootstrap paired per-example recall differences against gist-only."""
    rng = random.Random(seed)
    test = [row for row in rows if row["split"] == "test"]
    lookup = {
        (row["dataset"], row["example_id"], row["condition"]): row
        for row in test
    }
    effects = []
    for dataset in ("hotpotqa", "qasper"):
        identities = sorted(
            {row["example_id"] for row in test if row["dataset"] == dataset}
        )
        baseline = {
            identity: lookup[(dataset, identity, "B0_gist")]["evidence_recall"]
            for identity in identities
        }
        for condition in ("B1_bm25", "B2_exact", "B3_weighted", "H5_iterative_hybrid"):
            differences = [
                lookup[(dataset, identity, condition)]["evidence_recall"]
                - baseline[identity]
                for identity in identities
            ]
            draws = sorted(
                sum(rng.choice(differences) for _ in differences) / len(differences)
                for _ in range(samples)
            )
            positive = sum(value > 0 for value in differences)
            negative = sum(value < 0 for value in differences)
            effects.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "baseline": "B0_gist",
                    "mean_difference": sum(differences) / len(differences),
                    "bootstrap_ci95_low": draws[int(0.025 * samples)],
                    "bootstrap_ci95_high": draws[int(0.975 * samples)],
                    "positive_examples": positive,
                    "negative_examples": negative,
                    "ties": len(differences) - positive - negative,
                }
            )
    return effects


def _token_spans(tokenizer, chunks: list[str]) -> tuple[str, list[tuple[int, int]]]:
    source = " || ".join(chunks)
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    spans = []
    cursor = 0
    for text in chunks:
        start = source.index(text, cursor)
        end = start + len(text)
        rows = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        spans.append((min(rows), max(rows) + 1))
        cursor = end
    return source, spans


def _synthetic_feature(tokenizer, index: int, perturbation: str) -> tuple[dict, dict]:
    """Create a chain where a first-hop chunk exposes the second-hop address."""
    entry = f"Meridian{index}"
    bridge = f"Cobalt{index}"
    target = bridge
    if perturbation == "case":
        target = bridge.upper()
    elif perturbation == "punctuation":
        target = f"{bridge},"
    elif perturbation == "typo":
        target = f"Coblat{index}"
    chunks = [
        f"{entry} points to {target}",
        f"Unrelated{index} contains generic words",
        f"Distractor{index} discusses another topic",
        f"{bridge} stores payload Zeta{index}",
    ]
    if perturbation == "confidently_wrong":
        chunks[0] = f"{entry} mentions Wrong{index} but Correct{index} is relevant"
        chunks[1] = f"Wrong{index} stores a plausible distractor"
        chunks[3] = f"Correct{index} stores payload Zeta{index}"
    source, spans = _token_spans(tokenizer, chunks)
    feature = {
        "split": "synthetic",
        "dataset": "synthetic",
        "example_id": f"{perturbation}-{index}",
        "chunk_spans": spans,
        "memory_gists": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.7, 0.0, 0.7, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        "positive_mask": torch.tensor([False, False, False, True]),
        "queries": {"question_exp_h2.0": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    }
    example = {
        "dataset": "synthetic",
        "id": feature["example_id"],
        "source": source,
        "question": f"Locate {entry}",
    }
    return feature, example


def _synthetic_study(tokenizer, weights, budget: int) -> tuple[list[dict], dict]:
    rows = []
    confidence = []
    conditions = {
        "B0_gist": "gist_only",
        "B1_bm25": "bm25",
        "B2_exact": "token_exact",
        "B3_weighted": "token_weighted",
        "B4_approx": "token_approx",
        "H5_iterative_hybrid": "iterative_hybrid",
    }
    for perturbation in ("clean", "case", "punctuation", "typo", "confidently_wrong"):
        for index in range(20):
            feature, example = _synthetic_feature(tokenizer, index, perturbation)
            for condition, mode in conditions.items():
                row, candidates = _route_case(
                    tokenizer, feature, example, condition, mode, weights, budget
                )
                row["perturbation"] = perturbation
                rows.append(row)
                if condition == "H5_iterative_hybrid":
                    confidence.extend(
                        (score, label, channel, perturbation)
                        for score, label, channel in candidates
                    )
    groups = {}
    for row in rows:
        groups.setdefault((row["perturbation"], row["condition"]), []).append(row)
    summary = [
        {
            "perturbation": perturbation,
            "condition": condition,
            "examples": len(group),
            "evidence_recall": sum(row["evidence_recall"] for row in group) / len(group),
            "precision": sum(row["precision"] for row in group) / len(group),
        }
        for (perturbation, condition), group in sorted(groups.items())
    ]
    wrong = [row for row in confidence if row[3] == "confidently_wrong" and not row[1]]
    diagnostics = {
        "wrong_candidate_mean_raw_confidence": (
            sum(row[0] for row in wrong) / len(wrong) if wrong else 0.0
        ),
        "wrong_candidate_count": len(wrong),
        "interpretation": "High-overlap contradictory names test graceful degradation; no gold metadata enters the index.",
    }
    return summary, diagnostics


def _plot(summary: list[dict], synthetic: list[dict], output_dir: Path) -> None:
    test = [row for row in summary if row["split"] == "test"]
    conditions = list(CONDITIONS) + ["A1_broad_semantic", "O1_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        values = {
            row["condition"]: row["evidence_recall"]
            for row in test
            if row["dataset"] == dataset
        }
        axis.bar(range(len(conditions)), [values.get(name, 0.0) for name in conditions])
        axis.set_title(dataset.upper())
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Supporting-evidence recall")
        axis.set_xticks(range(len(conditions)), [name.split("_")[0] for name in conditions], rotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "natural_recall.png", dpi=180)
    fig.savefig(output_dir / "natural_recall.pdf")
    plt.close(fig)

    conditions = ["B0_gist", "B1_bm25", "B2_exact", "B3_weighted", "B4_approx", "H5_iterative_hybrid"]
    perturbations = ["clean", "case", "punctuation", "typo", "confidently_wrong"]
    values = {
        (row["perturbation"], row["condition"]): row["evidence_recall"]
        for row in synthetic
    }
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = 0.13
    centers = list(range(len(perturbations)))
    for condition_index, condition in enumerate(conditions):
        offsets = [value + (condition_index - 2.5) * width for value in centers]
        axis.bar(offsets, [values[(perturbation, condition)] for perturbation in perturbations], width, label=condition.split("_")[0])
    axis.set_xticks(centers, perturbations)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Target-evidence recall")
    axis.set_title("Controlled second-hop address recovery")
    axis.legend(ncol=3)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "synthetic_perturbations.png", dpi=180)
    fig.savefig(output_dir / "synthetic_perturbations.pdf")
    plt.close(fig)


def _plot_calibration(confidence, calibration, output_dir: Path) -> None:
    """Plot held-out reliability after validation-only histogram calibration."""
    fig, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="ideal")
    for condition in ("B0_gist", "B1_bm25", "B2_exact", "H5_iterative_hybrid"):
        bins = calibration[condition]["validation_bins"]
        values = [
            (_calibrate(score, bins), label)
            for score, label, _ in confidence[("test", condition)]
        ]
        points = []
        for lower in [index / 10 for index in range(10)]:
            group = [(score, label) for score, label in values if lower <= score < lower + 0.1]
            if group:
                points.append(
                    (
                        sum(score for score, _ in group) / len(group),
                        sum(label for _, label in group) / len(group),
                    )
                )
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=condition.split("_")[0],
            )
    axis.set_xlabel("Calibrated confidence")
    axis.set_ylabel("Empirical candidate validity")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "confidence_reliability.png", dpi=180)
    fig.savefig(output_dir / "confidence_reliability.pdf")
    plt.close(fig)


def run(args) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
    )
    feature_dir = args.feature_dir
    examples_by_split = {}
    features_by_split = {}
    for split, (offset, count) in SPLITS.items():
        examples = load_split_examples(args.cache_dir, count, offset, args.seed)
        examples_by_split[split] = {(row["dataset"], row["id"]): row for row in examples}
        features = torch.load(
            feature_dir / f"router_features_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        features_by_split[split] = features

    weight_grid = [(entry, later) for entry in (0.4, 0.6, 0.8) for later in (0.1, 0.3, 0.5)]
    validation_scores = []
    for weights in weight_grid:
        recalls = []
        for feature in features_by_split["validation"]:
            example = examples_by_split["validation"][(feature["dataset"], feature["example_id"])]
            row, _ = _route_case(
                tokenizer, feature, example, "H5_iterative_hybrid", "iterative_hybrid", weights, args.budget
            )
            recalls.append(row["evidence_recall"])
        validation_scores.append((sum(recalls) / len(recalls), weights))
    _, selected_weights = max(
        validation_scores,
        key=lambda row: (row[0], row[1][0], -row[1][1]),
    )

    rows = []
    confidence: dict[tuple[str, str], list[tuple[float, int, str]]] = {}
    all_conditions = {**CONDITIONS, "A1_broad_semantic": "gist_only", "O1_oracle": "oracle"}
    for split in SPLITS:
        for feature in features_by_split[split]:
            example = examples_by_split[split][(feature["dataset"], feature["example_id"])]
            for condition, mode in all_conditions.items():
                row, candidate_rows = _route_case(
                    tokenizer,
                    feature,
                    example,
                    condition,
                    mode,
                    selected_weights,
                    args.budget,
                )
                rows.append(row)
                confidence.setdefault((split, condition), []).extend(candidate_rows)

    calibration = {}
    for condition in all_conditions:
        bins = _fit_bins(confidence.get(("validation", condition), []))
        calibration[condition] = {
            "validation_bins": bins,
            "test": _calibration_metrics(confidence.get(("test", condition), []), bins),
        }
    _plot_calibration(confidence, calibration, output_dir)
    summary = _aggregate(rows)
    paired_effects = _paired_effects(rows, args.seed)
    synthetic_summary, wrong_reference = _synthetic_study(
        tokenizer, selected_weights, budget=2
    )
    _plot(summary, synthetic_summary, output_dir)
    with (output_dir / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "synthetic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(synthetic_summary[0]))
        writer.writeheader()
        writer.writerows(synthetic_summary)
    with (output_dir / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_effects[0]))
        writer.writeheader()
        writer.writerows(paired_effects)
    result = {
        "protocol": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "feature_source": "frozen Paper-2 attention-input hidden states",
            "routing_layer": 27,
            "chunk_tokens": 32,
            "budget": args.budget,
            "broad_semantic_budget": 8,
            "materialization_performed": False,
            "selected_hybrid_weights": {
                "entry_semantic": selected_weights[0],
                "later_semantic": selected_weights[1],
            },
            "validation_grid": [
                {"recall": score, "entry": weights[0], "later": weights[1]}
                for score, weights in validation_scores
            ],
            "seed": args.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "summary": summary,
        "paired_effects": paired_effects,
        "synthetic_summary": synthetic_summary,
        "wrong_reference": wrong_reference,
        "calibration": calibration,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["protocol"], indent=2))
