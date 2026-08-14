"""Diagnose oracle convergence and true evidence-edge rank for Paper 2.5.

This module is intentionally diagnostic.  It reuses the frozen Gate-2/Gate-3
routers for selection and only forces an annotated source evidence group when
measuring the rank of its successor.  Oracle target identities never enter a
router score or the offline adaptive competition policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_oracle_memory_use import _oracle_selections
from experiments.paper2_5_iterative_pra.run_gate2_local_closure import (
    _evaluate as evaluate_semantic,
)
from experiments.paper2_5_iterative_pra.run_gate3_native_qk_closure import (
    NATIVE_VARIANTS,
    _evaluate_native,
    _native_index,
    _normalize_baseline,
)
from pra_hf.native_closure import native_local_qk_scores
from pra_torch.hf import load_hf_routing_projection


SEEDS = (11, 23, 37, 53, 71)
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40)
METHODS = (
    "one_shot_parent",
    "parent_closure",
    "local_gist_closure",
    "native_qk_max_topk_p20",
    "native_qk_top4_topk_p20",
)
GEOMETRIES = ("parent_semantic", "local_semantic", "native_qk")


def _parent_id(example_id: str, index: int) -> str:
    return f"{example_id}#parent={index}"


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(int(left[0]), int(right[0])) < min(int(left[1]), int(right[1]))


def evidence_parent_groups(feature: dict) -> list[set[int]]:
    """Map each annotation to all overlapping parents, preserving annotation order."""
    groups: list[set[int]] = []
    for evidence_span in feature["evidence_spans"]:
        group = {
            index
            for index, parent_span in enumerate(feature["parent_spans"])
            if _overlaps(tuple(parent_span), tuple(evidence_span))
        }
        if group and group not in groups:
            groups.append(group)
    return groups


def canonical_oracle_parent_indices(feature: dict, layer_id: int = 27) -> set[int]:
    """Call the Paper-2 oracle and return its parent identities unchanged."""
    chunks = [
        SimpleNamespace(
            chunk_id=_parent_id(feature["example_id"], index),
            logical_start=int(span[0]),
            logical_end=int(span[1]),
        )
        for index, span in enumerate(feature["parent_spans"])
    ]
    entry = SimpleNamespace(
        uri=f"memory://{feature['example_id']}",
        layer_memory={layer_id: SimpleNamespace(chunks=chunks)},
    )
    selected = _oracle_selections(entry, layer_id, list(feature["evidence_spans"]))
    oracle = {int(hit.chunk_id.rsplit("=", 1)[1]) for hit in selected}
    mask_oracle = set(
        torch.nonzero(feature["parent_positive_mask"], as_tuple=False)
        .flatten()
        .tolist()
    )
    if oracle != mask_oracle:
        raise ValueError(
            f"Paper-2 oracle mismatch for {feature['example_id']}: "
            f"canonical={sorted(oracle)}, cached={sorted(mask_oracle)}"
        )
    return oracle


def oracle_set_metrics(selected: set[int], oracle: set[int]) -> dict[str, float]:
    """Compute set recovery metrics with explicit empty-set behavior."""
    intersection = selected & oracle
    union = selected | oracle
    return {
        "oracle_recall": len(intersection) / len(oracle) if oracle else 1.0,
        "oracle_precision": len(intersection) / len(selected) if selected else 0.0,
        "oracle_jaccard": len(intersection) / len(union) if union else 1.0,
        "complete_oracle": float(oracle <= selected),
    }


def _selected_indices(graph: dict, example_id: str) -> set[int]:
    selected = set()
    for node in graph["nodes"]:
        if not node.get("final_selected"):
            continue
        parent_id = node.get("parent_chunk_id") or node["node_id"]
        prefix = f"{example_id}#parent="
        if parent_id.startswith(prefix):
            selected.add(int(parent_id[len(prefix) :]))
    return selected


def competition_rank(
    scores: torch.Tensor,
    targets: set[int],
    excluded: set[int] | None = None,
) -> dict:
    """Rank the best target against candidates; equal scores share rank."""
    excluded = excluded or set()
    candidates = [
        index
        for index in range(scores.numel())
        if index not in excluded and math.isfinite(float(scores[index]))
    ]
    valid_targets = sorted(targets.intersection(candidates))
    if not valid_targets:
        raise ValueError("No target remains in the candidate set.")
    target = max(valid_targets, key=lambda index: (float(scores[index]), -index))
    target_score = float(scores[target])
    rank = 1 + sum(float(scores[index]) > target_score for index in candidates)
    distractors = [index for index in candidates if index not in targets]
    distractor = max(
        distractors, key=lambda index: (float(scores[index]), -index), default=None
    )
    distractor_score = float(scores[distractor]) if distractor is not None else float("-inf")
    ordered = sorted(candidates, key=lambda index: (-float(scores[index]), index))
    top_gap = (
        float(scores[ordered[0]] - scores[ordered[1]]) if len(ordered) > 1 else float("inf")
    )
    probabilities = torch.softmax(scores[candidates].float(), dim=0)
    entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
    return {
        "target_parent": target,
        "target_rank": rank,
        "target_score": target_score,
        "top_distractor_parent": distractor,
        "best_distractor_score": distractor_score,
        "oracle_margin": target_score - distractor_score,
        "top1_top2_gap": top_gap,
        "score_entropy": entropy,
        "candidate_count": len(candidates),
    }


def parent_semantic_scores(
    parent_memory: torch.Tensor,
    parent_query: torch.Tensor,
    source_group: set[int],
) -> torch.Tensor:
    """Return strongest projected query-to-memory edge from a source group."""
    source = F.normalize(parent_query[sorted(source_group)].float(), dim=-1)
    target = F.normalize(parent_memory.float(), dim=-1)
    return source @ target.T if source.ndim == 1 else (source @ target.T).amax(dim=0)


def local_semantic_scores(
    local_memory: torch.Tensor,
    local_query: torch.Tensor,
    local_parent_indices: torch.Tensor,
    source_group: set[int],
    parent_count: int,
) -> torch.Tensor:
    """Reduce all source-local to target-local similarities by target parent."""
    parent_indices = local_parent_indices.to(local_memory.device)
    source_mask = torch.zeros_like(parent_indices, dtype=torch.bool)
    for parent in source_group:
        source_mask |= parent_indices == parent
    source = F.normalize(local_query[source_mask].float(), dim=-1)
    memory = F.normalize(local_memory.float(), dim=-1)
    pair_scores = source @ memory.T
    scores = torch.full((parent_count,), float("-inf"), device=memory.device)
    for parent in range(parent_count):
        mask = parent_indices == parent
        if bool(mask.any()):
            scores[parent] = pair_scores[:, mask].max()
    return scores


def native_qk_parent_scores(
    feature: dict,
    source_group: set[int],
    device: torch.device,
    *,
    token_reduction: str = "max",
    head_reduction: str = "max",
    top_m: int = 4,
) -> tuple[torch.Tensor, int]:
    """Apply the exact Gate-3 scorer and max-reduce local edges by parent."""
    parent_indices = feature["local_parent_indices"].to(device)
    source_mask = torch.zeros_like(parent_indices, dtype=torch.bool)
    for parent in source_group:
        source_mask |= parent_indices == parent
    queries = feature["local_pre_query"].to(device)[source_mask]
    query_mask = feature["local_token_mask"].to(device)[source_mask]
    keys = feature["local_pre_key"].to(device)
    key_mask = feature["local_token_mask"].to(device)
    local_scores = native_local_qk_scores(
        queries,
        keys,
        query_mask,
        key_mask,
        token_reduction=token_reduction,
        head_reduction=head_reduction,
        top_m=top_m,
    )
    scores = torch.full(
        (len(feature["parent_spans"]),), float("-inf"), device=device
    )
    for parent in range(scores.numel()):
        mask = parent_indices == parent
        if bool(mask.any()):
            scores[parent] = local_scores.scores[:, mask].max()
    return scores, local_scores.dot_products


def validation_partition(example_id: str) -> str:
    """Assign examples stably without looking at scores, ranks, or labels."""
    digest = hashlib.sha256(example_id.encode("utf-8")).digest()
    return "validation" if digest[0] % 2 == 0 else "test"


def fit_adaptive_threshold(rows: list[dict]) -> dict:
    """Choose a validation-only margin threshold, preferring lower expansion cost."""
    validation = [row for row in rows if row["partition"] == "validation"]
    if not validation:
        raise ValueError("Adaptive policy needs validation rows.")
    gaps = sorted({float(row["top1_top2_gap"]) for row in validation})
    candidates = [float("-inf"), *gaps, float("inf")]
    records = []
    for threshold in candidates:
        success = []
        widths = []
        for row in validation:
            width = 1 if float(row["top1_top2_gap"]) > threshold else 4
            widths.append(width)
            success.append(float(row["target_rank"] <= width))
        records.append(
            {
                "threshold": threshold,
                "success": statistics.fmean(success),
                "mean_k": statistics.fmean(widths),
            }
        )
    return max(records, key=lambda row: (row["success"], -row["mean_k"], -row["threshold"]))


def evaluate_adaptive_policy(rows: list[dict], threshold: float) -> dict:
    """Evaluate a frozen threshold on held-out examples only."""
    test = [row for row in rows if row["partition"] == "test"]
    if not test:
        raise ValueError("Adaptive policy needs held-out test rows.")
    widths = [1 if float(row["top1_top2_gap"]) > threshold else 4 for row in test]
    return {
        "test_edges": len(test),
        "success": statistics.fmean(
            float(row["target_rank"] <= width) for row, width in zip(test, widths)
        ),
        "recall_at_1": statistics.fmean(float(row["target_rank"] <= 1) for row in test),
        "recall_at_4": statistics.fmean(float(row["target_rank"] <= 4) for row in test),
        "mean_k": statistics.fmean(widths),
    }


def _edge_summary(rows: list[dict]) -> list[dict]:
    output = []
    for geometry in GEOMETRIES:
        values = [row for row in rows if row["geometry"] == geometry]
        if geometry == "native_qk":
            values = list(
                {
                    (row["example_id"], row["transition"]): row for row in values
                }.values()
            )
        ranks = [int(row["target_rank"]) for row in values]
        if not ranks:
            continue
        quantiles = statistics.quantiles(ranks, n=4, method="inclusive")
        output.append(
            {
                "geometry": geometry,
                "edges": len(values),
                "mean_rank": statistics.fmean(ranks),
                "median_rank": statistics.median(ranks),
                "q1_rank": quantiles[0],
                "q3_rank": quantiles[2],
                "recall_at_1": statistics.fmean(rank <= 1 for rank in ranks),
                "recall_at_2": statistics.fmean(rank <= 2 for rank in ranks),
                "recall_at_4": statistics.fmean(rank <= 4 for rank in ranks),
                "recall_at_8": statistics.fmean(rank <= 8 for rank in ranks),
                "recall_at_16": statistics.fmean(rank <= 16 for rank in ranks),
                "mrr": statistics.fmean(1.0 / rank for rank in ranks),
                "mean_oracle_margin": statistics.fmean(
                    float(row["oracle_margin"]) for row in values
                ),
            }
        )
    return output


def _aggregate_convergence(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["fraction"], row["method"])].append(row)
    output = []
    for (dataset, fraction, method), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "fraction": fraction,
                "method": method,
                "rows": len(values),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in values)
                    for metric in (
                        "oracle_recall",
                        "oracle_precision",
                        "oracle_jaccard",
                        "complete_oracle",
                        "evidence_group_chain",
                        "routing_comparisons",
                        "semantic_gist_comparisons",
                        "native_qk_dot_products",
                        "materialized_kv_tokens",
                        "materialized_kv_fraction",
                    )
                },
            }
        )
    return output


def _minimum_complete_fractions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["example_id"], row["seed"], row["method"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        feasible = [
            float(row["fraction"])
            for row in values
            if row["complete_oracle"] and row["oracle_feasible"]
        ]
        output.append(
            {
                "dataset": key[0],
                "example_id": key[1],
                "seed": key[2],
                "method": key[3],
                "minimum_complete_fraction": min(feasible) if feasible else None,
            }
        )
    return output


def _geometry_comparison(edge_rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Compare the best semantic rank with native QK for every transition/seed."""
    keyed = {
        (row["example_id"], row["seed"], row["transition"], row["geometry"]): row
        for row in edge_rows
    }
    output = []
    for key, parent in keyed.items():
        if key[3] != "parent_semantic":
            continue
        prefix = key[:3]
        local = keyed[(*prefix, "local_semantic")]
        native = keyed[(*prefix, "native_qk")]
        semantic_rank = min(int(parent["target_rank"]), int(local["target_rank"]))
        native_rank = int(native["target_rank"])
        if semantic_rank <= 4 and native_rank <= 4:
            classification = "both_near_top"
        elif semantic_rank <= 4:
            classification = "semantic_wins"
        elif native_rank <= 4:
            classification = "native_qk_wins"
        else:
            classification = "both_poor"
        output.append(
            {
                "example_id": key[0],
                "seed": key[1],
                "transition": key[2],
                "parent_semantic_rank": parent["target_rank"],
                "local_semantic_rank": local["target_rank"],
                "native_qk_rank": native["target_rank"],
                "classification": classification,
            }
        )
    counts = dict(sorted(Counter(row["classification"] for row in output).items()))
    return output, counts


def _top4_signal_cases(
    convergence_rows: list[dict], edge_rows: list[dict]
) -> list[dict]:
    """Audit cases where Gate-3 Top-4 reduction completes a chain and max does not."""
    keyed = {
        (row["example_id"], row["seed"], row["fraction"], row["method"]): row
        for row in convergence_rows
        if row["dataset"] == "hotpotqa"
    }
    native_edges = {
        (row["example_id"], row["seed"], row["transition"]): row
        for row in edge_rows
        if row["geometry"] == "native_qk"
    }
    top4_edges = {
        (row["example_id"], row["seed"], row["transition"]): row
        for row in edge_rows
        if row["geometry"] == "native_qk_top4_reduction"
    }
    output = []
    for key, top4 in keyed.items():
        if key[2] != 0.20 or key[3] != "native_qk_top4_topk_p20":
            continue
        primary = keyed[(*key[:3], "native_qk_max_topk_p20")]
        if not top4["evidence_group_chain"] or primary["evidence_group_chain"]:
            continue
        edges = [
            row
            for edge_key, row in native_edges.items()
            if edge_key[0] == key[0] and edge_key[1] == key[1]
        ]
        for edge in edges:
            top4_edge = top4_edges[(key[0], key[1], edge["transition"])]
            if edge["target_rank"] > 1 and top4_edge["target_rank"] == 1:
                mechanism = "top4_reduction_reorders_oracle_to_top1"
            elif top4_edge["target_rank"] > 1:
                mechanism = "root_or_reverse_order_selection_effect"
            else:
                mechanism = "oracle_already_top1"
            output.append(
                {
                    "example_id": key[0],
                    "seed": key[1],
                    "transition": edge["transition"],
                    "target_rank": edge["target_rank"],
                    "target_score": edge["target_score"],
                    "top4_reduction_target_rank": top4_edge["target_rank"],
                    "top4_reduction_target_score": top4_edge["target_score"],
                    "top4_reduction_oracle_margin": top4_edge["oracle_margin"],
                    "top1_distractor_parent": edge["top_distractor_parent"],
                    "top1_distractor_score": edge["best_distractor_score"],
                    "oracle_margin": edge["oracle_margin"],
                    "top1_top2_gap": edge["top1_top2_gap"],
                    "distractor_is_oracle_parent": float(
                        edge["top_distractor_parent"]
                        in set(json.loads(primary["oracle_parent_ids"]))
                    ),
                    "same_parent": 0.0,
                    "entity_topic_annotation_available": 0.0,
                    "observed_mechanism": mechanism,
                    "primary_selected_parent_ids": primary["selected_parent_ids"],
                    "top4_selected_parent_ids": top4["selected_parent_ids"],
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_convergence(aggregate: list[dict], output_dir: Path) -> None:
    labels = {
        "one_shot_parent": "One-shot semantic",
        "parent_closure": "Parent iterative",
        "local_gist_closure": "Local iterative",
        "native_qk_max_topk_p20": "Native QK max",
        "native_qk_top4_topk_p20": "Native QK Top-4 reduction",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for method, label in labels.items():
            values = sorted(
                [
                    row
                    for row in aggregate
                    if row["dataset"] == dataset and row["method"] == method
                ],
                key=lambda row: row["fraction"],
            )
            axis.plot(
                [100 * row["fraction"] for row in values],
                [row["oracle_recall"] for row in values],
                marker="o",
                label=label,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Active parent/KV budget (%)")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Annotated-oracle parent recall")
    axes[1].legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"oracle_convergence.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_edge_rank(summary: list[dict], output_dir: Path) -> None:
    labels = [row["geometry"].replace("_", " ") for row in summary]
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    x = torch.arange(len(summary)).numpy()
    width = 0.25
    for offset, metric, label in (
        (-width, "recall_at_1", "R@1"),
        (0.0, "recall_at_4", "R@4"),
        (width, "recall_at_8", "R@8"),
    ):
        axis.bar(x + offset, [row[metric] for row in summary], width, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("True oracle-edge recall")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"oracle_edge_rank.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _convergence_row(
    feature: dict, row: dict, graph: dict, oracle: set[int], fraction: float
) -> dict:
    selected = _selected_indices(graph, feature["example_id"])
    groups = evidence_parent_groups(feature)
    comparisons = float(row.get("semantic_gist_comparisons", 0)) + float(
        row.get("native_qk_dot_products", 0)
    )
    return {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": row["seed"],
        "fraction": fraction,
        "method": row["condition"],
        "budget_parents": row["budget_parents"],
        "oracle_parent_ids": json.dumps(sorted(oracle)),
        "selected_parent_ids": json.dumps(sorted(selected)),
        "oracle_size": len(oracle),
        "oracle_feasible": float(len(oracle) <= row["budget_parents"]),
        **oracle_set_metrics(selected, oracle),
        "evidence_group_chain": float(
            bool(groups) and all(bool(selected & group) for group in groups)
        ),
        "routing_comparisons": comparisons,
        "semantic_gist_comparisons": row.get("semantic_gist_comparisons", 0),
        "native_qk_dot_products": row.get("native_qk_dot_products", 0),
        "materialized_kv_tokens": row["materialized_kv_tokens"],
        "materialized_kv_fraction": row["materialized_kv_fraction"],
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_file, weights_only=False)
    convergence_rows: list[dict] = []
    edge_rows: list[dict] = []
    native_edge_cache: dict[tuple[str, int], dict] = {}

    for seed in args.seeds:
        checkpoint = args.projection_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in features:
            oracle = canonical_oracle_parent_indices(feature)
            groups = evidence_parent_groups(feature)
            with torch.no_grad():
                root = projection.project_query(
                    feature["query_hidden"].to(device).unsqueeze(0)
                )[0]
                parent_hidden = feature["parent_hidden"].to(device)
                local_hidden = feature["local_hidden"].to(device)
                pm = projection.project_memory(parent_hidden)
                pq = projection.project_query(parent_hidden)
                lm = projection.project_memory(local_hidden)
                lq = projection.project_query(local_hidden)
            native_index = _native_index(feature, pm, lm, lq, device)
            for fraction in args.fractions:
                for condition in METHODS[:3]:
                    row, graph = evaluate_semantic(
                        feature,
                        root,
                        pm,
                        pq,
                        lm,
                        lq,
                        seed=seed,
                        fraction=fraction,
                        condition=condition,
                    )
                    row = _normalize_baseline(row, len(feature["local_spans"]))
                    convergence_rows.append(
                        _convergence_row(feature, row, graph, oracle, fraction)
                    )
                for variant in (NATIVE_VARIANTS[0], NATIVE_VARIANTS[1]):
                    row, graph = _evaluate_native(
                        feature,
                        root,
                        native_index,
                        seed=seed,
                        fraction=fraction,
                        candidate_fraction=0.20,
                        variant=variant,
                    )
                    convergence_rows.append(
                        _convergence_row(feature, row, graph, oracle, fraction)
                    )

            if feature["dataset"] != "hotpotqa" or len(groups) < 2:
                continue
            for transition_index, (source_group, target_group) in enumerate(
                zip(groups, groups[1:])
            ):
                score_sets = {
                    "parent_semantic": (
                        parent_semantic_scores(pm, pq, source_group),
                        len(source_group) * len(feature["parent_spans"]),
                    ),
                    "local_semantic": (
                        local_semantic_scores(
                            lm,
                            lq,
                            feature["local_parent_indices"],
                            source_group,
                            len(feature["parent_spans"]),
                        ),
                        sum(
                            int(parent) in source_group
                            for parent in feature["local_parent_indices"].tolist()
                        )
                        * len(feature["local_spans"]),
                    ),
                }
                native_key = (feature["example_id"], transition_index)
                if native_key not in native_edge_cache:
                    native_scores, dots = native_qk_parent_scores(
                        feature, source_group, device
                    )
                    top4_scores, top4_dots = native_qk_parent_scores(
                        feature,
                        source_group,
                        device,
                        token_reduction="top_m_mean",
                        head_reduction="top_m_mean",
                        top_m=4,
                    )
                    native_edge_cache[native_key] = {
                        "native_qk": {
                            **competition_rank(native_scores, target_group, source_group),
                            "routing_comparisons": dots,
                        },
                        "native_qk_top4_reduction": {
                            **competition_rank(top4_scores, target_group, source_group),
                            "routing_comparisons": top4_dots,
                        },
                    }
                for geometry, (scores, comparisons) in score_sets.items():
                    rank = competition_rank(scores, target_group, source_group)
                    edge_rows.append(
                        {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "seed": seed,
                            "transition": transition_index,
                            "source_oracle_parents": json.dumps(sorted(source_group)),
                            "target_oracle_parents": json.dumps(sorted(target_group)),
                            "geometry": geometry,
                            "partition": validation_partition(feature["example_id"]),
                            **rank,
                            "routing_comparisons": comparisons,
                        }
                    )
                for native_geometry, native_rank in native_edge_cache[native_key].items():
                    edge_rows.append(
                        {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "seed": seed,
                            "transition": transition_index,
                            "source_oracle_parents": json.dumps(sorted(source_group)),
                            "target_oracle_parents": json.dumps(sorted(target_group)),
                            "geometry": native_geometry,
                            "partition": validation_partition(feature["example_id"]),
                            **native_rank,
                        }
                    )
        print(
            f"oracle diagnostic seed {seed}: {len(convergence_rows)} convergence rows, "
            f"{len(edge_rows)} edge rows",
            flush=True,
        )

    aggregate = _aggregate_convergence(convergence_rows)
    edge_summary = _edge_summary(edge_rows)
    geometry_rows, geometry_counts = _geometry_comparison(edge_rows)
    top4_cases = _top4_signal_cases(convergence_rows, edge_rows)
    adaptive = []
    for geometry in GEOMETRIES:
        adaptive_rows = [row for row in edge_rows if row["geometry"] == geometry]
        if geometry == "native_qk":
            adaptive_rows = list(
                {
                    (row["example_id"], row["transition"]): row
                    for row in adaptive_rows
                }.values()
            )
        fit = fit_adaptive_threshold(adaptive_rows)
        adaptive.append(
            {
                "geometry": geometry,
                "validation_threshold": fit["threshold"],
                "validation_success": fit["success"],
                "validation_mean_k": fit["mean_k"],
                **{f"test_{key}": value for key, value in evaluate_adaptive_policy(
                    adaptive_rows, fit["threshold"]
                ).items()},
            }
        )

    # Root discovery is evaluated at the primary 20% budget, while propagation
    # is the oracle-conditioned edge rank above.  This keeps the two failures separate.
    primary = [
        row
        for row in convergence_rows
        if row["dataset"] == "hotpotqa" and row["fraction"] == 0.20
    ]
    decomposition = []
    for method in METHODS:
        values = [row for row in primary if row["method"] == method]
        first_hits = []
        for row in values:
            feature = next(item for item in features if item["example_id"] == row["example_id"])
            groups = evidence_parent_groups(feature)
            selected = set(json.loads(row["selected_parent_ids"]))
            first_hits.append(float(bool(groups and selected & groups[0])))
        decomposition.append(
            {
                "method": method,
                "root_first_group_recall": statistics.fmean(first_hits),
                "full_oracle_recovery": statistics.fmean(
                    float(row["complete_oracle"]) for row in values
                ),
            }
        )

    output = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "canonical_oracle": "experiments.paper2_hf.qa.run_oracle_memory_use._oracle_selections",
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "methods": list(METHODS),
        "convergence": aggregate,
        "edge_rank": edge_summary,
        "geometry_comparison_counts": geometry_counts,
        "top4_signal_cases": len(top4_cases),
        "adaptive_competition": adaptive,
        "failure_decomposition": decomposition,
        "minimum_complete_fractions": _minimum_complete_fractions(convergence_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_convergence_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "oracle_convergence_rows.csv", convergence_rows)
    _write_csv(args.output_dir / "oracle_convergence_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "oracle_edge_rank_rows.csv", edge_rows)
    _write_csv(args.output_dir / "oracle_edge_rank_summary.csv", edge_summary)
    _write_csv(args.output_dir / "geometry_comparison_rows.csv", geometry_rows)
    _write_csv(args.output_dir / "top4_signal_cases.csv", top4_cases)
    _write_csv(args.output_dir / "adaptive_competition.csv", adaptive)
    _plot_convergence(aggregate, args.output_dir)
    _plot_edge_rank(edge_summary, args.output_dir)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/oracle_convergence",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    args.fractions = tuple(map(float, args.fractions.split(",")))
    return args


if __name__ == "__main__":
    artifact = run(parse_args())
    print(json.dumps({"edge_rank": artifact["edge_rank"]}, indent=2))
