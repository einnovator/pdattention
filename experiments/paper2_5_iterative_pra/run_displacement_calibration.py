"""Diagnose root-hit displacement and cross-family score calibration.

The runner is offline and additive. It calls the frozen Gate-2/Gate-3 routers,
reuses the exact Paper-2 oracle, and never modifies SDK routing or materializes
K/V during diagnosis. Calibration controls rerank only a fixed union of root
and iterative candidates already scored by those routers.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_5_iterative_pra.run_gate2_local_closure import (
    _evaluate as evaluate_semantic,
)
from experiments.paper2_5_iterative_pra.run_gate3_native_qk_closure import (
    NATIVE_VARIANTS,
    _evaluate_native,
    _native_index,
    _normalize_baseline,
)
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    FRACTIONS,
    METHODS,
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    oracle_set_metrics,
    validation_partition,
)
from pra_hf.iterative import IterativeGistRouter
from pra_hf.native_closure import native_local_qk_scores
from pra_torch.hf import load_hf_routing_projection


PRIMARY_FRACTION = 0.20
ITERATIVE_METHODS = METHODS[1:]
CALIBRATION_MODES = ("family_zscore", "family_quantile")


def _parent_index(parent_id: str) -> int:
    return int(parent_id.split("#parent=", 1)[1].split("#", 1)[0])


def _graph_nodes(graph: dict) -> dict[int, dict]:
    """Return one final node per parent, preferring the strongest path trace."""
    output: dict[int, dict] = {}
    for node in graph["nodes"]:
        if not node.get("final_selected"):
            continue
        parent = _parent_index(node.get("parent_chunk_id") or node["node_id"])
        if parent not in output or float(node["path_score"]) > float(
            output[parent]["path_score"]
        ):
            output[parent] = node
    return output


def _selection_order(graph: dict) -> list[int]:
    return list(_graph_nodes(graph))


def classify_displacement(
    oracle: set[int],
    one_shot_graph: dict,
    iterative_graph: dict,
    *,
    method: str,
    budget: int,
) -> list[dict]:
    """Classify each oracle parent selected by one-shot as preserved/displaced."""
    one_nodes = _graph_nodes(one_shot_graph)
    iterative_nodes = _graph_nodes(iterative_graph)
    one_order = list(one_nodes)
    iterative_selected = set(iterative_nodes)
    replacements = [
        parent for parent in iterative_nodes if parent not in set(one_order)
    ]
    root_scores = torch.tensor(
        [float(node["direct_query_score"]) for node in one_nodes.values()]
    )
    root_probabilities = torch.softmax(root_scores, dim=0)
    root_entropy = float(
        (-(root_probabilities * root_probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    root_margin = (
        float(root_scores[0] - root_scores[1])
        if root_scores.numel() > 1
        else None
    )
    initial_slots = sum(node["hop"] == 1 for node in iterative_nodes.values())
    rows = []
    for parent in [value for value in one_order if value in oracle]:
        preserved = parent in iterative_selected
        one_rank = one_order.index(parent) + 1
        replacing_nodes = [iterative_nodes[value] for value in replacements]
        if preserved:
            reason = "preserved"
        elif one_rank > initial_slots:
            reason = "root_slot_partition"
        else:
            reason = "iterative_candidate_reranking"
        rows.append(
            {
                "oracle_parent": parent,
                "status": "preserved" if preserved else "displaced",
                "root_oracle_displaced": float(not preserved),
                "one_shot_score": float(one_nodes[parent]["direct_query_score"]),
                "one_shot_rank": one_rank,
                "final_iterative_rank": (
                    list(iterative_nodes).index(parent) + 1 if preserved else None
                ),
                "replacing_parents": json.dumps(replacements),
                "replacing_scores": json.dumps(
                    [float(node["path_score"]) for node in replacing_nodes]
                ),
                "replacing_hops": json.dumps(
                    [int(node["hop"]) for node in replacing_nodes]
                ),
                "replacing_score_families": json.dumps(
                    [score_family(node) for node in replacing_nodes]
                ),
                "displacement_reason": reason,
                "budget_reason": (
                    f"{initial_slots}_of_{budget}_slots_reserved_for_root"
                ),
                "dedup_reason": "none_parent_identity_deduplicated",
                "root_top1_top2_margin": root_margin,
                "root_selected_entropy": root_entropy,
                "replacement_hop2_fraction": (
                    statistics.fmean(int(node["hop"] == 2) for node in replacing_nodes)
                    if replacing_nodes
                    else 0.0
                ),
                "candidate_pool_fraction": iterative_graph.get("budget", {}).get(
                    "candidate_pool_fraction", 1.0
                ),
                "method": method,
            }
        )
    return rows


def protected_root_selection(
    one_shot_order: list[int],
    iterative_order: list[int],
    oracle: set[int],
    budget: int,
) -> list[int]:
    """Protect known one-shot oracle hits, then fill from frozen iterative order."""
    selected = [parent for parent in one_shot_order if parent in oracle][:budget]
    for parent in (*iterative_order, *one_shot_order):
        if parent not in selected and len(selected) < budget:
            selected.append(parent)
    return selected


def score_family(node: dict) -> str:
    if int(node["hop"]) == 1 or node.get("projection_type") == "root_query":
        return "root_semantic"
    if node.get("representation_type") == "pre_rope_native_qk":
        return "native_qk"
    return "propagated_semantic"


def _candidate_representations(
    feature: dict,
    seed: int,
    method: str,
    oracle: set[int],
    one_shot_graph: dict,
    iterative_graph: dict,
) -> list[dict]:
    """Build the fixed identity/score pool used by offline calibration controls."""
    rows = []
    seen = set()
    one_selected = set(_graph_nodes(one_shot_graph))
    iterative_selected = set(_graph_nodes(iterative_graph))
    for graph in (one_shot_graph, iterative_graph):
        for parent, node in _graph_nodes(graph).items():
            family = score_family(node)
            key = (parent, family)
            if key in seen:
                continue
            seen.add(key)
            raw = (
                float(node["direct_query_score"])
                if family == "root_semantic"
                else float(node["edge_score"])
            )
            rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "seed": seed,
                    "method": method,
                    "partition": validation_partition(feature["example_id"]),
                    "candidate_parent": parent,
                    "score_family": family,
                    "raw_score": raw,
                    "is_oracle": float(parent in oracle),
                    "selected_by_one_shot": float(parent in one_selected),
                    "selected_by_iterative": float(parent in iterative_selected),
                    "source": "frozen_selected_union",
                }
            )
    return rows


def fit_family_calibration(rows: list[dict], method: str) -> dict:
    """Fit score-family moments/CDFs on validation examples only."""
    selected = [
        row
        for row in rows
        if row["method"] == method and row["partition"] == "validation"
    ]
    if not selected:
        raise ValueError(f"No validation calibration rows for {method}.")
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["score_family"]].append(float(row["raw_score"]))
    return {
        family: {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) or 1.0,
            "sorted": sorted(values),
            "count": len(values),
        }
        for family, values in grouped.items()
    }


def calibrated_score(row: dict, fit: dict, mode: str) -> float:
    """Transform one frozen score without consulting identity or oracle labels."""
    family = fit[row["score_family"]]
    value = float(row["raw_score"])
    if mode == "family_zscore":
        return (value - family["mean"]) / family["std"]
    if mode == "family_quantile":
        return bisect.bisect_right(family["sorted"], value) / family["count"]
    raise ValueError(f"Unknown calibration mode: {mode}")


def calibrated_selection(
    rows: list[dict], fit: dict, mode: str, budget: int
) -> list[int]:
    """Rank parent identities by their best transformed family representation."""
    parent_scores: dict[int, float] = {}
    for row in rows:
        parent = int(row["candidate_parent"])
        transformed = calibrated_score(row, fit, mode)
        parent_scores[parent] = max(parent_scores.get(parent, float("-inf")), transformed)
    return [
        parent
        for parent, _ in sorted(
            parent_scores.items(), key=lambda item: (-item[1], item[0])
        )[:budget]
    ]


def _score_rows(
    feature: dict,
    seed: int,
    method: str,
    family: str,
    source: int | None,
    scores: torch.Tensor,
    oracle: set[int],
    excluded: set[int],
) -> list[dict]:
    candidates = [
        index
        for index in range(scores.numel())
        if index not in excluded and math.isfinite(float(scores[index]))
    ]
    ordered = sorted(candidates, key=lambda index: (-float(scores[index]), index))
    if not ordered:
        return []
    probabilities = torch.softmax(scores[ordered].float(), dim=0)
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    top1_top2 = (
        float(scores[ordered[0]] - scores[ordered[1]])
        if len(ordered) > 1
        else float("inf")
    )
    top4 = ordered[:4]
    top4_spread = float(scores[top4[0]] - scores[top4[-1]])
    return [
        {
            "dataset": feature["dataset"],
            "example_id": feature["example_id"],
            "seed": seed,
            "method": method,
            "score_family": family,
            "source_parent": source,
            "candidate_parent": candidate,
            "raw_score": float(scores[candidate]),
            "rank": ordered.index(candidate) + 1,
            "score_quantile": 1.0 - ordered.index(candidate) / max(len(ordered) - 1, 1),
            "is_oracle": float(candidate in oracle),
            "top1_top2_margin": top1_top2,
            "top4_spread": top4_spread,
            "score_entropy": entropy,
            "candidate_count": len(ordered),
        }
        for candidate in ordered
    ]


def _best_root_local(
    direct_local: torch.Tensor, local_parent_indices: torch.Tensor, parent: int
) -> int:
    rows = torch.nonzero(local_parent_indices == parent, as_tuple=False).flatten()
    return int(rows[torch.argmax(direct_local[rows])])


def semantic_family_rows(
    feature: dict,
    seed: int,
    method: str,
    root: torch.Tensor,
    pm: torch.Tensor,
    pq: torch.Tensor,
    lm: torch.Tensor,
    lq: torch.Tensor,
    oracle: set[int],
    budget: int,
) -> list[dict]:
    """Capture full root and propagated semantic candidate distributions."""
    pmn, pqn = F.normalize(pm.float(), dim=-1), F.normalize(pq.float(), dim=-1)
    lmn, lqn = F.normalize(lm.float(), dim=-1), F.normalize(lq.float(), dim=-1)
    rootn = F.normalize(root.float(), dim=-1)
    direct_parent, direct_local = pmn @ rootn, lmn @ rootn
    per_hop = max(1, math.ceil(budget / 2))
    initial = IterativeGistRouter._topk(direct_parent, min(per_hop, budget))
    rows = _score_rows(
        feature, seed, method, "root_semantic", None, direct_parent, oracle, set()
    )
    if method == "parent_closure":
        for source in initial:
            scores = pmn @ pqn[source]
            rows.extend(
                _score_rows(
                    feature,
                    seed,
                    method,
                    "propagated_semantic",
                    source,
                    scores,
                    oracle,
                    set(initial),
                )
            )
    else:
        parent_indices = feature["local_parent_indices"].to(lm.device)
        for source in initial:
            source_local = _best_root_local(direct_local, parent_indices, source)
            local_scores = lmn @ lqn[source_local]
            parent_scores = direct_parent.new_full(
                (len(feature["parent_spans"]),), float("-inf")
            )
            for parent in range(parent_scores.numel()):
                parent_scores[parent] = local_scores[parent_indices == parent].max()
            rows.extend(
                _score_rows(
                    feature,
                    seed,
                    method,
                    "propagated_semantic",
                    source,
                    parent_scores,
                    oracle,
                    set(initial),
                )
            )
    return rows


def native_family_rows(
    feature: dict,
    seed: int,
    method: str,
    root: torch.Tensor,
    pm: torch.Tensor,
    lm: torch.Tensor,
    lq: torch.Tensor,
    oracle: set[int],
    budget: int,
    *,
    token_reduction: str,
    head_reduction: str,
    device: torch.device,
) -> list[dict]:
    """Capture root scores and exact narrowed Gate-3 native candidate logits."""
    pmn, lmn, lqn = (
        F.normalize(pm.float(), dim=-1),
        F.normalize(lm.float(), dim=-1),
        F.normalize(lq.float(), dim=-1),
    )
    rootn = F.normalize(root.float(), dim=-1)
    direct_parent, direct_local = pmn @ rootn, lmn @ rootn
    initial_count = min(max(1, math.ceil(budget / 2)), budget)
    initial = IterativeGistRouter._topk(direct_parent, initial_count)
    rows = _score_rows(
        feature, seed, method, "root_semantic", None, direct_parent, oracle, set()
    )
    parent_indices = feature["local_parent_indices"].to(device)
    source_locals = [
        _best_root_local(direct_local, parent_indices, parent) for parent in initial
    ]
    pool_size = max(1, math.ceil(len(feature["parent_spans"]) * 0.20))
    pre_q = feature["local_pre_query"].to(device)
    pre_k = feature["local_pre_key"].to(device)
    masks = feature["local_token_mask"].to(device)
    for source_parent, source_local in zip(initial, source_locals):
        semantic = lmn @ lqn[source_local]
        narrowed = direct_parent.new_full(
            (len(feature["parent_spans"]),), float("-inf")
        )
        for parent in range(narrowed.numel()):
            narrowed[parent] = semantic[parent_indices == parent].max()
        narrowed[list(initial)] = float("-inf")
        candidate_parents = IterativeGistRouter._topk(narrowed, pool_size)
        local_mask = torch.zeros_like(parent_indices, dtype=torch.bool)
        for parent in candidate_parents:
            local_mask |= parent_indices == parent
        candidate_locals = torch.nonzero(local_mask, as_tuple=False).flatten()
        if candidate_locals.numel() == 0:
            continue
        scored = native_local_qk_scores(
            pre_q[source_local : source_local + 1],
            pre_k[candidate_locals],
            masks[source_local : source_local + 1],
            masks[candidate_locals],
            token_reduction=token_reduction,
            head_reduction=head_reduction,
            top_m=4,
        )
        parent_scores = direct_parent.new_full(
            (len(feature["parent_spans"]),), float("-inf")
        )
        for parent in candidate_parents:
            packed = parent_indices[candidate_locals] == parent
            parent_scores[parent] = scored.scores[0, packed].max()
        rows.extend(
            _score_rows(
                feature,
                seed,
                method,
                "native_qk",
                source_parent,
                parent_scores,
                oracle,
                set(initial),
            )
        )
    return rows


def score_family_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["score_family"])].append(row)
    output = []
    for (dataset, method, family), values in sorted(grouped.items()):
        samples = torch.tensor([float(row["raw_score"]) for row in values])
        group_keys = {
            (row["example_id"], row["seed"], row["source_parent"]) for row in values
        }
        first_per_group = []
        oracle_quantiles = []
        distractor_quantiles = []
        for key in group_keys:
            group = [
                row
                for row in values
                if (row["example_id"], row["seed"], row["source_parent"]) == key
            ]
            first_per_group.append(group[0])
            oracle_quantiles.extend(
                float(row["score_quantile"]) for row in group if row["is_oracle"]
            )
            distractors = [row for row in group if not row["is_oracle"]]
            if distractors:
                distractor_quantiles.append(
                    max(float(row["score_quantile"]) for row in distractors)
                )
        quantiles = torch.quantile(
            samples.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
        ).tolist()
        margins = [
            float(row["top1_top2_margin"])
            for row in first_per_group
            if math.isfinite(float(row["top1_top2_margin"]))
        ]
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "score_family": family,
                "scores": len(values),
                "groups": len(group_keys),
                "mean": float(samples.mean()),
                "std": float(samples.std(unbiased=False)),
                "min": float(samples.min()),
                "q05": quantiles[0],
                "q25": quantiles[1],
                "median": quantiles[2],
                "q75": quantiles[3],
                "q95": quantiles[4],
                "max": float(samples.max()),
                "mean_top1_top2_margin": statistics.fmean(
                    margins
                ) if margins else None,
                "mean_top4_spread": statistics.fmean(
                    float(row["top4_spread"]) for row in first_per_group
                ),
                "mean_entropy": statistics.fmean(
                    float(row["score_entropy"]) for row in first_per_group
                ),
                "mean_oracle_score_quantile": (
                    statistics.fmean(oracle_quantiles) if oracle_quantiles else None
                ),
                "mean_best_distractor_quantile": (
                    statistics.fmean(distractor_quantiles)
                    if distractor_quantiles
                    else None
                ),
            }
        )
    return output


def native_saturation_summary(rows: list[dict]) -> list[dict]:
    """Quantify information compression caused by Gate-3's native sigmoid."""
    grouped = defaultdict(list)
    for row in rows:
        if row["score_family"] == "native_qk":
            grouped[(row["dataset"], row["method"])].append(float(row["raw_score"]))
    output = []
    for (dataset, method), values in sorted(grouped.items()):
        raw = torch.tensor(values)
        mapped = torch.sigmoid(raw)
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "scores": len(values),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(unbiased=False)),
                "sigmoid_mean": float(mapped.mean()),
                "sigmoid_std": float(mapped.std(unbiased=False)),
                "fraction_sigmoid_above_0_999": float((mapped > 0.999).float().mean()),
                "fraction_sigmoid_above_0_99999": float(
                    (mapped > 0.99999).float().mean()
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_prior_convergence(rows: list[dict], path: Path) -> None:
    """Require exact identity parity with the previously versioned diagnostic."""
    with path.open(encoding="utf-8", newline="") as stream:
        prior = list(csv.DictReader(stream))
    prior_by_key = {
        (
            row["dataset"],
            row["example_id"],
            int(row["seed"]),
            float(row["fraction"]),
            row["method"],
        ): row
        for row in prior
    }
    if len(prior_by_key) < len(rows):
        raise ValueError(
            f"Prior convergence has {len(prior_by_key)} rows; rerun has {len(rows)}."
        )
    for row in rows:
        key = (
            row["dataset"],
            row["example_id"],
            int(row["seed"]),
            float(row["fraction"]),
            row["method"],
        )
        expected = prior_by_key.get(key)
        if expected is None:
            raise ValueError(f"Missing prior convergence key: {key}")
        if json.loads(expected["selected_parent_ids"]) != json.loads(
            row["selected_parent_ids"]
        ):
            raise ValueError(f"Selection identity drift for {key}")
        for metric in (
            "oracle_recall",
            "oracle_precision",
            "oracle_jaccard",
            "complete_oracle",
        ):
            if not math.isclose(float(expected[metric]), float(row[metric]), abs_tol=1e-9):
                raise ValueError(f"Oracle metric drift for {key}: {metric}")


def _aggregate(rows: list[dict], dimensions: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if samples:
                record[metric] = statistics.fmean(samples)
        output.append(record)
    return output


def _plot_cost(convergence: list[dict], output_dir: Path) -> None:
    labels = {
        "one_shot_parent": "One-shot",
        "parent_closure": "Parent iterative",
        "local_gist_closure": "Local iterative",
        "native_qk_max_topk_p20": "Native max",
        "native_qk_top4_topk_p20": "Native Top-4 reduction",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for row_index, dataset in enumerate(("hotpotqa", "qasper")):
        for method, label in labels.items():
            values = sorted(
                [
                    row
                    for row in convergence
                    if row["dataset"] == dataset and row["method"] == method
                ],
                key=lambda row: row["fraction"],
            )
            axes[row_index, 0].plot(
                [100 * row["materialized_kv_fraction"] for row in values],
                [row["oracle_recall"] for row in values],
                marker="o",
                label=label,
            )
            axes[row_index, 1].plot(
                [row["routing_seconds"] for row in values],
                [row["oracle_recall"] for row in values],
                marker="o",
                label=label,
            )
        axes[row_index, 0].set_ylabel(f"{dataset}\nOracle recall")
        axes[row_index, 0].set_xlabel("Active K/V fraction (%)")
        axes[row_index, 1].set_xlabel("Measured routing time (s, log scale)")
        axes[row_index, 1].set_xscale("log")
    for axis in axes.flat:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"oracle_recall_cost.{suffix}", dpi=180, bbox_inches="tight"
        )
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_file, weights_only=False)
    convergence_rows: list[dict] = []
    displacement_rows: list[dict] = []
    protected_rows: list[dict] = []
    score_rows: list[dict] = []
    calibration_candidates: list[dict] = []

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
                ph, lh = feature["parent_hidden"].to(device), feature["local_hidden"].to(device)
                pm, pq = projection.project_memory(ph), projection.project_query(ph)
                lm, lq = projection.project_memory(lh), projection.project_query(lh)
            native_index = _native_index(feature, pm, lm, lq, device)
            for fraction in args.fractions:
                semantic_outputs = {}
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
                    semantic_outputs[condition] = (
                        _normalize_baseline(row, len(feature["local_spans"])),
                        graph,
                    )
                native_outputs = {}
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
                    native_outputs[row["condition"]] = (row, graph)
                outputs = {**semantic_outputs, **native_outputs}
                one_row, one_graph = outputs["one_shot_parent"]
                one_order = _selection_order(one_graph)
                budget = int(one_row["budget_parents"])
                for method, (row, graph) in outputs.items():
                    selected = set(_selection_order(graph))
                    metrics = oracle_set_metrics(selected, oracle)
                    convergence_rows.append(
                        {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "seed": seed,
                            "fraction": fraction,
                            "method": method,
                            "budget_parents": budget,
                            "oracle_parent_ids": json.dumps(sorted(oracle)),
                            "selected_parent_ids": json.dumps(sorted(selected)),
                            **metrics,
                            "chain_completion": float(
                                bool(groups)
                                and all(bool(selected & group) for group in groups)
                            ),
                            "materialized_kv_tokens": row["materialized_kv_tokens"],
                            "materialized_kv_fraction": row["materialized_kv_fraction"],
                            "semantic_gist_comparisons": row.get(
                                "semantic_gist_comparisons", 0
                            ),
                            "native_qk_dot_products": row.get(
                                "native_qk_dot_products", 0
                            ),
                            "routing_seconds": row["routing_seconds"],
                        }
                    )
                    if method == "one_shot_parent":
                        continue
                    details = classify_displacement(
                        oracle,
                        one_graph,
                        graph,
                        method=method,
                        budget=budget,
                    )
                    for detail in details:
                        displacement_rows.append(
                            {
                                "dataset": feature["dataset"],
                                "example_id": feature["example_id"],
                                "seed": seed,
                                "fraction": fraction,
                                "budget_parents": budget,
                                **detail,
                            }
                        )
                    protected = set(
                        protected_root_selection(
                            one_order,
                            _selection_order(graph),
                            oracle,
                            budget,
                        )
                    )
                    protected_rows.append(
                        {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "seed": seed,
                            "fraction": fraction,
                            "method": method,
                            "budget_parents": budget,
                            "actual_oracle_recall": metrics["oracle_recall"],
                            "protected_oracle_recall": oracle_set_metrics(
                                protected, oracle
                            )["oracle_recall"],
                            "actual_complete_oracle": metrics["complete_oracle"],
                            "protected_complete_oracle": oracle_set_metrics(
                                protected, oracle
                            )["complete_oracle"],
                            "protected_selected_ids": json.dumps(sorted(protected)),
                        }
                    )
                    if fraction == PRIMARY_FRACTION:
                        calibration_candidates.extend(
                            _candidate_representations(
                                feature,
                                seed,
                                method,
                                oracle,
                                one_graph,
                                graph,
                            )
                        )
                if fraction == PRIMARY_FRACTION:
                    score_rows.extend(
                        semantic_family_rows(
                            feature,
                            seed,
                            "parent_closure",
                            root,
                            pm,
                            pq,
                            lm,
                            lq,
                            oracle,
                            budget,
                        )
                    )
                    score_rows.extend(
                        semantic_family_rows(
                            feature,
                            seed,
                            "local_gist_closure",
                            root,
                            pm,
                            pq,
                            lm,
                            lq,
                            oracle,
                            budget,
                        )
                    )
                    for method, reductions in (
                        ("native_qk_max_topk_p20", ("max", "max")),
                        (
                            "native_qk_top4_topk_p20",
                            ("top_m_mean", "top_m_mean"),
                        ),
                    ):
                        score_rows.extend(
                            native_family_rows(
                                feature,
                                seed,
                                method,
                                root,
                                pm,
                                lm,
                                lq,
                                oracle,
                                budget,
                                token_reduction=reductions[0],
                                head_reduction=reductions[1],
                                device=device,
                            )
                        )
        print(
            f"displacement/calibration seed {seed}: "
            f"{len(convergence_rows)} convergence, {len(displacement_rows)} hit rows",
            flush=True,
        )

    displacement_aggregate = _aggregate(
        displacement_rows,
        ("dataset", "fraction", "method"),
        ("root_oracle_displaced",),
    )
    displacement_characteristics = _aggregate(
        displacement_rows,
        ("dataset", "fraction", "method", "status"),
        (
            "root_top1_top2_margin",
            "root_selected_entropy",
            "replacement_hop2_fraction",
            "candidate_pool_fraction",
        ),
    )
    protected_aggregate = _aggregate(
        protected_rows,
        ("dataset", "fraction", "method"),
        (
            "actual_oracle_recall",
            "protected_oracle_recall",
            "actual_complete_oracle",
            "protected_complete_oracle",
        ),
    )
    convergence_aggregate = _aggregate(
        convergence_rows,
        ("dataset", "fraction", "method"),
        (
            "oracle_recall",
            "oracle_precision",
            "oracle_jaccard",
            "complete_oracle",
            "chain_completion",
            "materialized_kv_tokens",
            "materialized_kv_fraction",
            "semantic_gist_comparisons",
            "native_qk_dot_products",
            "routing_seconds",
        ),
    )
    calibration_rows = []
    for method in ITERATIVE_METHODS:
        fit = fit_family_calibration(calibration_candidates, method)
        examples = defaultdict(list)
        for row in calibration_candidates:
            if row["method"] == method and row["partition"] == "test":
                examples[(row["dataset"], row["example_id"], row["seed"])].append(row)
        for key, values in examples.items():
            feature = next(item for item in features if item["example_id"] == key[1])
            oracle = canonical_oracle_parent_indices(feature)
            groups = evidence_parent_groups(feature)
            budget = max(1, math.ceil(len(feature["parent_spans"]) * PRIMARY_FRACTION))
            one_shot_hits = {
                int(row["candidate_parent"])
                for row in values
                if row["selected_by_one_shot"] and row["is_oracle"]
            }
            selections = {
                "actual": {
                    int(row["candidate_parent"])
                    for row in values
                    if row["selected_by_iterative"]
                },
                **{
                    mode: set(calibrated_selection(values, fit, mode, budget))
                    for mode in CALIBRATION_MODES
                },
            }
            for mode, selected in selections.items():
                calibration_rows.append(
                    {
                        "dataset": key[0],
                        "example_id": key[1],
                        "seed": key[2],
                        "method": method,
                        "mode": mode,
                        "budget_parents": budget,
                        **oracle_set_metrics(selected, oracle),
                        "chain_completion": float(
                            bool(groups)
                            and all(bool(selected & group) for group in groups)
                        ),
                        "one_shot_oracle_hit_preservation": (
                            len(selected & one_shot_hits) / len(one_shot_hits)
                            if one_shot_hits
                            else 1.0
                        ),
                        "selected_parent_ids": json.dumps(sorted(selected)),
                    }
                )
    calibration_aggregate = _aggregate(
        calibration_rows,
        ("dataset", "method", "mode"),
        (
            "oracle_recall",
            "oracle_precision",
            "oracle_jaccard",
            "complete_oracle",
            "chain_completion",
            "one_shot_oracle_hit_preservation",
        ),
    )
    family_summary = score_family_summary(score_rows)
    saturation = native_saturation_summary(score_rows)
    validate_prior_convergence(convergence_rows, args.prior_convergence_file)
    root_hit_rows = []
    for row in convergence_rows:
        if row["method"] != "one_shot_parent":
            continue
        selected = set(json.loads(row["selected_parent_ids"]))
        oracle = set(json.loads(row["oracle_parent_ids"]))
        root_hit_rows.append(
            {
                "dataset": row["dataset"],
                "example_id": row["example_id"],
                "seed": row["seed"],
                "fraction": row["fraction"],
                "any_oracle_hit": float(bool(selected & oracle)),
            }
        )
    root_hit_aggregate = _aggregate(
        root_hit_rows, ("dataset", "fraction"), ("any_oracle_hit",)
    )
    qasper_primary_displacement = max(
        float(row["root_oracle_displaced"])
        for row in displacement_aggregate
        if row["dataset"] == "qasper" and row["fraction"] == PRIMARY_FRACTION
    )
    qasper_primary_protected_gain = max(
        float(row["protected_oracle_recall"])
        - float(row["actual_oracle_recall"])
        for row in protected_aggregate
        if row["dataset"] == "qasper" and row["fraction"] == PRIMARY_FRACTION
    )
    recommendation = (
        "protected_root"
        if qasper_primary_displacement >= 0.10
        and qasper_primary_protected_gain >= 0.03
        else "adaptive_competition"
    )
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_routing_changed": False,
        "canonical_oracle": "experiments.paper2_hf.qa.run_oracle_memory_use._oracle_selections",
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "primary_fraction": PRIMARY_FRACTION,
        "score_competition_audit": {
            "root_parents_evicted_after_admission": False,
            "root_allocation": "ceil(final_budget/2)_for_iterative_methods",
            "parent_local_semantic_path": "raw cosine -> 0.25 root anchor -> [0,1] affinity -> product path",
            "native_path": "raw pre-RoPE QK -> sigmoid -> 0.25 direct affinity anchor -> source-affinity product",
            "raw_native_and_root_scores_share_topk": False,
            "normalization_candidate_pool": "union of frozen one-shot and iterative selected identities",
        },
        "prior_oracle_convergence_identity_parity": True,
        "root_hit_rate": root_hit_aggregate,
        "displacement": displacement_aggregate,
        "displacement_characteristics": displacement_characteristics,
        "protected_root": protected_aggregate,
        "score_families": family_summary,
        "native_sigmoid_saturation": saturation,
        "calibration_controls": calibration_aggregate,
        "recommendation": {
            "next_gate": recommendation,
            "qasper_primary_max_displacement": qasper_primary_displacement,
            "qasper_primary_max_protected_recall_gain": qasper_primary_protected_gain,
            "sdk_change_in_this_iteration": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_competition_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "convergence_rows.csv", convergence_rows)
    _write_csv(args.output_dir / "convergence_aggregate.csv", convergence_aggregate)
    _write_csv(args.output_dir / "displacement_rows.csv", displacement_rows)
    _write_csv(args.output_dir / "displacement_aggregate.csv", displacement_aggregate)
    _write_csv(
        args.output_dir / "displacement_characteristics.csv",
        displacement_characteristics,
    )
    _write_csv(args.output_dir / "protected_root_rows.csv", protected_rows)
    _write_csv(args.output_dir / "protected_root_aggregate.csv", protected_aggregate)
    _write_csv(args.output_dir / "score_family_rows.csv", score_rows)
    _write_csv(args.output_dir / "score_family_summary.csv", family_summary)
    _write_csv(args.output_dir / "native_sigmoid_saturation.csv", saturation)
    _write_csv(args.output_dir / "root_hit_aggregate.csv", root_hit_aggregate)
    _write_csv(args.output_dir / "calibration_candidate_rows.csv", calibration_candidates)
    _write_csv(args.output_dir / "calibration_control_rows.csv", calibration_rows)
    _write_csv(args.output_dir / "calibration_control_aggregate.csv", calibration_aggregate)
    _plot_cost(convergence_aggregate, args.output_dir)
    return artifact


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
        "--prior-convergence-file",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/oracle_convergence/oracle_convergence_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/oracle_competition_diagnostics",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    args.fractions = tuple(map(float, args.fractions.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"displacement": result["displacement"]}, indent=2))
