"""Evaluate one-shot and single ROOT_SELECTED callback request/reply policies.

The study consumes the measured full action surface.  Policy selection uses
only the inherited 58-example validation partition; the 74-example test
partition is never used for tuning.  Dataset identity and evaluator labels are
excluded from controller features.  Five bootstrap seeds expose controller
variance, and leave-one-dataset-out (LODO) runs test transfer explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SEEDS = (1, 7, 21, 42, 87)
DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
QUERY_ONLY = (
    "query_tokens",
    "query_terms",
    "query_idf_mean",
    "query_idf_max",
    "rare_token_count",
    "query_rare_fraction",
    "named_entity_count",
    "numeric_marker",
    "url_marker",
    "question_markers",
    "id_marker",
    "query_region_layout",
)
ROOT_SCALARS = (
    "root_top1_score",
    "root_score_gap",
    "candidate_entropy",
    "channel_agreement",
    "channel_disagreement",
    "address_count",
    "address_rarity",
    "facet_agreement",
    "root_dispersion",
    "evidence_proxy",
    "searched_fraction",
    "remaining_search_fraction",
    "remaining_kv_fraction",
)
NO_GRAPH = {"global", "structural", "multiscale"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["split"]), str(row["dataset"]), str(row["example_id"])


def _quality_key(row: Mapping[str, Any]) -> tuple:
    """Order measured actions without using noisy wall-clock timing."""

    return (
        -float(row["evidence_recall"]),
        -float(row["complete_recovery"]),
        -float(row["precision"]),
        -float(row["mrr"]),
        float(row["active_fraction"]),
        float(row["root_comparisons"]) + float(row["successor_comparisons"]),
        float(row["graph_calls"]),
        str(row["complete_action"]),
    )


def _best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("An oracle candidate set cannot be empty.")
    return min(rows, key=_quality_key)


class RidgeClassifier:
    """Small deterministic bootstrap ridge classifier for policy comparisons."""

    def __init__(self, classes, mean, scale, weights):
        self.classes, self.mean, self.scale, self.weights = classes, mean, scale, weights

    @classmethod
    def fit(cls, features: np.ndarray, labels: Sequence[str], seed: int):
        classes = tuple(sorted(set(labels)))
        rng = random.Random(seed)
        sampled = [rng.randrange(len(labels)) for _ in labels]
        x = features[sampled]
        y = [labels[index] for index in sampled]
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.0
        design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
        target = np.zeros((len(x), len(classes)))
        lookup = {label: index for index, label in enumerate(classes)}
        for row_index, label in enumerate(y):
            target[row_index, lookup[label]] = 1.0
        penalty = np.eye(design.shape[1]) * 0.1
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return cls(classes, mean, scale, weights)

    def predict_one(self, feature: np.ndarray) -> str:
        design = np.append((feature - self.mean) / self.scale, 1.0)
        return self.classes[int(np.argmax(design @ self.weights))]


def _load_states(path: Path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    states, embeddings = {}, {}
    for key, value in raw.items():
        state = value["state"]
        identity = tuple(key.split("|", 3)[:3])
        initial = key.split("|", 3)[3]
        states[(*identity, initial)] = state
        embeddings[(*identity, initial)] = value["root_embedding"].float().numpy()
    return states, embeddings


def _query_names(states: Mapping) -> tuple[str, ...]:
    names = sorted({name for state in states.values() for name in state.query_features})
    preferred = [name for name in QUERY_ONLY if name in names]
    return tuple(preferred or names)


def _query_vector(state, names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(state.query_features.get(name, 0.0)) for name in names])


def _callback_vector(state, embedding: np.ndarray, names: Sequence[str], ablation: str) -> np.ndarray:
    query = _query_vector(state, names)
    scalars = np.asarray([float(getattr(state, name)) for name in ROOT_SCALARS])
    if ablation == "query_only":
        return query
    if ablation == "query_root_scores":
        return np.concatenate((query, scalars[:3]))
    if ablation == "query_root_channels":
        return np.concatenate((query, scalars[:5]))
    if ablation == "compact_state":
        return np.concatenate((query, scalars))
    if ablation == "compact_state_embedding":
        # Fixed projections avoid fitting thousands of nuisance parameters.
        width = 16
        boundaries = np.linspace(0, len(embedding), width + 1, dtype=int)
        pooled = np.asarray([
            float(embedding[boundaries[i]:boundaries[i + 1]].mean())
            for i in range(width)
        ])
        return np.concatenate((query, scalars, pooled))
    raise ValueError(f"Unknown ablation={ablation!r}.")


def _refined_initial(initial: str, available: set[str]) -> str | None:
    profile, root_method = initial.rsplit(".", 1)
    mode, count = profile.split(".f", 1)
    if mode == "structural":
        candidate = f"structural_graph.f{count}.{root_method}"
    elif mode in {"global", "multiscale"}:
        candidate = f"graph.f{max(2, int(count))}.{root_method}"
    else:
        return None
    return candidate if candidate in available else None


def _emit(baseline: str, seed: int, row: Mapping[str, Any], oracle: Mapping[str, Any], **extra):
    selected = set(filter(None, str(row["selected_chunk_ids"]).split("|")))
    oracle_selected = set(filter(None, str(oracle["selected_chunk_ids"]).split("|")))
    return {
        "baseline": baseline,
        "router_seed": seed,
        "split": row["split"],
        "dataset": row["dataset"],
        "example_id": row["example_id"],
        "initial_action": row["initial_action"],
        "complete_action": row["complete_action"],
        "oracle_action": oracle["complete_action"],
        "evidence_recall": float(row["evidence_recall"]),
        "precision": float(row["precision"]),
        "mrr": float(row["mrr"]),
        "complete_recovery": float(row["complete_recovery"]),
        "active_fraction": float(row["active_fraction"]),
        "comparisons": float(row["root_comparisons"]) + float(row["successor_comparisons"]),
        "graph_calls": float(row["graph_calls"]),
        "facet_latency_ms": float(row["facet_latency_ms"]),
        "retrieval_latency_ms": float(row["total_retrieval_latency_ms"]),
        "selected_chunk_ids": row["selected_chunk_ids"],
        "positive_chunk_ids": row["positive_chunk_ids"],
        "oracle_selected_overlap": len(selected & oracle_selected) / max(len(oracle_selected), 1),
        **extra,
    }


def _group(rows: Sequence[Mapping[str, Any]]):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[_identity(row)][str(row["initial_action"])][str(row["successor_method"])] = row
    return grouped


def _fixed_action(validation: Sequence[Mapping[str, Any]], predicate=lambda row: True) -> str:
    grouped = defaultdict(list)
    for row in validation:
        if predicate(row):
            grouped[str(row["complete_action"])].append(row)
    return min(
        grouped,
        key=lambda action: (
            -statistics.fmean(float(row["evidence_recall"]) for row in grouped[action]),
            -statistics.fmean(float(row["precision"]) for row in grouped[action]),
            statistics.fmean(float(row["active_fraction"]) for row in grouped[action]),
            action,
        ),
    )


def _threshold_uncertain(state, gap: float, entropy: float, disagreement: float) -> bool:
    return (
        float(state.root_score_gap) < gap
        or float(state.candidate_entropy) > entropy
        or float(state.channel_disagreement) >= disagreement
    )


def _tune_threshold_policy(grouped, states, train_ids, names, initial_model):
    successor_methods = tuple(next(iter(grouped[train_ids[0]].values())))
    candidates = []
    for gap in (0.0, 0.02, 0.05, 0.10):
        for entropy in (0.70, 0.85, 0.95, 1.10):
            for disagreement in (1.0, 2.0, 3.0, 99.0):
                for keep_method in successor_methods:
                    for refine_method in successor_methods:
                        selected = []
                        refinements = 0
                        for identity in train_ids:
                            query_state = states[(*identity, next(iter(grouped[identity])))]
                            initial = initial_model.predict_one(_query_vector(query_state, names))
                            state = states[(*identity, initial)]
                            refined = _refined_initial(initial, set(grouped[identity]))
                            use_refine = bool(
                                refined
                                and _threshold_uncertain(
                                    state, gap, entropy, disagreement
                                )
                            )
                            chosen = refined if use_refine else initial
                            method = refine_method if use_refine else keep_method
                            selected.append(grouped[identity][chosen][method])
                            refinements += int(use_refine)
                        candidates.append(
                            (
                                (
                                    -statistics.fmean(float(row["evidence_recall"]) for row in selected),
                                    -statistics.fmean(float(row["precision"]) for row in selected),
                                    statistics.fmean(float(row["active_fraction"]) for row in selected),
                                    refinements / len(selected),
                                    gap,
                                    entropy,
                                    disagreement,
                                    keep_method,
                                    refine_method,
                                ),
                                (gap, entropy, disagreement, keep_method, refine_method),
                            )
                        )
    return min(candidates, key=lambda value: value[0])[1]


def _study(rows, states, embeddings, *, train_datasets: set[str], test_datasets: set[str], protocol: str):
    grouped = _group(rows)
    train_ids = sorted(identity for identity in grouped if identity[0] == "validation" and identity[1] in train_datasets)
    test_ids = sorted(identity for identity in grouped if identity[0] == "test" and identity[1] in test_datasets)
    names = _query_names(states)
    validation_rows = [row for row in rows if row["split"] == "validation" and row["dataset"] in train_datasets]
    fixed = _fixed_action(validation_rows, lambda row: row["facet_mode"] == "global")
    fixed_graph = _fixed_action(validation_rows, lambda row: row["facet_mode"] == "structural_graph")
    complete_by_id = {
        identity: {row["complete_action"]: row for methods in grouped[identity].values() for row in methods.values()}
        for identity in grouped
    }
    outputs, targets, ablations = [], [], []

    for seed in SEEDS:
        complete_models = {}
        for policy, allow_graph in (("all", True), ("no_graph", False)):
            complete_x, complete_labels = [], []
            for identity in train_ids:
                candidates = [
                    row for row in complete_by_id[identity].values()
                    if allow_graph or row["facet_mode"] in NO_GRAPH
                ]
                target = _best(candidates)
                state = states[(*identity, str(target["initial_action"]))]
                complete_x.append(_query_vector(state, names))
                complete_labels.append(str(target["complete_action"]))
            complete_models[policy] = RidgeClassifier.fit(
                np.stack(complete_x), complete_labels, seed
            )

        # Initial query-only policies: all actions and graph-free actions.
        initial_models = {}
        for policy, allow_graph in (("all", True), ("no_graph", False)):
            train_x, labels = [], []
            for identity in train_ids:
                candidates = [
                    row for methods in grouped[identity].values() for row in methods.values()
                    if allow_graph or row["facet_mode"] in NO_GRAPH
                ]
                target = _best(candidates)
                state = states[(*identity, str(target["initial_action"]))]
                train_x.append(_query_vector(state, names))
                labels.append(str(target["initial_action"]))
                targets.append({"protocol": protocol, "seed": seed, "stage": f"initial_{policy}", "dataset": identity[1], "example_id": identity[2], "target": labels[-1]})
            initial_models[policy] = RidgeClassifier.fit(np.stack(train_x), labels, seed)

        callback_models = {}
        for policy, allow_graph, conditional in (
            ("B3_callback_no_graph", False, False),
            ("B4_callback_graph", True, False),
            ("B5_conditional_graph", False, True),
        ):
            for ablation in ("query_only", "query_root_scores", "query_root_channels", "compact_state", "compact_state_embedding"):
                train_x, labels = [], []
                for identity in train_ids:
                    available = set(grouped[identity])
                    for initial, methods in grouped[identity].items():
                        sample = next(iter(methods.values()))
                        if not allow_graph and sample["facet_mode"] not in NO_GRAPH:
                            continue
                        candidates = list(methods.values())
                        if conditional:
                            refined = _refined_initial(initial, available)
                            if refined:
                                candidates += list(grouped[identity][refined].values())
                        target = _best(candidates)
                        label = (
                            ("refine:" if str(target["initial_action"]) != initial else "keep:")
                            + str(target["successor_method"])
                        )
                        state = states[(*identity, initial)]
                        train_x.append(_callback_vector(state, embeddings[(*identity, initial)], names, ablation))
                        labels.append(label)
                callback_models[(policy, ablation)] = RidgeClassifier.fit(np.stack(train_x), labels, seed)
        threshold_policy = _tune_threshold_policy(
            grouped,
            states,
            train_ids,
            names,
            initial_models["no_graph"],
        )

        for identity in test_ids:
            oracle = _best(list(complete_by_id[identity].values()))
            outputs.append(_emit("B0_validation_fixed", seed, complete_by_id[identity][fixed], oracle, protocol=protocol, callback_invoked=0, graph_refined=0))
            outputs.append(_emit("B0b_structural_graph_fixed", seed, complete_by_id[identity][fixed_graph], oracle, protocol=protocol, callback_invoked=0, graph_refined=0))
            outputs.append(_emit("B6_complete_action_oracle", seed, oracle, oracle, protocol=protocol, callback_invoked=0, graph_refined=0))
            # A stage-wise evaluator oracle searches the same measured Cartesian
            # action surface, so it must equal the complete-action oracle.
            stage_initial = min(grouped[identity], key=lambda action: _quality_key(_best(list(grouped[identity][action].values()))))
            stage_oracle = _best(list(grouped[identity][stage_initial].values()))
            outputs.append(_emit("B7_stagewise_oracle", seed, stage_oracle, oracle, protocol=protocol, callback_invoked=1, graph_refined=int("graph" in stage_initial)))

            query_state = states[(*identity, next(iter(grouped[identity])))]
            for baseline, policy in (("B1_one_shot_no_graph", "no_graph"), ("B2_one_shot_graph", "all")):
                one_action = complete_models[policy].predict_one(_query_vector(query_state, names))
                one_row = complete_by_id[identity][one_action]
                outputs.append(_emit(baseline, seed, one_row, oracle, protocol=protocol, callback_invoked=0, graph_refined=0))

            for policy, initial_key in (("B3_callback_no_graph", "no_graph"), ("B4_callback_graph", "all"), ("B5_conditional_graph", "no_graph")):
                initial = initial_models[initial_key].predict_one(_query_vector(query_state, names))
                state = states[(*identity, initial)]
                for ablation in ("query_only", "query_root_scores", "query_root_channels", "compact_state", "compact_state_embedding"):
                    label = callback_models[(policy, ablation)].predict_one(
                        _callback_vector(state, embeddings[(*identity, initial)], names, ablation)
                    )
                    disposition, successor = label.split(":", 1)
                    chosen_initial = initial
                    if disposition == "refine":
                        chosen_initial = _refined_initial(initial, set(grouped[identity])) or initial
                    row = grouped[identity][chosen_initial].get(successor)
                    if row is None:
                        row = _best(list(grouped[identity][chosen_initial].values()))
                    emitted = _emit(policy, seed, row, oracle, protocol=protocol, feature_ablation=ablation, callback_invoked=1, graph_refined=int(chosen_initial != initial))
                    ablations.append(emitted)
                    if ablation == "compact_state":
                        outputs.append(emitted)
            gap, entropy, disagreement, keep_method, refine_method = threshold_policy
            initial = initial_models["no_graph"].predict_one(_query_vector(query_state, names))
            state = states[(*identity, initial)]
            refined = _refined_initial(initial, set(grouped[identity]))
            use_refine = bool(
                refined
                and _threshold_uncertain(state, gap, entropy, disagreement)
            )
            chosen = refined if use_refine else initial
            method = refine_method if use_refine else keep_method
            outputs.append(
                _emit(
                    "B5t_threshold_graph",
                    seed,
                    grouped[identity][chosen][method],
                    oracle,
                    protocol=protocol,
                    callback_invoked=1,
                    graph_refined=int(use_refine),
                    threshold_gap=gap,
                    threshold_entropy=entropy,
                    threshold_disagreement=disagreement,
                )
            )
    comparisons = {
        "B3_callback_no_graph": "B1_one_shot_no_graph",
        "B4_callback_graph": "B2_one_shot_graph",
        "B5_conditional_graph": "B1_one_shot_no_graph",
        "B5t_threshold_graph": "B1_one_shot_no_graph",
    }
    lookup = {
        (row["router_seed"], row["dataset"], row["example_id"], row["baseline"]): row
        for row in outputs
    }
    for row in outputs:
        counterpart = comparisons.get(row["baseline"])
        if counterpart is None:
            row["recall_delta_vs_one_shot"] = 0.0
            row["corrected_vs_one_shot"] = 0
            row["harmed_vs_one_shot"] = 0
            continue
        one_shot = lookup[(row["router_seed"], row["dataset"], row["example_id"], counterpart)]
        delta = float(row["evidence_recall"]) - float(one_shot["evidence_recall"])
        row["recall_delta_vs_one_shot"] = delta
        row["corrected_vs_one_shot"] = int(delta > 1e-12)
        row["harmed_vs_one_shot"] = int(delta < -1e-12)
    return outputs, targets, ablations


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], row["baseline"], row.get("feature_ablation", "default"), row["dataset"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append({
            "protocol": key[0], "baseline": key[1], "feature_ablation": key[2], "dataset": key[3], "rows": len(values),
            **{name: statistics.fmean(float(row.get(name, 0.0)) for row in values) for name in ("evidence_recall", "precision", "mrr", "complete_recovery", "active_fraction", "comparisons", "graph_calls", "facet_latency_ms", "retrieval_latency_ms", "callback_invoked", "graph_refined", "recall_delta_vs_one_shot", "corrected_vs_one_shot", "harmed_vs_one_shot")},
        })
    return output


def _facet_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["dataset"], row["facet_mode"], row["requested_facet_count"])].append(row)
    metrics = (
        "construction_latency_ms",
        "graph_construction_ms",
        "graph_clustering_ms",
        "graph_calls",
        "graph_nodes",
        "graph_edges",
        "graph_density",
        "graph_memory_bytes",
        "pairwise_similarity_evaluations",
        "tree_node_count",
        "mean_facet_overlap",
    )
    return [
        {
            "split": key[0],
            "dataset": key[1],
            "facet_mode": key[2],
            "requested_facet_count": key[3],
            "examples": len(values),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in metrics
            },
        }
        for key, values in sorted(grouped.items())
    ]


def _seed_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], row["baseline"], row.get("feature_ablation", "default"), row["router_seed"], row["dataset"])].append(row)
    return [
        {
            "protocol": key[0],
            "baseline": key[1],
            "feature_ablation": key[2],
            "router_seed": key[3],
            "dataset": key[4],
            "examples": len(values),
            **{
                metric: statistics.fmean(float(row.get(metric, 0.0)) for row in values)
                for metric in (
                    "evidence_recall",
                    "precision",
                    "mrr",
                    "complete_recovery",
                    "active_fraction",
                    "comparisons",
                    "graph_calls",
                    "recall_delta_vs_one_shot",
                    "corrected_vs_one_shot",
                    "harmed_vs_one_shot",
                )
            },
        }
        for key, values in sorted(grouped.items())
    ]


def _plots(rows, output: Path) -> None:
    aggregate = defaultdict(list)
    for row in rows:
        if row["protocol"] == "standard" and row.get("feature_ablation", "default") in {"default", "compact_state"}:
            aggregate[row["baseline"]].append(row)
    preferred = (
        "B0_validation_fixed",
        "B0b_structural_graph_fixed",
        "B1_one_shot_no_graph",
        "B2_one_shot_graph",
        "B3_callback_no_graph",
        "B4_callback_graph",
        "B5_conditional_graph",
        "B5t_threshold_graph",
        "B6_complete_action_oracle",
        "B7_stagewise_oracle",
    )
    labels = [label for label in preferred if label in aggregate]
    recalls = [statistics.fmean(float(row["evidence_recall"]) for row in aggregate[label]) for label in labels]
    graphs = [statistics.fmean(float(row["graph_calls"]) for row in aggregate[label]) for label in labels]
    seed_means = defaultdict(lambda: defaultdict(list))
    for label, values in aggregate.items():
        for row in values:
            seed_means[label][row["router_seed"]].append(float(row["evidence_recall"]))
    errors = [
        statistics.pstdev(
            statistics.fmean(values) for values in seed_means[label].values()
        )
        if len(seed_means[label]) > 1 else 0.0
        for label in labels
    ]
    short = [label.split("_", 1)[0] for label in labels]
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    bars = axis.bar(short, recalls, color="#287271")
    axis.errorbar(short, recalls, yerr=errors, fmt="none", ecolor="#202020", capsize=3)
    axis.set_ylabel("Evidence recall")
    axis.set_ylim(0, 1)
    axis.set_title("Request/reply policies on the frozen test cohort")
    axis.bar_label(bars, fmt="%.3f", fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"callback_quality.{suffix}", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    axis.scatter(graphs, recalls, color="#C1666B")
    for label, x, y in zip(short, graphs, recalls):
        axis.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Graph calls per example")
    axis.set_ylabel("Evidence recall")
    axis.set_title("Quality versus graph construction")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"callback_graph_cost.{suffix}", dpi=200)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.plots_only:
        rows = _read(args.output / "controller_rows.csv")
        _plots(rows, args.output)
        return {"plots_only": True, "rows": len(rows)}
    rows = _read(args.surface / "full_action_rows.csv")
    facet_rows = _read(args.surface / "facet_construction_rows.csv")
    states, embeddings = _load_states(args.state_cache)
    outputs, targets, ablations = _study(rows, states, embeddings, train_datasets=set(DATASETS), test_datasets=set(DATASETS), protocol="standard")
    lodo = []
    for dataset in DATASETS:
        local, local_targets, local_ablations = _study(rows, states, embeddings, train_datasets=set(DATASETS) - {dataset}, test_datasets={dataset}, protocol=f"lodo_{dataset}")
        lodo.extend(local)
        targets.extend(local_targets)
        ablations.extend(local_ablations)
    all_rows = outputs + lodo
    summary = _summaries(all_rows)
    seed_summary = _seed_summaries(all_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "controller_rows.csv", all_rows)
    _write(args.output / "controller_summary.csv", summary)
    _write(args.output / "controller_seed_summary.csv", seed_summary)
    _write(args.output / "controller_targets.csv", targets)
    _write(args.output / "callback_feature_ablations.csv", ablations)
    _write(args.output / "facet_construction_summary.csv", _facet_summaries(facet_rows))
    _plots(all_rows, args.output)
    b6 = [row for row in outputs if row["baseline"] == "B6_complete_action_oracle"]
    b7 = [row for row in outputs if row["baseline"] == "B7_stagewise_oracle"]
    headline = {}
    for baseline in sorted({row["baseline"] for row in outputs}):
        values = [row for row in outputs if row["baseline"] == baseline]
        headline[baseline] = {
            metric: statistics.fmean(float(row.get(metric, 0.0)) for row in values)
            for metric in (
                "evidence_recall",
                "precision",
                "mrr",
                "complete_recovery",
                "active_fraction",
                "graph_calls",
                "graph_refined",
                "recall_delta_vs_one_shot",
                "corrected_vs_one_shot",
                "harmed_vs_one_shot",
            )
        }
    findings = {
        "schema_version": "1.0",
        "seeds": list(SEEDS),
        "validation_identities": len({_identity(row) for row in rows if row["split"] == "validation"}),
        "test_identities": len({_identity(row) for row in rows if row["split"] == "test"}),
        "measured_actions": len(rows),
        "stagewise_oracle_equals_complete_oracle": all(abs(float(left["evidence_recall"]) - float(right["evidence_recall"])) < 1e-12 for left, right in zip(b6, b7)),
        "controller_inputs_exclude_dataset_and_gold": True,
        "callback_events_per_example": 1,
        "lodo_protocols": [f"lodo_{name}" for name in DATASETS],
        "standard_test_headline": headline,
        "interpretation": (
            "The stage-wise and complete-action evaluator oracles coincide; "
            "learned callbacks do not beat matched one-shot control, and the "
            "threshold callback remains below the validation-fixed policy."
        ),
    }
    (args.output / "callback_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/root_callback"
    parser.add_argument("--surface", type=Path, default=base / "surface")
    parser.add_argument("--state-cache", type=Path, default=base / "surface/root_states.pt")
    parser.add_argument("--output", type=Path, default=base / "controller")
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
