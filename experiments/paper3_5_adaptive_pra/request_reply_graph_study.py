"""Strengthened one-shot request/reply analysis with query-graph facets.

The runner consumes frozen measured retrieval rows.  It creates a new
identity-disjoint controller split inside the 74-example Paper 2.7 cohort;
this split is post-hoc and is reported as such. Dataset identity and evaluator
labels are never controller inputs. Full root/successor traversal results from
the preceding study are retained in a separate scope table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROUTER_SEEDS = (1, 7, 21, 42, 87)
DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa", "musique")
LABELS = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
    "musique": "MuSiQue",
}
ACTION_SPACES = {
    "B1_existing_request_reply": (
        "global_semantic", "paper26_bm25", "paper26_hybrid",
    ),
    "B2_graph_facet_choice": (
        "global_semantic", "paper25_clause", "paper25_multiscale",
        "graph_cc", "syntactic_graph",
    ),
    "B3_facet_root_matched_successor": (
        "global_semantic", "paper25_clause", "paper25_multiscale",
        "paper26_bm25", "paper26_hybrid", "graph_cc",
        "graph_label_propagation", "graph_cc_hybrid",
        "graph_label_propagation_hybrid", "syntactic_graph",
        "syntactic_graph_hybrid",
    ),
}
FEATURE_NAMES = (
    "facet_count", "mean_token_count", "mean_unique_tokens", "mean_rare_fraction",
    "mean_entities", "mean_relation_cues", "mean_confidence", "mean_component_nodes",
    "mean_component_density", "mean_component_weight", "mean_token_spans",
    "entity_facet_fraction", "relational_facet_fraction", "mixed_facet_fraction",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["example_id"])


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else 0.0


def _split(identities: Sequence[tuple[str, str]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    train, heldout = set(), set()
    for dataset in DATASETS:
        local = [identity for identity in identities if identity[0] == dataset]
        local.sort(key=lambda value: hashlib.sha256(value[1].encode()).hexdigest())
        cut = max(1, round(0.6 * len(local)))
        train.update(local[:cut])
        heldout.update(local[cut:])
    return train, heldout


def _best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -float(row.get("evidence_recall", 0.0)),
            -float(row.get("precision", 0.0)),
            -float(row.get("mrr", 0.0)),
            float(row.get("comparisons", row.get("pairwise_similarity_evaluations", 0.0))),
            float(row.get("graph_calls", 0.0)),
            str(row.get("condition", row.get("root_method", ""))),
        ),
    )


class RidgeClassifier:
    def __init__(self, classes, mean, scale, weights):
        self.classes, self.mean, self.scale, self.weights = classes, mean, scale, weights

    @classmethod
    def fit(cls, features: np.ndarray, targets: Sequence[str], seed: int):
        classes = tuple(sorted(set(targets)))
        rng = random.Random(seed)
        sampled = [rng.randrange(len(targets)) for _ in targets]
        x = features[sampled]
        y = [targets[index] for index in sampled]
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.0
        design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
        target = np.zeros((len(x), len(classes)))
        indices = {name: index for index, name in enumerate(classes)}
        for row, name in enumerate(y):
            target[row, indices[name]] = 1.0
        penalty = np.eye(design.shape[1]) * 0.1
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return cls(classes, mean, scale, weights)

    def predict(self, features: np.ndarray) -> list[str]:
        design = np.column_stack(((features - self.mean) / self.scale, np.ones(len(features))))
        scores = design @ self.weights
        return [self.classes[int(index)] for index in scores.argmax(1)]


def _identity_features(facet_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in facet_rows:
        grouped[_identity(row)].setdefault(str(row["facet_id"]), row)
    output = {}
    for identity, facets_by_id in grouped.items():
        facets = list(facets_by_id.values())
        types = Counter(str(row["facet_type"]) for row in facets)
        count = len(facets)
        output[identity] = np.asarray(
            [
                count,
                *(
                    _mean(facets, field)
                    for field in (
                        "token_count", "unique_token_count", "rare_token_fraction",
                        "entity_count", "relation_cue_count", "facet_confidence",
                        "component_nodes", "component_density",
                        "component_mean_edge_weight", "token_span_count",
                    )
                ),
                types["entity"] / count,
                types["relational"] / count,
                types["mixed"] / count,
            ],
            dtype=np.float64,
        )
    return output


def _evaluate_action_controllers(
    natural: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[str, str], np.ndarray],
    train: set[tuple[str, str]],
    heldout: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in natural:
        grouped[_identity(row)][str(row["condition"])] = row
    output, targets = [], []

    def emit(baseline, seed, identity, selected, oracle):
        row = grouped[identity][selected]
        output.append(
            {
                "baseline": baseline,
                "router_seed": seed,
                "dataset": identity[0],
                "example_id": identity[1],
                "selected_action": selected,
                "oracle_action": str(oracle["condition"]),
                "evidence_recall": float(row["evidence_recall"]),
                "precision": float(row["precision"]),
                "mrr": float(row["mrr"]),
                "routing_ms": float(row["routing_ms"]),
                "graph_calls": float(row["graph_calls"]),
                "pairwise_similarity_evaluations": float(row["pairwise_similarity_evaluations"]),
            }
        )

    all_actions = ACTION_SPACES["B3_facet_root_matched_successor"]
    for identity in sorted(heldout):
        oracle = _best([grouped[identity][name] for name in all_actions])
        emit("B0_static_hybrid", 0, identity, "paper26_hybrid", oracle)
        emit("B7_oracle_one_shot", 0, identity, str(oracle["condition"]), oracle)

    ordered_train, ordered_test = sorted(train), sorted(heldout)
    train_x = np.stack([features[identity] for identity in ordered_train])
    test_x = np.stack([features[identity] for identity in ordered_test])
    for baseline, actions in ACTION_SPACES.items():
        labels = []
        for identity in ordered_train:
            oracle = _best([grouped[identity][name] for name in actions])
            labels.append(str(oracle["condition"]))
            targets.append(
                {"baseline": baseline, "partition": "controller_train", "dataset": identity[0],
                 "example_id": identity[1], "oracle_action": str(oracle["condition"])}
            )
        for seed in ROUTER_SEEDS:
            classifier = RidgeClassifier.fit(train_x, labels, seed)
            for identity, selected in zip(ordered_test, classifier.predict(test_x)):
                oracle = _best([grouped[identity][name] for name in actions])
                emit(baseline, seed, identity, selected, oracle)
    return output, targets


def _rrf(rows: Sequence[Mapping[str, Any]], budget: int = 4) -> tuple[list[str], set[str]]:
    scores: dict[str, float] = defaultdict(float)
    positives: set[str] = set()
    for row in rows:
        positives.update(value for value in str(row["positive_chunk_ids"]).split("|") if value)
        for rank, chunk_id in enumerate(str(row["selected_chunk_ids"]).split("|"), 1):
            if chunk_id:
                scores[chunk_id] += 1.0 / (60 + rank)
    selected = sorted(scores, key=lambda item: (-scores[item], item))[:budget]
    return selected, positives


def _fused_metrics(selected: Sequence[str], positives: set[str]) -> dict[str, float]:
    hits = [index for index, value in enumerate(selected) if value in positives]
    return {
        "evidence_recall": len({selected[index] for index in hits}) / len(positives) if positives else 1.0,
        "precision": len(hits) / len(selected) if selected else 0.0,
        "mrr": 1.0 / (hits[0] + 1) if hits else 0.0,
    }


def _type_method(facet_type: str) -> str:
    return {"entity": "exact", "relational": "semantic", "mixed": "hybrid"}.get(
        facet_type, "hybrid"
    )


def _evaluate_per_facet(
    rows: Sequence[Mapping[str, Any]],
    train: set[tuple[str, str]],
    heldout: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[_identity(row)][str(row["facet_id"])].append(row)
    train_facets = [
        (identity, facet_id, candidates)
        for identity in sorted(train)
        for facet_id, candidates in grouped[identity].items()
    ]
    feature_fields = (
        "token_count", "unique_token_count", "rare_token_fraction", "entity_count",
        "relation_cue_count", "facet_confidence", "component_nodes",
        "component_density", "component_mean_edge_weight", "token_span_count",
    )
    train_x = np.asarray(
        [[float(candidates[0][field]) for field in feature_fields] for _, _, candidates in train_facets]
    )
    labels = [str(_best(candidates)["root_method"]) for _, _, candidates in train_facets]
    classifiers = {
        seed: RidgeClassifier.fit(train_x, labels, seed) for seed in ROUTER_SEEDS
    }
    output = []
    for identity in sorted(heldout):
        facets = grouped[identity]
        for policy, seed in (
            ("B4_fixed_semantic", 0),
            ("B4_type_rule", 0),
            ("B4_oracle_per_facet", 0),
            *(("B4_learned_per_facet", value) for value in ROUTER_SEEDS),
        ):
            selected_rows = []
            for candidates in facets.values():
                by_method = {str(row["root_method"]): row for row in candidates}
                if policy == "B4_fixed_semantic":
                    method = "semantic"
                elif policy == "B4_type_rule":
                    method = _type_method(str(candidates[0]["facet_type"]))
                elif policy == "B4_oracle_per_facet":
                    method = str(_best(candidates)["root_method"])
                else:
                    values = np.asarray([[float(candidates[0][field]) for field in feature_fields]])
                    method = classifiers[seed].predict(values)[0]
                selected_rows.append(by_method[method])
            selected, positives = _rrf(selected_rows)
            output.append(
                {
                    "baseline": policy,
                    "router_seed": seed,
                    "dataset": identity[0],
                    "example_id": identity[1],
                    "selected_chunks": "|".join(selected),
                    "facet_count": len(facets),
                    "fusion_method": "reciprocal_rank",
                    **_fused_metrics(selected, positives),
                }
            )
    return output


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["baseline"]), int(row["router_seed"]), str(row["dataset"]))].append(row)
        grouped[(str(row["baseline"]), int(row["router_seed"]), "all")].append(row)
    return [
        {
            "baseline": key[0], "router_seed": key[1], "dataset": key[2], "examples": len(group),
            "recall": _mean(group, "evidence_recall"), "precision": _mean(group, "precision"),
            "mrr": _mean(group, "mrr"),
            "routing_ms": _mean(group, "routing_ms") if "routing_ms" in group[0] else "",
        }
        for key, group in sorted(grouped.items())
    ]


def _plots(summary, graph_metrics, output):
    policies = (
        "B0_static_hybrid", "B2_graph_facet_choice",
        "B3_facet_root_matched_successor", "B4_learned_per_facet", "B7_oracle_one_shot",
    )
    values = []
    errors = []
    for policy in policies:
        samples = [float(row["recall"]) for row in summary if row["baseline"] == policy and row["dataset"] == "all"]
        values.append(statistics.fmean(samples))
        errors.append(statistics.pstdev(samples) if len(samples) > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.bar(range(len(policies)), values, yerr=errors, capsize=3, color="#2f6690")
    axis.set(xticks=range(len(policies)), xticklabels=[value.split("_")[0] for value in policies], ylabel="Evidence recall", title="Strengthened one-shot request/reply baselines")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "request_reply_baselines.png", dpi=180)
    fig.savefig(output / "request_reply_baselines.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for condition in ("graph_cc", "syntactic_graph", "graph_cc_hybrid", "syntactic_graph_hybrid"):
        rows = [row for row in graph_metrics if row["condition"] == condition]
        axis.scatter([float(row["construction_ms"]) for row in rows], [float(row["evidence_recall"]) for row in rows], label=condition.replace("_", " "), s=48)
    axis.set(xlabel="Graph construction + clustering (ms/example)", ylabel="Evidence recall", title="Graph-facet quality and construction cost")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / "graph_facet_quality_cost.png", dpi=180)
    fig.savefig(output / "graph_facet_quality_cost.pdf")
    plt.close(fig)


def run(args):
    natural = _read_csv(args.natural_dir / "natural_retrieval_rows.csv")
    facet_rows = _read_csv(args.natural_dir / "per_facet_method_rows.csv")
    features = _identity_features(facet_rows)
    identities = sorted({_identity(row) for row in natural})
    train, heldout = _split(identities)
    controller_rows, targets = _evaluate_action_controllers(natural, features, train, heldout)
    per_facet_rows = _evaluate_per_facet(facet_rows, train, heldout)
    combined = [*controller_rows, *per_facet_rows]
    summary = _summary(combined)

    natural_summary = _read_csv(args.natural_dir / "natural_retrieval_summary.csv")
    graph_metrics = [
        {
            "dataset": row["dataset"], "condition": row["condition"],
            "evidence_recall": row["evidence_recall"],
            "construction_ms": float(row["mean_graph_ms"]) + float(row["mean_cluster_ms"]),
            "graph_calls": row["mean_graph_calls"], "graph_density": row["mean_graph_density"],
            "pairwise_similarity_evaluations": row["mean_pairwise_similarity_evaluations"],
            "facet_overlap": row["mean_facet_overlap"], "graph_facets": row["mean_graph_facets"],
        }
        for row in natural_summary if row["condition"] in {
            "graph_cc", "graph_cc_hybrid", "syntactic_graph", "syntactic_graph_hybrid"
        }
    ]
    inherited = json.loads(args.inherited_method_findings.read_text(encoding="utf-8"))
    full = inherited["heldout_policy_summary"]
    scope_rows = [
        {"baseline": "B3", "scope": "inherited_full_root_successor", **full["S6_adaptive_both"]},
        {"baseline": "B7", "scope": "inherited_full_root_successor", **full["S7_oracle_both"]},
    ]
    b_table = []
    for baseline in ("B2_graph_facet_choice", "B3_facet_root_matched_successor", "B4_learned_per_facet", "B7_oracle_one_shot"):
        rows = [row for row in summary if row["baseline"] == baseline and row["dataset"] == "all"]
        b_table.append(
            {"baseline": baseline.split("_")[0], "policy": baseline, "scope": "root_discovery_at_four_chunks", "seeds": len(rows),
             "recall": _mean(rows, "recall"), "precision": _mean(rows, "precision"), "mrr": _mean(rows, "mrr")}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "request_reply_controller_rows.csv", controller_rows)
    _write_csv(args.output_dir / "request_reply_controller_targets.csv", targets)
    _write_csv(args.output_dir / "per_facet_policy_rows.csv", per_facet_rows)
    _write_csv(args.output_dir / "request_reply_summary.csv", summary)
    _write_csv(args.output_dir / "graph_specific_metrics.csv", graph_metrics)
    _write_csv(args.output_dir / "b2_b3_b4_b7_comparison.csv", b_table)
    _write_csv(args.output_dir / "full_method_scope_reference.csv", scope_rows)
    _plots(summary, graph_metrics, args.output_dir)
    findings = {
        "schema_version": "1.0",
        "scope": "post_hoc_identity_disjoint_request_reply_controller_split",
        "controller_features": list(FEATURE_NAMES),
        "dataset_identity_used": False,
        "train_identities": len(train),
        "heldout_identities": len(heldout),
        "router_seeds": list(ROUTER_SEEDS),
        "action_spaces": {key: list(value) for key, value in ACTION_SPACES.items()},
        "root_discovery_comparison": b_table,
        "full_root_successor_reference": scope_rows,
        "comparability_note": "Root-discovery B rows and inherited full traversal rows are separate scopes.",
    }
    (args.output_dir / "request_reply_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    base = Path("docs/papers/shared/results/paper3_5_adaptive_pra/request_reply_graph")
    parser.add_argument("--natural-dir", type=Path, default=base / "natural_replay")
    parser.add_argument("--output-dir", type=Path, default=base / "controller")
    parser.add_argument(
        "--inherited-method-findings", type=Path,
        default=Path("docs/papers/shared/results/paper3_5_adaptive_pra/adaptive_search_methods/paper3_5_findings.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
