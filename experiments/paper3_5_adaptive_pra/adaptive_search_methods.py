"""Adaptive root/successor search-method study for Paper 3.5.

The runner consumes frozen Paper 2.6 channel traces and the existing Paper 3.5
factorized surface. It trains small validation-only selectors, evaluates them
on identity-disjoint held-out examples, decomposes method-switch retry
headroom, and preserves unlike search and K/V costs as separate columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.adaptive_search import (  # noqa: E402
    MATCHED_SUCCESSOR,
    ROOT_METHODS,
    ROOT_METHOD_ALIASES,
    SUCCESSOR_METHODS,
    load_search_method_action_spec,
    method_cost_accounting,
    select_method_oracle,
    validate_method_feature_names,
)


FEATURE_NAMES = (
    "query_tokens",
    "query_terms",
    "rare_token_count",
    "query_rare_fraction",
    "query_idf_mean",
    "query_idf_max",
    "named_entity_count",
    "numeric_marker",
    "url_marker",
    "id_marker",
    "question_markers",
    "exact_top_score",
    "exact_score_gap",
    "bm25_top_score",
    "bm25_score_gap",
    "approx_top_score",
    "approx_score_gap",
    "gist_top_score",
    "semantic_score_gap",
    "hybrid_score_gap",
    "channel_disagreement",
    "mean_selected_jaccard",
    "query_region_layout",
    "facet_disagreement",
)
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "qasper": "QASPER",
    "2wikimultihopqa": "2Wiki",
    "musique": "MuSiQue",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row.get(field, 0.0)) for row in rows) if rows else 0.0


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    split = str(row.get("split", row.get("partition", "")))
    return split, str(row["dataset"]), str(row["example_id"])


def _canonical_root(name: str) -> str:
    return ROOT_METHOD_ALIASES.get(name, name)


def _root_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**row, "root_method": _canonical_root(str(row["channel"]))}


def _successor_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "root_method": _canonical_root(str(row["root_channel"])),
        "successor_method": str(row["successor_channel"]),
    }


@dataclass
class RidgeClassifier:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: Sequence[str],
        classes: Sequence[str],
        ridge: float = 0.1,
    ) -> "RidgeClassifier":
        if features.ndim != 2 or len(features) != len(targets) or not len(features):
            raise ValueError("Classifier needs aligned nonempty two-dimensional features.")
        names = tuple(classes)
        class_index = {name: index for index, name in enumerate(names)}
        if set(targets) - set(names):
            raise ValueError("A target is outside the declared method space.")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
        target = np.zeros((len(features), len(names)))
        for index, name in enumerate(targets):
            target[index, class_index[name]] = 1.0
        if design.shape[1] <= design.shape[0]:
            penalty = np.eye(design.shape[1]) * ridge
            penalty[-1, -1] = 0.0
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        else:
            weights = design.T @ np.linalg.solve(
                design @ design.T + np.eye(len(design)) * ridge, target
            )
        return cls(names, mean, scale, weights)

    def scores(self, features: np.ndarray) -> np.ndarray:
        design = np.column_stack(((features - self.mean) / self.scale, np.ones(len(features))))
        return design @ self.weights

    def predict(self, features: np.ndarray) -> list[str]:
        return [self.classes[int(index)] for index in self.scores(features).argmax(axis=1)]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        scores = self.scores(features)
        scores -= scores.max(axis=1, keepdims=True)
        values = np.exp(scores)
        return values / values.sum(axis=1, keepdims=True)


def _feature_matrix(
    identities: Sequence[tuple[str, str, str]],
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    validate_method_feature_names(names)
    return np.asarray(
        [[float(feature_map[identity].get(name, 0.0)) for name in names] for identity in identities],
        dtype=np.float64,
    )


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_identity(row)].append(dict(row))
    return grouped


def _best_fixed_pair(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["root_method"]), str(row["successor_method"]))].append(row)
    return min(
        grouped,
        key=lambda pair: (
            -len(grouped[pair]),
            -_mean(grouped[pair], "recall"),
            -_mean(grouped[pair], "precision"),
            _mean(grouped[pair], "comparisons"),
            pair,
        ),
    )


def _pair_row(
    rows: Sequence[Mapping[str, Any]], root_method: str, successor_method: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["root_method"] == root_method and row["successor_method"] == successor_method
    ]
    if not matches:
        raise KeyError(f"Missing measured pair {root_method}->{successor_method}.")
    return matches[0]


def _oracle_rows(
    root_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    successor_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root_output, successor_output, transitions = [], [], []
    for identity in sorted(root_groups):
        root = select_method_oracle(root_groups[identity])
        root_method = str(root["root_method"])
        root_output.append(
            {
                "split": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "oracle_root_method": root_method,
                "oracle_recall": root["recall"],
                "oracle_precision": root["precision"],
                "oracle_mrr": root["mrr"],
                "comparisons": root["comparisons"],
                "latency_ms": root["latency_ms"],
            }
        )
        pairs = list(successor_groups.get(identity, ()))
        if not pairs:
            continue
        conditional = [row for row in pairs if row["root_method"] == root_method]
        successor = select_method_oracle(conditional or pairs)
        combined = select_method_oracle(pairs)
        successor_output.append(
            {
                "split": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "conditioned_root_method": root_method,
                "oracle_successor_method": successor["successor_method"],
                "oracle_recall": successor["recall"],
                "oracle_precision": successor["precision"],
                "oracle_mrr": successor["mrr"],
                "combined_root_method": combined["root_method"],
                "combined_successor_method": combined["successor_method"],
                "combined_recall": combined["recall"],
                "combined_precision": combined["precision"],
                "combined_mrr": combined["mrr"],
                "comparisons": successor["comparisons"],
                "latency_ms": successor["latency_ms"],
                "mapping_semantics": successor["mapping_semantics"],
            }
        )
        transitions.append(
            {
                "split": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "root_method": combined["root_method"],
                "successor_method": combined["successor_method"],
                "recall": combined["recall"],
                "precision": combined["precision"],
            }
        )
    return root_output, successor_output, transitions


def _selector_features(
    features: Sequence[Mapping[str, Any]],
    root_oracles: Sequence[Mapping[str, Any]],
    successor_oracles: Sequence[Mapping[str, Any]],
    useful_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    by_identity = {_identity(row): dict(row) for row in features}
    root = {_identity(row): row for row in root_oracles}
    successor = {_identity(row): row for row in successor_oracles}
    useful_group = _group_rows(useful_rows)
    output = []
    for identity in sorted(by_identity):
        addresses = useful_group.get(identity, ())
        positive_counts = [
            float(row["minimum_candidate_count"])
            for row in addresses
            if float(row["minimum_candidate_count"]) > 0
        ]
        address = {
            "address_exposed": max((float(row["exposed"]) for row in addresses), default=0.0),
            "address_count": max((float(row["address_count"]) for row in addresses), default=0.0),
            "address_candidate_count": min(positive_counts, default=0.0),
            "address_successor_rank": min(
                (float(row["minimum_successor_rank"]) for row in addresses), default=0.0
            ),
            "useful_address": max((float(row["useful_address"]) for row in addresses), default=0.0),
            "iterative_gain": max((float(row["iterative_gain"]) for row in addresses), default=0.0),
        }
        row = {
            **by_identity[identity],
            **address,
            "oracle_root_method": root[identity]["oracle_root_method"],
            "oracle_successor_method": successor.get(identity, {}).get("oracle_successor_method", ""),
            "combined_root_method": successor.get(identity, {}).get("combined_root_method", ""),
            "combined_successor_method": successor.get(identity, {}).get("combined_successor_method", ""),
        }
        output.append(row)
        by_identity[identity] = row
    return output, by_identity


def _evaluate_policy(
    name: str,
    identities: Sequence[tuple[str, str, str]],
    roots: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    pairs: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    choices: Mapping[tuple[str, str, str], tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for identity in identities:
        root_method, successor_method = choices[identity]
        measured = _pair_row(pairs[identity], root_method, successor_method)
        root = next(row for row in roots[identity] if row["root_method"] == root_method)
        costs = method_cost_accounting(root, measured)
        rows.append(
            {
                "split": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "policy": name,
                "root_method": root_method,
                "successor_method": successor_method,
                "recall": measured["recall"],
                "precision": measured["precision"],
                "mrr": measured["mrr"],
                "complete_recovery": measured["complete_recovery"],
                "requested_chunks": 4,
                **costs,
            }
        )
    return rows


def _summarize_policy(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        for dataset in [*sorted({str(row["dataset"]) for row in selected}), "all"]:
            values = selected if dataset == "all" else [row for row in selected if row["dataset"] == dataset]
            output.append(
                {
                    "policy": policy,
                    "dataset": dataset,
                    "n": len(values),
                    "recall": _mean(values, "recall"),
                    "precision": _mean(values, "precision"),
                    "mrr": _mean(values, "mrr"),
                    "complete_recovery": _mean(values, "complete_recovery"),
                    "root_comparisons": _mean(values, "root_comparisons"),
                    "successor_comparisons": _mean(values, "successor_comparisons"),
                    "token_span_operations": _mean(values, "token_span_operations"),
                    "materialized_kv_tokens": _mean(values, "materialized_kv_tokens"),
                }
            )
    return output


def _fit_selectors(
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    successor_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    root_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
    successor_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[RidgeClassifier, RidgeClassifier, RidgeClassifier]:
    validation = sorted(
        identity
        for identity in feature_map
        if identity[0] == "validation" and identity in successor_groups
    )
    matrix = _feature_matrix(validation, feature_map)
    root_targets = [str(root_oracles[identity]["oracle_root_method"]) for identity in validation]
    root_model = RidgeClassifier.fit(matrix, root_targets, ROOT_METHODS)
    successor_validation = [identity for identity in validation if identity in successor_oracles]
    successor_matrix = _feature_matrix(successor_validation, feature_map)
    combined_targets = [
        str(successor_oracles[identity]["combined_successor_method"])
        for identity in successor_validation
    ]
    root_one_hot = np.asarray(
        [
            [float(root_oracles[identity]["oracle_root_method"] == name) for name in ROOT_METHODS]
            for identity in successor_validation
        ]
    )
    successor_model = RidgeClassifier.fit(
        np.column_stack((successor_matrix, root_one_hot)), combined_targets, SUCCESSOR_METHODS
    )
    fixed_pair = _best_fixed_pair(
        [row for identity in successor_validation for row in successor_groups[identity]]
    )
    fixed_root = fixed_pair[0]
    fixed_targets = []
    for identity in successor_validation:
        rows = [row for row in successor_groups[identity] if row["root_method"] == fixed_root]
        fixed_targets.append(str(select_method_oracle(rows)["successor_method"]))
    fixed_successor_model = RidgeClassifier.fit(matrix, fixed_targets, SUCCESSOR_METHODS)
    return root_model, successor_model, fixed_successor_model


def _study_policies(
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    root_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    successor_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    root_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
    successor_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = sorted(
        identity for identity in feature_map if identity[0] == "validation" and identity in successor_oracles
    )
    heldout = sorted(
        identity for identity in feature_map if identity[0] == "test" and identity in successor_oracles
    )
    validation_rows = [row for identity in validation for row in successor_groups[identity]]
    fixed_pair = _best_fixed_pair(validation_rows)
    fixed_same = _best_fixed_pair(
        [
            row
            for row in validation_rows
            if row["successor_method"] == MATCHED_SUCCESSOR[row["root_method"]]
        ]
    )
    dataset_pairs = {
        dataset: _best_fixed_pair(
            [row for identity in validation if identity[1] == dataset for row in successor_groups[identity]]
        )
        for dataset in sorted({identity[1] for identity in validation})
    }
    root_model, successor_model, fixed_successor_model = _fit_selectors(
        feature_map, successor_groups, root_oracles, successor_oracles
    )
    heldout_matrix = _feature_matrix(heldout, feature_map)
    predicted_roots = root_model.predict(heldout_matrix)
    root_one_hot = np.asarray(
        [[float(root == method) for method in ROOT_METHODS] for root in predicted_roots]
    )
    predicted_successors = successor_model.predict(np.column_stack((heldout_matrix, root_one_hot)))
    fixed_successors = fixed_successor_model.predict(heldout_matrix)
    choices: dict[str, dict[tuple[str, str, str], tuple[str, str]]] = {
        "S0_fixed_same": {identity: fixed_same for identity in heldout},
        "S1_dataset_fixed_diagnostic": {identity: dataset_pairs[identity[1]] for identity in heldout},
        "S2_static_hybrid": {identity: ("hybrid", "hybrid_state") for identity in heldout},
        "S3_fixed_pair": {identity: fixed_pair for identity in heldout},
        "S4_adaptive_root": {
            identity: (predicted_roots[index], fixed_pair[1]) for index, identity in enumerate(heldout)
        },
        "S5_adaptive_successor": {
            identity: (fixed_pair[0], fixed_successors[index]) for index, identity in enumerate(heldout)
        },
        "S6_adaptive_both": {
            identity: (predicted_roots[index], predicted_successors[index])
            for index, identity in enumerate(heldout)
        },
        "S7_oracle_both": {
            identity: (
                str(successor_oracles[identity]["combined_root_method"]),
                str(successor_oracles[identity]["combined_successor_method"]),
            )
            for identity in heldout
        },
    }
    choices["C0_nonlearned_cascade"] = {}
    for identity in heldout:
        feature = feature_map[identity]
        if float(feature["exact_score_gap"]) >= 0.25:
            pair = ("exact", "exact_new_address")
        elif float(feature["bm25_score_gap"]) >= 0.2:
            pair = ("bm25", "native_semantic")
        elif float(feature["channel_disagreement"]) >= 3:
            pair = ("semantic", "approximate_new_address")
        else:
            pair = fixed_pair
        choices["C0_nonlearned_cascade"][identity] = pair
    rows = [
        row
        for name, selected in choices.items()
        for row in _evaluate_policy(name, heldout, root_groups, successor_groups, selected)
    ]
    return rows, {
        "fixed_same": fixed_same,
        "fixed_pair": fixed_pair,
        "dataset_pairs": dataset_pairs,
        "root_model": root_model,
        "successor_model": successor_model,
        "heldout": heldout,
        "choices": choices,
    }


def _cross_dataset(
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    root_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    datasets = sorted({identity[1] for identity in feature_map})
    for heldout_dataset in datasets:
        train = sorted(
            identity
            for identity in feature_map
            if identity[0] == "validation" and identity[1] != heldout_dataset
        )
        evaluate = sorted(
            identity
            for identity in feature_map
            if identity[0] == "test" and identity[1] == heldout_dataset
        )
        model = RidgeClassifier.fit(
            _feature_matrix(train, feature_map),
            [str(root_oracles[identity]["oracle_root_method"]) for identity in train],
            ROOT_METHODS,
        )
        predictions = model.predict(_feature_matrix(evaluate, feature_map))
        rows.append(
            {
                "heldout_dataset": heldout_dataset,
                "train_datasets": "|".join(dataset for dataset in datasets if dataset != heldout_dataset),
                "train_n": len(train),
                "test_n": len(evaluate),
                "root_method_accuracy": statistics.fmean(
                    float(prediction == root_oracles[identity]["oracle_root_method"])
                    for prediction, identity in zip(predictions, evaluate)
                ),
                "dataset_feature_used": 0,
            }
        )
    return rows


def _self_embedding_gate(
    representation_file: Path,
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    root_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not representation_file.exists():
        return []
    import torch

    cached = torch.load(representation_file, map_location="cpu", weights_only=False)
    embeddings = {
        (str(row["dataset"]), str(row["example_id"])): row["representations"]["S2_embed_last"]["vector"]
        .float()
        .numpy()
        for row in cached
    }
    identities = sorted(identity for identity in feature_map if identity[1:] in embeddings)
    observable = _feature_matrix(identities, feature_map)
    embed = np.asarray([embeddings[identity[1:]] for identity in identities])
    targets = [str(root_oracles[identity]["oracle_root_method"]) for identity in identities]
    outputs = []
    conditions = (
        ("R0_observable", observable),
        ("S2_embed_last", embed),
        ("R0_plus_S2_embed_last", np.column_stack((observable, embed))),
    )
    for name, values in conditions:
        predictions = [""] * len(identities)
        for fold in range(4):
            train_indices = [index for index in range(len(identities)) if index % 4 != fold]
            test_indices = [index for index in range(len(identities)) if index % 4 == fold]
            model = RidgeClassifier.fit(
                values[train_indices], [targets[index] for index in train_indices], ROOT_METHODS
            )
            for index, prediction in zip(test_indices, model.predict(values[test_indices])):
                predictions[index] = prediction
        outputs.append(
            {
                "representation": name,
                "validation_n": 0,
                "heldout_n": len(identities),
                "datasets": "hotpotqa|qasper",
                "root_method_accuracy": statistics.fmean(
                    float(prediction == root_oracles[identity]["oracle_root_method"])
                    for prediction, identity in zip(predictions, identities)
                ),
                "coverage_note": "Four-fold identity-grouped OOF; cache covers HotpotQA/QASPER test identities only",
            }
        )
    return outputs


def _useful_address_study(
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    names = (
        "query_idf_max",
        "named_entity_count",
        "approx_top_score",
        "approx_score_gap",
        "semantic_score_gap",
        "address_exposed",
        "address_count",
        "address_candidate_count",
        "address_successor_rank",
    )
    validation = sorted(identity for identity in feature_map if identity[0] == "validation")
    heldout = sorted(identity for identity in feature_map if identity[0] == "test")
    targets = ["useful" if float(feature_map[i]["useful_address"]) else "not_useful" for i in validation]
    model = RidgeClassifier.fit(
        _feature_matrix(validation, feature_map, names),
        targets,
        ("not_useful", "useful"),
        ridge=0.5,
    )
    probabilities = model.probabilities(_feature_matrix(heldout, feature_map, names))[:, 1]
    return [
        {
            "split": identity[0],
            "dataset": identity[1],
            "example_id": identity[2],
            **{name: feature_map[identity][name] for name in names},
            "useful_address": feature_map[identity]["useful_address"],
            "iterative_gain": feature_map[identity]["iterative_gain"],
            "predicted_useful_probability": float(probability),
            "predicted_useful": int(probability >= 0.5),
        }
        for identity, probability in zip(heldout, probabilities)
    ]


def _retry_study(
    policy_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    successor_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    feature_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial = {_identity(row): row for row in policy_rows if row["policy"] == "S6_adaptive_both"}
    oracle = {_identity(row): row for row in policy_rows if row["policy"] == "S7_oracle_both"}
    upper = []
    for identity in sorted(initial):
        before, target = initial[identity], oracle[identity]
        rows = successor_groups[identity]
        target_recall = float(target["recall"])
        same_root = select_method_oracle(
            [row for row in rows if row["root_method"] == before["root_method"]]
        )
        same_successor = select_method_oracle(
            [row for row in rows if row["successor_method"] == before["successor_method"]]
        )
        if float(before["recall"]) >= target_recall - 1e-12:
            action, corrected = "stop", 0
        elif float(same_root["recall"]) >= target_recall - 1e-12:
            action, corrected = "change_successor_method", 1
        elif float(same_successor["recall"]) >= target_recall - 1e-12:
            action, corrected = "change_root_method", 1
        else:
            action, corrected = "combined_method_switch", 1
        upper.append(
            {
                "split": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "initial_root_method": before["root_method"],
                "initial_successor_method": before["successor_method"],
                "initial_recall": before["recall"],
                "oracle_root_method": target["root_method"],
                "oracle_successor_method": target["successor_method"],
                "oracle_recall": target["recall"],
                "correction_action": action,
                "corrected": corrected,
                "recall_gain": target_recall - float(before["recall"]),
            }
        )

    validation = sorted(
        identity
        for identity in feature_map
        if identity[0] == "validation" and identity in successor_groups
    )
    matrix = _feature_matrix(validation, feature_map)
    roots = context["root_model"].predict(matrix)
    one_hot = np.asarray([[float(root == method) for method in ROOT_METHODS] for root in roots])
    successors = context["successor_model"].predict(np.column_stack((matrix, one_hot)))
    labels = []
    retry_root_targets = []
    retry_successor_targets = []
    retry_features = []
    for index, (identity, root, successor) in enumerate(zip(validation, roots, successors)):
        rows = successor_groups[identity]
        try:
            before = _pair_row(rows, root, successor)
        except KeyError:
            # One validation example omits this measured cross-product. It is
            # excluded from retry training rather than assigned a fake score.
            continue
        retry_features.append(matrix[index])
        target = select_method_oracle(rows)
        retry_root_targets.append(str(target["root_method"]))
        retry_successor_targets.append(str(target["successor_method"]))
        if float(before["recall"]) >= float(target["recall"]) - 1e-12:
            labels.append("stop")
            continue
        same_root = select_method_oracle([row for row in rows if row["root_method"] == root])
        same_successor = select_method_oracle(
            [row for row in rows if row["successor_method"] == successor]
        )
        if float(same_root["recall"]) >= float(target["recall"]) - 1e-12:
            labels.append("change_successor_method")
        elif float(same_successor["recall"]) >= float(target["recall"]) - 1e-12:
            labels.append("change_root_method")
        else:
            labels.append("combined_method_switch")
    action_classes = (
        "stop",
        "change_root_method",
        "change_successor_method",
        "combined_method_switch",
    )
    retry_model = RidgeClassifier.fit(
        np.asarray(retry_features), labels, action_classes, ridge=0.5
    )
    retry_root_model = RidgeClassifier.fit(
        np.asarray(retry_features), retry_root_targets, ROOT_METHODS, ridge=0.5
    )
    retry_successor_model = RidgeClassifier.fit(
        np.asarray(retry_features), retry_successor_targets, SUCCESSOR_METHODS, ridge=0.5
    )
    heldout_features = _feature_matrix(context["heldout"], feature_map)
    predicted = retry_model.predict(heldout_features)
    predicted_roots = retry_root_model.predict(heldout_features)
    predicted_successors = retry_successor_model.predict(heldout_features)
    upper_map = {_identity(row): row for row in upper}
    targeted = []
    for identity, action, retry_root, retry_successor in zip(
        context["heldout"], predicted, predicted_roots, predicted_successors
    ):
        before, target = initial[identity], oracle[identity]
        selected_root = str(before["root_method"])
        selected_successor = str(before["successor_method"])
        if action == "stop":
            retry_pair = (selected_root, selected_successor)
        elif action == "change_root_method":
            retry_pair = (retry_root, selected_successor)
        elif action == "change_successor_method":
            retry_pair = (selected_root, retry_successor)
        else:
            retry_pair = (retry_root, retry_successor)
        try:
            learned_after = _pair_row(successor_groups[identity], *retry_pair)
        except KeyError:
            learned_after = before
            retry_pair = (selected_root, selected_successor)

        alternate_pair = context["fixed_pair"]
        if alternate_pair == (selected_root, selected_successor):
            alternate_pair = context["fixed_same"]
        always_after = _pair_row(successor_groups[identity], *alternate_pair)
        common = {
            "split": identity[0],
            "dataset": identity[1],
            "example_id": identity[2],
            "initial_recall": before["recall"],
            "oracle_recall": target["recall"],
            "oracle_action": upper_map[identity]["correction_action"],
        }
        targeted.extend(
            (
                {
                    **common,
                    "policy": "initial_no_retry",
                    "predicted_action": "stop",
                    "selected_root_method": selected_root,
                    "selected_successor_method": selected_successor,
                    "final_recall": before["recall"],
                    "action_correct": "",
                },
                {
                    **common,
                    "policy": "always_switch_channel",
                    "predicted_action": "always_switch_channel",
                    "selected_root_method": alternate_pair[0],
                    "selected_successor_method": alternate_pair[1],
                    "final_recall": always_after["recall"],
                    "action_correct": "",
                },
                {
                    **common,
                    "policy": "learned_targeted_retry",
                    "predicted_action": action,
                    "selected_root_method": retry_pair[0],
                    "selected_successor_method": retry_pair[1],
                    "final_recall": learned_after["recall"],
                    "action_correct": int(action == upper_map[identity]["correction_action"]),
                },
                {
                    **common,
                    "policy": "oracle_correction",
                    "predicted_action": upper_map[identity]["correction_action"],
                    "selected_root_method": target["root_method"],
                    "selected_successor_method": target["successor_method"],
                    "final_recall": target["recall"],
                    "action_correct": 1,
                },
            )
        )
    return upper, targeted


def _factorized_join(
    factorized_path: Path,
    regret_path: Path,
    successor_oracles: Mapping[tuple[str, str, str], Mapping[str, Any]],
    successor_groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    fixed_pair: tuple[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    method_by_example = {
        (identity[1], identity[2]): row for identity, row in successor_oracles.items()
    }
    pairs_by_example = {
        (identity[1], identity[2]): rows for identity, rows in successor_groups.items()
    }
    factorized = _read_csv(factorized_path)
    joined = []
    for row in factorized:
        identity = _identity(row)
        example_key = identity[1:]
        if example_key not in method_by_example:
            continue
        method = method_by_example[example_key]
        joined.append(
            {
                **row,
                "root_method": method["combined_root_method"],
                "successor_method": method["combined_successor_method"],
                "method_recall": method["combined_recall"],
                "method_precision": method["combined_precision"],
                "joint_quality_floor": min(float(row["chain_complete"]), float(method["combined_recall"])),
                "composition_only": 1,
            }
        )
    regret_source = _read_csv(regret_path)
    regret = []
    for row in regret_source:
        identity = _identity(row)
        example_key = identity[1:]
        if example_key not in method_by_example:
            continue
        method = method_by_example[example_key]
        fixed = _pair_row(pairs_by_example[example_key], *fixed_pair)
        regret.append(
            {
                "partition": identity[0],
                "dataset": identity[1],
                "example_id": identity[2],
                "seed": row["seed"],
                "factorized_cost_savings": row["quantization_regret"],
                "fixed_method_recall": fixed["recall"],
                "oracle_method_recall": method["combined_recall"],
                "adaptive_method_quality_gain": float(method["combined_recall"]) - float(fixed["recall"]),
                "combined_sources_reported_separately": 1,
            }
        )
    search_admission = [
        {
            "partition": row["partition"],
            "dataset": row["dataset"],
            "example_id": row["example_id"],
            "seed": row["seed"],
            "root_method": row["root_method"],
            "successor_method": row["successor_method"],
            "search_parent_budget": row["search_budget"],
            "kv_parent_budget": row["kv_budget"],
            "materialized_kv_tokens": row["materialized_kv_tokens"],
            "conceptual_chain_complete": row["conceptual_chain_complete"],
            "physical_chain_complete": row["chain_complete"],
            "paper2_6_method_materialization_performed": 0,
        }
        for row in joined
    ]
    return joined, regret, search_admission


def _cost_accounting(
    root_rows: Sequence[Mapping[str, Any]], successor_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for stage, field, methods, rows in (
        ("root", "root_method", ROOT_METHODS, root_rows),
        ("successor", "successor_method", SUCCESSOR_METHODS, successor_rows),
    ):
        for method in methods:
            selected = [row for row in rows if row[field] == method]
            output.append(
                {
                    "stage": stage,
                    "method": method,
                    "n": len(selected),
                    "comparisons": _mean(selected, "comparisons"),
                    "index_lookups": _mean(selected, "index_lookups"),
                    "token_span_operations": _mean(selected, "token_span_operations"),
                    "latency_ms_exhaustive_python": _mean(selected, "latency_ms"),
                    "index_memory_bytes": _mean(selected, "index_memory_bytes"),
                    "placement": selected[0]["placement"] if selected else "",
                    "cross_method_scalar_cost": "not_defined",
                }
            )
    return output


def _query_region_interaction(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    layouts = sorted({str(row["query_region_layout"]) for row in features})
    counts = {layout: sum(str(row["query_region_layout"]) == layout for row in features) for layout in layouts}
    identifiable = len(layouts) > 1 and min(counts.values()) >= 8
    return [
        {
            "query_region_layout": layout,
            "examples": counts[layout],
            "distinct_layouts": len(layouts),
            "method_interaction_identifiable": int(identifiable),
            "decision": "estimate interaction" if identifiable else "defer interaction-aware head",
        }
        for layout in layouts
    ]


def _save_figure(figure, output: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=190)
    figure.savefig(output / f"{name}.pdf")
    plt.close(figure)


def _plots(
    output: Path,
    root_oracles: Sequence[Mapping[str, Any]],
    successor_oracles: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    selector_features: Sequence[Mapping[str, Any]],
    useful: Sequence[Mapping[str, Any]],
    search_admission: Sequence[Mapping[str, Any]],
    cross_dataset: Sequence[Mapping[str, Any]],
    retry: Sequence[Mapping[str, Any]],
    regret: Sequence[Mapping[str, Any]],
) -> None:
    held_root = [row for row in root_oracles if row["split"] == "test"]
    held_successor = [row for row in successor_oracles if row["split"] == "test"]
    conditions = (
        (held_root, "oracle_root_method", ROOT_METHODS, "root_method_oracle_distribution", "Root-method oracle"),
        (held_successor, "oracle_successor_method", SUCCESSOR_METHODS, "successor_method_oracle_distribution", "Successor-method oracle"),
    )
    for rows, field, methods, name, title in conditions:
        figure, axis = plt.subplots(figsize=(7.6, 4.2))
        datasets = sorted({str(row["dataset"]) for row in rows})
        width = 0.15
        x = np.arange(len(methods))
        for index, dataset in enumerate(datasets):
            values = [sum(row[field] == method and row["dataset"] == dataset for row in rows) for method in methods]
            axis.bar(x + (index - 1.5) * width, values, width, label=DATASET_LABELS.get(dataset, dataset))
        axis.set_xticks(x, [method.replace("_", "\n") for method in methods], fontsize=8)
        axis.set_ylabel("held-out examples")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        _save_figure(figure, output, name)

    matrix = np.zeros((len(ROOT_METHODS), len(SUCCESSOR_METHODS)))
    for row in transitions:
        if row["split"] == "test":
            matrix[ROOT_METHODS.index(row["root_method"]), SUCCESSOR_METHODS.index(row["successor_method"])] += 1
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(SUCCESSOR_METHODS)), [x.replace("_", "\n") for x in SUCCESSOR_METHODS], fontsize=8)
    axis.set_yticks(range(len(ROOT_METHODS)), ROOT_METHODS)
    axis.set_xlabel("successor method")
    axis.set_ylabel("root method")
    for i in range(len(ROOT_METHODS)):
        for j in range(len(SUCCESSOR_METHODS)):
            axis.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="held-out oracle transitions")
    _save_figure(figure, output, "root_successor_method_heatmap")

    overall = [row for row in policy_summary if row["dataset"] == "all"]
    figure, axis = plt.subplots(figsize=(7.6, 4.5))
    for row in overall:
        cost = float(row["root_comparisons"]) + float(row["successor_comparisons"])
        axis.scatter(cost, float(row["recall"]), s=55)
        axis.annotate(str(row["policy"]).replace("_", " "), (cost, float(row["recall"])), fontsize=7)
    axis.set_xlabel("search comparisons (operation count, not latency-equivalent)")
    axis.set_ylabel("held-out evidence recall")
    axis.grid(alpha=0.25)
    _save_figure(figure, output, "fixed_adaptive_quality_cost_frontier")

    root_lookup = {_identity(row): row for row in root_oracles}
    held_features = [row for row in selector_features if row["split"] == "test"]
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    for method in ROOT_METHODS:
        rows = [row for row in held_features if root_lookup[_identity(row)]["oracle_root_method"] == method]
        axis.scatter(
            [float(row["channel_disagreement"]) for row in rows],
            [float(row["mean_selected_jaccard"]) for row in rows],
            label=method,
            alpha=0.75,
        )
    axis.set_xlabel("channel disagreement count")
    axis.set_ylabel("mean selected-set Jaccard")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    _save_figure(figure, output, "channel_disagreement_selected_method")

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.scatter(
        [float(row["predicted_useful_probability"]) for row in useful],
        [float(row["iterative_gain"]) for row in useful],
        c=[float(row["useful_address"]) for row in useful],
        cmap="coolwarm",
        alpha=0.75,
    )
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("predicted useful-address probability")
    axis.set_ylabel("lexical-successor iterative gain")
    axis.grid(alpha=0.25)
    _save_figure(figure, output, "useful_address_confidence_gain")

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.scatter(
        [float(row["search_parent_budget"]) for row in search_admission],
        [float(row["kv_parent_budget"]) for row in search_admission],
        c=[float(row["materialized_kv_tokens"]) for row in search_admission],
        cmap="viridis",
        alpha=0.55,
    )
    axis.plot([0, 8], [0, 8], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("conceptual search parent budget")
    axis.set_ylabel("K/V admission parent budget")
    axis.grid(alpha=0.25)
    _save_figure(figure, output, "search_breadth_kv_admission_breadth")

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(
        [DATASET_LABELS.get(row["heldout_dataset"], row["heldout_dataset"]) for row in cross_dataset],
        [float(row["root_method_accuracy"]) for row in cross_dataset],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("leave-one-dataset-out root-method accuracy")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output, "method_cross_dataset_generalization")

    counts = Counter(str(row["correction_action"]) for row in retry)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar([name.replace("_", "\n") for name in counts], list(counts.values()))
    axis.set_ylabel("held-out examples")
    axis.set_title("Oracle corrective action from adaptive root+successor")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output, "retry_correction_by_action")

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    datasets = sorted({str(row["dataset"]) for row in regret})
    x = np.arange(len(datasets))
    factorized = [
        statistics.fmean(
            float(row["factorized_cost_savings"])
            for row in regret
            if row["dataset"] == dataset and str(row["factorized_cost_savings"]).strip()
        )
        for dataset in datasets
    ]
    method = [
        statistics.fmean(
            float(row["adaptive_method_quality_gain"])
            for row in regret
            if row["dataset"] == dataset
        )
        for dataset in datasets
    ]
    axis.bar(x - 0.18, factorized, 0.36, label="factorized cost savings")
    second = axis.twinx()
    second.bar(x + 0.18, method, 0.36, color="tab:orange", label="method recall gain")
    axis.set_xticks(x, [DATASET_LABELS.get(d, d) for d in datasets])
    axis.set_ylabel("abstract cost savings")
    second.set_ylabel("evidence recall gain")
    axis.set_title("Independent sources of adaptive headroom")
    _save_figure(figure, output, "factorized_method_oracle_regret")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_search_method_action_spec(args.paper2_dir / "search_method_action_spec.json")
    action_space = {
        **spec.to_dict(),
        "control_vector": [
            "Q_regions", "S_root", "S_succ", "F", "R", "K", "H",
            "B_search", "B_KV", "theta", "L", "G", "M",
        ],
        "interpret": ["Q_regions", "F", "S_root"],
        "search": ["R", "K", "H", "S_succ"],
        "admit": ["B_search", "B_KV", "theta", "L", "G", "M"],
        "root_successor_independent": True,
    }
    (args.output_dir / "search_method_action_space.json").write_text(
        json.dumps(action_space, indent=2, sort_keys=True), encoding="utf-8"
    )

    root_rows = [
        normalized
        for row in _read_csv(args.paper2_dir / "root_channel_results.csv")
        for normalized in (_root_row(row),)
        if normalized["root_method"] in ROOT_METHODS
    ]
    successor_rows = [
        _successor_row(row) for row in _read_csv(args.paper2_dir / "successor_channel_results.csv")
    ]
    features = _read_csv(args.paper2_dir / "selector_observable_features.csv")
    useful_rows = _read_csv(args.paper2_dir / "iterative_useful_address.csv")
    root_groups = _group_rows(root_rows)
    successor_groups = _group_rows(successor_rows)
    root_oracle_rows, successor_oracle_rows, transitions = _oracle_rows(root_groups, successor_groups)
    _write_csv(args.output_dir / "root_method_oracle.csv", root_oracle_rows)
    _write_csv(args.output_dir / "successor_method_oracle.csv", successor_oracle_rows)
    _write_csv(args.output_dir / "root_successor_transition.csv", transitions)

    selector_rows, feature_map = _selector_features(
        features, root_oracle_rows, successor_oracle_rows, useful_rows
    )
    _write_csv(args.output_dir / "method_selector_features.csv", selector_rows)
    root_oracles = {_identity(row): row for row in root_oracle_rows}
    successor_oracles = {_identity(row): row for row in successor_oracle_rows}
    policy_rows, context = _study_policies(
        feature_map, root_groups, successor_groups, root_oracles, successor_oracles
    )
    policy_summary = _summarize_policy(policy_rows)
    embedding_rows = _self_embedding_gate(args.representation_file, feature_map, root_oracles)
    _write_csv(args.output_dir / "method_selector_results.csv", [*policy_summary, *embedding_rows])

    cross_dataset = _cross_dataset(feature_map, root_oracles)
    _write_csv(args.output_dir / "method_cross_dataset_results.csv", cross_dataset)
    useful = _useful_address_study(feature_map)
    _write_csv(args.output_dir / "useful_address_features.csv", useful)
    retry_upper, targeted_retry = _retry_study(policy_rows, context, successor_groups, feature_map)
    _write_csv(args.output_dir / "method_retry_upper_bound.csv", retry_upper)
    _write_csv(args.output_dir / "method_targeted_retry.csv", targeted_retry)

    factorized, regret, search_admission = _factorized_join(
        args.factorized_oracles,
        args.profile_regret,
        successor_oracles,
        successor_groups,
        context["fixed_pair"],
    )
    _write_csv(args.output_dir / "factorized_method_oracle.csv", factorized)
    _write_csv(args.output_dir / "method_profile_quantization_regret.csv", regret)
    _write_csv(args.output_dir / "search_vs_admission_results.csv", search_admission)
    costs = _cost_accounting(root_rows, successor_rows)
    _write_csv(args.output_dir / "method_cost_accounting.csv", costs)
    query_interaction = _query_region_interaction(selector_rows)
    _write_csv(args.output_dir / "query_region_method_interaction.csv", query_interaction)

    _plots(
        args.output_dir,
        root_oracle_rows,
        successor_oracle_rows,
        transitions,
        policy_summary,
        selector_rows,
        useful,
        search_admission,
        cross_dataset,
        retry_upper,
        regret,
    )

    overall = {row["policy"]: row for row in policy_summary if row["dataset"] == "all"}
    validation_root_methods = {
        method: statistics.fmean(
            float(row["recall"])
            for row in root_rows
            if row["split"] == "validation" and row["root_method"] == method
        )
        for method in ROOT_METHODS
    }
    fixed_root_method = max(validation_root_methods, key=validation_root_methods.get)
    heldout_fixed_root_recall = statistics.fmean(
        float(row["recall"])
        for row in root_rows
        if row["split"] == "test" and row["root_method"] == fixed_root_method
    )
    heldout_root_oracle_recall = statistics.fmean(
        float(row["oracle_recall"]) for row in root_oracle_rows if row["split"] == "test"
    )
    heldout_successor_conditional_oracle = statistics.fmean(
        float(row["oracle_recall"]) for row in successor_oracle_rows if row["split"] == "test"
    )
    learned_retry = [row for row in targeted_retry if row["policy"] == "learned_targeted_retry"]
    retry_summary = {
        policy: statistics.fmean(
            float(row["final_recall"]) for row in targeted_retry if row["policy"] == policy
        )
        for policy in sorted({str(row["policy"]) for row in targeted_retry})
    }
    predicted_useful = [row for row in useful if int(row["predicted_useful"])]
    useful_true_positive = sum(
        int(row["predicted_useful"]) and int(float(row["useful_address"])) for row in useful
    )
    useful_metrics = {
        "positive_rate": _mean(useful, "useful_address"),
        "predicted_positive_rate": len(predicted_useful) / max(len(useful), 1),
        "precision": useful_true_positive / max(len(predicted_useful), 1),
        "recall": useful_true_positive / max(sum(int(float(row["useful_address"])) for row in useful), 1),
    }
    transition_counts = Counter(
        (row["root_method"], row["successor_method"])
        for row in transitions
        if row["split"] == "test"
    )
    findings = {
        "schema_version": "1.0",
        "study": "adaptive_search_method_selection",
        "examples": len(feature_map),
        "validation_examples": sum(identity[0] == "validation" for identity in feature_map),
        "heldout_examples": sum(identity[0] == "test" for identity in feature_map),
        "paper2_6_action_spec_sha256": spec.source_sha256,
        "materialization_performed_by_method_study": False,
        "root_successor_independent": True,
        "fixed_pair": list(context["fixed_pair"]),
        "root_method_headroom": {
            "validation_selected_method": fixed_root_method,
            "heldout_fixed_recall": heldout_fixed_root_recall,
            "heldout_oracle_recall": heldout_root_oracle_recall,
            "oracle_gain": heldout_root_oracle_recall - heldout_fixed_root_recall,
        },
        "successor_method_headroom": {
            "heldout_conditional_oracle_recall": heldout_successor_conditional_oracle,
            "heldout_combined_oracle_recall": overall["S7_oracle_both"]["recall"],
            "heldout_fixed_pair_recall": overall["S3_fixed_pair"]["recall"],
        },
        "heldout_policy_summary": overall,
        "heldout_oracle_transition_counts": {
            f"{root}->{successor}": count
            for (root, successor), count in sorted(transition_counts.items())
        },
        "cross_dataset": cross_dataset,
        "self_embedding_gate": embedding_rows,
        "useful_address_gate": useful_metrics,
        "retry_final_recall": retry_summary,
        "learned_retry_action_accuracy": _mean(learned_retry, "action_correct"),
        "factorized_join_examples": len({(row["dataset"], row["example_id"]) for row in factorized}),
        "factorized_join_is_composition_only": True,
        "query_region_method_interaction": query_interaction,
        "claims": [
            "retrieval representation is a separate adaptive action",
            "root and successor method choices need not match",
            "method oracle and validation instability are evaluator-side diagnostics",
            "search comparisons and physical K/V admission remain separate",
            "no generated-answer or materialization gain is claimed",
        ],
    }
    (args.output_dir / "paper3_5_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    aggregate_path = args.output_dir.parent / "paper3_5_findings.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    aggregate["adaptive_search_method_selection"] = findings
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "adaptive_search_claim_audit.md").write_text(
        "# Adaptive search-method claim audit\n\n"
        "- The study replays frozen Paper 2.6 discovery rows; it does not rerun generation.\n"
        "- Root and successor requests are two chunks each and are evaluated at matched budgets.\n"
        "- Oracle labels select rows only in offline analysis and never enter deployment features.\n"
        "- Dataset identity is metadata and is excluded from every learned selector.\n"
        "- The Paper 3.5 factorized join reports both measured surfaces and a conservative floor; "
        "it is not an end-to-end joint execution.\n"
        "- Paper 2.6 performs no K/V materialization. Search and K/V costs are not collapsed into "
        "a cross-system scalar.\n",
        encoding="utf-8",
    )
    return findings


def _paper2_default() -> Path:
    candidates = (
        ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection",
        ROOT.parent / "pdattention" / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    output = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/adaptive_search_methods"
    parser.add_argument("--paper2-dir", type=Path, default=_paper2_default())
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument(
        "--representation-file",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/self_router_representations.pt",
    )
    parser.add_argument(
        "--factorized-oracles",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/factorized_oracle_rows.csv",
    )
    parser.add_argument(
        "--profile-regret",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/profile_quantization_regret.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
