"""Build the measurement-only Paper 2.5 final metrics gate.

The runner consumes frozen feature caches and previously emitted execution
rows.  Oracle identities are used only after a facet or graph condition has
been selected, never as selector inputs or graph-search controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import _feature_example
from pra_hf.cross_dataset_diagnostics import all_offset_multiscale_facets
from pra_hf.final_metrics import (
    decompose_path_survival,
    facet_confidence_features,
    fit_linear_selector,
    pareto_flags,
    predict_linear_selector,
    require_disjoint_identifiers,
    selected_facet_group_rank,
)
from pra_hf.layerwise_graph import pearson, spearman
from pra_hf.natural_reasoning_graph import map_example_to_parents


SCALES = ("global", "1", "2", "4", "8", "16")
LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
REPRESENTATIVE_LAYERS = (0, 12, 27)
REPRESENTATIVE_CHUNKS = (32, 128, 256)
RECALL_K = (1, 2, 4, 8)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _clean(value) for key, value in row.items()} for row in rows)


def _mean(rows, field):
    values = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        if isinstance(value, str) and value.lower() in ("true", "false"):
            values.append(float(value.lower() == "true"))
        else:
            values.append(float(value))
    return statistics.fmean(values) if values else None


def _scale(provenance: dict) -> str:
    return "global" if provenance["kind"] == "global" else str(int(provenance["scale"]))


def _position_bin(relative_midpoint: float) -> str:
    if relative_midpoint < 1 / 3:
        return "early"
    if relative_midpoint < 2 / 3:
        return "middle"
    return "late"


def _question_features(feature: dict, facet_hidden: torch.Tensor) -> list[float]:
    norms = facet_hidden.float().norm(dim=1)
    local = norms[1:] if norms.numel() > 1 else norms
    question = feature["question"].strip().lower()
    return [
        float(feature["question_tokens"]),
        float(norms.numel()),
        float(norms[0]),
        float(local.mean()),
        float(local.std(unbiased=False)),
        float(local.max()),
        float(local.min()),
        float(question.startswith(("who", "what", "which"))),
        float(question.startswith(("when", "where"))),
        float(question.startswith(("how", "why"))),
        float(question.startswith(("is", "are", "was", "were", "did", "do", "does"))),
    ]


def _facet_index_for_scale(
    scale: str, facets: list[dict], hidden_norms: torch.Tensor
) -> int:
    candidates = [index for index, row in enumerate(facets) if _scale(row) == scale]
    if not candidates:
        return int(torch.argmax(hidden_norms))
    return max(candidates, key=lambda index: (float(hidden_norms[index]), -index))


def _selector_indices(
    facets: list[dict],
    scores: torch.Tensor,
    hidden_norms: torch.Tensor,
    *,
    validation_scale: str,
    linear_scale: str,
) -> dict[str, int]:
    local = [index for index, row in enumerate(facets) if row["kind"] != "global"]
    confidence = facet_confidence_features(scores)
    return {
        "global": 0,
        "earliest": min(local, key=lambda index: (facets[index]["token_start"], index)),
        "latest": max(local, key=lambda index: (facets[index]["token_end"], -index)),
        "highest_hidden_norm": int(torch.argmax(hidden_norms)),
        "highest_parent_margin": int(torch.argmax(confidence["top_parent_margin"])),
        "lowest_parent_entropy": int(
            torch.argmin(confidence["normalized_parent_score_entropy"])
        ),
        "validation_most_common_scale": _facet_index_for_scale(
            validation_scale, facets, hidden_norms
        ),
        "validation_linear_scale": _facet_index_for_scale(linear_scale, facets, hidden_norms),
    }


def _winning_facets(args, output_dir: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    target_rows = _read_csv(args.natural_dir / "multiscale_target_ranks.csv")
    features = torch.load(args.natural_feature_file, map_location="cpu", weights_only=False)
    cached = torch.load(args.facet_cache, map_location="cpu", weights_only=False)
    feature_by_id = {row["example_id"]: row for row in features}
    cache_by_id = {row["example_id"]: row for row in cached}
    if set(feature_by_id) != set(cache_by_id):
        raise ValueError("natural feature and multiscale cache identities differ")

    prepared = {}
    for example_id, feature in feature_by_id.items():
        cache = cache_by_id[example_id]
        facets = all_offset_multiscale_facets(
            feature["query_hidden_states"].float(), tuple(feature["question_span"])
        )
        provenance = [row.__dict__ for row in facets.provenance]
        if provenance != cache["facet_provenance"]:
            raise ValueError(f"facet provenance mismatch for {example_id}")
        example = _feature_example(feature)
        mapping = map_example_to_parents(
            example,
            int(feature["source_tokens"]),
            feature["node_token_spans"],
            chunk_size=128,
        )
        prepared[example_id] = {
            "feature": feature,
            "cache": cache,
            "facets": provenance,
            "hidden": facets.hidden,
            "hidden_norms": facets.hidden.float().norm(dim=1),
            "mapping": mapping,
            "question_features": _question_features(feature, facets.hidden),
        }

    validation_labels = defaultdict(list)
    for row in target_rows:
        if row["partition"] == "validation" and row["role"] == "root":
            validation_labels[row["example_id"]].append(row["winning_scale"])
    scale_index = {scale: index for index, scale in enumerate(SCALES)}
    train_ids = sorted(validation_labels)
    heldout_ids = sorted(
        {
            row["example_id"]
            for row in target_rows
            if row["partition"] == "test" and row["role"] == "root"
        }
    )
    require_disjoint_identifiers(train_ids, heldout_ids)
    train_x = torch.tensor([prepared[key]["question_features"] for key in train_ids])
    train_y = torch.tensor(
        [
            scale_index[
                min(
                    Counter(validation_labels[key]).items(),
                    key=lambda item: (-item[1], scale_index[item[0]]),
                )[0]
            ]
            for key in train_ids
        ]
    )
    linear_model = fit_linear_selector(train_x, train_y, class_count=len(SCALES), ridge=2.0)
    all_ids = sorted(prepared)
    predicted = predict_linear_selector(
        linear_model,
        torch.tensor([prepared[key]["question_features"] for key in all_ids]),
    )
    predicted_scale = {key: SCALES[int(value)] for key, value in zip(all_ids, predicted)}

    common_scale = {}
    for dataset in ("musique", "2wikimultihopqa"):
        rows = [
            row
            for row in target_rows
            if row["dataset"] == dataset
            and row["partition"] == "validation"
            and row["role"] == "root"
        ]
        common_scale[dataset] = min(
            Counter(row["winning_scale"] for row in rows).items(),
            key=lambda item: (-item[1], scale_index[item[0]]),
        )[0]

    winning_rows = []
    for row in target_rows:
        item = prepared[row["example_id"]]
        facet = int(row["winning_facet"])
        provenance = item["facets"][facet]
        scores = item["cache"]["scores_by_seed"][int(row["seed"])]
        confidence = facet_confidence_features(scores)
        support_start, support_end = map(int, item["feature"]["question_span"])
        support_length = max(support_end - support_start, 1)
        midpoint = (provenance["token_start"] + provenance["token_end"]) / 2
        relative_midpoint = (midpoint - support_start) / support_length
        winning_rows.append(
            {
                **row,
                "winning_relative_midpoint": relative_midpoint,
                "winning_position_bin": _position_bin(relative_midpoint),
                "winning_hidden_norm": float(item["hidden_norms"][facet]),
                "winning_top_parent_score": float(confidence["top_parent_score"][facet]),
                "winning_top_parent_margin": float(confidence["top_parent_margin"][facet]),
                "winning_parent_score_entropy": float(
                    confidence["parent_score_entropy"][facet]
                ),
                "winning_normalized_parent_score_entropy": float(
                    confidence["normalized_parent_score_entropy"][facet]
                ),
                "question_tokens": item["feature"]["question_tokens"],
                "selector_feature_scope": "query states/provenance plus unlabeled parent scores",
            }
        )

    selector_rows = []
    root_targets = [row for row in target_rows if row["role"] == "root"]
    for row in root_targets:
        item = prepared[row["example_id"]]
        scores = item["cache"]["scores_by_seed"][int(row["seed"])]
        indices = _selector_indices(
            item["facets"],
            scores,
            item["hidden_norms"],
            validation_scale=common_scale[row["dataset"]],
            linear_scale=predicted_scale[row["example_id"]],
        )
        group = item["mapping"].node_parent_groups[row["node_id"]]
        for selector, facet in indices.items():
            rank = selected_facet_group_rank(scores, facet, group)
            selector_rows.append(
                {
                    "dataset": row["dataset"],
                    "example_id": row["example_id"],
                    "partition": row["partition"],
                    "seed": row["seed"],
                    "node_id": row["node_id"],
                    "selector": selector,
                    "selected_facet": facet,
                    "selected_scale": _scale(item["facets"][facet]),
                    "target_rank": rank,
                    **{f"R_at_{k}": int(rank <= k) for k in RECALL_K},
                    "target_used_during_selection": 0,
                }
            )

    selector_example_rows = []
    grouped_selectors = defaultdict(list)
    for row in selector_rows:
        grouped_selectors[
            (
                row["dataset"],
                row["example_id"],
                row["partition"],
                row["seed"],
                row["selector"],
            )
        ].append(row)
    for key, rows in grouped_selectors.items():
        rank = min(int(row["target_rank"]) for row in rows)
        selector_example_rows.append(
            {
                "dataset": key[0],
                "example_id": key[1],
                "partition": key[2],
                "seed": key[3],
                "selector": key[4],
                "best_root_rank": rank,
                **{f"R_at_{k}": int(rank <= k) for k in RECALL_K},
                "target_used_during_selection": 0,
                "aggregation": "post-hoc best annotated root per example",
            }
        )

    summary = []
    for partition in ("validation", "test"):
        for dataset in ("musique", "2wikimultihopqa"):
            for selector in sorted({row["selector"] for row in selector_rows}):
                rows = [
                    row
                    for row in selector_example_rows
                    if row["partition"] == partition
                    and row["dataset"] == dataset
                    and row["selector"] == selector
                ]
                summary.append(
                    {
                        "summary_type": "diagnostic_selector",
                        "partition": partition,
                        "dataset": dataset,
                        "role": "root",
                        "selector": selector,
                        "rows": len(rows),
                        **{f"R_at_{k}": _mean(rows, f"R_at_{k}") for k in RECALL_K},
                    }
                )
    for partition in ("validation", "test"):
        for dataset in ("musique", "2wikimultihopqa"):
            for role in ("root", "intermediate", "terminal"):
                base = [
                    row
                    for row in winning_rows
                    if row["partition"] == partition
                    and row["dataset"] == dataset
                    and row["role"] == role
                ]
                if not base:
                    continue
                for scale in SCALES:
                    rows = [row for row in base if row["winning_scale"] == scale]
                    summary.append(
                        {
                            "summary_type": "winning_scale_distribution",
                            "partition": partition,
                            "dataset": dataset,
                            "role": role,
                            "winning_scale": scale,
                            "rows": len(rows),
                            "fraction": len(rows) / len(base),
                        }
                    )
                for position in ("early", "middle", "late"):
                    rows = [row for row in base if row["winning_position_bin"] == position]
                    summary.append(
                        {
                            "summary_type": "winning_position_distribution",
                            "partition": partition,
                            "dataset": dataset,
                            "role": role,
                            "position_bin": position,
                            "rows": len(rows),
                            "fraction": len(rows) / len(base),
                        }
                    )
    for dataset in ("musique", "2wikimultihopqa"):
        fields = ("annotated_hops",) if dataset == "musique" else ("question_type", "graph_type")
        for field in fields:
            values = sorted(
                {
                    row[field]
                    for row in winning_rows
                    if row["dataset"] == dataset
                    and row["partition"] == "test"
                    and row["role"] == "root"
                }
            )
            for value in values:
                base = [
                    row
                    for row in winning_rows
                    if row["dataset"] == dataset
                    and row["partition"] == "test"
                    and row["role"] == "root"
                    and row[field] == value
                ]
                for scale in SCALES:
                    rows = [row for row in base if row["winning_scale"] == scale]
                    summary.append(
                        {
                            "summary_type": f"winning_scale_by_{field}",
                            "partition": "test",
                            "dataset": dataset,
                            "role": "root",
                            field: value,
                            "winning_scale": scale,
                            "rows": len(rows),
                            "fraction": len(rows) / len(base),
                        }
                    )

    model_metadata = {
        "feature_names": [
            "question_tokens",
            "facet_count",
            "global_hidden_norm",
            "local_hidden_norm_mean",
            "local_hidden_norm_std",
            "local_hidden_norm_max",
            "local_hidden_norm_min",
            "who_what_which",
            "when_where",
            "how_why",
            "auxiliary_yes_no",
        ],
        "classes": list(SCALES),
        "fit_partition": "validation",
        "fit_examples": len(train_ids),
        "target": "majority oracle-winning root scale per validation example",
        "target_or_parent_identity_in_features": False,
        "ridge": 2.0,
        "validation_most_common_scale": common_scale,
    }
    _write_csv(output_dir / "winning_facet_selector_rows.csv", selector_rows)
    _write_csv(
        output_dir / "winning_facet_selector_example_rows.csv", selector_example_rows
    )
    return winning_rows, summary, selector_rows, model_metadata


def _visibility(winning_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    role_rows, hop_rows = [], []
    for dataset in ("musique", "2wikimultihopqa"):
        for role in ("root", "intermediate", "terminal"):
            rows = [
                row
                for row in winning_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["role"] == role
            ]
            if not rows:
                continue
            counts = Counter(row["winning_scale"] for row in rows)
            role_rows.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "rows": len(rows),
                    **{
                        f"oracle_facet_R_at_{k}": statistics.fmean(
                            int(row["semantic_best_rank"]) <= k for row in rows
                        )
                        for k in RECALL_K
                    },
                    **{
                        f"lexical_R_at_{k}": statistics.fmean(
                            int(row["lexical_rank"]) <= k for row in rows
                        )
                        for k in RECALL_K
                    },
                    "most_common_winning_scale": min(
                        counts.items(), key=lambda item: (-item[1], SCALES.index(item[0]))
                    )[0],
                    "mean_winning_relative_midpoint": _mean(
                        rows, "winning_relative_midpoint"
                    ),
                }
            )
        values = sorted(
            {
                int(row["annotated_step"])
                for row in winning_rows
                if row["dataset"] == dataset and row["partition"] == "test"
            }
        )
        for step in values:
            rows = [
                row
                for row in winning_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and int(row["annotated_step"]) == step
            ]
            counts = Counter(row["winning_scale"] for row in rows)
            hop_rows.append(
                {
                    "dataset": dataset,
                    "annotated_step": step,
                    "roles": "|".join(sorted({row["role"] for row in rows})),
                    "rows": len(rows),
                    **{
                        f"oracle_facet_R_at_{k}": statistics.fmean(
                            int(row["semantic_best_rank"]) <= k for row in rows
                        )
                        for k in RECALL_K
                    },
                    **{
                        f"lexical_R_at_{k}": statistics.fmean(
                            int(row["lexical_rank"]) <= k for row in rows
                        )
                        for k in RECALL_K
                    },
                    "most_common_winning_scale": min(
                        counts.items(), key=lambda item: (-item[1], SCALES.index(item[0]))
                    )[0],
                    "mean_winning_relative_midpoint": _mean(
                        rows, "winning_relative_midpoint"
                    ),
                }
            )
    return role_rows, hop_rows


def _aggregate_example_layer(rows, value_field, *, filters=None):
    grouped = defaultdict(list)
    for row in rows:
        if filters and not filters(row):
            continue
        value = row.get(value_field, "")
        if value in ("", None):
            continue
        grouped[(row["example_id"], int(row["layer"]))].append(float(value))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _bootstrap_layer_correlation(
    context: dict,
    graph: dict,
    *,
    replicates: int = 1000,
    seed: int = 20260815,
) -> dict:
    identities = sorted({key[0] for key in context}.intersection(key[0] for key in graph))
    usable = [
        identity
        for identity in identities
        if all((identity, layer) in context and (identity, layer) in graph for layer in LAYERS)
    ]
    if len(usable) < 2:
        return {"examples": len(usable), "pearson": None, "spearman": None,
                "pearson_low": None, "pearson_high": None,
                "spearman_low": None, "spearman_high": None}
    base_left = [
        statistics.fmean(context[(identity, layer)] for identity in usable)
        for layer in LAYERS
    ]
    base_right = [
        statistics.fmean(graph[(identity, layer)] for identity in usable)
        for layer in LAYERS
    ]
    generator = torch.Generator().manual_seed(seed)
    pearsons, spearmans = [], []
    for _ in range(replicates):
        sample = [usable[int(index)] for index in torch.randint(len(usable), (len(usable),), generator=generator)]
        left = [statistics.fmean(context[(identity, layer)] for identity in sample) for layer in LAYERS]
        right = [statistics.fmean(graph[(identity, layer)] for identity in sample) for layer in LAYERS]
        p, s = pearson(left, right), spearman(left, right)
        if math.isfinite(p):
            pearsons.append(p)
        if math.isfinite(s):
            spearmans.append(s)
    pearsons.sort(); spearmans.sort()
    return {
        "examples": len(usable),
        "bootstrap_replicates": replicates,
        "pearson": pearson(base_left, base_right),
        "spearman": spearman(base_left, base_right),
        "pearson_low": pearsons[int(0.025 * len(pearsons))] if pearsons else None,
        "pearson_high": pearsons[max(0, int(0.975 * len(pearsons)) - 1)] if pearsons else None,
        "spearman_low": spearmans[int(0.025 * len(spearmans))] if spearmans else None,
        "spearman_high": spearmans[max(0, int(0.975 * len(spearmans)) - 1)] if spearmans else None,
    }


def _layer_synthesis(layer_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    joined = _read_csv(layer_dir / "layerwise_joined_summary.csv")
    depth = _read_csv(layer_dir / "layerwise_musique_depth_summary.csv")
    by_depth = {(int(row["layer"]), int(row["annotated_depth"])): row for row in depth}
    summary = []
    for row in joined:
        layer = int(row["layer"])
        summary.append(
            {
                **row,
                **{
                    f"D{d}_minimum_native_depth": by_depth[(layer, d)][
                        "mean_minimum_native_depth_if_complete"
                    ]
                    for d in (2, 3, 4)
                },
                **{
                    f"D{d}_complete_recovery": by_depth[(layer, d)]["complete_recovery"]
                    for d in (2, 3, 4)
                },
            }
        )

    context_rows = _read_csv(layer_dir / "layerwise_context_rows.csv")
    transition_rows = _read_csv(layer_dir / "layerwise_transition_rows.csv")
    graph_rows = _read_csv(layer_dir / "layerwise_graph_rows.csv")
    context_maps = {
        "attention_contribution_ratio": _aggregate_example_layer(
            context_rows,
            "attention_contribution_ratio",
            filters=lambda row: row["partition"] == "test" and row["token_class"] == "all" and row["context_radius"] == "full",
        ),
        "attention_rotation": _aggregate_example_layer(
            context_rows,
            "post_attention_rotation",
            filters=lambda row: row["partition"] == "test" and row["token_class"] == "all" and row["context_radius"] == "full",
        ),
        "intervention_contextualization": _aggregate_example_layer(
            context_rows,
            "intervention_displacement",
            filters=lambda row: row["partition"] == "test" and row["token_class"] == "all" and row["context_radius"] == "32",
        ),
        "attention_entropy": _aggregate_example_layer(
            context_rows,
            "attention_entropy",
            filters=lambda row: row["partition"] == "test" and row["token_class"] == "all" and row["context_radius"] == "full",
        ),
        "effective_attention_support": _aggregate_example_layer(
            context_rows,
            "effective_attention_support",
            filters=lambda row: row["partition"] == "test" and row["token_class"] == "all" and row["context_radius"] == "full",
        ),
    }
    graph_maps = {
        "edge_R6": _aggregate_example_layer(
            transition_rows,
            "recovered_at_6",
            filters=lambda row: row["dataset"] == "2wikimultihopqa" and row["partition"] == "test" and row["mapping_status"] == "preserved",
        ),
        "edge_MRR": _aggregate_example_layer(
            transition_rows,
            "reciprocal_rank",
            filters=lambda row: row["dataset"] == "2wikimultihopqa" and row["partition"] == "test" and row["mapping_status"] == "preserved",
        ),
        "depth_contraction": _aggregate_example_layer(
            graph_rows,
            "depth_contraction",
            filters=lambda row: row["dataset"] == "musique" and row["partition"] == "test",
        ),
        "shortcut_rate": _aggregate_example_layer(
            graph_rows,
            "shortcut_rate",
            filters=lambda row: row["partition"] == "test",
        ),
        "branching_factor": _aggregate_example_layer(
            graph_rows,
            "effective_branching_factor",
            filters=lambda row: row["partition"] == "test",
        ),
    }
    correlations = []
    for context_name, context in context_maps.items():
        for graph_name, graph in graph_maps.items():
            uncertainty = _bootstrap_layer_correlation(context, graph)
            correlations.append(
                {
                    "context_metric": context_name,
                    "graph_metric": graph_name,
                    "pearson": uncertainty.pop("pearson"),
                    "spearman": uncertainty.pop("spearman"),
                    "layer_points": len(LAYERS),
                    **uncertainty,
                }
            )

    cross_source = _read_csv(layer_dir / "layer_granularity_summary.csv")
    cross = []
    for layer in REPRESENTATIVE_LAYERS:
        for chunk in REPRESENTATIVE_CHUNKS:
            edge = next(
                row for row in cross_source
                if row["dataset"] == "2wikimultihopqa" and int(row["layer"]) == layer and int(row["chunk_size"]) == chunk
            )
            depth_row = next(
                row for row in cross_source
                if row["dataset"] == "musique_D4"
                and int(row["layer"]) == layer
                and int(row["chunk_size"]) == chunk
            )
            cross.append(
                {
                    "layer": layer,
                    "chunk_size": chunk,
                    "2wiki_edge_R6": edge["edge_R6"],
                    "2wiki_edge_MRR": edge["edge_MRR"],
                    "2wiki_strict_path_K6": edge["strict_path_K6"],
                    "musique_complete_recovery": depth_row["complete_recovery"],
                    "musique_node_recall": depth_row["node_recall"],
                    "musique_minimum_native_depth_if_complete": depth_row[
                        "minimum_native_depth_if_complete"
                    ],
                    "musique_annotated_depth": 4,
                }
            )
    return summary, correlations, cross


def _edge_decomposition(natural_dir: Path) -> list[dict]:
    rows = _read_csv(natural_dir / "transition_path_by_granularity.csv")
    result = []
    for row in rows:
        if row["dataset"] != "2wikimultihopqa":
            continue
        metrics = decompose_path_survival(
            float(row["edge_recall"]),
            float(row["product_model_path_survival"]),
            float(row["preserved_path_survival"]),
        )
        result.append(
            {
                "dataset": row["dataset"],
                "chunk_size": row["chunk_size"],
                "K": row["K"],
                "edge_R_at_K": row["edge_recall"],
                "product_expected_path_survival": row["product_model_path_survival"],
                "observed_preserved_path_survival": row["preserved_path_survival"],
                "observed_strict_path_survival": row["strict_complete_path_survival"],
                "raw_product_minus_observed": float(row["product_model_path_survival"])
                - float(row["preserved_path_survival"]),
                **metrics,
            }
        )
    return result


def _shortcut_synthesis(layer_dir: Path) -> list[dict]:
    rows = _read_csv(layer_dir / "layer_granularity_rows.csv")
    result = []
    for layer in REPRESENTATIVE_LAYERS:
        for chunk in REPRESENTATIVE_CHUNKS:
            for depth in (2, 3, 4):
                selected = [
                    row for row in rows
                    if row["dataset"] == "musique"
                    and row["partition"] == "test"
                    and int(row["layer"]) == layer
                    and int(row["chunk_size"]) == chunk
                    and int(row["annotated_hops"]) == depth
                ]
                completed = [row for row in selected if int(row["complete_recovery"]) == 1]
                mean_depth = _mean(completed, "minimum_native_depth")
                result.append(
                    {
                        "dataset": "musique",
                        "annotated_depth": depth,
                        "layer": layer,
                        "chunk_size": chunk,
                        "examples": len(selected),
                        "complete_recovery": _mean(selected, "complete_recovery"),
                        "mean_minimum_native_depth_if_complete": mean_depth,
                        "DeltaD_if_complete": depth - mean_depth if mean_depth is not None else None,
                        "interpretation": "joint representation-depth and chunk-coarse-graining condition",
                    }
                )
    baseline = {
        (int(row["annotated_depth"]), int(row["layer"]), int(row["chunk_size"])): row
        for row in result
    }
    for row in result:
        depth, layer, chunk = int(row["annotated_depth"]), int(row["layer"]), int(row["chunk_size"])
        delta = row["DeltaD_if_complete"]
        early = baseline[(depth, 0, chunk)]["DeltaD_if_complete"]
        fine = baseline[(depth, layer, 32)]["DeltaD_if_complete"]
        row["layer_effect_vs_layer0_same_chunk"] = (
            delta - early if delta is not None and early is not None else None
        )
        row["chunk_effect_vs_32_same_layer"] = (
            delta - fine if delta is not None and fine is not None else None
        )
    return result


def _aggregate_search_rows(rows: list[dict], dataset: str, *, hotpot: bool) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if (
            row["partition"] != "test"
            or row.get("H", "").lower() == "none"
            or row.get("B", "").lower() == "none"
            or row.get("K", "").lower() == "none"
            or int(row["H"]) != 4
        ):
            continue
        key = (int(row["chunk_size"]), int(row["K"]), int(row["B"]))
        grouped[key].append(row)
    result = []
    for (chunk, k, b), values in sorted(grouped.items()):
        complete_field = "complete_oracle" if hotpot else "complete_graph"
        recall_field = "oracle_recall" if hotpot else "oracle_node_recall"
        selected_field = "active_kv_fraction"
        result.append(
            {
                "dataset": dataset,
                "chunk_size": chunk,
                "root_breadth": 1 if hotpot else _mean(values, "R_root"),
                "K": k,
                "B": b,
                "H": 4,
                "selected_source_fraction": _mean(values, selected_field),
                "evidence_density": _mean(values, "evidence_density"),
                "evidence_recall": _mean(values, recall_field),
                "complete_recovery_or_path_survival": _mean(values, complete_field),
                "nodes_expanded": _mean(values, "nodes_expanded"),
                "search_latency_seconds": _mean(values, "search_seconds"),
                "adjacency_index_seconds": _mean(values, "adjacency_build_seconds"),
                "peak_cuda_allocated_bytes": _mean(values, "peak_gpu_allocated_bytes"),
                "logical_tokens": _mean(values, "logical_reference_tokens"),
                "selected_tokens": _mean(values, "selected_parent_tokens"),
                "conceptual_active_parents": _mean(values, "conceptual_active_parents"),
                "materialized_kv_tokens": _mean(values, "native_kv_tokens"),
                "counterfactual_native_kv_tokens": _mean(values, "counterfactual_native_kv_tokens"),
            }
        )
    return result


def _frontier(args) -> tuple[list[dict], list[dict]]:
    hotpot = _read_csv(args.chunk_dir / "discovery_surface_rows.csv")
    natural = _read_csv(args.natural_dir / "natural_graph_oracle_search_rows.csv")
    rows = _aggregate_search_rows(hotpot, "hotpotqa", hotpot=True)
    for dataset in ("musique", "2wikimultihopqa"):
        rows.extend(
            _aggregate_search_rows(
                [row for row in natural if row["dataset"] == dataset], dataset, hotpot=False
            )
        )
    for dataset in ("hotpotqa", "musique", "2wikimultihopqa"):
        selected = [row for row in rows if row["dataset"] == dataset]
        flags = pareto_flags(
            selected,
            maximize=("complete_recovery_or_path_survival", "evidence_recall"),
            minimize=("selected_source_fraction", "search_latency_seconds"),
        )
        for row, flag in zip(selected, flags):
            row["pareto"] = int(flag)
            row["operating_point"] = ""
        pareto = [row for row in selected if row["pareto"]]
        conservative_pool = [row for row in pareto if row["complete_recovery_or_path_survival"] >= 0.5]
        conservative = min(
            conservative_pool or pareto,
            key=lambda row: (row["selected_source_fraction"], -row["evidence_recall"]),
        )
        balanced = max(
            pareto,
            key=lambda row: (
                row["complete_recovery_or_path_survival"]
                + row["evidence_recall"]
                - row["selected_source_fraction"],
                -row["search_latency_seconds"],
            ),
        )
        high = max(
            pareto,
            key=lambda row: (
                row["complete_recovery_or_path_survival"],
                row["evidence_recall"],
                -row["selected_source_fraction"],
            ),
        )
        for label, row in (("conservative", conservative), ("balanced", balanced), ("high_recall", high)):
            row["operating_point"] = "|".join(filter(None, (row["operating_point"], label)))

    natural_systems = _read_csv(args.natural_dir / "natural_graph_system_rows.csv")
    hotpot_systems = _read_csv(args.chunk_dir / "systems_scaling_rows.csv")
    for row in rows:
        systems = (
            hotpot_systems
            if row["dataset"] == "hotpotqa"
            else [item for item in natural_systems if item["dataset"] == row["dataset"]]
        )
        matched = [
            item
            for item in systems
            if int(item["chunk_size"]) == int(row["chunk_size"])
            and (row["dataset"] != "hotpotqa" or item["partition"] == "test")
        ]
        row["peak_cuda_allocated_bytes"] = _mean(matched, "peak_gpu_allocated_bytes")
        row["peak_cuda_reserved_bytes"] = _mean(matched, "peak_gpu_reserved_bytes")
        row["h2d_bytes"] = _mean(matched, "h2d_bytes")
        row["h2d_seconds"] = _mean(matched, "h2d_seconds")
    memory = []
    for dataset in ("hotpotqa", "musique", "2wikimultihopqa"):
        source = hotpot_systems if dataset == "hotpotqa" else [row for row in natural_systems if row["dataset"] == dataset]
        for chunk in sorted({int(row["chunk_size"]) for row in source}):
            values = [row for row in source if int(row["chunk_size"]) == chunk and (dataset != "hotpotqa" or row["partition"] == "test")]
            point_values = [row for row in rows if row["dataset"] == dataset and row["chunk_size"] == chunk]
            memory.append(
                {
                    "dataset": dataset,
                    "chunk_size": chunk,
                    "logical_tokens": _mean(point_values, "logical_tokens"),
                    "selected_tokens": _mean(point_values, "selected_tokens"),
                    "selected_source_fraction": _mean(point_values, "selected_source_fraction"),
                    "conceptual_active_parents": _mean(point_values, "conceptual_active_parents"),
                    "materialized_kv_tokens": 0,
                    "materialized_kv_bytes": 0,
                    "materialization_performed": 0,
                    "counterfactual_native_kv_tokens": _mean(point_values, "counterfactual_native_kv_tokens"),
                    "peak_cuda_allocated_bytes": _mean(values, "peak_gpu_allocated_bytes"),
                    "peak_cuda_reserved_bytes": _mean(values, "peak_gpu_reserved_bytes"),
                    "routing_score_cache_bytes": _mean(values, "routing_search_cache_bytes"),
                    "h2d_bytes": _mean(values, "h2d_bytes"),
                    "h2d_seconds": _mean(values, "h2d_seconds"),
                    "adjacency_index_seconds": _mean(values, "adjacency_build_seconds"),
                    "search_latency_seconds": _mean(point_values, "search_latency_seconds"),
                    "ttft_tpot_concurrency_measured": 0,
                }
            )
    return rows, memory


def _cross_dataset(args) -> list[dict]:
    routing = {row["dataset"]: row for row in _read_csv(args.natural_dir / "routing_ceiling_table.csv")}
    central = {(row["dataset"], int(row["chunk_size"])): row for row in _read_csv(args.natural_dir / "cross_dataset_granularity.csv")}
    query = _read_csv(args.query_dir / "query_entry_summary.csv")
    qasper = next(row for row in query if row["partition"] == "test" and row["dataset"] == "qasper" and row["fraction"] == "0.2" and row["variant"] == "B_multi_span_semantic")
    hotpot_global = next(row for row in query if row["partition"] == "test" and row["dataset"] == "hotpotqa" and row["fraction"] == "0.2" and row["variant"] == "A_global_semantic")
    hotpot_chunk = next(row for row in _read_csv(args.chunk_dir / "chunk_k_table.csv") if row["chunk_size"] == "128" and row["K"] == "4")
    hotpot_density = next(row for row in _read_csv(args.chunk_dir / "evidence_density_table.csv") if row["chunk_size"] == "128")
    result = [
        {
            "dataset": "qasper",
            "role": "direct-retrieval control",
            "routed_root_R_at_4": qasper["recall_at_4"],
            "oracle_facet_R_at_4": None,
            "oracle_facet_scope": "not measured: all-offset audit not run",
            "edge_R_at_6": None,
            "complete_recovery": qasper["complete_oracle"],
            "selected_source_fraction": qasper["active_final_kv_fraction"],
            "main_limitation": "small direct-retrieval control; no annotated transition graph",
        },
        {
            "dataset": "hotpotqa",
            "role": "shallow/granularity",
            "routed_root_R_at_4": hotpot_global["recall_at_4"],
            "oracle_facet_R_at_4": None,
            "oracle_facet_scope": "not measured: all-offset audit not run",
            "edge_R_at_6": None,
            "complete_recovery": hotpot_chunk["complete_recovery"],
            "selected_source_fraction": hotpot_density["mean_active_kv_fraction"],
            "main_limitation": "coarse parents recover evidence but approach half-source activation",
        },
    ]
    for dataset, role in (("2wikimultihopqa", "annotated edge/path"), ("musique", "depth/topology/shortcut")):
        row, graph = routing[dataset], central[(dataset, 128)]
        result.append(
            {
                "dataset": dataset,
                "role": role,
                "routed_root_R_at_4": row["current_routed_R_at_4"],
                "oracle_facet_R_at_4": row["oracle_multiscale_root_R_at_4"],
                "oracle_facet_scope": "all-offset scales 1/2/4/8/16/global",
                "edge_R_at_6": graph["edge_R_at_6"],
                "complete_recovery": graph["complete_recovery"],
                "selected_source_fraction": graph["selected_source_fraction"],
                "main_limitation": (
                    "executable facet selection remains below oracle ceiling"
                    if dataset == "2wikimultihopqa"
                    else "native contraction does not identify a calibrated terminal rule"
                ),
            }
        )
    return result


def _negative_registry(output_dir: Path) -> None:
    text = """# Paper 2.5 Negative-Results Registry

This registry freezes measured boundaries; it is not a list of failed software runs.

| Finding | Canonical artifact |
|---|---|
| Iterative parent closure is not reliably above matched one-shot retrieval. | `../iterative_closure_aggregate.csv` |
| Static query reranking hurts under the matched controlled gate. | `../monotonic_adaptive_competition/heldout_effects_vs_one_shot.csv` |
| Dynamic Q/A reconstruction fails its predeclared improvement gate. | `../dynamic_query_discovery/dynamic_query_gate_results.json` |
| Terminal max-facet threshold does not calibrate robustly. | `../semantic_graph_search/goal_threshold_audit.csv` |
| Facet complementarity does not separate plausible false terminals. | `../semantic_graph_search/false_goal_review.csv` |
| Native head splitting is inconsistent across datasets. | `../query_entry_facets/head_selection_audit.csv` |
| Fine chunks reduce native annotated-edge quality despite lower payload. | `../natural_graph_depth/transition_path_by_granularity.csv` |
| Measured contextualization alone weakly predicts graph quality. | `../layerwise_graph/layerwise_correlations.csv` |
| Executable facet competition remains far below the oracle all-offset ceiling. | `../natural_graph_depth/routing_ceiling_table.csv` and `winning_facet_summary.csv` |

The final gate adds no learned production router, adaptive layer fusion, adaptive search
budget, terminal stopping rule, or materialization mechanism. Those are deferred to Papers 3
and 3.5.
"""
    (output_dir / "negative_results_registry.md").write_text(text, encoding="utf-8")


def _plots(
    output_dir: Path, winning_summary, layer_cross, edge_rows, frontier, correlations
) -> None:
    selector = [row for row in winning_summary if row["summary_type"] == "diagnostic_selector" and row["partition"] == "test"]
    methods = ["global", "highest_hidden_norm", "highest_parent_margin", "lowest_parent_entropy", "validation_most_common_scale", "validation_linear_scale"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    x = torch.arange(len(methods)).numpy()
    for offset, (dataset, label) in zip((-0.18, 0.18), (("musique", "MuSiQue"), ("2wikimultihopqa", "2Wiki"))):
        values = [next(row for row in selector if row["dataset"] == dataset and row["selector"] == method)["R_at_4"] for method in methods]
        axes[0].bar(x + offset, values, width=0.36, label=label)
    axes[0].set_xticks(x, ["global", "norm", "margin", "entropy", "common scale", "linear scale"], rotation=25, ha="right")
    axes[0].set(ylabel="Selected-facet root R@4", ylim=(0, 1.03))
    axes[0].legend(frameon=False)
    for dataset, label in (("musique", "MuSiQue"), ("2wikimultihopqa", "2Wiki")):
        rows = [row for row in frontier if row["dataset"] == dataset and row["pareto"]]
        axes[1].scatter([row["selected_source_fraction"] for row in rows], [row["complete_recovery_or_path_survival"] for row in rows], label=label, alpha=0.8)
    axes[1].set(xlabel="Selected source fraction", ylabel="Complete recovery", ylim=(-0.02, 1.03))
    axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output_dir / "facet_predictability_frontier.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    matrix = torch.tensor([[float(next(row for row in layer_cross if row["layer"] == layer and row["chunk_size"] == chunk)["2wiki_edge_R6"]) for chunk in REPRESENTATIVE_CHUNKS] for layer in REPRESENTATIVE_LAYERS])
    image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set(xticks=range(3), xticklabels=REPRESENTATIVE_CHUNKS, yticks=range(3), yticklabels=REPRESENTATIVE_LAYERS, xlabel="Chunk tokens", ylabel="Layer", title="2Wiki edge R@6")
    for i in range(3):
        for j in range(3): axes[0].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] < .65 else "black")
    figure.colorbar(image, ax=axes[0], fraction=.046)
    selected = [row for row in edge_rows if int(row["K"]) == 6]
    for chunk in REPRESENTATIVE_CHUNKS:
        row = next(item for item in selected if int(item["chunk_size"]) == chunk)
        axes[1].plot(["edge", "product", "observed"], [float(row["edge_R_at_K"]), float(row["product_expected_path_survival"]), float(row["observed_preserved_path_survival"])], marker="o", label=f"{chunk} tokens")
    axes[1].set(ylabel="2Wiki survival at K=6", ylim=(0, 1.03))
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.25)
    figure.tight_layout(); figure.savefig(output_dir / "layer_chunk_edge_decomposition.png", dpi=180); plt.close(figure)

    matched = [
        row
        for row in correlations
        if row["context_metric"] == "intervention_contextualization"
    ]
    labels = ["edge R@6", "MRR", "depth contraction", "shortcut rate", "branching"]
    graph_names = ["edge_R6", "edge_MRR", "depth_contraction", "shortcut_rate", "branching_factor"]
    ordered = [next(row for row in matched if row["graph_metric"] == name) for name in graph_names]
    centers = torch.tensor([float(row["pearson"]) for row in ordered])
    lower = centers - torch.tensor([float(row["pearson_low"]) for row in ordered])
    upper = torch.tensor([float(row["pearson_high"]) for row in ordered]) - centers
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    axis.errorbar(
        range(len(labels)), centers.numpy(), yerr=torch.stack((lower, upper)).numpy(),
        fmt="o", capsize=4, color="#176b87",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xticks=range(len(labels)), xticklabels=labels,
        ylabel="Pearson correlation across layers", ylim=(-1.05, 1.05),
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "layer_context_matched_correlations.png", dpi=180)
    plt.close(figure)


def run(args) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    winning, winning_summary, selector_rows, selector_model = _winning_facets(args, output_dir)
    role_visibility, hop_visibility = _visibility(winning)
    layer_summary, correlations, layer_cross = _layer_synthesis(args.layer_dir)
    edge_rows = _edge_decomposition(args.natural_dir)
    shortcut_rows = _shortcut_synthesis(args.layer_dir)
    frontier, memory = _frontier(args)
    cross_dataset = _cross_dataset(args)
    artifacts = (
        ("winning_facet_rows.csv", winning),
        ("winning_facet_summary.csv", winning_summary),
        ("root_terminal_visibility.csv", role_visibility),
        ("per_hop_facet_visibility.csv", hop_visibility),
        ("layer_graph_context_summary.csv", layer_summary),
        ("layer_context_correlations.csv", correlations),
        ("layer_chunk_interaction.csv", layer_cross),
        ("edge_search_decomposition.csv", edge_rows),
        ("musique_shortcut_synthesis.csv", shortcut_rows),
        ("sparse_quality_frontier.csv", frontier),
        ("memory_serving_audit.csv", memory),
        ("cross_dataset_summary.csv", cross_dataset),
    )
    for name, rows in artifacts:
        _write_csv(output_dir / name, rows)
    _negative_registry(output_dir)
    _plots(output_dir, winning_summary, layer_cross, edge_rows, frontier, correlations)
    heldout_selectors = [row for row in winning_summary if row["summary_type"] == "diagnostic_selector" and row["partition"] == "test"]
    result = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "measurement_only": True,
        "training_performed": False,
        "new_retrieval_or_search_architecture": False,
        "oracle_labels_available_to_facet_selectors": False,
        "oracle_labels_available_during_graph_search": False,
        "oracle_evaluation_post_hoc": True,
        "selector_model": selector_model,
        "heldout_selector_summary": heldout_selectors,
        "row_counts": {name: len(rows) for name, rows in artifacts},
        "experiment_freeze_recommendation": (
            "Freeze Paper 2.5: query views contain a high oracle ceiling, but inexpensive "
            "selectors do not close it; graph quality is non-monotonic in layer and interacts "
            "with chunk size. Defer learned routing, adaptive topology, and serving materialization."
        ),
    }
    (output_dir / "final_metrics_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    shared = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument("--natural-dir", type=Path, default=shared / "natural_graph_depth")
    parser.add_argument("--layer-dir", type=Path, default=shared / "layerwise_graph")
    parser.add_argument("--chunk-dir", type=Path, default=shared / "chunk_granularity")
    parser.add_argument("--query-dir", type=Path, default=shared / "query_entry_facets")
    parser.add_argument("--natural-feature-file", type=Path, default=shared / "natural_graph_depth/natural_graph_features.pt")
    parser.add_argument("--facet-cache", type=Path, default=shared / "natural_graph_depth/natural_multiscale_query_facet_cache.pt")
    parser.add_argument("--output-dir", type=Path, default=shared / "final_metrics")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
