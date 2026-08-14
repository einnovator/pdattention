"""Evaluate contextual query facets and native head-split root discovery.

Variant A exactly reproduces the learned final-token semantic root selector.
Variant B changes only its query-side representation. Variants C/D use real
layer-27 pre-RoPE Q/K heads; native mean-head controls isolate head reduction
from that representation change. All variants share one final parent Top-B.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    oracle_set_metrics,
    validation_partition,
)
from pra_hf.query_facets import (
    FacetScoreResult,
    QueryFacetSet,
    build_contextual_query_facets,
    global_query_facet,
    pool_parent_native_keys,
    score_native_query_facets,
    score_semantic_query_facets,
    select_bounded_parents,
    target_rank_metrics,
)
from pra_torch.hf import load_hf_routing_projection


FRACTIONS = (0.10, 0.20, 0.30, 0.40)
PRIMARY_FRACTION = 0.20


@dataclass(frozen=True)
class QueryConfig:
    """One contextual query-set definition selected without test labels."""

    name: str
    window: int | None
    stride: int | None
    facet_reduction: str


@dataclass(frozen=True)
class HeadConfig:
    """One native-head reduction with optional independent nominations."""

    name: str
    head_reduction: str
    top_m: int = 4
    nomination_k: int = 0


GLOBAL = QueryConfig("global", None, None, "max")
QUERY_CONFIGS = tuple(
    QueryConfig(
        f"w{window}_s{max(1, window // 2)}_{reduction}",
        window,
        max(1, window // 2),
        reduction,
    )
    for window in (4, 8, 16)
    for reduction in ("max", "top_m_mean")
)
HEAD_CONFIGS = (
    HeadConfig("mean", "mean"),
    HeadConfig("max", "max"),
    HeadConfig("top4_mean", "top_m_mean", top_m=4),
    HeadConfig("per_head_top1", "max", nomination_k=1),
    HeadConfig("per_head_top2", "max", nomination_k=2),
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _query_facets(query_feature: dict, config: QueryConfig) -> QueryFacetSet:
    hidden = query_feature["query_hidden_states"].float()
    native = query_feature["query_pre_query"].float()
    if config.window is None:
        return global_query_facet(hidden, native)
    return build_contextual_query_facets(
        hidden,
        tuple(query_feature["question_span"]),
        window=config.window,
        stride=int(config.stride),
        native_query=native,
    )


def _variant_name(
    representation: str,
    query_config: QueryConfig,
    head_config: HeadConfig | None,
) -> str:
    if representation == "semantic":
        return f"semantic_{query_config.name}"
    if head_config is None:
        raise ValueError("Native variants require a head configuration.")
    return f"native_{query_config.name}_{head_config.name}"


def _decode_span(tokenizer, query_feature: dict, start: int, end: int) -> str:
    ids = query_feature["prompt_input_ids"][start:end]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _rows_for_score(
    feature: dict,
    query_feature: dict,
    tokenizer,
    facets: QueryFacetSet,
    result: FacetScoreResult,
    *,
    variant: str,
    representation: str,
    query_config: QueryConfig,
    head_config: HeadConfig | None,
    seed: int | None,
    fractions: tuple[float, ...],
    scoring_seconds: float,
) -> list[dict]:
    groups = evidence_parent_groups(feature)
    if not groups:
        raise ValueError(f"No ordered evidence group for {feature['example_id']}")
    first_group = groups[0]
    oracle = canonical_oracle_parent_indices(feature)
    rows = []
    for fraction in fractions:
        budget = max(1, math.ceil(len(feature["parent_spans"]) * fraction))
        started = time.perf_counter()
        selection = select_bounded_parents(
            result,
            budget,
            per_head_nomination_k=(head_config.nomination_k if head_config else 0),
        )
        selected = set(selection.parent_indices)
        rank = target_rank_metrics(result.scores, first_group, selected)
        oracle_metrics = oracle_set_metrics(selected, oracle)
        target = int(rank["target_parent"])
        winning_facet = int(result.winning_facet[target])
        winning_head = int(result.winning_head[target])
        provenance = facets.provenance[winning_facet]
        selected_tokens = sum(
            int(feature["parent_spans"][parent][1])
            - int(feature["parent_spans"][parent][0])
            for parent in selected
        )
        row = {
            "example_id": feature["example_id"],
            "dataset": feature["dataset"],
            "partition": validation_partition(feature["example_id"]),
            "seed": seed,
            "fraction": fraction,
            "budget": budget,
            "variant": variant,
            "representation": representation,
            "query_window_size": query_config.window,
            "query_stride": query_config.stride,
            "query_facet_reduction": query_config.facet_reduction,
            "query_facet_count": len(facets.provenance),
            "head_mode": head_config.name if head_config else "not_applicable",
            "active_head_count": int(result.component_scores.shape[1]),
            "oracle_root_parent": target,
            "oracle_root_rank": int(rank["target_rank"]),
            "oracle_root_present": float(rank["target_present"]),
            "mrr": float(rank["mrr"]),
            "recall_at_1": float(rank["recall_at_1"]),
            "recall_at_2": float(rank["recall_at_2"]),
            "recall_at_4": float(rank["recall_at_4"]),
            "recall_at_8": float(rank["recall_at_8"]),
            "winning_query_facet": winning_facet,
            "winning_query_span_kind": provenance.kind,
            "winning_query_span": json.dumps(
                [provenance.token_start, provenance.token_end]
            ),
            "winning_query_span_text": _decode_span(
                tokenizer,
                query_feature,
                provenance.token_start,
                provenance.token_end,
            ),
            "winning_head": winning_head if winning_head >= 0 else None,
            "oracle_score": float(rank["target_score"]),
            "best_distractor_parent": rank["best_distractor_parent"],
            "best_distractor_score": float(rank["best_distractor_score"]),
            "oracle_margin": float(rank["oracle_margin"]),
            "score_entropy": float(rank["score_entropy"]),
            "false_positive_parent_count": int(len(selected - oracle)),
            "selected_parent_ids": json.dumps(selection.parent_indices),
            "nominated_parent_ids": json.dumps(selection.nominated_parent_indices),
            "oracle_recall": oracle_metrics["oracle_recall"],
            "oracle_precision": oracle_metrics["oracle_precision"],
            "oracle_jaccard": oracle_metrics["oracle_jaccard"],
            "complete_oracle": oracle_metrics["complete_oracle"],
            "search_comparisons": result.comparisons,
            "raw_span_head_comparisons": result.comparisons,
            "candidate_parent_comparisons": (
                result.component_scores.shape[0] * result.component_scores.shape[2]
            ),
            "deduplicated_candidates": selection.deduplicated_candidates,
            "final_parent_budget": selection.final_budget,
            "active_final_kv_tokens": selected_tokens,
            "active_final_kv_fraction": selected_tokens
            / max(int(feature["source_tokens"]), 1),
            "scoring_wall_time": scoring_seconds,
            "selection_wall_time": time.perf_counter() - started,
            "wall_time": scoring_seconds + time.perf_counter() - started,
            "full_query_forward_count": query_feature["full_query_forward_count"],
            "independent_window_forward_count": query_feature[
                "independent_window_forward_count"
            ],
        }
        if len(selection.parent_indices) != selection.final_budget:
            raise AssertionError("Facet/head search inflated or under-filled Top-B.")
        rows.append(row)
    return rows


def _metric_mean(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return statistics.fmean(values) if values else None


SUMMARY_METRICS = (
    "oracle_root_present",
    "oracle_root_rank",
    "mrr",
    "recall_at_1",
    "recall_at_2",
    "recall_at_4",
    "recall_at_8",
    "oracle_margin",
    "score_entropy",
    "false_positive_parent_count",
    "oracle_recall",
    "oracle_precision",
    "oracle_jaccard",
    "complete_oracle",
    "query_facet_count",
    "active_head_count",
    "search_comparisons",
    "candidate_parent_comparisons",
    "deduplicated_candidates",
    "final_parent_budget",
    "active_final_kv_tokens",
    "active_final_kv_fraction",
    "scoring_wall_time",
    "selection_wall_time",
    "wall_time",
)


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["partition"],
                row["dataset"],
                float(row["fraction"]),
                row["variant"],
                row["representation"],
                row["query_window_size"],
                row["query_stride"],
                row["query_facet_reduction"],
                row["head_mode"],
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        record = {
            "partition": key[0],
            "dataset": key[1],
            "fraction": key[2],
            "variant": key[3],
            "representation": key[4],
            "query_window_size": key[5],
            "query_stride": key[6],
            "query_facet_reduction": key[7],
            "head_mode": key[8],
            "rows": len(values),
            "identities": len({row["example_id"] for row in values}),
            "seeds": len({row["seed"] for row in values if row["seed"] is not None}),
        }
        for metric in SUMMARY_METRICS:
            record[metric] = _metric_mean(values, metric)
        output.append(record)
    return output


def _summary_lookup(
    summary: list[dict], partition: str, dataset: str, variant: str
) -> dict:
    return next(
        row
        for row in summary
        if row["partition"] == partition
        and row["dataset"] == dataset
        and row["fraction"] == PRIMARY_FRACTION
        and row["variant"] == variant
    )


def _choose_query_config(summary: list[dict]) -> tuple[QueryConfig, list[dict]]:
    baseline = _summary_lookup(summary, "validation", "qasper", "semantic_global")
    audit = []
    for config in QUERY_CONFIGS:
        variant = f"semantic_{config.name}"
        hotpot = _summary_lookup(summary, "validation", "hotpotqa", variant)
        qasper = _summary_lookup(summary, "validation", "qasper", variant)
        preserved = qasper["oracle_root_present"] >= baseline["oracle_root_present"] - 0.02
        audit.append(
            {
                "candidate": config.name,
                "hotpot_root_present": hotpot["oracle_root_present"],
                "hotpot_mrr": hotpot["mrr"],
                "qasper_root_present": qasper["oracle_root_present"],
                "qasper_preserved": preserved,
                "search_comparisons": hotpot["search_comparisons"],
            }
        )
    eligible = [row for row in audit if row["qasper_preserved"]] or audit
    chosen = max(
        eligible,
        key=lambda row: (
            row["hotpot_root_present"],
            row["hotpot_mrr"],
            row["qasper_root_present"],
            -row["search_comparisons"],
            row["candidate"],
        ),
    )
    return next(config for config in QUERY_CONFIGS if config.name == chosen["candidate"]), audit


def _choose_head_config(summary: list[dict]) -> tuple[HeadConfig, list[dict]]:
    baseline = _summary_lookup(summary, "validation", "qasper", "native_global_mean")
    audit = []
    for config in HEAD_CONFIGS:
        if config.name == "mean":
            continue
        variant = f"native_global_{config.name}"
        hotpot = _summary_lookup(summary, "validation", "hotpotqa", variant)
        qasper = _summary_lookup(summary, "validation", "qasper", variant)
        preserved = qasper["oracle_root_present"] >= baseline["oracle_root_present"] - 0.02
        audit.append(
            {
                "candidate": config.name,
                "hotpot_root_present": hotpot["oracle_root_present"],
                "hotpot_mrr": hotpot["mrr"],
                "qasper_root_present": qasper["oracle_root_present"],
                "qasper_preserved": preserved,
                "search_comparisons": hotpot["search_comparisons"],
                "deduplicated_candidates": hotpot["deduplicated_candidates"],
            }
        )
    eligible = [row for row in audit if row["qasper_preserved"]] or audit
    chosen = max(
        eligible,
        key=lambda row: (
            row["hotpot_root_present"],
            row["hotpot_mrr"],
            row["qasper_root_present"],
            -row["search_comparisons"],
            row["candidate"],
        ),
    )
    return next(config for config in HEAD_CONFIGS if config.name == chosen["candidate"]), audit


def _rename_row(row: dict, variant: str) -> dict:
    return {**row, "variant": variant}


def _diagnostics(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    primary = [
        row
        for row in rows
        if row["partition"] == "test" and row["fraction"] == PRIMARY_FRACTION
    ]
    lookup = {
        (row["dataset"], row["example_id"], row["seed"], row["variant"]): row
        for row in primary
    }
    facets, heads, interactions = [], [], []
    for key, baseline in lookup.items():
        dataset, example_id, seed, variant = key
        if variant == "A_global_semantic":
            multi = lookup.get((dataset, example_id, seed, "B_multi_span_semantic"))
            if multi and not baseline["oracle_root_present"] and multi["oracle_root_present"]:
                facets.append(
                    {
                        "dataset": dataset,
                        "example_id": example_id,
                        "seed": seed,
                        "winning_query_span": multi["winning_query_span"],
                        "winning_query_span_text": multi["winning_query_span_text"],
                        "oracle_root_parent": multi["oracle_root_parent"],
                        "global_query_rank": baseline["oracle_root_rank"],
                        "facet_query_rank": multi["oracle_root_rank"],
                        "facet_score": multi["oracle_score"],
                        "best_distractor_score": multi["best_distractor_score"],
                    }
                )
        if variant == "E_global_native_mean":
            split = lookup.get((dataset, example_id, seed, "C_global_split_head"))
            if split and not baseline["oracle_root_present"] and split["oracle_root_present"]:
                heads.append(
                    {
                        "dataset": dataset,
                        "example_id": example_id,
                        "query_mode": "global",
                        "winning_head": split["winning_head"],
                        "oracle_root_parent": split["oracle_root_parent"],
                        "head_specific_oracle_rank": split["oracle_root_rank"],
                        "aggregated_oracle_rank": baseline["oracle_root_rank"],
                        "best_distractor_parent": split["best_distractor_parent"],
                        "score_margin": split["oracle_margin"],
                    }
                )
    identity_keys = {
        (row["dataset"], row["example_id"])
        for row in primary
        if row["variant"] == "E_global_native_mean"
    }
    for dataset, example_id in sorted(identity_keys):
        values = {
            variant: lookup[(dataset, example_id, None, variant)]["oracle_root_present"]
            for variant in (
                "E_global_native_mean",
                "F_multi_span_native_mean",
                "C_global_split_head",
                "D_multi_span_split_head",
            )
        }
        baseline = bool(values["E_global_native_mean"])
        span = bool(values["F_multi_span_native_mean"])
        head = bool(values["C_global_split_head"])
        combined = bool(values["D_multi_span_split_head"])
        category = (
            "synergistic_span_head_gain"
            if combined and not span and not head and not baseline
            else "span_only_gain"
            if span and not head and not baseline
            else "head_only_gain"
            if head and not span and not baseline
            else "no_unique_gain"
        )
        interactions.append(
            {
                "dataset": dataset,
                "example_id": example_id,
                "interaction_class": category,
                **values,
            }
        )
    return facets, heads, interactions


def _synthetic_controls() -> list[dict]:
    controls = []
    semantic_query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    semantic_memory = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    global_semantic = score_semantic_query_facets(
        semantic_query[:1], semantic_memory
    )
    facet_semantic = score_semantic_query_facets(semantic_query, semantic_memory)
    controls.append(
        {
            "control": "span_dilution",
            "global_selected": select_bounded_parents(
                global_semantic, 1
            ).parent_indices[0],
            "faceted_selected": select_bounded_parents(
                facet_semantic, 1
            ).parent_indices[0],
            "passed": select_bounded_parents(global_semantic, 1).parent_indices[0]
            == 1
            and select_bounded_parents(facet_semantic, 1).parent_indices[0] == 0,
        }
    )

    query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    keys = torch.tensor(
        [
            [[10.0, 0.0], [0.0, 0.0]],
            [[6.0, 0.0], [6.0, 0.0]],
        ]
    )
    mean = score_native_query_facets(query, keys, head_reduction="mean")
    split = score_native_query_facets(query, keys, head_reduction="max")
    controls.append(
        {
            "control": "head_specialization",
            "aggregated_selected": select_bounded_parents(mean, 1).parent_indices[0],
            "split_selected": select_bounded_parents(split, 1).parent_indices[0],
            "passed": select_bounded_parents(mean, 1).parent_indices[0] == 1
            and select_bounded_parents(split, 1).parent_indices[0] == 0,
        }
    )

    combined_query = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    global_mean = score_native_query_facets(
        combined_query[:1], keys, head_reduction="mean"
    )
    facet_mean = score_native_query_facets(
        combined_query,
        torch.tensor(
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[6.0, 0.0], [6.0, 0.0]],
            ]
        ),
        head_reduction="mean",
    )
    global_split = score_native_query_facets(
        combined_query[:1],
        torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[6.0, 0.0], [6.0, 0.0]],
            ]
        ),
        head_reduction="max",
    )
    combined_split = score_native_query_facets(
        combined_query,
        torch.tensor(
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[6.0, 0.0], [6.0, 0.0]],
            ]
        ),
        head_reduction="max",
    )
    selections = [
        select_bounded_parents(value, 1).parent_indices[0]
        for value in (global_mean, facet_mean, global_split, combined_split)
    ]
    controls.append(
        {
            "control": "combined_span_head",
            "global_aggregated_selected": selections[0],
            "facet_aggregated_selected": selections[1],
            "global_split_selected": selections[2],
            "combined_selected": selections[3],
            "passed": selections == [1, 1, 1, 0],
        }
    )
    return controls


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: list[dict], output_dir: Path) -> None:
    test = [row for row in summary if row["partition"] == "test"]
    variants = (
        "A_global_semantic",
        "B_multi_span_semantic",
        "C_global_split_head",
        "D_multi_span_split_head",
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    for dataset in ("hotpotqa", "qasper"):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for variant, color in zip(variants, colors):
            values = sorted(
                (
                    row["fraction"],
                    row["oracle_root_present"],
                )
                for row in test
                if row["dataset"] == dataset and row["variant"] == variant
            )
            axis.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                marker="o",
                label=variant.split("_", 1)[0],
                color=color,
            )
        axis.set_xlabel("Final parent fraction")
        axis.set_ylabel("First evidence group in root Top-B")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(ncol=4)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{dataset}_root_entry.{suffix}", dpi=180)
        plt.close(figure)

    primary = [row for row in test if row["fraction"] == PRIMARY_FRACTION]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = list(range(len(variants)))
    width = 0.36
    for offset, dataset, color in (
        (-width / 2, "hotpotqa", "#4c78a8"),
        (width / 2, "qasper", "#f58518"),
    ):
        lookup = {
            row["variant"]: row
            for row in primary
            if row["dataset"] == dataset
        }
        axes[0].bar(
            [value + offset for value in x],
            [lookup[variant]["oracle_root_present"] for variant in variants],
            width,
            label=dataset,
            color=color,
        )
        axes[1].bar(
            [value + offset for value in x],
            [lookup[variant]["search_comparisons"] for variant in variants],
            width,
            label=dataset,
            color=color,
        )
    for axis, label in zip(
        axes, ("Root entry probability", "Raw query-parent/head comparisons")
    ):
        axis.set_xticks(x, ("A", "B", "C", "D"))
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"primary_quality_cost.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    source_features = torch.load(
        args.source_feature_file, map_location="cpu", weights_only=False
    )
    query_features = torch.load(
        args.query_feature_file, map_location="cpu", weights_only=False
    )
    query_by_id = {row["example_id"]: row for row in query_features}
    if [row["example_id"] for row in source_features] != [
        row["example_id"] for row in query_features
    ]:
        raise ValueError("Source and query feature identities/order do not match.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    projections = {}
    for seed in args.seeds:
        checkpoint = args.projection_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projections[seed] = load_hf_routing_projection(checkpoint, device=device)

    rows = []
    for feature_index, feature in enumerate(source_features, start=1):
        query_feature = query_by_id[feature["example_id"]]
        baseline_error = float(
            (
                query_feature["query_hidden_states"][-1].float()
                - feature["query_hidden"].float()
            )
            .abs()
            .max()
        )
        if baseline_error != 0.0:
            raise AssertionError("Variant A query differs from the frozen root cache.")
        facet_sets = {
            config.name: _query_facets(query_feature, config)
            for config in (GLOBAL, *QUERY_CONFIGS)
        }
        parent_keys = pool_parent_native_keys(
            feature["local_pre_key"],
            feature["local_token_mask"],
            feature["local_parent_indices"],
            len(feature["parent_spans"]),
        ).to(device)

        for seed, projection in projections.items():
            parent_memory = projection.project_memory(
                feature["parent_hidden"].to(device)
            )
            for config in (GLOBAL, *QUERY_CONFIGS):
                facets = facet_sets[config.name]
                _synchronize(device)
                started = time.perf_counter()
                projected_query = projection.project_query(facets.hidden.to(device))
                result = score_semantic_query_facets(
                    projected_query,
                    parent_memory,
                    facet_reduction=config.facet_reduction,
                    top_m=2,
                )
                _synchronize(device)
                scoring_seconds = time.perf_counter() - started
                rows.extend(
                    _rows_for_score(
                        feature,
                        query_feature,
                        tokenizer,
                        facets,
                        result,
                        variant=_variant_name("semantic", config, None),
                        representation="learned_semantic_128d",
                        query_config=config,
                        head_config=None,
                        seed=seed,
                        fractions=args.fractions,
                        scoring_seconds=scoring_seconds,
                    )
                )

        for config in (GLOBAL, *QUERY_CONFIGS):
            facets = facet_sets[config.name]
            for head in HEAD_CONFIGS:
                _synchronize(device)
                started = time.perf_counter()
                result = score_native_query_facets(
                    facets.native_query.to(device),
                    parent_keys,
                    facet_reduction=config.facet_reduction,
                    head_reduction=head.head_reduction,
                    top_m=head.top_m,
                )
                _synchronize(device)
                scoring_seconds = time.perf_counter() - started
                rows.extend(
                    _rows_for_score(
                        feature,
                        query_feature,
                        tokenizer,
                        facets,
                        result,
                        variant=_variant_name("native", config, head),
                        representation="layer27_pre_rope_native_qk",
                        query_config=config,
                        head_config=head,
                        seed=None,
                        fractions=args.fractions,
                        scoring_seconds=scoring_seconds,
                    )
                )
        del parent_keys
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[query-entry-score {feature_index}/{len(source_features)}] "
            f"{feature['dataset']} {feature['example_id']}",
            flush=True,
        )

    sweep_summary = _aggregate(rows)
    selected_query, query_audit = _choose_query_config(sweep_summary)
    selected_head, head_audit = _choose_head_config(sweep_summary)
    source_variants = {
        "A_global_semantic": "semantic_global",
        "B_multi_span_semantic": f"semantic_{selected_query.name}",
        "E_global_native_mean": "native_global_mean",
        "F_multi_span_native_mean": f"native_{selected_query.name}_mean",
        "C_global_split_head": f"native_global_{selected_head.name}",
        "D_multi_span_split_head": (
            f"native_{selected_query.name}_{selected_head.name}"
        ),
    }
    final_rows = [
        _rename_row(row, final_name)
        for final_name, source_name in source_variants.items()
        for row in rows
        if row["variant"] == source_name
    ]
    final_summary = _aggregate(final_rows)
    facet_diagnostics, head_diagnostics, interaction = _diagnostics(final_rows)
    synthetic = _synthetic_controls()
    if not all(row["passed"] for row in synthetic):
        raise AssertionError("One or more synthetic query/head controls failed.")

    baseline_all_fraction = [
        row
        for row in final_rows
        if row["partition"] == "test"
        and row["dataset"] == "hotpotqa"
        and row["variant"] == "A_global_semantic"
    ]
    reproduced_first_root = statistics.fmean(
        float(row["oracle_root_present"]) for row in baseline_all_fraction
    )
    prior = json.loads(args.prior_result_file.read_text(encoding="utf-8"))
    canonical_first_root = 1.0 - float(
        prior["hotpot_failure_decomposition"]["first_root_absent_rate"]
    )
    if not math.isclose(reproduced_first_root, canonical_first_root, abs_tol=1e-12):
        raise AssertionError(
            "Variant A did not reproduce the canonical all-fraction Hotpot root rate."
        )

    primary = [
        row
        for row in final_summary
        if row["partition"] == "test" and row["fraction"] == PRIMARY_FRACTION
    ]
    baseline_by_dataset = {
        dataset: next(
            row
            for row in primary
            if row["dataset"] == dataset and row["variant"] == "A_global_semantic"
        )
        for dataset in ("hotpotqa", "qasper")
    }
    for row in primary:
        baseline = baseline_by_dataset[row["dataset"]]
        delta_recall = row["oracle_root_present"] - baseline["oracle_root_present"]
        delta_cost = row["search_comparisons"] - baseline["search_comparisons"]
        row["root_recall_gain_per_extra_comparison"] = (
            delta_recall / delta_cost if delta_cost > 0 else None
        )

    candidates = [
        row
        for row in primary
        if row["dataset"] == "hotpotqa"
        and row["variant"]
        in {
            "B_multi_span_semantic",
            "C_global_split_head",
            "D_multi_span_split_head",
        }
    ]
    best = max(candidates, key=lambda row: (row["oracle_root_present"], row["mrr"]))
    qasper_best = next(
        row
        for row in primary
        if row["dataset"] == "qasper" and row["variant"] == best["variant"]
    )
    hotpot_gain = (
        best["oracle_root_present"]
        - baseline_by_dataset["hotpotqa"]["oracle_root_present"]
    )
    qasper_loss = (
        baseline_by_dataset["qasper"]["oracle_root_present"]
        - qasper_best["oracle_root_present"]
    )
    materially_improved = hotpot_gain >= args.material_gain and qasper_loss <= 0.05
    recommendation = (
        "A_use_multi_span_query_facets"
        if best["variant"] == "B_multi_span_semantic" and materially_improved
        else "B_use_head_split_query_search"
        if best["variant"] == "C_global_split_head" and materially_improved
        else "C_use_combined_span_head_search"
        if best["variant"] == "D_multi_span_split_head" and materially_improved
        else "D_make_faceted_search_adaptive"
        if hotpot_gain >= args.material_gain and qasper_loss > 0.05
        else "E_move_to_learned_or_explicit_query_decomposition"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_default_changed": False,
        "training_performed": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "primary_fraction": PRIMARY_FRACTION,
        "memory_representation_frozen": True,
        "source_feature": str(args.source_feature_file.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "query_feature": str(args.query_feature_file.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "query_window_ladder": [asdict(config) for config in QUERY_CONFIGS],
        "head_policy_ladder": [asdict(config) for config in HEAD_CONFIGS],
        "selected_query_config": asdict(selected_query),
        "selected_head_config": asdict(selected_head),
        "query_selection_audit": query_audit,
        "head_selection_audit": head_audit,
        "variant_sources": source_variants,
        "canonical_baseline_first_root_all_fraction": canonical_first_root,
        "reproduced_baseline_first_root_all_fraction": reproduced_first_root,
        "primary_heldout_summary": primary,
        "synthetic_controls": synthetic,
        "facet_success_count": len(facet_diagnostics),
        "head_specialization_success_count": len(head_diagnostics),
        "interaction_counts": {
            label: sum(row["interaction_class"] == label for row in interaction)
            for label in (
                "span_only_gain",
                "head_only_gain",
                "synergistic_span_head_gain",
                "no_unique_gain",
            )
        },
        "best_root_variant": best["variant"],
        "best_hotpot_root_gain": hotpot_gain,
        "best_qasper_root_loss": qasper_loss,
        "material_root_improvement": materially_improved,
        "propagation_confirmation_required": materially_improved,
        "propagation_confirmation_run": False,
        "recommendation": recommendation,
        "sdk_change_in_this_iteration": False,
    }
    (args.output_dir / "query_entry_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "query_entry_rows.csv", final_rows)
    _write_csv(args.output_dir / "query_entry_summary.csv", final_summary)
    _write_csv(args.output_dir / "query_selection_audit.csv", query_audit)
    _write_csv(args.output_dir / "head_selection_audit.csv", head_audit)
    _write_csv(args.output_dir / "query_facet_successes.csv", facet_diagnostics)
    _write_csv(args.output_dir / "head_specialization_successes.csv", head_diagnostics)
    _write_csv(args.output_dir / "span_head_interactions.csv", interaction)
    _write_csv(args.output_dir / "synthetic_controls.csv", synthetic)
    _plot(final_summary, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--material-gain", type=float, default=0.10)
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/"
        "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--query-feature-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/"
        "query_entry_facets/query_entry_features.pt",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--prior-result-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/"
        "monotonic_adaptive_competition/adaptive_competition_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/query_entry_facets",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    args.fractions = tuple(map(float, args.fractions.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "best_root_variant": result["best_root_variant"],
                "material_root_improvement": result["material_root_improvement"],
                "recommendation": result["recommendation"],
            },
            indent=2,
        )
    )
