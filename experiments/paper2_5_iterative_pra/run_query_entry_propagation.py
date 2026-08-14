"""Reconnect the selected query-facet root to frozen bounded propagation.

This confirmation is intentionally narrow.  It reuses the validation-selected
4-token contextual query facets and the previously frozen native-rank monotonic
policy at the primary 20% parent budget.  The model, memory gists, transition
geometry, propagation breadth, payload, and final materialization budget remain
unchanged.
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
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_5_iterative_pra.run_monotonic_adaptive_competition import (
    FrozenTrace,
    _transition_for_source,
    evaluate_policy,
)
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    oracle_set_metrics,
    validation_partition,
)
from pra_hf.adaptive_competition import (
    RootLockConfig,
    TransitionPolicyConfig,
    deterministic_topk,
)
from pra_hf.query_facets import (
    build_contextual_query_facets,
    score_semantic_query_facets,
)
from pra_torch.hf import load_hf_routing_projection


PRIMARY_FRACTION = 0.20
ROOT_POLICY_NAME = "zscore_1"
ROOT_POLICY = RootLockConfig("zscore", threshold=1.0)
TRANSITION_POLICY_NAME = "fixed_k1"
TRANSITION_POLICY = TransitionPolicyConfig("fixed", fixed_k=1)
TRANSITION_GEOMETRY = "native_rank"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_trace(
    feature: dict,
    query_feature: dict,
    projection,
    *,
    device: torch.device,
    candidate_fraction: float,
) -> tuple[FrozenTrace, int, int, float]:
    """Build only transitions reachable from the selected facet root Top-B."""
    facets = build_contextual_query_facets(
        query_feature["query_hidden_states"].float(),
        tuple(query_feature["question_span"]),
        window=4,
        stride=2,
        native_query=query_feature["query_pre_query"].float(),
    )
    with torch.no_grad():
        projected_facets = projection.project_query(facets.hidden.to(device))
        parent_hidden = feature["parent_hidden"].to(device)
        local_hidden = feature["local_hidden"].to(device)
        parent_memory = F.normalize(
            projection.project_memory(parent_hidden).float(), dim=-1
        )
        local_memory = F.normalize(
            projection.project_memory(local_hidden).float(), dim=-1
        )
        local_query = F.normalize(
            projection.project_query(local_hidden).float(), dim=-1
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        root_started = time.perf_counter()
        root_result = score_semantic_query_facets(
            projected_facets,
            parent_memory,
            facet_reduction="max",
            top_m=2,
        )
        local_result = score_semantic_query_facets(
            projected_facets,
            local_memory,
            facet_reduction="max",
            top_m=2,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        root_scoring_seconds = time.perf_counter() - root_started
        budget = max(1, math.ceil(len(feature["parent_spans"]) * PRIMARY_FRACTION))
        sources = deterministic_topk(root_result.scores, budget)
        transitions = {}
        transition_traces = {}
        for source in sources:
            scores, trace = _transition_for_source(
                feature,
                source,
                local_result.scores,
                local_memory,
                local_query,
                device=device,
                candidate_fraction=candidate_fraction,
            )
            transitions[source] = scores
            transition_traces[source] = trace
    return (
        FrozenTrace(root_result.scores.detach().cpu(), transitions, transition_traces),
        int(root_result.comparisons),
        len(facets.provenance),
        root_scoring_seconds,
    )


def _root_only_row(
    feature: dict,
    root_scores: torch.Tensor,
    seed: int,
    *,
    root_comparisons: int,
    facet_count: int,
    root_scoring_seconds: float,
) -> dict:
    budget = max(1, math.ceil(len(feature["parent_spans"]) * PRIMARY_FRACTION))
    selected = set(deterministic_topk(root_scores, budget))
    groups = evidence_parent_groups(feature)
    oracle = canonical_oracle_parent_indices(feature)
    metrics = oracle_set_metrics(selected, oracle)
    selected_tokens = sum(
        int(feature["parent_spans"][parent][1])
        - int(feature["parent_spans"][parent][0])
        for parent in selected
    )
    return {
        "method": "new_query_facets_root_only",
        "partition": validation_partition(feature["example_id"]),
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "fraction": PRIMARY_FRACTION,
        "budget": budget,
        "first_oracle_group_present": float(bool(groups and selected & groups[0])),
        "chain_complete": float(
            bool(groups) and all(bool(selected & group) for group in groups)
        ),
        **metrics,
        "selected_parent_ids": json.dumps(sorted(selected)),
        "query_facet_count": facet_count,
        "query_facet_comparisons": root_comparisons,
        "total_routing_comparisons": root_comparisons,
        "active_final_kv_tokens": selected_tokens,
        "active_final_kv_fraction": selected_tokens
        / max(int(feature["source_tokens"]), 1),
        "propagation_activation": 0.0,
        "wall_time": root_scoring_seconds,
        "estimated_full_routing_seconds": root_scoring_seconds,
    }


def _confirmation_rows(
    source_features: list[dict],
    query_features: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    device = torch.device(args.device)
    query_by_id = {row["example_id"]: row for row in query_features}
    root_rows: list[dict] = []
    propagated_rows: list[dict] = []
    for seed in args.seeds:
        checkpoint = args.projection_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in source_features:
            trace, root_comparisons, facet_count, root_scoring_seconds = _build_trace(
                feature,
                query_by_id[feature["example_id"]],
                projection,
                device=device,
                candidate_fraction=args.candidate_fraction,
            )
            root_rows.append(
                _root_only_row(
                    feature,
                    trace.root_scores,
                    seed,
                    root_comparisons=root_comparisons,
                    facet_count=facet_count,
                    root_scoring_seconds=root_scoring_seconds,
                )
            )
            row = evaluate_policy(
                feature,
                seed,
                PRIMARY_FRACTION,
                trace,
                root_policy_name=ROOT_POLICY_NAME,
                root_policy=ROOT_POLICY,
                transition_policy_name=TRANSITION_POLICY_NAME,
                transition_policy=TRANSITION_POLICY,
                geometry=TRANSITION_GEOMETRY,
                agreement={},
                stage="query_facet_propagation_confirmation",
            )
            row["method"] = "new_query_facets_plus_monotonic_propagation"
            row["query_facet_count"] = facet_count
            row["query_facet_comparisons"] = root_comparisons
            row["query_facet_scoring_seconds"] = root_scoring_seconds
            row["total_routing_comparisons"] += root_comparisons - row["root_comparisons"]
            row["root_comparisons"] = root_comparisons
            row["estimated_full_routing_seconds"] += root_scoring_seconds
            propagated_rows.append(row)
        print(f"query-facet propagation: seed {seed} complete", flush=True)
    return root_rows, propagated_rows


def _mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.fmean(values) if values else 0.0


def _summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["partition"] == "test":
            grouped[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "identities": len({row["example_id"] for row in values}),
                "seeds": len({str(row["seed"]) for row in values}),
                "rows": len(values),
                "first_oracle_group_present": _mean(
                    values, "first_oracle_group_present"
                ),
                "oracle_recall": _mean(values, "oracle_recall"),
                "complete_oracle": _mean(values, "complete_oracle"),
                "chain_complete": _mean(values, "chain_complete"),
                "active_final_kv_fraction": _mean(
                    values, "active_final_kv_fraction"
                ),
                "total_routing_comparisons": _mean(
                    values, "total_routing_comparisons"
                ),
                "propagation_activation": _mean(values, "propagation_activation"),
                "estimated_full_routing_seconds": _mean(
                    values, "estimated_full_routing_seconds"
                ),
            }
        )
    return output


def _comparison_rows(
    new_root_rows: list[dict],
    new_propagated_rows: list[dict],
    prior_rows_path: Path,
    query_rows_path: Path,
    source_features: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    feature_by_id = {feature["example_id"]: feature for feature in source_features}
    with query_rows_path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            if (
                source["partition"] == "test"
                and float(source["fraction"]) == PRIMARY_FRACTION
                and source["variant"] == "A_global_semantic"
            ):
                feature = feature_by_id[source["example_id"]]
                groups = evidence_parent_groups(feature)
                selected = set(json.loads(source["selected_parent_ids"]))
                rows.append(
                    {
                        **source,
                        "method": "current_one_shot",
                        "first_oracle_group_present": source["oracle_root_present"],
                        "chain_complete": float(
                            bool(groups)
                            and all(bool(selected & group) for group in groups)
                        ),
                        "total_routing_comparisons": source["search_comparisons"],
                        "propagation_activation": 0.0,
                        "estimated_full_routing_seconds": source["wall_time"],
                    }
                )
    with prior_rows_path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            if (
                source["partition"] == "test"
                and float(source["fraction"]) == PRIMARY_FRACTION
                and source["policy_role"] == "exploration"
                and source["transition_geometry"] == TRANSITION_GEOMETRY
            ):
                rows.append({**source, "method": "current_best_monotonic_iterative"})
    rows.extend(new_root_rows)
    rows.extend(new_propagated_rows)
    return rows


def _plot(summary: list[dict], output_dir: Path) -> None:
    methods = (
        "current_one_shot",
        "current_best_monotonic_iterative",
        "new_query_facets_root_only",
        "new_query_facets_plus_monotonic_propagation",
    )
    labels = ("One-shot", "Prior iterative", "Facets", "Facets + propagation")
    lookup = {(row["dataset"], row["method"]): row for row in summary}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    width = 0.36
    x = list(range(len(methods)))
    for offset, dataset, color in (
        (-width / 2, "hotpotqa", "#4c78a8"),
        (width / 2, "qasper", "#f58518"),
    ):
        axes[0].bar(
            [value + offset for value in x],
            [lookup[(dataset, method)]["first_oracle_group_present"] for method in methods],
            width,
            label=dataset,
            color=color,
            edgecolor="#202020",
            linewidth=0.4,
        )
        axes[1].bar(
            [value + offset for value in x],
            [lookup[(dataset, method)]["chain_complete"] for method in methods],
            width,
            label=dataset,
            color=color,
            edgecolor="#202020",
            linewidth=0.4,
        )
    for axis, title in zip(axes, ("Entry evidence in root Top-B", "Chain completion")):
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out rate")
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"propagation_confirmation.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    source_features = torch.load(
        args.source_feature_file, map_location="cpu", weights_only=False
    )
    query_features = torch.load(
        args.query_feature_file, map_location="cpu", weights_only=False
    )
    if [row["example_id"] for row in source_features] != [
        row["example_id"] for row in query_features
    ]:
        raise ValueError("Source and query feature identities/order do not match.")
    new_root, new_propagated = _confirmation_rows(
        source_features, query_features, args
    )
    comparison_rows = _comparison_rows(
        new_root,
        new_propagated,
        args.prior_policy_rows,
        args.query_rows,
        source_features,
    )
    summary = _summary(comparison_rows)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_default_changed": False,
        "training_performed": False,
        "fraction": PRIMARY_FRACTION,
        "query_config": {
            "window": 4,
            "stride": 2,
            "facet_reduction": "max",
            "contextual_encoding_passes": 1,
            "independent_window_encoding_passes": 0,
        },
        "propagation_config": {
            "root_lock": ROOT_POLICY_NAME,
            "transition_geometry": TRANSITION_GEOMETRY,
            "transition_policy": TRANSITION_POLICY_NAME,
            "candidate_fraction": args.candidate_fraction,
        },
        "heldout_comparison": summary,
        "recommendation": "retain_query_facets_as_research_option",
        "sdk_change_in_this_iteration": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "propagation_confirmation.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "propagation_confirmation_rows.csv", comparison_rows)
    _write_csv(args.output_dir / "propagation_confirmation_summary.csv", summary)
    _plot(summary, args.output_dir)

    result_path = args.output_dir / "query_entry_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["propagation_confirmation_run"] = True
    result["propagation_confirmation"] = artifact
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--candidate-fraction", type=float, default=0.20)
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--query-feature-file",
        type=Path,
        default=result_root / "query_entry_facets/query_entry_features.pt",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--prior-policy-rows",
        type=Path,
        default=result_root / "monotonic_adaptive_competition/heldout_policy_rows.csv",
    )
    parser.add_argument(
        "--query-rows",
        type=Path,
        default=result_root / "query_entry_facets/query_entry_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "query_entry_facets",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["heldout_comparison"], indent=2))
