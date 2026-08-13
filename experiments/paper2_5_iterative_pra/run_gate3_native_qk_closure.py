"""Gate 3: semantically narrowed tokenwise native-QK propagation."""

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

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_5_iterative_pra.run_gate2_local_closure import (
    _evaluate as evaluate_semantic,
)
from pra_hf.native_closure import (
    NativeLocalQKRouter,
    NativeQKIndex,
    NativeQKRoutingConfig,
)
from pra_torch.hf import load_hf_routing_projection


SEEDS = (11, 23, 37, 53, 71)
FINAL_FRACTIONS = (0.10, 0.20, 0.30)
CANDIDATE_FRACTIONS = (0.10, 0.20)
BASELINES = ("one_shot_parent", "parent_closure", "local_gist_closure")
NATIVE_VARIANTS = (
    ("max_topk", "max", "max", "topk"),
    ("top4_topk", "top_m_mean", "top_m_mean", "topk"),
    ("max_threshold1", "max", "max", "threshold"),
)


def _groups(mask: torch.Tensor) -> list[set[int]]:
    groups: list[set[int]] = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        if not groups or index != max(groups[-1]) + 1:
            groups.append(set())
        groups[-1].add(index)
    return groups


def _native_index(feature, pm, lm, lq, device: torch.device) -> NativeQKIndex:
    return NativeQKIndex(
        parent_ids=tuple(
            f"{feature['example_id']}#parent={index}"
            for index in range(len(feature["parent_spans"]))
        ),
        parent_spans=tuple(tuple(span) for span in feature["parent_spans"]),
        parent_memory_gists=pm,
        local_spans=tuple(tuple(span) for span in feature["local_spans"]),
        local_parent_indices=feature["local_parent_indices"].to(device),
        local_memory_gists=lm,
        local_query_gists=lq,
        local_pre_query=feature["local_pre_query"].to(device),
        local_pre_key=feature["local_pre_key"].to(device),
        token_mask=feature["local_token_mask"].to(device),
        layer_id=27,
    )


def _native_condition(variant: str, candidate_fraction: float) -> str:
    return f"native_qk_{variant}_p{round(100 * candidate_fraction)}"


def _evaluate_native(
    feature,
    root,
    index,
    *,
    seed: int,
    fraction: float,
    candidate_fraction: float,
    variant: tuple[str, str, str, str],
):
    name, token_reduction, head_reduction, transition_mode = variant
    parent_count = len(feature["parent_spans"])
    budget = max(1, math.ceil(parent_count * fraction))
    evidence = set(
        torch.nonzero(feature["parent_positive_mask"]).flatten().tolist()
    )
    groups = _groups(feature["parent_positive_mask"])
    evidence_ids = {
        f"{feature['example_id']}#parent={index}" for index in evidence
    }
    config = NativeQKRoutingConfig(
        max_unique_parents=budget,
        candidate_pool_fraction=candidate_fraction,
        initial_parent_count=max(1, math.ceil(budget / 2)),
        branch_top_k=max(1, budget - max(1, math.ceil(budget / 2))),
        root_anchor_alpha=0.25,
        token_reduction=token_reduction,
        head_reduction=head_reduction,
        top_m=4,
        transition_mode=transition_mode,
        threshold_lambda=1.0,
    )
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    started = time.perf_counter()
    result = NativeLocalQKRouter(index).route(
        root,
        config,
        example_id=feature["example_id"],
        evidence_parent_ids=evidence_ids,
    )
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    elapsed = time.perf_counter() - started
    selected = set(result.selected_indices)
    selected_tokens = sum(
        int(feature["parent_spans"][index][1])
        - int(feature["parent_spans"][index][0])
        for index in selected
    )
    costs = result.graph.costs
    row = {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "fraction": fraction,
        "condition": _native_condition(name, candidate_fraction),
        "candidate_fraction": candidate_fraction,
        "token_reduction": token_reduction,
        "head_reduction": head_reduction,
        "transition_mode": transition_mode,
        "budget_parents": budget,
        "candidate_parents_total": parent_count,
        "candidate_parents": costs["candidate_parents"],
        "candidate_local_regions": costs["candidate_local_regions"],
        "candidate_tokens": costs["candidate_tokens"],
        "unique_parents_selected": len(selected),
        "any_evidence": float(bool(selected & evidence)),
        "exact_evidence_identity": float(bool(evidence) and evidence <= selected),
        "chain_completion": float(
            bool(groups) and all(bool(selected & group) for group in groups)
        ),
        "evidence_coverage": len(selected & evidence) / max(len(evidence), 1),
        "exact_all_evidence_feasible": float(len(evidence) <= budget),
        "chain_budget_feasible": float(len(groups) <= budget),
        "materialized_kv_tokens": selected_tokens,
        "materialized_kv_fraction": selected_tokens
        / max(feature["source_tokens"], 1),
        "routing_seconds": elapsed,
        "semantic_gist_comparisons": costs["semantic_gist_comparisons"],
        "native_qk_dot_products": costs["native_qk_dot_products"],
        "native_qk_comparisons": costs["native_qk_comparisons"],
        "accepted_native_transitions": costs["accepted_native_transitions"],
        "proposed_native_transitions": costs["proposed_native_transitions"],
        "kv_materializations_during_closure": costs[
            "kv_materializations_during_closure"
        ],
    }
    graph = result.graph.to_dict()
    graph.update(
        {
            "condition": row["condition"],
            "seed": seed,
            "fraction": fraction,
            "candidate_fraction": candidate_fraction,
        }
    )
    _validate_graph(graph)
    return row, graph


def _normalize_baseline(row: dict, local_region_count: int) -> dict:
    """Add Gate-3 cost columns without changing Gate-2 measurements."""
    semantic_comparisons = row["semantic_gist_comparisons"]
    if row["condition"] == "local_gist_closure":
        semantic_comparisons += local_region_count
    return {
        **row,
        "semantic_gist_comparisons": semantic_comparisons,
        "candidate_fraction": 0.0,
        "token_reduction": "none",
        "head_reduction": "none",
        "transition_mode": "none",
        "candidate_parents_total": row["candidate_parents"],
        "candidate_local_regions": 0,
        "candidate_tokens": 0,
        "native_qk_dot_products": 0,
        "accepted_native_transitions": 0,
        "proposed_native_transitions": 0,
        "kv_materializations_during_closure": 0,
    }


def _validate_graph(graph: dict) -> None:
    if graph["schema_version"] != "2.0":
        raise ValueError("Gate-3 graphs must remain schema-v2 compatible.")
    if graph["costs"].get("kv_materializations_during_closure", 0) != 0:
        raise ValueError("Native closure materialized K/V before final selection.")
    if graph["costs"].get("unique_parents_selected", 0) > graph["budget"][
        "max_unique_parents"
    ]:
        raise ValueError("Native closure exceeded the final parent budget.")
    for edge in graph["edges"]:
        if edge["edge_type"] == "native_qk":
            required = (
                "source_span",
                "target_span",
                "query_head",
                "kv_head",
                "score",
                "semantic_candidate_rank",
            )
            if any(edge.get(field) is None for field in required):
                raise ValueError("Native graph edge lacks required diagnostics.")


def _aggregate(rows: list[dict]) -> list[dict]:
    dimensions = {
        "dataset",
        "example_id",
        "seed",
        "fraction",
        "condition",
        "token_reduction",
        "head_reduction",
        "transition_mode",
    }
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["fraction"], row["condition"])].append(row)
    output = []
    for (dataset, fraction, condition), values in sorted(grouped.items()):
        record = {
            "dataset": dataset,
            "fraction": fraction,
            "condition": condition,
            "examples_x_seeds": len(values),
        }
        metrics = sorted({key for row in values for key in row}.difference(dimensions))
        for metric in metrics:
            samples = [
                float(row[metric])
                for row in values
                if metric in row and isinstance(row[metric], (int, float))
            ]
            if samples:
                record[metric] = statistics.fmean(samples)
        output.append(record)
    return output


def _paired(rows: list[dict]) -> list[dict]:
    metrics = (
        "any_evidence",
        "exact_evidence_identity",
        "chain_completion",
        "evidence_coverage",
        "semantic_gist_comparisons",
        "native_qk_dot_products",
        "routing_seconds",
    )
    keyed = {
        (
            row["dataset"],
            row["example_id"],
            row["seed"],
            row["fraction"],
            row["condition"],
        ): row
        for row in rows
    }
    output = []
    native_conditions = sorted(
        {row["condition"] for row in rows if row["condition"].startswith("native_qk")}
    )
    for dataset in sorted({row["dataset"] for row in rows}):
        for fraction in sorted(
            {row["fraction"] for row in rows if row["dataset"] == dataset}
        ):
            for left in native_conditions:
                for right in ("one_shot_parent", "local_gist_closure"):
                    pairs = []
                    for key, left_row in keyed.items():
                        if key[0] == dataset and key[3] == fraction and key[4] == left:
                            right_row = keyed.get((*key[:4], right))
                            if right_row is not None:
                                pairs.append((left_row, right_row))
                    if not pairs:
                        continue
                    record = {
                        "dataset": dataset,
                        "fraction": fraction,
                        "comparison": f"{left}_minus_{right}",
                        "pairs": len(pairs),
                    }
                    for metric in metrics:
                        record[f"delta_{metric}"] = statistics.fmean(
                            float(left_row.get(metric, 0))
                            - float(right_row.get(metric, 0))
                            for left_row, right_row in pairs
                        )
                    output.append(record)
    return output


def _synthetic(seed: int, examples: int, device: torch.device) -> list[dict]:
    """Build a native-only bridge plus a stronger semantic distractor."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for example in range(examples):
        basis, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        root = basis[0].to(device)
        pm = torch.stack((
            basis[0],
            0.7 * basis[0] + 0.7141428 * basis[3],
            basis[2],
            basis[4],
        )).to(device)
        pq = torch.stack((basis[3], basis[3], basis[0], basis[4])).to(device)
        lm = torch.stack((basis[0], basis[1], basis[2], basis[4])).to(device)
        lq = torch.stack((basis[1], basis[3], basis[0], basis[4])).to(device)
        pre_q = torch.zeros(4, 2, 4, 2, device=device)
        pre_k = torch.zeros(4, 2, 2, 2, device=device)
        pre_q[0, 0, 0] = torch.tensor([3.0, 0.0], device=device)
        pre_k[1, 0, 0] = torch.tensor([0.0, 3.0], device=device)
        pre_k[2, 0, 0] = torch.tensor([3.0, 0.0], device=device)
        pre_k[3, 0, 0] = torch.tensor([-3.0, 0.0], device=device)
        feature = {
            "dataset": "synthetic_native_bridge",
            "example_id": f"native-{seed}-{example}",
            "parent_spans": [(0, 256), (256, 512), (512, 768), (768, 1024)],
            "parent_positive_mask": torch.tensor([True, False, True, False]),
            "local_spans": [(0, 32), (256, 288), (512, 544), (768, 800)],
            "local_parent_indices": torch.arange(4),
            "source_tokens": 1024,
            "local_pre_query": pre_q,
            "local_pre_key": pre_k,
            "local_token_mask": torch.ones(4, 2, dtype=torch.bool, device=device),
        }
        for condition in BASELINES:
            row, _ = evaluate_semantic(
                feature,
                root,
                pm,
                pq,
                lm,
                lq,
                seed=seed,
                fraction=0.5,
                condition=condition,
            )
            rows.append(_normalize_baseline(row, len(feature["local_spans"])))
        index = _native_index(feature, pm, lm, lq, device)
        row, _ = _evaluate_native(
            feature,
            root,
            index,
            seed=seed,
            fraction=0.5,
            candidate_fraction=0.75,
            variant=NATIVE_VARIANTS[0],
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plots(aggregate: list[dict], output_dir: Path) -> None:
    conditions = {
        "one_shot_parent": "One-shot semantic",
        "parent_closure": "Q-frontier parent",
        "local_gist_closure": "Local semantic",
        "native_qk_max_topk_p10": "Native QK, pool 10%",
        "native_qk_max_topk_p20": "Native QK, pool 20%",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    for row_index, dataset in enumerate(("hotpotqa", "qasper")):
        for condition, label in conditions.items():
            values = sorted(
                (
                    row
                    for row in aggregate
                    if row["dataset"] == dataset and row["condition"] == condition
                ),
                key=lambda row: row["fraction"],
            )
            for column, metric in enumerate(("chain_completion", "evidence_coverage")):
                axes[row_index, column].plot(
                    [100 * row["fraction"] for row in values],
                    [row[metric] for row in values],
                    marker="o",
                    label=label,
                )
        axes[row_index, 0].set_ylabel(f"{dataset}\nRecall")
    for axis, title in zip(axes[0], ("Evidence-group chain completion", "Evidence coverage")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Final parent/KV budget (%)")
    for axis in axes.flat:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7, loc="lower right")
    figure.suptitle("Gate 3: semantic narrowing plus native local QK closure")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"gate3_native_qk_quality.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    primary = [
        row
        for row in aggregate
        if row["condition"] in {"native_qk_max_topk_p10", "native_qk_max_topk_p20"}
        and row["fraction"] == 0.20
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    labels = [f"{row['dataset']} {round(100 * row['candidate_fraction'])}%" for row in primary]
    for axis, metric, title in zip(
        axes,
        ("semantic_gist_comparisons", "native_qk_dot_products", "routing_seconds"),
        ("Semantic comparisons", "Native token/head dots", "Routing wall time (s)"),
    ):
        axis.bar(labels, [row[metric] for row in primary])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Gate 3 routing cost at 20% final budget")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"gate3_native_qk_cost.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.output_dir / "native_qk_features_test.pt", weights_only=False)
    rows, graphs = [], []
    for seed in args.seeds:
        checkpoint = args.feature_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in features:
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
            index = _native_index(feature, pm, lm, lq, device)
            for fraction in args.final_fractions:
                for condition in BASELINES:
                    row, _ = evaluate_semantic(
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
                    rows.append(
                        _normalize_baseline(row, len(feature["local_spans"]))
                    )
                for candidate_fraction in args.candidate_fractions:
                    for variant in NATIVE_VARIANTS:
                        row, graph = _evaluate_native(
                            feature,
                            root,
                            index,
                            seed=seed,
                            fraction=fraction,
                            candidate_fraction=candidate_fraction,
                            variant=variant,
                        )
                        rows.append(row)
                        if (
                            fraction == 0.20
                            and candidate_fraction == 0.20
                            and variant[0] == "max_topk"
                        ):
                            graphs.append(graph)
        print(f"gate 3 seed {seed}: {len(rows)} natural rows", flush=True)
    synthetic = []
    for seed in args.seeds:
        synthetic.extend(_synthetic(seed, args.synthetic_examples, device))
    all_rows = rows + synthetic
    aggregate = _aggregate(all_rows)
    paired = _paired(all_rows)
    artifact = {
        "schema_version": "2.0",
        "gate": 3,
        "runtime": runtime_metadata(),
        "seeds": list(args.seeds),
        "final_fractions": list(args.final_fractions),
        "candidate_fractions": list(args.candidate_fractions),
        "native_variants": [
            {
                "name": row[0],
                "token_reduction": row[1],
                "head_reduction": row[2],
                "transition_mode": row[3],
            }
            for row in NATIVE_VARIANTS
        ],
        "routing_representation": "layer_27_tokenwise_pre_rope_qk",
        "semantic_narrowing_first": True,
        "materialization_changed": False,
        "memory_use_adapter": None,
        "rows": all_rows,
        "aggregate": aggregate,
        "paired_deltas": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate3_native_qk_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "gate3_native_qk_rows.csv", all_rows)
    _write_csv(args.output_dir / "gate3_native_qk_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "gate3_native_qk_paired.csv", paired)
    with (args.output_dir / "gate3_retrieval_graphs.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for graph in graphs:
            _validate_graph(graph)
            stream.write(json.dumps(graph, sort_keys=True) + "\n")
    _plots(aggregate, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument(
        "--final-fractions", default=",".join(map(str, FINAL_FRACTIONS))
    )
    parser.add_argument(
        "--candidate-fractions", default=",".join(map(str, CANDIDATE_FRACTIONS))
    )
    parser.add_argument("--synthetic-examples", type=int, default=64)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    args.final_fractions = tuple(map(float, args.final_fractions.split(",")))
    args.candidate_fractions = tuple(map(float, args.candidate_fractions.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"gate": 3, "rows": len(result["rows"])}, indent=2))
