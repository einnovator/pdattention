"""Evaluate projection-correct and local associative closure in staged gates."""

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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.iterative import GistIndex, IterativeGistRouter, IterativeRoutingConfig
from pra_torch.hf import load_hf_routing_projection


SEEDS = (11, 23, 37, 53, 71)
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
CONDITIONS = {
    "one_shot": {"depth": 1, "projection": "memory", "alpha": 1.0},
    "chunk_closure_memory_frontier": {
        "depth": 2,
        "projection": "memory",
        "alpha": 0.25,
    },
    "chunk_closure_query_frontier": {
        "depth": 2,
        "projection": "query",
        "alpha": 0.25,
    },
}


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _groups(mask: torch.Tensor) -> list[set[int]]:
    groups: list[set[int]] = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        if not groups or index != max(groups[-1]) + 1:
            groups.append(set())
        groups[-1].add(index)
    return groups


def _index(feature: dict, memory: torch.Tensor, query_memory: torch.Tensor) -> GistIndex:
    records = []
    for index, span in enumerate(feature["chunk_spans"]):
        entry = SimpleNamespace(uri=f"memory://{feature['example_id']}")
        chunk = SimpleNamespace(
            chunk_id=f"{feature['example_id']}#chunk={index}",
            token_start=int(span[0]),
            token_end=int(span[1]),
            logical_start=int(span[0]),
            logical_end=int(span[1]),
            routing_gist=SimpleNamespace(k=memory[index : index + 1]),
        )
        records.append((entry, chunk))
    mask = torch.ones((len(records), 1), device=memory.device, dtype=torch.bool)
    return GistIndex(
        layer_id=27,
        records=tuple(records),
        gists=F.normalize(memory.float(), dim=-1).unsqueeze(1),
        gist_mask=mask,
        query_gists=F.normalize(query_memory.float(), dim=-1).unsqueeze(1),
    )


def _routing_config(label: str, budget: int) -> IterativeRoutingConfig:
    condition = CONDITIONS[label]
    if label == "one_shot":
        return IterativeRoutingConfig(
            depth=1,
            branch_top_k=budget,
            beam_size=budget,
            max_unique_chunks=budget,
            root_anchor_alpha=1.0,
            path_score_mode="direct",
        )
    per_hop = max(1, math.ceil(budget / int(condition["depth"])))
    return IterativeRoutingConfig(
        depth=int(condition["depth"]),
        branch_top_k=per_hop,
        beam_size=per_hop,
        max_unique_chunks=budget,
        root_anchor_alpha=float(condition["alpha"]),
        frontier_projection=str(condition["projection"]),
        path_score_mode="product",
    )


def _evaluate(
    feature: dict,
    root: torch.Tensor,
    memory: torch.Tensor,
    query_memory: torch.Tensor,
    *,
    seed: int,
    fraction: float,
    condition: str,
) -> tuple[dict, dict]:
    budget = max(1, math.ceil(fraction * len(memory)))
    router = IterativeGistRouter(_index(feature, memory, query_memory))
    evidence = set(torch.nonzero(feature["positive_mask"], as_tuple=False).flatten().tolist())
    evidence_ids = {router.index.chunk_ids[index] for index in evidence}
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    started = time.perf_counter()
    result = router.route(
        root,
        _routing_config(condition, budget),
        example_id=feature["example_id"],
        evidence_chunk_ids=evidence_ids,
    )
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    elapsed = time.perf_counter() - started
    selected = set(result.selected_indices)
    groups = _groups(feature["positive_mask"])
    selected_tokens = sum(
        int(feature["chunk_spans"][index][1]) - int(feature["chunk_spans"][index][0])
        for index in selected
    )
    row = {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "fraction": fraction,
        "condition": condition,
        "depth": CONDITIONS[condition]["depth"],
        "frontier_projection": CONDITIONS[condition]["projection"],
        "budget": budget,
        "candidate_chunks": len(memory),
        "unique_selected_chunks": len(selected),
        "any_evidence": float(bool(selected & evidence)),
        "exact_evidence_identity": float(bool(evidence) and evidence <= selected),
        "chain_completion": float(
            bool(groups) and all(bool(selected & group) for group in groups)
        ),
        "evidence_coverage": len(selected & evidence) / max(len(evidence), 1),
        "materialized_kv_tokens": selected_tokens,
        "materialized_kv_fraction": selected_tokens / max(feature["source_tokens"], 1),
        "gist_comparisons": result.graph.costs["semantic_gist_comparisons"],
        "routing_seconds": elapsed,
    }
    graph = result.graph.to_dict()
    graph["condition"] = condition
    graph["seed"] = seed
    graph["fraction"] = fraction
    return row, graph


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "any_evidence",
        "exact_evidence_identity",
        "chain_completion",
        "evidence_coverage",
        "unique_selected_chunks",
        "materialized_kv_tokens",
        "materialized_kv_fraction",
        "gist_comparisons",
        "routing_seconds",
    )
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
        record.update({metric: _mean(values, metric) for metric in metrics})
        output.append(record)
    return output


def _paired(rows: list[dict]) -> list[dict]:
    metrics = (
        "any_evidence",
        "exact_evidence_identity",
        "chain_completion",
        "evidence_coverage",
        "gist_comparisons",
        "routing_seconds",
    )
    keyed = {
        (row["dataset"], row["example_id"], row["seed"], row["fraction"], row["condition"]): row
        for row in rows
    }
    output = []
    comparisons = (
        ("chunk_closure_query_frontier", "one_shot"),
        ("chunk_closure_query_frontier", "chunk_closure_memory_frontier"),
        ("chunk_closure_memory_frontier", "one_shot"),
    )
    for dataset in sorted({row["dataset"] for row in rows}):
        for fraction in sorted({row["fraction"] for row in rows}):
            for left, right in comparisons:
                pairs = []
                for key, left_row in keyed.items():
                    if key[0] != dataset or key[3] != fraction or key[4] != left:
                        continue
                    right_row = keyed[(key[0], key[1], key[2], key[3], right)]
                    pairs.append((left_row, right_row))
                record = {
                    "dataset": dataset,
                    "fraction": fraction,
                    "comparison": f"{left}_minus_{right}",
                    "pairs": len(pairs),
                }
                for metric in metrics:
                    record[f"delta_{metric}"] = statistics.fmean(
                        left_row[metric] - right_row[metric]
                        for left_row, right_row in pairs
                    )
                output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(aggregate: list[dict], output_dir: Path) -> None:
    labels = {
        "one_shot": "One-shot parent",
        "chunk_closure_memory_frontier": "Memory frontier (historical)",
        "chunk_closure_query_frontier": "Query frontier (corrected)",
    }
    for dataset in ("hotpotqa", "qasper"):
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        metrics = ("any_evidence", "chain_completion", "evidence_coverage")
        for condition, label in labels.items():
            values = [
                row for row in aggregate
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            values.sort(key=lambda row: row["fraction"])
            for axis, metric in zip(axes, metrics):
                axis.plot(
                    [100 * row["fraction"] for row in values],
                    [row[metric] for row in values],
                    marker="o",
                    label=label,
                )
        for axis, title in zip(axes, ("Any evidence", "Chain completion", "Coverage")):
            axis.set_title(title)
            axis.set_xlabel("Final chunk budget (%)")
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Recall")
        axes[-1].legend(fontsize=7)
        figure.suptitle(f"{dataset}: asymmetric frontier audit")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(
                output_dir / f"{dataset}_projection_gate.{suffix}",
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_dir / "router_features_test.pt", weights_only=False)
    rows, graphs = [], []
    for seed in args.seeds:
        checkpoint = (
            args.feature_dir
            / "checkpoints"
            / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in features:
            hidden = feature["memory_gists"].to(device)
            with torch.no_grad():
                root = projection.project_query(
                    feature["queries"]["last"].to(device).unsqueeze(0)
                )[0]
                memory = projection.project_memory(hidden)
                query_memory = projection.project_query(hidden)
            for fraction in args.fractions:
                for condition in CONDITIONS:
                    row, graph = _evaluate(
                        feature,
                        root,
                        memory,
                        query_memory,
                        seed=seed,
                        fraction=fraction,
                        condition=condition,
                    )
                    rows.append(row)
                    if fraction == 0.10:
                        graphs.append(graph)
        print(f"gate 1 seed {seed}: {len(rows)} rows", flush=True)
    aggregate = _aggregate(rows)
    paired = _paired(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "2.0",
        "gate": 1,
        "runtime": runtime_metadata(),
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "frozen_baseline_artifact": str(
            (ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/iterative_closure_results.json").relative_to(ROOT)
        ),
        "projection_contract": {
            "root": "W_q h_q",
            "stored_memory": "W_m h_B",
            "historical_frontier": "W_m h_A",
            "corrected_frontier": "W_q h_A",
            "retraining": False,
        },
        "rows": rows,
        "aggregate": aggregate,
        "paired_deltas": paired,
    }
    (args.output_dir / "gate1_projection_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "gate1_projection_rows.csv", rows)
    _write_csv(args.output_dir / "gate1_projection_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "gate1_projection_paired.csv", paired)
    with (args.output_dir / "gate1_retrieval_graphs.jsonl").open("w", encoding="utf-8") as stream:
        for graph in graphs:
            stream.write(json.dumps(graph, sort_keys=True) + "\n")
    _plot(aggregate, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/local_associative_closure",
    )
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    args.fractions = tuple(float(value) for value in args.fractions.split(","))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"gate": result["gate"], "rows": len(result["rows"])}, indent=2))
