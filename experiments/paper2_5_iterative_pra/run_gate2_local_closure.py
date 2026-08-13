"""Compare parent and local-gist closure under matched parent/KV budgets."""

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
from pra_hf.iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingConfig,
)
from pra_torch.hf import load_hf_routing_projection


SEEDS = (11, 23, 37, 53, 71)
FRACTIONS = (0.10, 0.20, 0.30)
CONDITIONS = ("one_shot_parent", "parent_closure", "local_gist_closure")


def _groups(mask: torch.Tensor) -> list[set[int]]:
    groups: list[set[int]] = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        if not groups or index != max(groups[-1]) + 1:
            groups.append(set())
        groups[-1].add(index)
    return groups


def _parent_index(feature, memory, query_memory) -> GistIndex:
    records = []
    for index, span in enumerate(feature["parent_spans"]):
        entry = SimpleNamespace(uri=f"memory://{feature['example_id']}")
        chunk = SimpleNamespace(
            chunk_id=f"{feature['example_id']}#parent={index}",
            token_start=int(span[0]), token_end=int(span[1]),
            logical_start=int(span[0]), logical_end=int(span[1]),
            routing_gist=SimpleNamespace(k=memory[index:index + 1]),
        )
        records.append((entry, chunk))
    return GistIndex(
        27, tuple(records), F.normalize(memory.float(), dim=-1).unsqueeze(1),
        torch.ones((len(records), 1), device=memory.device, dtype=torch.bool),
        F.normalize(query_memory.float(), dim=-1).unsqueeze(1),
    )


def _hierarchical_index(feature, pm, pq, lm, lq) -> HierarchicalGistIndex:
    return HierarchicalGistIndex(
        parent_ids=tuple(
            f"{feature['example_id']}#parent={index}"
            for index in range(len(feature["parent_spans"]))
        ),
        parent_spans=tuple(tuple(span) for span in feature["parent_spans"]),
        parent_memory_gists=pm,
        parent_query_gists=pq,
        local_spans=tuple(tuple(span) for span in feature["local_spans"]),
        local_parent_indices=feature["local_parent_indices"].to(pm.device),
        local_memory_gists=lm,
        local_query_gists=lq,
        layer_id=27,
    )


def _config(condition: str, budget: int) -> IterativeRoutingConfig:
    if condition == "one_shot_parent":
        return IterativeRoutingConfig(
            depth=1, branch_top_k=budget, beam_size=budget,
            max_unique_chunks=budget, root_anchor_alpha=1.0,
            path_score_mode="direct",
        )
    per_hop = max(1, math.ceil(budget / 2))
    return IterativeRoutingConfig(
        depth=2, branch_top_k=per_hop, beam_size=per_hop,
        max_unique_chunks=budget, root_anchor_alpha=0.25,
        frontier_projection="query", path_score_mode="product",
    )


def _evaluate(feature, root, pm, pq, lm, lq, *, seed, fraction, condition):
    parent_count = len(feature["parent_spans"])
    budget = max(1, math.ceil(parent_count * fraction))
    evidence = set(torch.nonzero(feature["parent_positive_mask"]).flatten().tolist())
    groups = _groups(feature["parent_positive_mask"])
    evidence_ids = {f"{feature['example_id']}#parent={index}" for index in evidence}
    config = _config(condition, budget)
    if root.device.type == "cuda": torch.cuda.synchronize(root.device)
    started = time.perf_counter()
    if condition == "local_gist_closure":
        result = HierarchicalLocalGistRouter(
            _hierarchical_index(feature, pm, pq, lm, lq)
        ).route(root, config, example_id=feature["example_id"], evidence_parent_ids=evidence_ids)
    else:
        router = IterativeGistRouter(_parent_index(feature, pm, pq))
        result = router.route(root, config, example_id=feature["example_id"], evidence_chunk_ids=evidence_ids)
    if root.device.type == "cuda": torch.cuda.synchronize(root.device)
    elapsed = time.perf_counter() - started
    selected = set(result.selected_indices)
    selected_tokens = sum(
        int(feature["parent_spans"][index][1]) - int(feature["parent_spans"][index][0])
        for index in selected
    )
    costs = result.graph.costs
    row = {
        "dataset": feature["dataset"], "example_id": feature["example_id"],
        "seed": seed, "fraction": fraction, "condition": condition,
        "budget_parents": budget, "candidate_parents": parent_count,
        "unique_parents_selected": len(selected),
        "any_evidence": float(bool(selected & evidence)),
        "exact_evidence_identity": float(bool(evidence) and evidence <= selected),
        "chain_completion": float(bool(groups) and all(bool(selected & group) for group in groups)),
        "evidence_coverage": len(selected & evidence) / max(len(evidence), 1),
        "materialized_kv_tokens": selected_tokens,
        "materialized_kv_fraction": selected_tokens / max(feature["source_tokens"], 1),
        "routing_seconds": elapsed,
        "semantic_gist_comparisons": costs.get("semantic_gist_comparisons", costs.get("unique_gist_comparisons", 0)),
        "native_qk_comparisons": costs.get("native_qk_comparisons", 0),
        "local_nodes_explored": costs.get("local_nodes_explored", 0),
        "local_nodes_activated_per_parent": costs.get("local_nodes_activated_per_parent", 0),
        "cross_parent_transitions": costs.get("cross_parent_transitions", 0),
        "repeated_parent_transitions": costs.get("repeated_parent_transitions", 0),
        "local_entropy": costs.get("local_entropy", 0),
        "parent_entropy": costs.get("parent_entropy", 0),
        "path_depth": costs.get("path_depth", max((node.hop for node in result.graph.nodes), default=0)),
        "local_vs_parent_similarity_ratio": costs.get("local_vs_parent_similarity_ratio", 0),
        "bridge_locality_score": costs.get("bridge_locality_score", 0),
    }
    graph = result.graph.to_dict()
    graph.update({"condition": condition, "seed": seed, "fraction": fraction})
    return row, graph


def _aggregate(rows):
    metrics = tuple(key for key in rows[0] if key not in {
        "dataset", "example_id", "seed", "fraction", "condition"
    })
    grouped = defaultdict(list)
    for row in rows: grouped[(row["dataset"], row["fraction"], row["condition"])].append(row)
    output = []
    for (dataset, fraction, condition), values in sorted(grouped.items()):
        record = {"dataset": dataset, "fraction": fraction, "condition": condition, "examples_x_seeds": len(values)}
        for metric in metrics:
            record[metric] = statistics.fmean(float(row[metric]) for row in values)
        output.append(record)
    return output


def _paired(rows):
    metrics = ("any_evidence", "exact_evidence_identity", "chain_completion", "evidence_coverage", "semantic_gist_comparisons", "routing_seconds")
    keyed = {(r["dataset"], r["example_id"], r["seed"], r["fraction"], r["condition"]): r for r in rows}
    output = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for fraction in sorted({r["fraction"] for r in rows if r["dataset"] == dataset}):
            for left, right in (("local_gist_closure", "one_shot_parent"), ("local_gist_closure", "parent_closure"), ("parent_closure", "one_shot_parent")):
                pairs = []
                for key, left_row in keyed.items():
                    if key[0] == dataset and key[3] == fraction and key[4] == left:
                        pairs.append((left_row, keyed[(*key[:4], right)]))
                if not pairs:
                    continue
                record = {"dataset": dataset, "fraction": fraction, "comparison": f"{left}_minus_{right}", "pairs": len(pairs)}
                for metric in metrics:
                    record[f"delta_{metric}"] = statistics.fmean(a[metric] - b[metric] for a, b in pairs)
                output.append(record)
    return output


def _synthetic(seed: int, examples: int, device):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for example in range(examples):
        basis, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        root = basis[0].to(device)
        # Parent A is root-relevant but its mean query does not lead to B.
        # A's first local query is the hidden bridge to B's first local key.
        parent_memory = torch.stack((basis[0], 0.7 * basis[0] + 0.7141428 * basis[3], basis[2])).to(device)
        parent_query = torch.stack((basis[4], basis[3], basis[0])).to(device)
        local_memory = torch.stack((basis[0], basis[4], basis[3], basis[5], basis[1], basis[2])).to(device)
        local_query = torch.stack((basis[1], basis[4], basis[3], basis[5], basis[0], basis[2])).to(device)
        feature = {
            "dataset": "synthetic_local_bridge", "example_id": f"s{seed}-{example}",
            "parent_spans": [(0, 256), (256, 512), (512, 768)],
            "parent_positive_mask": torch.tensor([True, False, True]),
            "local_spans": [(0, 32), (32, 64), (256, 288), (288, 320), (512, 544), (544, 576)],
            "local_parent_indices": torch.tensor([0, 0, 1, 1, 2, 2]),
            "source_tokens": 768,
        }
        for condition in CONDITIONS:
            row, _ = _evaluate(feature, root, parent_memory, parent_query, local_memory, local_query, seed=seed, fraction=2/3, condition=condition)
            rows.append(row)
    return rows


def _write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _plots(aggregate, output_dir):
    labels = {"one_shot_parent": "One-shot parent", "parent_closure": "Parent closure", "local_gist_closure": "Local-gist closure"}
    for dataset in ("hotpotqa", "qasper"):
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for condition, label in labels.items():
            values = sorted((r for r in aggregate if r["dataset"] == dataset and r["condition"] == condition), key=lambda r: r["fraction"])
            for axis, metric in zip(axes, ("any_evidence", "chain_completion", "evidence_coverage")):
                axis.plot([100*r["fraction"] for r in values], [r[metric] for r in values], marker="o", label=label)
        for axis, title in zip(axes, ("Any evidence", "Chain completion", "Coverage")):
            axis.set_title(title); axis.set_xlabel("Final parent/KV budget (%)"); axis.set_ylim(0, 1.02); axis.grid(alpha=.25)
        axes[0].set_ylabel("Recall"); axes[-1].legend(fontsize=8)
        figure.suptitle(f"{dataset}: hierarchical local closure"); figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{dataset}_local_gate.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)


def run(args):
    device = torch.device(args.device)
    features = torch.load(args.output_dir / "local_router_features_test.pt", weights_only=False)
    rows, graphs = [], []
    for seed in args.seeds:
        checkpoint = args.feature_dir / "checkpoints" / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in features:
            with torch.no_grad():
                root = projection.project_query(feature["query_hidden"].to(device).unsqueeze(0))[0]
                ph, lh = feature["parent_hidden"].to(device), feature["local_hidden"].to(device)
                pm, pq = projection.project_memory(ph), projection.project_query(ph)
                lm, lq = projection.project_memory(lh), projection.project_query(lh)
            for fraction in args.fractions:
                for condition in CONDITIONS:
                    row, graph = _evaluate(feature, root, pm, pq, lm, lq, seed=seed, fraction=fraction, condition=condition)
                    rows.append(row)
                    if fraction == 0.20: graphs.append(graph)
        print(f"gate 2 seed {seed}: {len(rows)} rows", flush=True)
    synthetic = []
    for seed in args.seeds: synthetic.extend(_synthetic(seed, args.synthetic_examples, device))
    all_rows = rows + synthetic
    aggregate, paired = _aggregate(all_rows), _paired(all_rows)
    artifact = {
        "schema_version": "2.0", "gate": 2, "runtime": runtime_metadata(),
        "seeds": list(args.seeds), "fractions": list(args.fractions),
        "scales": {"contextual_encoding_tokens": 256, "associative_propagation_tokens": 32, "materialization_parent_tokens": 256},
        "local_windows_reencoded": False, "rows": all_rows, "aggregate": aggregate, "paired_deltas": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate2_local_results.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output_dir / "gate2_local_rows.csv", all_rows)
    _write_csv(args.output_dir / "gate2_local_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "gate2_local_paired.csv", paired)
    with (args.output_dir / "gate2_retrieval_graphs.jsonl").open("w", encoding="utf-8") as stream:
        for graph in graphs: stream.write(json.dumps(graph, sort_keys=True) + "\n")
    _plots(aggregate, args.output_dir)
    return artifact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--synthetic-examples", type=int, default=64)
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/local_associative_closure")
    args = parser.parse_args(); args.seeds = tuple(map(int, args.seeds.split(","))); args.fractions = tuple(map(float, args.fractions.split(","))); return args


if __name__ == "__main__":
    result = run(parse_args()); print(json.dumps({"gate": 2, "rows": len(result["rows"])}, indent=2))
