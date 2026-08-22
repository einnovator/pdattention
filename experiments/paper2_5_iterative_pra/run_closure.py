"""Run matched-budget iterative-gist retrieval on frozen Paper 2 features.

The pretrained transformer is not rerun.  Five independently trained Paper 2
semantic projections map root queries and cached hidden-state gists into one
routing space.  All compared methods then operate on the same projected tensors.
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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.iterative import GistIndex, IterativeGistRouter, IterativeRoutingConfig
from pra_torch.hf import load_hf_routing_projection


SEEDS = (11, 23, 37, 53, 71)
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
DEPTHS = (1, 2, 3, 4)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FRONTIERS = ("direct", "residual", "mean", "weighted_mean")
PATH_MODES = ("product", "logsum", "last", "min", "mean", "direct")


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _ci95(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    return 2.776 * statistics.stdev(values) / math.sqrt(len(values))


def _evidence_groups(mask: torch.Tensor) -> list[set[int]]:
    """Group contiguous evidence chunks in source order; this is not causal hop order."""
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    groups: list[set[int]] = []
    for index in indices:
        if not groups or index != max(groups[-1]) + 1:
            groups.append(set())
        groups[-1].add(index)
    return groups


def _index(memory: torch.Tensor, spans: list, example_id: str, device) -> GistIndex:
    records = []
    for index, span in enumerate(spans):
        entry = SimpleNamespace(uri=f"memory://{example_id}")
        chunk = SimpleNamespace(
            chunk_id=f"{example_id}#chunk={index}",
            token_start=int(span[0]),
            token_end=int(span[1]),
            logical_start=int(span[0]),
            logical_end=int(span[1]),
            routing_gist=SimpleNamespace(k=memory[index : index + 1]),
        )
        records.append((entry, chunk))
    gists = F.normalize(memory.to(device=device, dtype=torch.float32), dim=-1).unsqueeze(1)
    return GistIndex(
        layer_id=27,
        records=tuple(records),
        gists=gists,
        gist_mask=torch.ones((len(records), 1), device=device, dtype=torch.bool),
    )


def _config(method: str, depth: int, budget: int, alpha: float, frontier: str, path: str):
    if method == "one_shot":
        return IterativeRoutingConfig(
            depth=1,
            branch_top_k=budget,
            beam_size=budget,
            max_unique_chunks=budget,
            root_anchor_alpha=1.0,
            frontier_mode="direct",
            path_score_mode="direct",
        )
    per_round = max(1, math.ceil(budget / max(depth, 1)))
    return IterativeRoutingConfig(
        depth=depth,
        branch_top_k=per_round,
        beam_size=per_round,
        max_unique_chunks=budget,
        root_anchor_alpha=alpha,
        frontier_mode=frontier,
        path_score_mode=path,
    )


def _evaluate_one(feature, root, memory, *, seed, method, depth, fraction, alpha, frontier, path):
    candidate_count = int(memory.shape[0])
    budget = max(1, math.ceil(fraction * candidate_count))
    router = IterativeGistRouter(_index(memory, feature["chunk_spans"], feature["example_id"], root.device))
    evidence = set(torch.nonzero(feature["positive_mask"], as_tuple=False).flatten().tolist())
    evidence_ids = {router.index.chunk_ids[index] for index in evidence}
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    started = time.perf_counter()
    result = router.route(
        root,
        _config(method, depth, budget, alpha, frontier, path),
        example_id=feature["example_id"],
        evidence_chunk_ids=evidence_ids,
    )
    if root.device.type == "cuda":
        torch.cuda.synchronize(root.device)
    elapsed = time.perf_counter() - started
    selected = set(result.selected_indices)
    selected_ids = [router.index.chunk_ids[index] for index in result.selected_indices]
    selected_by_hop = [
        [node.node_id for node in result.graph.nodes if node.hop == hop and node.final_selected]
        for hop in range(1, depth + 1)
    ]
    groups = _evidence_groups(feature["positive_mask"])
    direct = torch.tensor(result.direct_scores)
    direct_order = torch.argsort(direct, descending=True).tolist()
    evidence_ranks = [direct_order.index(index) + 1 for index in evidence]
    selected_tokens = sum(
        int(feature["chunk_spans"][index][1]) - int(feature["chunk_spans"][index][0])
        for index in selected
    )
    relational_gaps = []
    normalized_memory = F.normalize(memory.float(), dim=-1)
    for target in evidence:
        peers = evidence - {target}
        if peers:
            peer_score = max(
                float(normalized_memory[target] @ normalized_memory[peer]) for peer in peers
            )
            relational_gaps.append(peer_score - float(direct[target]))
    row = {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "method": method,
        "depth": depth,
        "fraction": fraction,
        "budget": budget,
        "alpha": alpha,
        "frontier": frontier,
        "path_score": path,
        "candidate_chunks": candidate_count,
        "evidence_chunks": len(evidence),
        "recovered_evidence_chunks": len(selected & evidence),
        "selected_chunk_ids": json.dumps(selected_ids),
        "evidence_chunk_ids": json.dumps(sorted(evidence_ids)),
        "selected_chunk_ids_by_hop": json.dumps(selected_by_hop),
        "selected_unique_chunks": len(selected),
        "selected_fraction": len(selected) / candidate_count,
        "any_evidence": float(bool(selected & evidence)),
        "all_evidence": float(bool(evidence) and evidence <= selected),
        "chain_completion": float(bool(groups) and all(bool(selected & group) for group in groups)),
        "evidence_coverage": len(selected & evidence) / max(len(evidence), 1),
        "hop1_recall": float(bool(groups) and bool(selected & groups[0])),
        "hop2_recall": float(len(groups) > 1 and bool(selected & groups[1])) if len(groups) > 1 else None,
        "direct_mrr": 1.0 / min(evidence_ranks) if evidence_ranks else 0.0,
        "exact_all_feasible": float(len(evidence) <= budget),
        "chain_feasible": float(len(groups) <= budget),
        "oracle_exact_all": float(len(evidence) <= budget),
        "oracle_chain_completion": float(len(groups) <= budget),
        "materialized_kv_tokens": selected_tokens,
        "materialized_kv_fraction": selected_tokens / max(int(feature["source_tokens"]), 1),
        "routing_seconds": elapsed,
        "gist_comparisons": result.graph.costs.get("unique_gist_comparisons", 0),
        "candidate_proposals": result.graph.costs.get("candidate_proposals", 0),
        "duplicate_proposals": result.graph.costs.get("duplicate_proposals", 0),
        "candidate_overlap": result.graph.costs.get("candidate_overlap_mean", 0.0),
        "unique_parents": result.graph.costs.get("unique_parents", 0),
        "relational_gap": statistics.fmean(relational_gaps) if relational_gaps else None,
        "stop_reason": result.graph.stop_reason,
    }
    return row, result.graph.to_dict()


def _aggregate(rows):
    keys = (
        "any_evidence", "all_evidence", "chain_completion", "evidence_coverage", "hop1_recall",
        "hop2_recall", "direct_mrr", "selected_unique_chunks", "selected_fraction",
        "exact_all_feasible", "chain_feasible", "oracle_exact_all", "oracle_chain_completion",
        "materialized_kv_tokens", "materialized_kv_fraction", "routing_seconds",
        "gist_comparisons", "candidate_proposals", "duplicate_proposals",
        "candidate_overlap", "unique_parents", "relational_gap",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["depth"], row["fraction"], row["alpha"], row["frontier"], row["path_score"])].append(row)
    output = []
    for condition, values in sorted(groups.items(), key=str):
        record = dict(zip(("dataset", "method", "depth", "fraction", "alpha", "frontier", "path_score"), condition))
        record["examples_x_seeds"] = len(values)
        for key in keys:
            record[key] = _mean(values, key)
        output.append(record)
    return output


def _seed_summary(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[(row["dataset"], row["method"], row["depth"], row["fraction"], row["alpha"], row["frontier"], row["path_score"], row["seed"])].append(row)
    condition = defaultdict(list)
    for key, values in by_seed.items():
        condition[key[:-1]].append({
            "seed": key[-1],
            "any_evidence": _mean(values, "any_evidence"),
            "all_evidence": _mean(values, "all_evidence"),
            "chain_completion": _mean(values, "chain_completion"),
            "evidence_coverage": _mean(values, "evidence_coverage"),
        })
    output = []
    for key, seed_rows in sorted(condition.items(), key=str):
        row = dict(zip(("dataset", "method", "depth", "fraction", "alpha", "frontier", "path_score"), key))
        row["seeds"] = len(seed_rows)
        for metric in ("any_evidence", "all_evidence", "chain_completion", "evidence_coverage"):
            values = [value[metric] for value in seed_rows]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = _ci95(values)
        output.append(row)
    return output


def _auc(rows, metric):
    points = sorted((float(row["fraction"]), float(row[metric])) for row in rows)
    return sum((right[0] - left[0]) * (right[1] + left[1]) / 2 for left, right in zip(points, points[1:])) / 0.25


def _threshold(rows, metric, target):
    for row in sorted(rows, key=lambda value: value["fraction"]):
        if row[metric] >= target:
            return row["fraction"]
    return None


def _curve_summary(aggregates):
    groups = defaultdict(list)
    for row in aggregates:
        groups[(row["dataset"], row["method"], row["depth"], row["alpha"], row["frontier"], row["path_score"])].append(row)
    return [
        {
            "dataset": key[0], "method": key[1], "depth": key[2], "alpha": key[3],
            "frontier": key[4], "path_score": key[5],
            "any_auc_5_30": _auc(values, "any_evidence"),
            "all_auc_5_30": _auc(values, "all_evidence"),
            "chain_auc_5_30": _auc(values, "chain_completion"),
            "any_f80": _threshold(values, "any_evidence", 0.8),
            "any_f90": _threshold(values, "any_evidence", 0.9),
            "all_f80": _threshold(values, "all_evidence", 0.8),
            "all_f90": _threshold(values, "all_evidence", 0.9),
            "chain_f80": _threshold(values, "chain_completion", 0.8),
            "chain_f90": _threshold(values, "chain_completion", 0.9),
        }
        for key, values in sorted(groups.items(), key=str)
    ]


def _synthetic(seed: int, hops: int, examples: int, width: int, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for example in range(examples):
        basis, _ = torch.linalg.qr(torch.randn(width, width, generator=generator))
        root = basis[0]
        # The root can enter through A (cos=0.8), while B/C are increasingly
        # weak direct matches but retain a 0.6 edge to their predecessor.
        chain = [F.normalize(0.8 * root + 0.6 * basis[1], dim=0)]
        for hop in range(1, hops):
            chain.append(
                F.normalize(0.6 * chain[-1] + 0.8 * basis[hop + 1], dim=0)
            )
        # Distractors outrank indirect B/C from the root but not the next true
        # edge from a chain frontier.  One-shot and closure receive the same B.
        distractors = [
            F.normalize(0.7 * root + math.sqrt(1.0 - 0.7**2) * basis[hops + 1 + i], dim=0)
            for i in range(8)
        ]
        memory = torch.stack([*chain, *distractors]).to(device)
        feature = {
            "dataset": f"synthetic_{hops}hop", "example_id": f"s{seed}-{hops}-{example}",
            "positive_mask": torch.tensor([True] * hops + [False] * len(distractors)),
            "chunk_spans": [(i, i + 1) for i in range(len(memory))], "source_tokens": len(memory),
        }
        for method, depth in (("one_shot", 1), ("iterative", hops)):
            row, _ = _evaluate_one(
                feature, root.to(device), memory, seed=seed, method=method, depth=depth,
                fraction=hops / len(memory), alpha=0.0, frontier="direct", path="product",
            )
            rows.append(row)
    return rows


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plots(aggregates, output_dir):
    for dataset in ("hotpotqa", "qasper"):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        conditions = [
            ("One-shot Top-B", "one_shot", 1, 1.0, "direct", "direct"),
            ("Closure D=2", "iterative", 2, 0.25, "direct", "product"),
            ("Closure D=3", "iterative", 3, 0.25, "direct", "product"),
            ("Closure D=4", "iterative", 4, 0.25, "direct", "product"),
        ]
        for label, method, depth, alpha, frontier, path in conditions:
            rows = [row for row in aggregates if row["dataset"] == dataset and row["method"] == method and row["depth"] == depth and row["alpha"] == alpha and row["frontier"] == frontier and row["path_score"] == path]
            rows.sort(key=lambda row: row["fraction"])
            if not rows:
                continue
            axes[0].plot([100 * row["fraction"] for row in rows], [row["any_evidence"] for row in rows], marker="o", label=label)
            axes[1].plot([100 * row["fraction"] for row in rows], [row["chain_completion"] for row in rows], marker="o", label=label)
        axes[0].set_title("Any evidence")
        axes[1].set_title("Evidence-group chain completion")
        for axis in axes:
            axis.set_xlabel("Final unique-chunk budget (%)")
            axis.set_ylabel("Recall")
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        figure.suptitle(f"{dataset}: matched-budget associative closure")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{dataset}_closure_recall.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)


def run(args):
    device = torch.device(args.device)
    features = torch.load(args.feature_dir / "router_features_test.pt", weights_only=False)
    rows, graphs = [], []
    for seed in args.seeds:
        checkpoint = args.feature_dir / "checkpoints" / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature in features:
            with torch.no_grad():
                root = projection.project_query(feature["queries"]["last"].to(device).unsqueeze(0))[0]
                memory = projection.project_memory(feature["memory_gists"].to(device))
            # Main matched-budget curves.
            for fraction in args.fractions:
                row, graph = _evaluate_one(feature, root, memory, seed=seed, method="one_shot", depth=1, fraction=fraction, alpha=1.0, frontier="direct", path="direct")
                rows.append(row)
                if fraction == 0.10:
                    graphs.append(graph)
                for depth in args.depths:
                    if depth == 1:
                        continue
                    row, graph = _evaluate_one(feature, root, memory, seed=seed, method="iterative", depth=depth, fraction=fraction, alpha=0.25, frontier="direct", path="product")
                    rows.append(row)
                    if fraction == 0.10 and depth == 2:
                        graphs.append(graph)
            # Targeted ablations use 10% and D=2 to bound combinatorics.
            for alpha in ALPHAS:
                if alpha == 0.25:
                    continue
                row, _ = _evaluate_one(feature, root, memory, seed=seed, method="iterative", depth=2, fraction=0.10, alpha=alpha, frontier="direct", path="product")
                rows.append(row)
            for frontier in FRONTIERS:
                if frontier == "direct":
                    continue
                row, _ = _evaluate_one(feature, root, memory, seed=seed, method="iterative", depth=2, fraction=0.10, alpha=0.25, frontier=frontier, path="product")
                rows.append(row)
            for path in PATH_MODES:
                if path == "product":
                    continue
                row, _ = _evaluate_one(feature, root, memory, seed=seed, method="iterative", depth=2, fraction=0.10, alpha=0.25, frontier="direct", path=path)
                rows.append(row)
        print(f"seed {seed}: {len(rows)} rows", flush=True)
    synthetic = []
    for seed in args.seeds:
        synthetic.extend(_synthetic(seed, 2, args.synthetic_examples, 16, device))
        synthetic.extend(_synthetic(seed, 3, args.synthetic_examples, 16, device))
    rows.extend(synthetic)
    aggregates = _aggregate(rows)
    seed_summary = _seed_summary(rows)
    curves = _curve_summary([row for row in aggregates if row["dataset"] in {"hotpotqa", "qasper"} and (row["method"] == "one_shot" or (row["alpha"] == 0.25 and row["frontier"] == "direct" and row["path_score"] == "product"))])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "runtime": runtime_metadata(), "seeds": list(args.seeds), "fractions": list(args.fractions),
        "depths": list(args.depths), "primary_alpha": 0.25,
        "feature_artifact": str((args.feature_dir / "router_features_test.pt").relative_to(ROOT)),
        "protocol_notes": [
            "Each seed is an independently trained semantic projection evaluated on the same held-out examples.",
            "All-evidence requires every evidence-overlapping chunk; hop1/hop2 are source-ordered contiguous evidence groups, not annotated causal hop order.",
            "Chain completion requires at least one selected chunk from every contiguous evidence group; oracle feasibility reports whether the final chunk budget can represent the target.",
            "Closure and one-shot use the same final unique-chunk cap; actual selected counts and K/V-token fractions are reported.",
            "No transformer rerun occurs during closure; downstream generation is left to the existing full-native-K/V integration gate.",
        ],
        "rows": rows, "aggregates": aggregates, "seed_summary": seed_summary, "curve_summary": curves,
    }
    (args.output_dir / "iterative_closure_results.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output_dir / "iterative_closure_rows.csv", rows)
    _write_csv(args.output_dir / "iterative_closure_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "iterative_closure_seed_summary.csv", seed_summary)
    _write_csv(args.output_dir / "iterative_closure_curve_summary.csv", curves)
    with (args.output_dir / "retrieval_graphs.jsonl").open("w", encoding="utf-8") as stream:
        for graph in graphs:
            stream.write(json.dumps(graph, sort_keys=True) + "\n")
    _plots(aggregates, args.output_dir)
    return artifact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    parser.add_argument("--synthetic-examples", type=int, default=64)
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra")
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    args.fractions = tuple(float(value) for value in args.fractions.split(","))
    args.depths = tuple(int(value) for value in args.depths.split(","))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"rows": len(result["rows"]), "aggregates": len(result["aggregates"])}, indent=2))
