"""Audit all-offset multiscale query visibility on frozen natural graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
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
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import (
    PRIMARY_CHUNK,
    _atomic_native,
    _feature_example,
    _node_recovery,
    _parent_hidden,
    _search,
    _selected_token_metrics,
)
from experiments.paper2_5_iterative_pra.run_oracle_convergence import SEEDS
from pra_hf.cross_dataset_diagnostics import (
    all_offset_multiscale_facets,
    best_group_facet_rank,
    bounded_multiscale_candidates,
    group_rank,
    token_jaccard_parent_scores,
)
from pra_hf.natural_reasoning_graph import load_2wiki, load_musique, map_example_to_parents
from pra_hf.query_facets import score_semantic_query_facets
from pra_hf.semantic_graph_search import build_native_parent_adjacency
from pra_torch.hf import load_hf_routing_projection


SCALES = (1, 2, 4, 8, 16)
RECALL_K = (1, 2, 4, 8)
PRACTICAL_ROOT_BUDGETS = (1, 2, 4, 8)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale(provenance) -> str:
    return "global" if provenance.kind == "global" else str(int(provenance.scale))


def _decoded_span(tokenizer, feature: dict, provenance) -> str:
    if provenance.kind == "global":
        ids = feature["question_input_ids"]
    else:
        ids = feature["prompt_input_ids"][provenance.token_start : provenance.token_end]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


def _node_depths(example) -> dict[str, int]:
    depths = {}
    for node in example.nodes:
        depths[node.node_id] = 1 + max(
            (depths.get(parent, 0) for parent in node.dependencies), default=0
        )
    return depths


def _target_roles(example, node_id: str, depths: dict[str, int]) -> tuple[str, ...]:
    outgoing = {source for source, _ in example.annotated_edges}
    roles = []
    if node_id in example.root_node_ids:
        roles.append("root")
    if node_id not in example.root_node_ids and node_id in outgoing:
        roles.append("intermediate")
    if depths[node_id] > 1 and node_id not in outgoing:
        roles.append("terminal")
    return tuple(roles or ("intermediate",))


def _load_sources(args) -> dict[str, object]:
    examples = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    return {example.example_id: example for example in examples}


def _current_routed(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _example_best(rows: list[dict], role: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["role"] == role:
            grouped[(row["dataset"], row["example_id"], row["partition"], row["seed"])].append(row)
    return [
        {
            "dataset": key[0],
            "example_id": key[1],
            "partition": key[2],
            "seed": key[3],
            "role": role,
            "semantic_rank": min(int(row["semantic_best_rank"]) for row in values),
            "lexical_rank": min(int(row["lexical_rank"]) for row in values),
            "winning_scale": min(values, key=lambda row: int(row["semantic_best_rank"]))[
                "winning_scale"
            ],
        }
        for key, values in grouped.items()
    ]


def _aggregate(target_rows, false_rows, scale_rows, practical_rows, routed_rows) -> dict:
    root = _example_best(target_rows, "root")
    terminal = _example_best(target_rows, "terminal")
    routing_ceiling, visibility, false_summary, scale_summary, practical = [], [], [], [], []
    for dataset in ("musique", "2wikimultihopqa"):
        root_test = [row for row in root if row["dataset"] == dataset and row["partition"] == "test"]
        terminal_test = [
            row for row in terminal if row["dataset"] == dataset and row["partition"] == "test"
        ]
        current = [
            row
            for row in routed_rows
            if row["dataset"] == dataset
            and row["partition"] == "test"
            and int(row["R_root"]) == 4
        ]
        scale_counts = Counter(row["winning_scale"] for row in root_test)
        routing_ceiling.append(
            {
                "dataset": dataset,
                "current_routed_R_at_4": _mean(current, "correct_root_present"),
                **{
                    f"oracle_multiscale_root_R_at_{k}": statistics.fmean(
                        int(row["semantic_rank"]) <= k for row in root_test
                    )
                    for k in RECALL_K
                },
                "best_root_scale": scale_counts.most_common(1)[0][0],
                "terminal_oracle_multiscale_R_at_4": (
                    statistics.fmean(int(row["semantic_rank"]) <= 4 for row in terminal_test)
                    if terminal_test
                    else None
                ),
                "lexical_root_R_at_4": statistics.fmean(
                    int(row["lexical_rank"]) <= 4 for row in root_test
                ),
                "lexical_terminal_R_at_4": (
                    statistics.fmean(int(row["lexical_rank"]) <= 4 for row in terminal_test)
                    if terminal_test
                    else None
                ),
                "root_rows": len(root_test),
                "terminal_rows": len(terminal_test),
            }
        )
        for role in ("root", "intermediate", "terminal"):
            rows = [
                row
                for row in target_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["role"] == role
            ]
            if not rows:
                continue
            visibility.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "mean_annotated_step": _mean(rows, "annotated_step"),
                    **{
                        f"semantic_R_at_{k}": statistics.fmean(
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
                    "rows": len(rows),
                }
            )
        negatives = [
            row
            for row in false_rows
            if row["dataset"] == dataset and row["partition"] == "test"
        ]
        false_summary.append(
            {
                "dataset": dataset,
                "false_parent_best_rank_mean": _mean(negatives, "best_rank"),
                "false_parent_R_at_1": statistics.fmean(int(row["best_rank"]) <= 1 for row in negatives),
                "false_parent_R_at_4": statistics.fmean(int(row["best_rank"]) <= 4 for row in negatives),
                "rows": len(negatives),
            }
        )
        for scale in ("global", "1", "2", "4", "8", "16"):
            rows = [
                row
                for row in scale_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["scale"] == scale
            ]
            if rows:
                scale_summary.append(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "mean_score": _mean(rows, "mean_score"),
                        "mean_scale_max": _mean(rows, "scale_max"),
                        "facet_rows": sum(int(row["facet_count"]) for row in rows),
                    }
                )
        for budget in PRACTICAL_ROOT_BUDGETS:
            rows = [
                row
                for row in practical_rows
                if row["dataset"] == dataset
                and row["partition"] == "test"
                and row["root_budget"] == budget
            ]
            if rows:
                practical.append(
                    {
                        "dataset": dataset,
                        "root_budget": budget,
                        "root_recall": _mean(rows, "correct_root_present"),
                        "complete_recovery": _mean(rows, "complete_graph"),
                        "node_recall": _mean(rows, "oracle_node_recall"),
                        "candidate_count": _mean(rows, "candidate_count"),
                        "selected_source_fraction": _mean(rows, "active_kv_fraction"),
                        "selection_ms": 1000 * _mean(rows, "selection_seconds"),
                    }
                )
    validation_decisions = {}
    for row in routing_ceiling:
        dataset = row["dataset"]
        validation_roots = [
            item for item in root if item["dataset"] == dataset and item["partition"] == "validation"
        ]
        current_validation = [
            item
            for item in routed_rows
            if item["dataset"] == dataset
            and item["partition"] == "validation"
            and int(item["R_root"]) == 4
        ]
        ceiling = statistics.fmean(int(item["semantic_rank"]) <= 4 for item in validation_roots)
        baseline = _mean(current_validation, "correct_root_present")
        validation_decisions[dataset] = {
            "oracle_multiscale_R_at_4": ceiling,
            "current_routed_R_at_4": baseline,
            "absolute_gain": ceiling - baseline,
            "run_bounded_router": ceiling - baseline >= 0.10,
            "criterion": "validation-only oracle ceiling gain >= 0.10",
        }
    return {
        "routing_ceiling": routing_ceiling,
        "visibility_by_role": visibility,
        "false_parent_multiple_comparison": false_summary,
        "score_distribution_by_scale": scale_summary,
        "bounded_multiscale_router": practical,
        "validation_decisions": validation_decisions,
    }


def _plots(aggregate: dict, output_dir: Path) -> None:
    ceiling = aggregate["routing_ceiling"]
    visibility = aggregate["visibility_by_role"]
    practical = aggregate["bounded_multiscale_router"]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    labels = ["MuSiQue", "2Wiki"]
    x = torch.arange(2).numpy()
    axes[0].bar(x - 0.25, [row["current_routed_R_at_4"] for row in ceiling], 0.25, label="Current")
    axes[0].bar(x, [row["oracle_multiscale_root_R_at_4"] for row in ceiling], 0.25, label="Oracle facet")
    axes[0].bar(x + 0.25, [row["lexical_root_R_at_4"] for row in ceiling], 0.25, label="Lexical")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Root recall at 4", ylim=(0, 1.03))
    axes[0].legend(frameon=False, fontsize=8)
    for dataset, label in (("musique", "MuSiQue"), ("2wikimultihopqa", "2Wiki")):
        rows = [row for row in visibility if row["dataset"] == dataset]
        axes[1].plot(
            [row["role"] for row in rows],
            [row["semantic_R_at_4"] for row in rows],
            marker="o",
            label=label,
        )
        selected = [row for row in practical if row["dataset"] == dataset]
        if selected:
            axes[2].plot(
                [row["selected_source_fraction"] for row in selected],
                [row["complete_recovery"] for row in selected],
                marker="o",
                label=label,
            )
    axes[1].set(ylabel="Oracle-facet recall at 4", ylim=(0, 1.03))
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set(xlabel="Selected source fraction", ylabel="Complete recovery", ylim=(0, 1.03))
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"natural_multiscale_query_audit.{suffix}", dpi=180)
    plt.close(figure)


def run(args) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_file, map_location="cpu", weights_only=False)
    sources = _load_sources(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    projections = {
        seed: load_hf_routing_projection(
            args.projection_dir
            / "checkpoints"
            / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt",
            device=device,
        )
        for seed in args.seeds
    }
    routed_rows = _current_routed(args.routed_file)
    target_rows, false_rows, scale_rows, prepared, cache = [], [], [], [], []
    for index, feature in enumerate(features, start=1):
        example = _feature_example(feature)
        source_example = sources[feature["example_id"]]
        source_ids = tokenizer(source_example.source, add_special_tokens=False).input_ids
        if len(source_ids) != int(feature["source_tokens"]):
            raise ValueError(f"Source-token alignment failed for {feature['example_id']}.")
        mapping = map_example_to_parents(
            example,
            int(feature["source_tokens"]),
            feature["node_token_spans"],
            chunk_size=PRIMARY_CHUNK,
        )
        parent_hidden = _parent_hidden(feature["token_hidden"], mapping.parent_spans)
        facets = all_offset_multiscale_facets(
            feature["query_hidden_states"].float(),
            tuple(feature["question_span"]),
            scales=SCALES,
        )
        decoded = tuple(_decoded_span(tokenizer, feature, row) for row in facets.provenance)
        lexical = token_jaccard_parent_scores(
            feature["question_input_ids"].tolist(), source_ids, mapping.parent_spans
        )
        q, k, mask, local_parents = _atomic_native(feature, PRIMARY_CHUNK)
        adjacency = build_native_parent_adjacency(
            q.to(device),
            k.to(device),
            mask.to(device),
            local_parents.to(device),
            len(mapping.parent_spans),
            token_reduction="top_m_mean",
            head_reduction="top_m_mean",
            top_m=4,
        ).scores.detach().cpu()
        depths = _node_depths(example)
        scores_by_seed = {}
        for seed, projection in projections.items():
            started = time.perf_counter()
            scored = score_semantic_query_facets(
                projection.project_query(facets.hidden.to(device)),
                projection.project_memory(parent_hidden.to(device)),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            score_seconds = time.perf_counter() - started
            scores = scored.component_scores[:, 0, :].detach().cpu()
            scores_by_seed[seed] = scores
            for scale in ("global", "1", "2", "4", "8", "16"):
                facet_ids = [
                    facet_id
                    for facet_id, row in enumerate(facets.provenance)
                    if _scale(row) == scale
                ]
                if not facet_ids:
                    continue
                values = scores[facet_ids].float()
                scale_rows.append(
                    {
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "partition": feature["partition"],
                        "seed": seed,
                        "scale": scale,
                        "facet_count": len(facet_ids),
                        "mean_score": float(values.mean()),
                        "score_std": float(values.std(unbiased=False)),
                        "scale_max": float(values.max()),
                    }
                )
            oracle_parents = set(mapping.oracle_parent_ids)
            for node in example.nodes:
                group = mapping.node_parent_groups.get(node.node_id, ())
                if not group:
                    continue
                ranked = best_group_facet_rank(scores, group)
                provenance = facets.provenance[ranked.facet_index]
                for role in _target_roles(example, node.node_id, depths):
                    target_rows.append(
                        {
                            "dataset": feature["dataset"],
                            "example_id": feature["example_id"],
                            "partition": feature["partition"],
                            "question_type": feature["question_type"],
                            "annotated_hops": feature["annotated_hops"],
                            "graph_type": feature["graph_type"],
                            "seed": seed,
                            "node_id": node.node_id,
                            "role": role,
                            "annotated_step": depths[node.node_id],
                            "semantic_best_rank": ranked.rank,
                            "lexical_rank": group_rank(lexical, group),
                            "winning_facet": ranked.facet_index,
                            "winning_scale": _scale(provenance),
                            "winning_token_start": provenance.token_start,
                            "winning_token_end": provenance.token_end,
                            "winning_question_offset": max(
                                0, provenance.token_start - int(feature["question_span"][0])
                            ),
                            "winning_decoded_span": decoded[ranked.facet_index],
                            "target_score": ranked.target_score,
                            "best_distractor_score": ranked.best_distractor_score,
                            "oracle_margin": ranked.margin,
                            "facet_count": scores.shape[0],
                            "score_seconds": score_seconds,
                        }
                    )
            for parent in range(len(mapping.parent_spans)):
                if parent in oracle_parents:
                    continue
                rank = best_group_facet_rank(scores, (parent,))
                false_rows.append(
                    {
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "partition": feature["partition"],
                        "seed": seed,
                        "parent_id": parent,
                        "best_rank": rank.rank,
                        "winning_scale": _scale(facets.provenance[rank.facet_index]),
                        "best_score": rank.target_score,
                    }
                )
        cache.append(
            {
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "partition": feature["partition"],
                "chunk_size": PRIMARY_CHUNK,
                "facet_provenance": [row.__dict__ for row in facets.provenance],
                "decoded_spans": decoded,
                "scores_by_seed": {seed: scores.half() for seed, scores in scores_by_seed.items()},
            }
        )
        prepared.append(
            {
                "feature": feature,
                "mapping": mapping,
                "edge_scores": adjacency,
                "scores_by_seed": scores_by_seed,
            }
        )
        print(f"[multiscale {index}/{len(features)}] {feature['dataset']} {feature['example_id']}", flush=True)

    provisional = _aggregate(target_rows, false_rows, scale_rows, [], routed_rows)
    practical_rows = []
    for item in prepared:
        feature, mapping = item["feature"], item["mapping"]
        if not provisional["validation_decisions"][feature["dataset"]]["run_bounded_router"]:
            continue
        for seed, scores in item["scores_by_seed"].items():
            for root_budget in PRACTICAL_ROOT_BUDGETS:
                started = time.perf_counter()
                roots, candidates = bounded_multiscale_candidates(
                    scores, proposal_width=1, global_budget=root_budget
                )
                selection_seconds = time.perf_counter() - started
                result = _search(item["edge_scores"], roots, 6, 4, 16)
                recall, complete, _ = _node_recovery(result.visited, mapping)
                practical_rows.append(
                    {
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "partition": feature["partition"],
                        "seed": seed,
                        "proposal_width_per_facet": 1,
                        "root_budget": root_budget,
                        "candidate_count": candidates,
                        "correct_root_present": int(
                            bool(set(roots).intersection(mapping.root_parent_ids))
                        ),
                        "oracle_node_recall": recall,
                        "complete_graph": int(complete),
                        "visited_count": len(result.visited),
                        "selection_seconds": selection_seconds,
                        "search_seconds": result.search_seconds,
                        **_selected_token_metrics(result.visited, mapping, feature),
                    }
                )
    aggregate = _aggregate(target_rows, false_rows, scale_rows, practical_rows, routed_rows)
    gate1 = json.loads(args.gate1_file.read_text(encoding="utf-8"))["aggregate"]
    unified = []
    for row in gate1["recommended_operating_points"]:
        unified.append(
            {
                "dataset": row["dataset"],
                "entry_mode": "oracle_annotated_entry",
                "operating_point": row["operating_point"],
                "R_root": "oracle_group_count",
                "chunk_size": row["chunk_size"],
                "native_K": row["K"],
                "B": row["B"],
                "complete_recovery": row["complete_recovery"],
                "node_recall": row["node_recall"],
                "root_recall": 1.0,
                "selected_source_fraction": row["selected_source_fraction"],
                "oracle_entry": True,
            }
        )
    for row in aggregate["bounded_multiscale_router"]:
        unified.append(
            {
                "dataset": row["dataset"],
                "entry_mode": "bounded_multiscale_union",
                "operating_point": "executable_diagnostic",
                "R_root": row["root_budget"],
                "chunk_size": PRIMARY_CHUNK,
                "native_K": 6,
                "B": 16,
                "complete_recovery": row["complete_recovery"],
                "node_recall": row["node_recall"],
                "root_recall": row["root_recall"],
                "selected_source_fraction": row["selected_source_fraction"],
                "oracle_entry": False,
            }
        )
    aggregate["unified_sparse_recovery"] = unified
    aggregate["next_architecture_gate"] = {
        "graph_representation": (
            "retain a contextual 128-token discovery representation while exposing "
            "16/32-token payload identities; do not train search control around missing fine edges"
        ),
        "entry_routing": (
            "learn or calibrate one bounded facet selector/competition rule; the high oracle "
            "ceiling and failed max-union router do not justify more unbounded facets"
        ),
        "materialization_boundary": (
            "keep conceptual parent selection separate from native-KV materialization for Paper 3"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "natural_multiscale_query_facet_cache.pt"
    torch.save(cache, cache_path)
    for name, rows in (
        ("multiscale_target_ranks.csv", target_rows),
        ("multiscale_false_parent_ranks.csv", false_rows),
        ("multiscale_scale_scores.csv", scale_rows),
        ("bounded_multiscale_router_rows.csv", practical_rows),
        ("routing_ceiling_table.csv", aggregate["routing_ceiling"]),
        ("query_visibility_by_role.csv", aggregate["visibility_by_role"]),
        ("false_parent_multiple_comparison.csv", aggregate["false_parent_multiple_comparison"]),
        ("multiscale_score_distribution.csv", aggregate["score_distribution_by_scale"]),
        ("bounded_multiscale_router_summary.csv", aggregate["bounded_multiscale_router"]),
        ("unified_sparse_recovery.csv", aggregate["unified_sparse_recovery"]),
    ):
        _write_csv(args.output_dir / name, rows)
    _plots(aggregate, args.output_dir)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "training_performed": False,
        "chunk_size": PRIMARY_CHUNK,
        "facet_scales": list(SCALES) + ["global"],
        "facet_stride": 1,
        "one_contextual_query_encoding": True,
        "independent_span_encoding": False,
        "oracle_labels_available_during_scoring_or_search": False,
        "oracle_evaluation_post_hoc": True,
        "bounded_router": {
            "proposal_width_per_facet": 1,
            "one_shared_global_root_budget": True,
            "per_facet_materialization_budget": False,
            "native_K": 6,
            "H": 4,
            "B": 16,
        },
        "cache": {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
            "tracked": False,
        },
        "aggregate": aggregate,
        "row_counts": {
            "target": len(target_rows),
            "false_parent": len(false_rows),
            "scale": len(scale_rows),
            "practical": len(practical_rows),
        },
    }
    (args.output_dir / "natural_multiscale_query_audit_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    data = ROOT / "data/.paper2_5_datasets"
    parser.add_argument("--musique-dev", type=Path, default=data / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=data / "2wiki/dev.json")
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth"
    parser.add_argument("--feature-file", type=Path, default=output / "natural_graph_features.pt")
    parser.add_argument("--routed-file", type=Path, default=output / "natural_graph_routed_rows.csv")
    parser.add_argument(
        "--gate1-file", type=Path, default=output / "natural_graph_depth_results.json"
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
