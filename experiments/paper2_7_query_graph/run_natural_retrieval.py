"""Matched-budget four-dataset query-graph retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_channel_geometry import (  # noqa: E402
    DATASETS,
    LABELS,
    WEIGHTS,
    _load_cases,
)
from experiments.paper2_6_hybrid_pra.run_study import _metrics, _records  # noqa: E402
from experiments.paper2_7_query_graph.helpers import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    git_metadata,
    resolve_artifact,
    write_csv,
    write_json,
)
from experiments.paper2_7_query_graph.run_algorithm_cross import EDGE_WEIGHTS  # noqa: E402
from pra_hf.adaptive_facets import (  # noqa: E402
    GraphFacetConfig,
    build_adaptive_query_facets,
)
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex  # noqa: E402
from pra_hf.iterative import IterativeGistRouter  # noqa: E402
from pra_hf.query_facets import (  # noqa: E402
    build_multiscale_query_facets,
    build_span_query_facets,
    deterministic_phrase_spans,
)
from pra_hf.query_graph import (  # noqa: E402
    QueryUnitProvenance,
    build_query_graph,
    graph_memory_bytes,
    lexical_feature_matrix,
)
from pra_hf.query_graph_cluster import (  # noqa: E402
    connected_components,
    deterministic_kmeans,
    weighted_label_propagation,
)
from pra_hf.query_graph_facets import pool_hard_graph_facets  # noqa: E402


CONDITIONS = (
    "global_semantic",
    "paper25_multiscale",
    "paper25_clause",
    "embedding_kmeans",
    "graph_cc",
    "graph_label_propagation",
    "syntactic_graph",
    "paper26_bm25",
    "paper26_hybrid",
    "graph_cc_hybrid",
    "graph_label_propagation_hybrid",
    "syntactic_graph_hybrid",
)
GRAPH_CONDITIONS = {
    "graph_cc",
    "graph_label_propagation",
    "graph_cc_hybrid",
    "graph_label_propagation_hybrid",
    "syntactic_graph",
    "syntactic_graph_hybrid",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest_hash(rows) -> str:
    identities = sorted(
        (str(row["dataset"]), str(row["partition"]), str(row["example_id"]))
        for row in rows
    )
    return hashlib.sha256(json.dumps(identities).encode("utf-8")).hexdigest()


def _query_state_map(query_entry_path: Path, natural_path: Path):
    entry_rows = torch.load(query_entry_path, map_location="cpu", weights_only=False)
    natural_rows = torch.load(
        natural_path, map_location="cpu", weights_only=False, mmap=True
    )
    mapping = {}
    for row in (*entry_rows, *natural_rows):
        if str(row.get("partition", "test")) != "test":
            continue
        mapping[(str(row["dataset"]), str(row["example_id"]))] = {
            "query_hidden_states": row["query_hidden_states"],
            "question_span": tuple(map(int, row["question_span"])),
            "prompt_input_ids": row["prompt_input_ids"],
            "query_pre_query": row.get("query_pre_query"),
        }
    return mapping, entry_rows, natural_rows


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _topk(scores: torch.Tensor, budget: int) -> list[int]:
    return IterativeGistRouter._topk(scores, budget)


def _semantic_scores(facets: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    if facets.ndim != 2 or memory.ndim != 2 or facets.shape[1] != memory.shape[1]:
        raise ValueError("Query facets and memory gists must be aligned rank-two tensors.")
    facets = torch.nn.functional.normalize(facets.float(), dim=-1, eps=1e-12)
    memory = torch.nn.functional.normalize(memory.float(), dim=-1, eps=1e-12)
    return torch.einsum("fd,cd->fc", facets, memory).max(dim=0).values


def _hybrid_selection(token_index, tokenizer, query_ids, semantic, mode, budget):
    candidates = token_index.score(
        query_ids,
        semantic.cpu(),
        tokenizer,
        HybridDiscoveryPolicy(
            mode=mode,
            semantic_weight=WEIGHTS[0],
            token_weight=1.0 - WEIGHTS[0],
        ),
        hop=1,
        parent_id="__root__",
    )
    return [index for index, _ in sorted(enumerate(candidates), key=lambda row: (row[1].rank, row[0]))[:budget]]


def _graph_facets(hidden, prompt_ids, span, tokenizer, policy, method, device):
    start, end = span
    support = hidden[start:end].float().to(device)
    token_ids = prompt_ids[start:end].tolist()
    texts = [str(value) for value in tokenizer.convert_ids_to_tokens(token_ids)]
    lexical = lexical_feature_matrix(texts, buckets=256, device=device)
    provenance = tuple(
        QueryUnitProvenance(
            unit_id=start + index,
            token_start=start + index,
            token_end=start + index + 1,
            text=text,
            layer=27,
        )
        for index, text in enumerate(texts)
    )
    alpha, beta, delta = EDGE_WEIGHTS[policy["edge_family"]]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    graph_started = time.perf_counter()
    graph = build_query_graph(
        support,
        lexical_features=lexical,
        provenance=provenance,
        contextual_weight=alpha,
        lexical_weight=beta,
        position_weight=delta,
        top_k=int(policy["top_k"]),
        threshold=float(policy["threshold"]),
        policy=str(policy["policy"]),
    )
    _synchronize(device)
    graph_ms = (time.perf_counter() - graph_started) * 1000.0
    cluster_started = time.perf_counter()
    result = (
        connected_components(graph)
        if method == "cc"
        else weighted_label_propagation(graph)
    )
    facets = pool_hard_graph_facets(
        graph, support, result, include_global=False
    )
    _synchronize(device)
    cluster_ms = (time.perf_counter() - cluster_started) * 1000.0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return facets, result, graph, graph_ms, cluster_ms, int(peak)


def _fixed_facets(hidden, prompt_ids, span, tokenizer, family, device):
    hidden = hidden.float().to(device)
    if family == "multiscale":
        return build_multiscale_query_facets(
            hidden, span, windows=(2, 4, 8, 16), include_global=False
        ).hidden
    token_texts = [str(value) for value in tokenizer.convert_ids_to_tokens(prompt_ids.tolist())]
    spans = deterministic_phrase_spans(token_texts, span, neighborhood=4)
    if not spans:
        spans = ((span[0], span[1], "question"),)
    return build_span_query_facets(hidden, spans, include_global=False).hidden


def _hierarchical_facets(hidden, prompt_ids, span, tokenizer, policy, device):
    """Build the request/reply syntactic-to-graph treatment from one query pass."""

    token_texts = [str(value) for value in tokenizer.convert_ids_to_tokens(prompt_ids.tolist())]
    config = GraphFacetConfig(
        similarity_mode="contextual",
        threshold=float(policy["threshold"]),
        top_k=int(policy["top_k"]),
        graph_policy=str(policy["policy"]),
        cluster_method="connected_components",
    )
    return build_adaptive_query_facets(
        hidden.float().to(device),
        token_texts,
        mode="syntactic_graph",
        support_span=tuple(map(int, span)),
        coarse_partition_mode="clause",
        graph_config=config,
    )


def _condition_summary(rows):
    output = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition]
            output.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "examples": len(group),
                    "evidence_recall": sum(row["evidence_recall"] for row in group) / len(group),
                    "precision": sum(row["precision"] for row in group) / len(group),
                    "mrr": sum(row["mrr"] for row in group) / len(group),
                    "path_completion": sum(row["path_completion"] for row in group) / len(group),
                    "requested_chunks": sum(row["requested_chunks"] for row in group) / len(group),
                    "requested_token_budget": sum(row["requested_token_budget"] for row in group) / len(group),
                    "mean_graph_ms": sum(row["graph_ms"] for row in group) / len(group),
                    "mean_cluster_ms": sum(row["cluster_ms"] for row in group) / len(group),
                    "mean_routing_ms": sum(row["routing_ms"] for row in group) / len(group),
                    "mean_graph_facets": sum(row["graph_facets"] for row in group) / len(group),
                    "mean_graph_edges": sum(row["graph_edges"] for row in group) / len(group),
                    "mean_graph_calls": sum(row["graph_calls"] for row in group) / len(group),
                    "mean_graph_density": sum(row["graph_density"] for row in group) / len(group),
                    "mean_facet_overlap": sum(row["facet_overlap"] for row in group) / len(group),
                    "mean_pairwise_similarity_evaluations": sum(
                        row["pairwise_similarity_evaluations"] for row in group
                    ) / len(group),
                }
            )
    return output


def _parity(rows, inherited_dir):
    inherited = []
    for dataset in DATASETS:
        stem = "2wiki" if dataset == "2wikimultihopqa" else dataset
        inherited.extend(_read_csv(inherited_dir / f"channel_results_{stem}.csv"))
    expected = {
        (row["dataset"], row["example_id"], row["channel"]): row
        for row in inherited
        if row["split"] == "test" and row["channel"] in {"gist", "hybrid"}
    }
    parity_rows = []
    for row in rows:
        channel = {"global_semantic": "gist", "paper26_hybrid": "hybrid"}.get(row["condition"])
        if channel is None:
            continue
        prior = expected[(row["dataset"], row["example_id"], channel)]
        prior_selected = tuple(value for value in prior["selected_chunk_ids"].split("|") if value)
        current_selected = tuple(value for value in row["selected_chunk_ids"].split("|") if value)
        parity_rows.append(
            {
                "dataset": row["dataset"],
                "example_id": row["example_id"],
                "condition": row["condition"],
                "recall_abs_error": abs(row["evidence_recall"] - float(prior["evidence_recall"])),
                "selection_exact_match": int(current_selected == prior_selected),
            }
        )
    return parity_rows


def _paired_effects(rows, *, seed: int = 20260822, draws: int = 10000):
    comparisons = (
        ("graph_cc", "paper25_multiscale"),
        ("graph_cc", "embedding_kmeans"),
        ("graph_cc_hybrid", "paper26_hybrid"),
        ("graph_label_propagation_hybrid", "paper26_hybrid"),
        ("syntactic_graph", "paper25_multiscale"),
        ("syntactic_graph_hybrid", "paper26_hybrid"),
    )
    rng = random.Random(seed)
    output = []
    by_key = {(row["dataset"], row["example_id"], row["condition"]): row for row in rows}
    for dataset in DATASETS:
        identities = sorted(
            {row["example_id"] for row in rows if row["dataset"] == dataset}
        )
        for treatment, baseline in comparisons:
            deltas = [
                by_key[(dataset, identity, treatment)]["evidence_recall"]
                - by_key[(dataset, identity, baseline)]["evidence_recall"]
                for identity in identities
            ]
            samples = sorted(
                sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                for _ in range(draws)
            )
            output.append(
                {
                    "dataset": dataset,
                    "treatment": treatment,
                    "baseline": baseline,
                    "examples": len(deltas),
                    "mean_recall_delta": sum(deltas) / len(deltas),
                    "ci95_low": samples[int(0.025 * draws)],
                    "ci95_high": samples[min(draws - 1, int(0.975 * draws))],
                    "positive_examples": sum(value > 0 for value in deltas),
                    "negative_examples": sum(value < 0 for value in deltas),
                    "zero_examples": sum(value == 0 for value in deltas),
                }
            )
    return output


def _plots(summary, effects, output):
    shown = (
        "global_semantic",
        "paper25_multiscale",
        "embedding_kmeans",
        "paper26_hybrid",
        "graph_cc",
        "graph_cc_hybrid",
    )
    colors = ["#59656f", "#d1495b", "#7a5195", "#e09f3e", "#2f6690", "#16817a"]
    fig, axis = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    width = 0.13
    for index, condition in enumerate(shown):
        values = [
            next(row["evidence_recall"] for row in summary if row["dataset"] == dataset and row["condition"] == condition)
            for dataset in DATASETS
        ]
        axis.bar(
            [position + (index - 2.5) * width for position in range(4)],
            values,
            width,
            label=condition.replace("_", " "),
            color=colors[index],
        )
    axis.set(
        xticks=range(4),
        xticklabels=[LABELS[dataset] for dataset in DATASETS],
        ylabel="Evidence recall",
        ylim=(0, 0.45),
        title="Held-out retrieval at four requested 32-token chunks",
    )
    axis.legend(ncols=2, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "natural_retrieval_recall.png", dpi=180)
    fig.savefig(output / "natural_retrieval_recall.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    gains = []
    errors = [[], []]
    for dataset in DATASETS:
        row = next(
            value for value in effects
            if value["dataset"] == dataset
            and value["treatment"] == "graph_cc_hybrid"
            and value["baseline"] == "paper26_hybrid"
        )
        gains.append(row["mean_recall_delta"])
        errors[0].append(row["mean_recall_delta"] - row["ci95_low"])
        errors[1].append(row["ci95_high"] - row["mean_recall_delta"])
    axis.bar(
        [LABELS[dataset] for dataset in DATASETS],
        gains,
        yerr=errors,
        capsize=3,
        color=["#2f6690" if value >= 0 else "#d1495b" for value in gains],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="Graph-CC hybrid - Paper 2.6 hybrid recall", title="Query-graph retrieval effect")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "natural_graph_gain.png", dpi=180)
    fig.savefig(output / "natural_graph_gain.pdf")
    plt.close(fig)


def run(args):
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    loader_args = argparse.Namespace(
        seed=args.cohort_seed,
        cache_dir=args.cache_dir,
        paper2_feature_dir=args.paper2_feature_dir,
        natural_features=args.natural_features,
        musique_dev=args.musique_dev,
        twowiki_dev=args.twowiki_dev,
    )
    cases = [row for row in _load_cases(loader_args) if row[0]["split"] == "test"]
    query_map, entry_rows, natural_rows = _query_state_map(
        args.query_entry_features, args.natural_features
    )
    if len(cases) != 74 or any((feature["dataset"], feature["example_id"]) not in query_map for feature, _ in cases):
        raise AssertionError("The frozen 74-identity held-out query-state cohort is incomplete.")

    rows = []
    assignments = []
    facet_method_rows = []
    for case_index, (feature, example) in enumerate(cases, 1):
        identity = (feature["dataset"], feature["example_id"])
        query = query_map[identity]
        hidden = query["query_hidden_states"]
        prompt_ids = query["prompt_input_ids"]
        span = query["question_span"]
        global_query = feature["queries"]["question_exp_h2.0"].float().to(device)
        query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
        index, positives = _records(feature, example)
        # GistIndex preserves the production lexicographic cache-identity order;
        # raw feature rows remain in numeric chunk order.
        memory = index.gists[:, 0].float().to(device)
        token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
        global_scores = _semantic_scores(global_query.unsqueeze(0), memory)

        graph_outputs = {}
        for method in ("cc", "label_propagation"):
            facets, clusters, graph, graph_ms, cluster_ms, peak = _graph_facets(
                hidden, prompt_ids, span, tokenizer, policy, method, device
            )
            _synchronize(device)
            score_started = time.perf_counter()
            facet_scores = _semantic_scores(
                torch.cat([global_query.unsqueeze(0), facets.hidden], dim=0), memory
            )
            _synchronize(device)
            score_ms = (time.perf_counter() - score_started) * 1000.0
            graph_outputs[method] = {
                "facets": facets,
                "clusters": clusters,
                "graph": graph,
                "graph_ms": graph_ms,
                "cluster_ms": cluster_ms,
                "peak": peak,
                "scores": facet_scores,
                "score_ms": score_ms,
            }
            assignments.append(
                {
                    "dataset": identity[0],
                    "example_id": identity[1],
                    "method": method,
                    "question_span_start": span[0],
                    "question_span_end": span[1],
                    "labels": clusters.labels.cpu().tolist(),
                    "member_unit_ids": [list(row.member_unit_ids) for row in facets.provenance],
                    "token_spans": [[list(value) for value in row.token_spans] for row in facets.provenance],
                }
            )

        hierarchy = _hierarchical_facets(
            hidden, prompt_ids, span, tokenizer, policy, device
        )
        hierarchy_metrics = hierarchy.metrics
        multiscale = _fixed_facets(hidden, prompt_ids, span, tokenizer, "multiscale", device)
        clause = _fixed_facets(hidden, prompt_ids, span, tokenizer, "clause", device)
        support = hidden[span[0] : span[1]].float().to(device)
        kmeans_count = max(1, min(6, round(math.sqrt(support.shape[0] / 2.0))))
        kmeans = deterministic_kmeans(support, kmeans_count)
        synthetic_graph = graph_outputs["cc"]["graph"]
        kmeans_facets = pool_hard_graph_facets(
            synthetic_graph, support, kmeans, include_global=False
        )

        # Measure every root channel independently for each hierarchical facet.
        # These rows support fixed, type-rule, learned, and evaluator-only oracle
        # per-facet policies without rerunning query encoding or memory scoring.
        hierarchy_nodes = [
            node for node in hierarchy.nodes if node.facet_id != hierarchy.root_id
        ]
        hierarchy_states = hierarchy.scoring_facets.hidden[1:]
        if len(hierarchy_nodes) != hierarchy_states.shape[0]:
            raise AssertionError("Hierarchical node/scoring provenance is misaligned.")
        for node, state in zip(hierarchy_nodes, hierarchy_states):
            node_scores = _semantic_scores(state.unsqueeze(0), memory)
            node_query_ids = [int(prompt_ids[index]) for index in node.token_indices]
            for root_method, discovery_mode in (
                ("semantic", None),
                ("exact", "token_exact"),
                ("bm25", "bm25"),
                ("hybrid", "token_semantic_rerank"),
            ):
                method_started = time.perf_counter()
                selected_indices = (
                    _topk(node_scores, args.budget)
                    if discovery_mode is None
                    else _hybrid_selection(
                        token_index,
                        tokenizer,
                        node_query_ids,
                        node_scores,
                        discovery_mode,
                        args.budget,
                    )
                )
                method_ms = (time.perf_counter() - method_started) * 1000.0
                selected_ids = [index.chunk_ids[value] for value in selected_indices]
                metric = _metrics(selected_ids, positives)
                facet_method_rows.append(
                    {
                        "split": "test",
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "facet_id": node.facet_id,
                        "facet_kind": node.kind,
                        "facet_type": node.facet_type,
                        "root_method": root_method,
                        "successor_method": {
                            "semantic": "native_semantic",
                            "exact": "exact_new_address",
                            "bm25": "bm25_state",
                            "hybrid": "hybrid_state",
                        }[root_method],
                        **metric,
                        "selected_chunk_ids": "|".join(selected_ids),
                        "positive_chunk_ids": "|".join(sorted(positives)),
                        "comparisons": len(index.chunk_ids),
                        "latency_ms": method_ms,
                        "token_count": node.lexical_features["token_count"],
                        "unique_token_count": node.lexical_features["unique_token_count"],
                        "rare_token_fraction": node.lexical_features["rare_token_fraction"],
                        "entity_count": node.lexical_features["entity_count"],
                        "relation_cue_count": node.lexical_features["relation_cue_count"],
                        "facet_confidence": node.confidence,
                        "component_nodes": node.graph_statistics.node_count,
                        "component_density": node.graph_statistics.density,
                        "component_mean_edge_weight": node.graph_statistics.mean_edge_weight,
                        "token_span_count": len(node.token_spans),
                    }
                )
        score_inputs = {
            "global_semantic": global_query.unsqueeze(0),
            "paper25_multiscale": torch.cat([global_query.unsqueeze(0), multiscale]),
            "paper25_clause": torch.cat([global_query.unsqueeze(0), clause]),
            "embedding_kmeans": torch.cat([global_query.unsqueeze(0), kmeans_facets.hidden]),
            "syntactic_graph": torch.cat(
                [global_query.unsqueeze(0), hierarchy.scoring_facets.hidden[1:]], dim=0
            ),
        }
        semantic_scores = {}
        score_ms = {}
        for condition, values in score_inputs.items():
            _synchronize(device)
            started = time.perf_counter()
            semantic_scores[condition] = _semantic_scores(values, memory)
            _synchronize(device)
            score_ms[condition] = (time.perf_counter() - started) * 1000.0
        semantic_scores.update(
            graph_cc=graph_outputs["cc"]["scores"],
            graph_label_propagation=graph_outputs["label_propagation"]["scores"],
        )
        score_ms.update(
            graph_cc=graph_outputs["cc"]["score_ms"],
            graph_label_propagation=graph_outputs["label_propagation"]["score_ms"],
        )
        selections = {}
        selection_ms = {}
        for condition, scores in semantic_scores.items():
            started = time.perf_counter()
            selections[condition] = _topk(scores, args.budget)
            selection_ms[condition] = (time.perf_counter() - started) * 1000.0
        hybrid_inputs = {
            "paper26_bm25": (global_scores, "bm25"),
            "paper26_hybrid": (global_scores, "token_semantic_rerank"),
            "graph_cc_hybrid": (graph_outputs["cc"]["scores"], "token_semantic_rerank"),
            "graph_label_propagation_hybrid": (graph_outputs["label_propagation"]["scores"], "token_semantic_rerank"),
            "syntactic_graph_hybrid": (semantic_scores["syntactic_graph"], "token_semantic_rerank"),
        }
        for condition, (scores, mode) in hybrid_inputs.items():
            started = time.perf_counter()
            selections[condition] = _hybrid_selection(
                token_index, tokenizer, query_ids, scores, mode, args.budget
            )
            selection_ms[condition] = (time.perf_counter() - started) * 1000.0
            score_ms[condition] = (
                graph_outputs["cc"]["score_ms"]
                if condition == "graph_cc_hybrid"
                else graph_outputs["label_propagation"]["score_ms"]
                if condition == "graph_label_propagation_hybrid"
                else score_ms["syntactic_graph"]
                if condition == "syntactic_graph_hybrid"
                else 0.0
            )

        positive_indices = {index.chunk_ids.index(value) for value in positives}
        for condition in CONDITIONS:
            selected_indices = selections[condition]
            selected_ids = [index.chunk_ids[value] for value in selected_indices]
            metric = _metrics(selected_ids, positives)
            graph_method = "cc" if "graph_cc" in condition else "label_propagation" if "graph_label_propagation" in condition else None
            graph_info = graph_outputs.get(graph_method, {})
            hierarchical = condition.startswith("syntactic_graph")
            graph_build_ms = (
                hierarchy_metrics.graph_construction_ms
                if hierarchical else float(graph_info.get("graph_ms", 0.0))
            )
            graph_cluster_ms = (
                hierarchy_metrics.graph_clustering_ms
                if hierarchical else float(graph_info.get("cluster_ms", 0.0))
            )
            routing_ms = graph_build_ms + graph_cluster_ms
            routing_ms += score_ms[condition] + selection_ms[condition]
            graph_node_count = (
                hierarchy_metrics.graph_nodes
                if hierarchical else int(graph_info["graph"].node_count) if graph_info else 0
            )
            graph_edge_count = (
                hierarchy_metrics.graph_edges
                if hierarchical else int(graph_info["graph"].edge_count) if graph_info else 0
            )
            possible_edges = graph_node_count * max(graph_node_count - 1, 0)
            rows.append(
                {
                    "split": "test",
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "condition": condition,
                    **metric,
                    "requested_chunks": len(selected_indices),
                    "requested_token_budget": len(selected_indices) * 32,
                    "materialized_chunks": 0,
                    "materialized_tokens": 0,
                    "selected_chunk_ids": "|".join(selected_ids),
                    "positive_chunk_ids": "|".join(sorted(positives)),
                    "positive_indices": "|".join(map(str, sorted(positive_indices))),
                    "graph_facets": (
                        hierarchy_metrics.facet_count - 1
                        if hierarchical else int(graph_info["facets"].hidden.shape[0]) if graph_info else 0
                    ),
                    "graph_nodes": graph_node_count,
                    "graph_edges": graph_edge_count,
                    "graph_density": graph_edge_count / possible_edges if possible_edges else 0.0,
                    "graph_calls": hierarchy_metrics.graph_calls if hierarchical else int(bool(graph_info)),
                    "graph_memory_bytes": (
                        hierarchy_metrics.graph_memory_bytes
                        if hierarchical else graph_memory_bytes(graph_info["graph"]) if graph_info else 0
                    ),
                    "graph_ms": graph_build_ms,
                    "cluster_ms": graph_cluster_ms,
                    "pairwise_similarity_evaluations": (
                        hierarchy_metrics.pairwise_similarity_evaluations
                        if hierarchical else graph_node_count * max(graph_node_count - 1, 0)
                    ),
                    "facet_overlap": hierarchy_metrics.mean_facet_overlap if hierarchical else 0.0,
                    "score_ms": score_ms[condition],
                    "selection_ms": selection_ms[condition],
                    "routing_ms": routing_ms,
                    "cluster_iterations": graph_info["clusters"].iterations if graph_info else 0,
                    "cluster_converged": int(graph_info["clusters"].converged) if graph_info else 0,
                    "peak_graph_gpu_bytes": int(graph_info.get("peak", 0)),
                    "device": str(device),
                }
            )
        print(f"[graph natural {case_index}/{len(cases)}] {identity[0]} {identity[1]}", flush=True)

    if {row["requested_chunks"] for row in rows} != {args.budget}:
        raise AssertionError("Natural conditions do not share one conceptual chunk budget.")
    summaries = _condition_summary(rows)
    parity = _parity(rows, args.paper2_6_channel_dir)
    effects = _paired_effects(rows)
    parity_pass = max(row["recall_abs_error"] for row in parity) <= 1e-9 and min(row["selection_exact_match"] for row in parity) == 1
    graph_gains = {}
    for dataset in DATASETS:
        graph = next(row["evidence_recall"] for row in summaries if row["dataset"] == dataset and row["condition"] == "graph_cc_hybrid")
        inherited = next(row["evidence_recall"] for row in summaries if row["dataset"] == dataset and row["condition"] == "paper26_hybrid")
        graph_gains[dataset] = graph - inherited
    gate3_pass = max(graph_gains.values()) > 0 and min(graph_gains.values()) >= -args.material_regression
    primary_effects = [
        row for row in effects
        if row["treatment"] == "graph_cc_hybrid"
        and row["baseline"] == "paper26_hybrid"
    ]
    gate3_statistical_pass = any(row["ci95_low"] > 0 for row in primary_effects)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "natural_retrieval_rows.csv", rows)
    write_csv(args.output_dir / "per_facet_method_rows.csv", facet_method_rows)
    write_csv(args.output_dir / "natural_retrieval_summary.csv", summaries)
    write_csv(args.output_dir / "paper2_6_parity.csv", parity)
    write_csv(args.output_dir / "paired_recall_effects.csv", effects)
    with (args.output_dir / "facet_assignments.jsonl").open("w", encoding="utf-8") as stream:
        for row in assignments:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    config = {
        "git": git_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": 27,
        "selected_graph_policy": policy,
        "budget": {"requested_chunks": args.budget, "chunk_tokens": 32, "requested_token_budget": args.budget * 32},
        "materialization_performed": False,
        "generation_performed": False,
        "device": str(device),
        "cohort_seed": args.cohort_seed,
        "query_entry_identity_hash": _manifest_hash([{**row, "partition": "test"} for row in entry_rows]),
        "natural_identity_hash": _manifest_hash(natural_rows),
        "threshold_selection": "controlled validation only",
    }
    write_json(args.output_dir / "run_config.json", config)
    findings = {
        "schema_version": "1.0",
        "config": config,
        "cohort": {dataset: sum(feature["dataset"] == dataset for feature, _ in cases) for dataset in DATASETS},
        "summary": summaries,
        "paper2_6_parity_pass": parity_pass,
        "paper2_6_parity_max_recall_error": max(row["recall_abs_error"] for row in parity),
        "graph_cc_hybrid_recall_gain": graph_gains,
        "paired_effects": effects,
        "gate3_directional_pass": gate3_pass,
        "gate3_statistical_pass": gate3_statistical_pass,
        "gate3_rule": f"positive gain on >=1 dataset and no dataset below -{args.material_regression:.3f}",
        "gate4_run": False,
        "gate5_run": False,
        "stop_reason": (
            "Natural paired intervals include zero; G4 algorithm expansion and G5 encoding-mode pilot remain gated."
            if gate3_pass and not gate3_statistical_pass
            else None
            if gate3_pass
            else "G3 failed; algorithm expansion and encoding-mode pilot remain gated."
        ),
    }
    write_json(args.output_dir / "natural_findings.json", findings)
    (args.output_dir / "claim_audit.md").write_text(
        "# Paper 2.7 natural-retrieval claim audit\n\n"
        "- Frozen Paper 2.5/2.6 query and memory states; no model forward or training.\n"
        "- Graph policy selected on controlled validation only.\n"
        "- All conditions request exactly four aligned 32-token chunks.\n"
        "- Physical native K/V materialization and answer generation were not run.\n"
        "- Query clustering is distinct from inherited memory traversal.\n"
        "- Attention and residual-update query edges were unavailable in this frozen cohort.\n"
        "- Causal hidden-state graphs are not described as bidirectional semantic encoders.\n"
        f"- Paper 2.6 route parity: {parity_pass}.\n"
        f"- G3 directional retrieval-budget proxy passed: {gate3_pass}.\n"
        f"- G3 paired-interval criterion passed: {gate3_statistical_pass}.\n",
        encoding="utf-8",
    )
    _plots(summaries, effects, args.output_dir)
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    base = ROOT / "docs/papers/shared/results/paper2_7_query_graph"
    adaptive_output = (
        ROOT
        / "docs/papers/shared/results/paper3_5_adaptive_pra/request_reply_graph/natural_replay"
    )
    inherited = Path(r"D:/git/rd/pdattention-iter-gist")
    local_channel_dir = ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_geometry"
    inherited_channel_dir = (
        Path(r"D:/git/rd/pdattention-hybrid")
        / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_geometry"
    )
    parser.add_argument("--policy", type=Path, default=base / "algorithm_cross/selected_graph_policy.json")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--cohort-seed", type=int, default=20260811)
    parser.add_argument("--material-regression", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--paper2-feature-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--natural-features", type=Path, default=resolve_artifact("docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt"))
    parser.add_argument("--query-entry-features", type=Path, default=resolve_artifact("docs/papers/shared/results/paper2_5_iterative_pra/query_entry_facets/query_entry_features.pt"))
    parser.add_argument("--musique-dev", type=Path, default=inherited / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "data/.paper2_5_datasets/2wiki/dev.json")
    parser.add_argument(
        "--paper2-6-channel-dir",
        type=Path,
        default=local_channel_dir if local_channel_dir.exists() else inherited_channel_dir,
    )
    parser.add_argument("--output-dir", type=Path, default=adaptive_output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, allow_nan=False))
