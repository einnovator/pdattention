"""Evaluate zero-config callable ingestion and bounded union discovery.

The runner tunes only deterministic fusion weights on the frozen validation
identities. Test rows are then used once for E1--E3 and record-boundary audits.
It does not load a generative model or execute any tool.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.python_tool_ingestion_cases import PAPER6_5_TOOL_CALLABLES
from data.semantic_concepts import canonical_concept_map
from data.semantic_hard_tools import SemanticHardQuery, semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf.agent_disclosure import ToolCapabilityGraph
from pra_hf.agent_resources import AgentResource
from pra_hf.context_records import tool_definition_record
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder, ExternalSemanticIndex, ToolSemanticCard
from pra_hf.tool_records import PythonTypeSchemaCache, ToolRecord, tool_record_from_callable
from pra_hf.union_discovery import ToolDiscoveryMode, ToolDiscoveryPolicy, UnionStrategy, discover_candidate_set


K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)
STRATEGIES = tuple(UnionStrategy)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, (frozenset, set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _record_payload(record: ToolRecord) -> dict[str, object]:
    return asdict(record)


def _full_query(row: SemanticHardQuery) -> str:
    return "\n".join(value for value in (row.context, row.query) if value)


def _score_resources(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
    encoder: CompactEmbeddingEncoder,
) -> tuple[dict[str, torch.Tensor], tuple[ToolSemanticCard, ...]]:
    cards = tuple(ToolSemanticCard.from_resource(row) for row in resources)
    tool_vectors = encoder.encode([row.structured_text for row in cards])
    query_vectors = encoder.encode([_full_query(row) for row in queries], query=True)
    index = ExternalSemanticIndex(resources, canonical_concept_map())
    tensors = {
        name: torch.zeros((len(queries), len(resources)), dtype=torch.float32)
        for name in ("token", "bm25", "dictionary", "tags")
    }
    for query_index, query in enumerate(queries):
        scored = index.score(query.query, context=query.context, language=query.language)
        for resource_index, score in enumerate(scored):
            tensors["token"][query_index, resource_index] = score.token
            tensors["bm25"][query_index, resource_index] = score.bm25
            tensors["dictionary"][query_index, resource_index] = score.dictionary
            tensors["tags"][query_index, resource_index] = score.tags
    tensors["lexical"] = torch.maximum(tensors.pop("token"), tensors["bm25"])
    tensors["embedding"] = ((query_vectors @ tool_vectors.T) + 1.0) / 2.0
    return tensors, cards


def _target_indices(resources: Sequence[AgentResource], queries: Sequence[SemanticHardQuery]) -> list[int]:
    by_name = {row.name: index for index, row in enumerate(resources)}
    return [by_name[row.required_tool] for row in queries]


def _rank_metrics(scores: torch.Tensor, targets: Sequence[int], indices: Sequence[int]) -> dict[str, float]:
    subset = scores[list(indices)]
    target = torch.tensor([targets[index] for index in indices]).reshape(-1, 1)
    order = torch.argsort(subset, dim=1, descending=True, stable=True)
    ranks = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1": float((ranks == 1).float().mean()),
        "recall_at_3": float((ranks <= 3).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
    }


def _weight_grid(channel_names: Sequence[str]) -> tuple[dict[str, float], ...]:
    units = range(5)
    rows = []
    for values in itertools.product(units, repeat=len(channel_names)):
        if sum(values) != 4:
            continue
        rows.append({name: value / 4.0 for name, value in zip(channel_names, values)})
    return tuple(rows)


def _fuse(channels: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    output = torch.zeros_like(next(iter(channels.values())))
    for name, weight in weights.items():
        output += float(weight) * channels[name]
    return output


def _select_weights(
    channels: Mapping[str, torch.Tensor],
    names: Sequence[str],
    targets: Sequence[int],
    validation_indices: Sequence[int],
) -> tuple[dict[str, float], torch.Tensor, dict[str, float]]:
    candidates = []
    for weights in _weight_grid(names):
        scores = _fuse(channels, weights)
        metrics = _rank_metrics(scores, targets, validation_indices)
        key = (metrics["top1"], metrics["mrr"], metrics["recall_at_3"], -metrics["mean_rank"])
        candidates.append((key, tuple(weights[name] for name in names), weights, scores, metrics))
    _, _, weights, scores, metrics = max(candidates, key=lambda row: (row[0], row[1]))
    return dict(weights), scores, metrics


def _condition_scores(
    channels: Mapping[str, torch.Tensor],
    targets: Sequence[int],
    validation_indices: Sequence[int],
    *,
    prefix: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    python_weights, python_scores, python_metrics = _select_weights(
        channels, ("lexical", "tags"), targets, validation_indices
    )
    dictionary_weights, dictionary_scores, dictionary_metrics = _select_weights(
        channels, ("lexical", "dictionary", "tags"), targets, validation_indices
    )
    hybrid_weights, hybrid_scores, hybrid_metrics = _select_weights(
        channels, ("lexical", "dictionary", "tags", "embedding"), targets, validation_indices
    )
    conditions = {
        f"{prefix}_lexical": channels["lexical"],
        f"{prefix}_python_only": python_scores,
        f"{prefix}_dictionary": dictionary_scores,
        f"{prefix}_embedding": channels["embedding"],
        f"{prefix}_hybrid": hybrid_scores,
    }
    selection = {
        "python_only": {"weights": python_weights, "validation": python_metrics},
        "dictionary": {"weights": dictionary_weights, "validation": dictionary_metrics},
        "hybrid": {"weights": hybrid_weights, "validation": hybrid_metrics},
    }
    return conditions, selection


def _semantic_rows(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
    conditions: Mapping[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows = []
    for condition, scores in conditions.items():
        orders = torch.argsort(scores, dim=1, descending=True, stable=True)
        for index, query in enumerate(queries):
            order = orders[index].tolist()
            rank = order.index(targets[index]) + 1
            rows.append({
                "query_id": query.query_id,
                "split": query.split,
                "hardness_level": query.hardness_level,
                "language": query.language,
                "condition": condition,
                "required_tool": query.required_tool,
                "top_tool": resources[order[0]].name,
                "rank": rank,
                "top1": int(rank == 1),
                "recall_at_3": int(rank <= 3),
                "recall_at_5": int(rank <= 5),
            })
    return rows


def _graph_scores(
    resources: Sequence[AgentResource],
    channels: Mapping[str, torch.Tensor],
    query_index: int,
) -> dict[str, float]:
    graph = ToolCapabilityGraph(resources)
    base = channels["lexical"][query_index] + channels["dictionary"][query_index] + channels["tags"][query_index] + channels["embedding"][query_index]
    roots = torch.argsort(base, descending=True, stable=True)[:2].tolist()
    scores: dict[str, float] = {}
    for root_index in roots:
        root_uri = resources[root_index].uri
        for edge in graph.outgoing.get(root_uri, ()):
            scores[edge.target_uri] = max(scores.get(edge.target_uri, 0.0), float(edge.weight))
    return scores


def _union_rows(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
    channels: Mapping[str, torch.Tensor],
    *,
    tokenizer,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_indices = [index for index, row in enumerate(queries) if row.split == "test"]
    uri_by_name = {row.name: row.uri for row in resources}
    resource_by_uri = {row.uri: row for row in resources}
    validation_indices = [index for index, row in enumerate(queries) if row.split == "validation"]
    channel_top1 = {
        name: _rank_metrics(values, _target_indices(resources, queries), validation_indices)["top1"]
        for name, values in channels.items()
    }
    best_channel = max(channel_top1, key=lambda name: (channel_top1[name], name))
    raw_rows = []
    for query_index in test_indices:
        query = queries[query_index]
        score_maps = {
            name: {resource.uri: float(values[query_index, index]) for index, resource in enumerate(resources)}
            for name, values in channels.items()
            if name in {"lexical", "dictionary", "tags", "embedding"}
        }
        score_maps["graph"] = _graph_scores(resources, channels, query_index)
        for strategy in STRATEGIES:
            for budget in K_VALUES:
                preferred = best_channel if strategy == UnionStrategy.SINGLE_CHANNEL else None
                candidates = discover_candidate_set(
                    _full_query(query),
                    resources,
                    score_maps,
                    ToolDiscoveryPolicy(
                        mode=ToolDiscoveryMode.TOP_K,
                        strategy=strategy,
                        max_candidates=budget,
                        graph=True,
                        preferred_channel=preferred,
                    ),
                )
                names = tuple(resource_by_uri[uri].name for uri in candidates.candidate_uris)
                required = query.required_tool in names
                useful = {query.required_tool, *query.useful_tools}
                selected_useful = useful & set(names)
                unsafe = set(query.unsafe_tools) & set(names)
                schema_tokens = sum(
                    len(tokenizer.encode(resource_by_uri[uri].content))
                    for uri in candidates.candidate_uris
                )
                raw_rows.append({
                    "query_id": query.query_id,
                    "hardness_level": query.hardness_level,
                    "language": query.language,
                    "strategy": strategy.value,
                    "max_candidates": budget,
                    "candidate_count": len(names),
                    "candidate_names": "|".join(names),
                    "required_tool": query.required_tool,
                    "required_recall": int(required),
                    "useful_recall": len(selected_useful) / max(len(useful), 1),
                    "useful_precision": len(selected_useful) / max(len(names), 1),
                    "unsafe_exposure": int(bool(unsafe)),
                    "unsafe_names": "|".join(sorted(unsafe)),
                    "schema_tokens": schema_tokens,
                    "provenance_diversity": sum(len(row.sources) for row in candidates.provenance) / max(len(candidates.provenance), 1),
                    "provenance": json.dumps([asdict(row) for row in candidates.provenance], sort_keys=True),
                    "best_single_channel": best_channel,
                })
    frontier = []
    for strategy in STRATEGIES:
        for budget in K_VALUES:
            subset = [row for row in raw_rows if row["strategy"] == strategy.value and row["max_candidates"] == budget]
            frontier.append({
                "strategy": strategy.value,
                "max_candidates": budget,
                "queries": len(subset),
                "required_recall": sum(float(row["required_recall"]) for row in subset) / len(subset),
                "useful_recall": sum(float(row["useful_recall"]) for row in subset) / len(subset),
                "useful_precision": sum(float(row["useful_precision"]) for row in subset) / len(subset),
                "unsafe_exposure": sum(float(row["unsafe_exposure"]) for row in subset) / len(subset),
                "mean_candidates": sum(float(row["candidate_count"]) for row in subset) / len(subset),
                "mean_schema_tokens": sum(float(row["schema_tokens"]) for row in subset) / len(subset),
                "provenance_diversity": sum(float(row["provenance_diversity"]) for row in subset) / len(subset),
            })
    return raw_rows, frontier


def _aggregate_semantics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    conditions = sorted({str(row["condition"]) for row in rows})
    strata = ("all", "H0", "H1", "H2", "H3", "H4", "H5", "multilingual")
    for condition in conditions:
        for stratum in strata:
            subset = [
                row for row in rows
                if row["condition"] == condition
                and row["split"] == "test"
                and (stratum == "all" or row["hardness_level"] == stratum or stratum == "multilingual" and row["language"] != "en")
            ]
            if not subset:
                continue
            output.append({
                "condition": condition,
                "stratum": stratum,
                "queries": len(subset),
                "top1": sum(int(row["top1"]) for row in subset) / len(subset),
                "recall_at_3": sum(int(row["recall_at_3"]) for row in subset) / len(subset),
                "recall_at_5": sum(int(row["recall_at_5"]) for row in subset) / len(subset),
                "mean_rank": sum(int(row["rank"]) for row in subset) / len(subset),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/auto_union_records",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concepts = canonical_concept_map()
    cache = PythonTypeSchemaCache()
    auto_records = tuple(
        tool_record_from_callable(
            function,
            namespace="paper6_5_auto",
            tenant_id="paper6_5",
            concept_map=concepts,
            type_cache=cache,
        )
        for function in PAPER6_5_TOOL_CALLABLES
    )
    auto_resources = tuple(row.to_agent_resource() for row in auto_records)
    manual_resources = realistic_tool_catalog()
    queries = semantic_hardness_queries()
    if tuple(row.name for row in auto_resources) != tuple(row.name for row in manual_resources):
        raise RuntimeError("Automatic and manual catalogs must preserve aligned tool names.")

    with (args.output_dir / "auto_tool_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in auto_records:
            stream.write(json.dumps(_record_payload(record), sort_keys=True, default=_json_default) + "\n")
    (args.output_dir / "python_type_schema_cache.json").write_text(
        json.dumps(cache.to_manifest(), indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    (args.output_dir / "auto_keyword_manifest.json").write_text(json.dumps({
        "schema_version": "1.0",
        "tools": [{
            "name": row.name,
            "manual_tags": sorted(row.manual_tags),
            "auto_tags": sorted(row.auto_tags),
            "keywords": sorted(row.keywords),
            "operations": sorted(row.operation_concepts),
            "objects": sorted(row.object_concepts),
            "evidence": [asdict(value) for value in row.keyword_evidence],
        } for row in auto_records],
    }, indent=2, sort_keys=True), encoding="utf-8")

    encoder = CompactEmbeddingEncoder(
        str(args.model_root / "bge-small-en-v1.5"),
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        device=args.device,
        query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    manual_channels, _ = _score_resources(manual_resources, queries, encoder)
    auto_channels, _ = _score_resources(auto_resources, queries, encoder)
    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    validation_indices = [index for index, row in enumerate(queries) if row.split == "validation"]
    manual_targets = _target_indices(manual_resources, queries)
    auto_targets = _target_indices(auto_resources, queries)
    manual_conditions, manual_selection = _condition_scores(
        manual_channels, manual_targets, validation_indices, prefix="manual"
    )
    auto_conditions, auto_selection = _condition_scores(
        auto_channels, auto_targets, validation_indices, prefix="auto"
    )
    semantic_rows = _semantic_rows(manual_resources, queries, manual_targets, manual_conditions)
    semantic_rows += _semantic_rows(auto_resources, queries, auto_targets, auto_conditions)
    semantic_summary = _aggregate_semantics(semantic_rows)
    _write_csv(args.output_dir / "auto_vs_manual_semantics.csv", semantic_rows)
    _write_csv(args.output_dir / "auto_vs_manual_summary.csv", semantic_summary)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    union_rows, frontier = _union_rows(auto_resources, queries, auto_channels, tokenizer=tokenizer)
    _write_csv(args.output_dir / "union_candidate_rows.csv", union_rows)
    _write_csv(args.output_dir / "union_recall_frontier.csv", frontier)

    boundary_rows = []
    for resource in auto_resources:
        record = tool_definition_record(resource)
        boundary_rows.append({
            "record_id": record.record_id,
            "record_type": record.record_type.value,
            "version": record.version,
            "source_fingerprint": record.source_fingerprint,
            "size_bytes": record.size_bytes,
            "boundaries": [asdict(row) for row in record.boundaries],
            "policy": asdict(record.policy),
        })
    (args.output_dir / "record_boundary_audit.json").write_text(
        json.dumps({"schema_version": "1.0", "records": boundary_rows}, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    from pra_hf.context_records import RecordType, default_record_policy
    (args.output_dir / "record_policy_manifest.json").write_text(json.dumps({
        "schema_version": "1.0",
        "paper7_progressive_detail": "not_implemented",
        "policies": {record_type.value: asdict(default_record_policy(record_type)) for record_type in RecordType},
    }, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

    def overall(condition: str, metric: str) -> float:
        return float(next(row[metric] for row in semantic_summary if row["condition"] == condition and row["stratum"] == "all"))

    lexical = overall("auto_lexical", "top1")
    manual = overall("manual_hybrid", "top1")
    automatic = overall("auto_hybrid", "top1")
    gain_retention = (automatic - lexical) / (manual - lexical) if manual != lexical else 0.0
    best_frontier = max(frontier, key=lambda row: (row["required_recall"], -row["mean_candidates"], -row["unsafe_exposure"]))
    findings = {
        "schema_version": "1.0",
        "manual_selection": manual_selection,
        "auto_selection": auto_selection,
        "test": {
            "lexical_top1": lexical,
            "manual_hybrid_top1": manual,
            "auto_hybrid_top1": automatic,
            "auto_gain_retention": gain_retention,
            "best_union_frontier": best_frontier,
        },
        "gates": {
            "auto_metadata_retains_most_manual_gain": gain_retention >= 0.75,
            "union_default_pending_jit": True,
            "record_atomic_default": True,
        },
        "device": args.device,
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "auto_union_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
