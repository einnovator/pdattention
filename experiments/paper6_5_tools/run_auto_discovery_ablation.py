"""Run Paper 6.5 A0--A7 automatic tool-discovery ablations.

One callable-derived ToolRecord catalog is frozen before scoring. Every policy
then reveals a subset of its evidence, so no condition can benefit from a
different registration representation or test-label-informed regeneration.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.python_tool_ingestion_cases import PAPER6_5_TOOL_CALLABLES
from data.semantic_concepts import canonical_concept_map, dictionary_sources_manifest
from data.semantic_hard_tools import SemanticHardQuery, semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.run_auto_union_records import _condition_scores, _score_resources
from pra_hf.agent_resources import AgentResource, DiscoveryRequest, PersistentResourceIndex
from pra_hf.auto_tool_discovery import (
    AutoEvidenceSource,
    AutoSemanticEvidence,
    AutoToolSemanticView,
    auto_tag_score,
    automatic_semantic_view,
    evidence_provenance_counts,
    inferred_concept_score,
    weighted_keyword_score,
)
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder
from pra_hf.tool_records import PythonTypeSchemaCache, ToolRecord, tool_record_from_callable
from pra_hf.union_discovery import ToolDiscoveryPolicy, UnionStrategy, discover_candidate_set


K_VALUES = (2, 3, 4, 5, 6, 8, 10)
EMBEDDING_REPRESENTATIONS = ("description", "name_description", "structured_card", "multi_vector")
BASE_KEYWORD_SOURCES = frozenset({
    AutoEvidenceSource.FUNCTION_NAME,
    AutoEvidenceSource.DOCSTRING,
    AutoEvidenceSource.PARAMETER_NAME,
    AutoEvidenceSource.PARAMETER_DESCRIPTION,
    AutoEvidenceSource.RETURN_DESCRIPTION,
    AutoEvidenceSource.TYPE_SCHEMA,
    AutoEvidenceSource.MODULE_NAMESPACE,
})


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


def _full_query(row: SemanticHardQuery) -> str:
    return "\n".join(value for value in (row.context, row.query) if value)


def _targets(views: Sequence[AutoToolSemanticView], queries: Sequence[SemanticHardQuery]) -> list[int]:
    by_name = {row.name: index for index, row in enumerate(views)}
    return [by_name[row.required_tool] for row in queries]


def _metrics(scores: torch.Tensor, targets: Sequence[int], indices: Sequence[int]) -> dict[str, float]:
    selected = scores[list(indices)]
    target = torch.tensor([targets[index] for index in indices]).reshape(-1, 1)
    order = torch.argsort(selected, dim=1, descending=True, stable=True)
    ranks = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1": float((ranks == 1).float().mean()),
        "recall_at_3": float((ranks <= 3).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
    }


def _score_rows(
    views: Sequence[AutoToolSemanticView],
    queries: Sequence[SemanticHardQuery],
    scorer,
) -> torch.Tensor:
    output = torch.zeros((len(queries), len(views)), dtype=torch.float32)
    for query_index, query in enumerate(queries):
        text = _full_query(query)
        for tool_index, view in enumerate(views):
            output[query_index, tool_index] = scorer(text, query.language, view)
    return output


def _raw_resources(records: Sequence[ToolRecord], views: Sequence[AutoToolSemanticView]) -> tuple[AgentResource, ...]:
    resources = []
    for record, view in zip(records, views):
        source = record.to_agent_resource()
        resources.append(replace(
            source,
            description=view.raw_text,
            aliases=(),
            metadata={"zero_config_raw_callable": True},
        ))
    return tuple(resources)


def _raw_lexical_scores(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
) -> tuple[torch.Tensor, torch.Tensor]:
    index = PersistentResourceIndex(resources)
    token = torch.zeros((len(queries), len(resources)), dtype=torch.float32)
    bm25 = torch.zeros_like(token)
    for query_index, query in enumerate(queries):
        request = DiscoveryRequest(query=_full_query(query), tenant_id="paper6_5", top_k=len(resources))
        by_uri = {row.uri: row for row in index.score(request, channels=("token", "index"))}
        for tool_index, resource in enumerate(resources):
            score = by_uri.get(resource.uri)
            token[query_index, tool_index] = 0.0 if score is None else score.token
            bm25[query_index, tool_index] = 0.0 if score is None else score.index
    return token, bm25


def _embedding_matrix(
    encoder: CompactEmbeddingEncoder,
    query_vectors: torch.Tensor,
    views: Sequence[AutoToolSemanticView],
    representation: str,
) -> torch.Tensor:
    values = [row.embedding_text(representation) for row in views]
    if representation == "multi_vector":
        flattened = [field for row in values for field in row]
        tool_vectors = encoder.encode(flattened).reshape(len(views), 3, -1)
        scores = torch.einsum("qd,rvd->qrv", query_vectors, tool_vectors).max(dim=2).values
    else:
        tool_vectors = encoder.encode([str(value) for value in values])
        scores = query_vectors @ tool_vectors.T
    return ((scores + 1.0) / 2.0).clamp(0.0, 1.0)


def _weight_grid(names: Sequence[str]) -> Iterable[dict[str, float]]:
    for values in itertools.product(range(5), repeat=len(names)):
        if sum(values) == 4:
            yield {name: value / 4.0 for name, value in zip(names, values)}


def _fuse(channels: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    output = torch.zeros_like(next(iter(channels.values())))
    for name, weight in weights.items():
        output += channels[name] * float(weight)
    return output


def _select_fusion(
    channels: Mapping[str, torch.Tensor],
    targets: Sequence[int],
    validation_indices: Sequence[int],
) -> tuple[dict[str, float], torch.Tensor, dict[str, float]]:
    names = tuple(channels)
    candidates = []
    for weights in _weight_grid(names):
        scores = _fuse(channels, weights)
        metrics = _metrics(scores, targets, validation_indices)
        key = (metrics["top1"], metrics["mrr"], metrics["recall_at_3"], -metrics["mean_rank"])
        candidates.append((key, tuple(weights[name] for name in names), weights, scores, metrics))
    _, _, weights, scores, metrics = max(candidates, key=lambda row: (row[0], row[1]))
    return dict(weights), scores, metrics


def _query_rows(
    views: Sequence[AutoToolSemanticView],
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
                "top_tool": views[order[0]].name,
                "rank": rank,
                "top1": int(rank == 1),
                "recall_at_3": int(rank <= 3),
                "recall_at_5": int(rank <= 5),
            })
    return rows


def _summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    strata = ("all", "H0", "H1", "H2", "H3", "H4", "H5", "multilingual")
    for condition in sorted({str(row["condition"]) for row in rows}):
        for stratum in strata:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["split"] == "test"
                and (
                    stratum == "all" or row["hardness_level"] == stratum
                    or stratum == "multilingual" and row["language"] != "en"
                )
            ]
            if not selected:
                continue
            output.append({
                "condition": condition,
                "stratum": stratum,
                "queries": len(selected),
                "top1": sum(int(row["top1"]) for row in selected) / len(selected),
                "recall_at_3": sum(int(row["recall_at_3"]) for row in selected) / len(selected),
                "recall_at_5": sum(int(row["recall_at_5"]) for row in selected) / len(selected),
                "mean_rank": sum(int(row["rank"]) for row in selected) / len(selected),
            })
    return output


def _variant_view(
    view: AutoToolSemanticView,
    *,
    remove_sources: Iterable[AutoEvidenceSource] = (),
    source_replacement: tuple[AutoEvidenceSource, str] | None = None,
    embedding_updates: Mapping[str, str] | None = None,
) -> AutoToolSemanticView:
    removed = frozenset(remove_sources)
    evidence = [row for row in view.evidence if row.source not in removed]
    if source_replacement is not None:
        source, surface = source_replacement
        evidence = [row for row in evidence if row.source != source]
        for term in surface.replace("_", " ").casefold().split():
            if len(term) > 1:
                evidence.append(AutoSemanticEvidence(term, source, surface, 1.0))
    fields = dict(view.embedding_fields)
    fields.update(dict(embedding_updates or {}))
    return replace(view, evidence=tuple(evidence), embedding_fields=tuple(fields.items()))


def _quality_rows(
    variants: Mapping[str, Sequence[AutoToolSemanticView]],
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
    encoder: CompactEmbeddingEncoder,
    query_vectors: torch.Tensor,
    selected_embedding: str,
    *,
    dimension: str,
) -> list[dict[str, object]]:
    test_indices = [index for index, row in enumerate(queries) if row.split == "test"]
    output = []
    for quality, views in variants.items():
        keyword = _score_rows(
            views, queries,
            lambda text, language, view: weighted_keyword_score(text, view, sources=BASE_KEYWORD_SOURCES),
        )
        embedding = _embedding_matrix(encoder, query_vectors, views, selected_embedding)
        for policy, scores in (("A1_auto_keywords", keyword), ("A5_embedding", embedding)):
            metrics = _metrics(scores, targets, test_indices)
            output.append({"dimension": dimension, "quality": quality, "policy": policy, **metrics})
    return output


def _source_ablation_rows(
    views: Sequence[AutoToolSemanticView],
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
) -> list[dict[str, object]]:
    groups = {
        "full_A1": frozenset(),
        "minus_function_name": {AutoEvidenceSource.FUNCTION_NAME},
        "minus_docstring": {AutoEvidenceSource.DOCSTRING},
        "minus_parameters": {AutoEvidenceSource.PARAMETER_NAME, AutoEvidenceSource.PARAMETER_DESCRIPTION},
        "minus_return": {AutoEvidenceSource.RETURN_DESCRIPTION},
        "minus_module_namespace": {AutoEvidenceSource.MODULE_NAMESPACE},
    }
    test_indices = [index for index, row in enumerate(queries) if row.split == "test"]
    output = []
    for condition, removed in groups.items():
        sources = BASE_KEYWORD_SOURCES - removed
        scores = _score_rows(
            views, queries,
            lambda text, language, view: weighted_keyword_score(text, view, sources=sources),
        )
        output.append({"condition": condition, "removed_sources": "|".join(sorted(row.value for row in removed)), **_metrics(scores, targets, test_indices)})
    return output


def _union_experiment(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
    channels: Mapping[str, torch.Tensor],
    tokenizer,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_uri = {row.uri: row for row in resources}
    strategies = (UnionStrategy.FUSED_SCORE, UnionStrategy.RAW_UNION, UnionStrategy.DIVERSITY_UNION)
    candidate_rows = []
    for split in ("validation", "test"):
        for query_index, query in enumerate(queries):
            if query.split != split:
                continue
            score_maps = {
                name: {resource.uri: float(values[query_index, index]) for index, resource in enumerate(resources)}
                for name, values in channels.items()
            }
            for strategy in strategies:
                for budget in K_VALUES:
                    candidates = discover_candidate_set(
                        _full_query(query), resources, score_maps,
                        ToolDiscoveryPolicy(
                            mode="union", strategy=strategy, max_candidates=budget,
                            allow_unsafe=True, graph=False, channels=tuple(channels),
                        ),
                    )
                    names = tuple(by_uri[uri].name for uri in candidates.candidate_uris)
                    useful = {query.required_tool, *query.useful_tools}
                    unsafe = set(query.unsafe_tools) & set(names)
                    candidate_rows.append({
                        "split": split,
                        "query_id": query.query_id,
                        "hardness_level": query.hardness_level,
                        "language": query.language,
                        "strategy": strategy.value,
                        "max_candidates": budget,
                        "candidate_count": len(names),
                        "candidate_names": "|".join(names),
                        "required_tool": query.required_tool,
                        "required_recall": int(query.required_tool in names),
                        "useful_precision": len(useful & set(names)) / max(len(names), 1),
                        "unsafe_exposure": int(bool(unsafe)),
                        "schema_tokens": sum(len(tokenizer.encode(by_uri[uri].content)) for uri in candidates.candidate_uris),
                        "channel_diversity": sum(len(row.sources) for row in candidates.provenance) / max(len(candidates.provenance), 1),
                        "provenance": json.dumps([asdict(row) for row in candidates.provenance], sort_keys=True),
                    })
    frontier = []
    for split in ("validation", "test"):
        for strategy in strategies:
            for budget in K_VALUES:
                subset = [row for row in candidate_rows if row["split"] == split and row["strategy"] == strategy.value and row["max_candidates"] == budget]
                frontier.append({
                    "split": split,
                    "strategy": strategy.value,
                    "max_candidates": budget,
                    "queries": len(subset),
                    "required_recall": sum(row["required_recall"] for row in subset) / len(subset),
                    "useful_precision": sum(row["useful_precision"] for row in subset) / len(subset),
                    "unsafe_exposure": sum(row["unsafe_exposure"] for row in subset) / len(subset),
                    "mean_candidates": sum(row["candidate_count"] for row in subset) / len(subset),
                    "mean_schema_tokens": sum(row["schema_tokens"] for row in subset) / len(subset),
                    "channel_diversity": sum(row["channel_diversity"] for row in subset) / len(subset),
                })

    complementarity = []
    test_indices = [index for index, row in enumerate(queries) if row.split == "test"]
    for index in test_indices:
        recovered = []
        for name, scores in channels.items():
            top = torch.argsort(scores[index], descending=True, stable=True)[:3].tolist()
            if targets[index] in top:
                recovered.append(name)
        category = "missed_by_all" if not recovered else f"only_{recovered[0]}" if len(recovered) == 1 else "multiple_channels"
        complementarity.append({
            "query_id": queries[index].query_id,
            "hardness_level": queries[index].hardness_level,
            "language": queries[index].language,
            "required_tool": queries[index].required_tool,
            "channel_top_k": 3,
            "recovered_channels": "|".join(recovered),
            "category": category,
        })
    return candidate_rows, frontier, complementarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_ablation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    concepts = canonical_concept_map()
    cache = PythonTypeSchemaCache()
    records = tuple(
        tool_record_from_callable(
            function, namespace="paper6_5_auto", tenant_id="paper6_5",
            concept_map=concepts, type_cache=cache,
        )
        for function in PAPER6_5_TOOL_CALLABLES
    )
    views = tuple(automatic_semantic_view(record, concepts=concepts) for record in records)
    resources = _raw_resources(records, views)
    queries = semantic_hardness_queries()
    targets = _targets(views, queries)
    validation_indices = [index for index, row in enumerate(queries) if row.split == "validation"]
    test_indices = [index for index, row in enumerate(queries) if row.split == "test"]

    manifest = {
        "schema_version": "1.0",
        "catalog_frozen_once": True,
        "manual_metadata_in_auto_conditions": False,
        "tools": [{
            "name": record.name,
            "manual_tags": sorted(record.manual_tags),
            "evidence": [{**asdict(row), "source": row.source.value} for row in view.evidence],
            "operations": sorted(view.operations),
            "objects": sorted(view.objects),
            "auto_tags": sorted(view.auto_tags),
            "embedding_fields": dict(view.embedding_fields),
            "field_provenance": record.field_provenance,
        } for record, view in zip(records, views)],
        "provenance_counts": evidence_provenance_counts(views),
        "dictionary": dictionary_sources_manifest(),
    }
    (args.output_dir / "auto_keyword_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    a0_token, a0_bm25 = _raw_lexical_scores(resources, queries)
    a1 = _score_rows(
        views, queries,
        lambda text, language, view: weighted_keyword_score(text, view, sources=BASE_KEYWORD_SOURCES),
    )
    a2_expanded = _score_rows(
        views, queries,
        lambda text, language, view: weighted_keyword_score(
            text, view,
            sources={*BASE_KEYWORD_SOURCES, AutoEvidenceSource.DICTIONARY_EXPANSION},
            concepts=concepts, language=language, expand_query=True,
        ),
    )
    a2_direct = _score_rows(
        views, queries,
        lambda text, language, view: weighted_keyword_score(
            text, view, sources={AutoEvidenceSource.DICTIONARY_EXPANSION},
            concepts=concepts, language=language, expand_query=True,
        ),
    )
    a2 = torch.maximum(a1, torch.maximum(a2_expanded, a2_direct))
    concept_only = _score_rows(views, queries, lambda text, language, view: inferred_concept_score(text, view, concepts, language=language))
    a3 = torch.maximum(a2, concept_only)
    tag_only = _score_rows(views, queries, lambda text, language, view: auto_tag_score(text, view, concepts, language=language))
    a4 = torch.maximum(a3, tag_only)

    encoder = CompactEmbeddingEncoder(
        str(args.model_root / "bge-small-en-v1.5"),
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        device=args.device,
        query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    query_vectors = encoder.encode([_full_query(row) for row in queries], query=True)
    embedding_scores = {
        representation: _embedding_matrix(encoder, query_vectors, views, representation)
        for representation in EMBEDDING_REPRESENTATIONS
    }
    embedding_validation = {name: _metrics(scores, targets, validation_indices) for name, scores in embedding_scores.items()}
    selected_embedding = max(
        EMBEDDING_REPRESENTATIONS,
        key=lambda name: (
            embedding_validation[name]["top1"], embedding_validation[name]["mrr"],
            embedding_validation[name]["recall_at_3"], -embedding_validation[name]["mean_rank"], name,
        ),
    )
    a5 = embedding_scores[selected_embedding]

    hybrid_channels = {
        "raw_bm25": a0_bm25,
        "auto_keyword": a1,
        "synonym": a2,
        "concept": a3,
        "auto_tag": a4,
        "embedding": a5,
    }
    hybrid_weights, a6, hybrid_validation = _select_fusion(hybrid_channels, targets, validation_indices)

    manual_resources = realistic_tool_catalog()
    manual_channels, _ = _score_resources(manual_resources, queries, encoder)
    manual_targets = [next(index for index, row in enumerate(manual_resources) if row.name == query.required_tool) for query in queries]
    manual_conditions, manual_selection = _condition_scores(manual_channels, manual_targets, validation_indices, prefix="manual")
    manual_ceiling = manual_conditions["manual_hybrid"]

    conditions = {
        "A0_token": a0_token,
        "A0_raw_bm25": a0_bm25,
        "A1_auto_keywords": a1,
        "A2_expanded_lexical": a2_expanded,
        "A2_direct_canonical": a2_direct,
        "A2_keywords_synonyms": a2,
        "A3_inferred_concepts": a3,
        "A4_auto_tags": a4,
        **{f"A5_embedding_{name}": scores for name, scores in embedding_scores.items()},
        "A5_embedding_selected": a5,
        "A6_auto_hybrid": a6,
        "manual_rich_ceiling": manual_ceiling,
    }
    rows = _query_rows(views, queries, targets, conditions)
    summaries = _summaries(rows)
    _write_csv(args.output_dir / "auto_discovery_rows.csv", rows)
    _write_csv(args.output_dir / "auto_discovery_by_hardness.csv", summaries)

    primary_order = (
        "A0_raw_bm25", "A1_auto_keywords", "A2_keywords_synonyms",
        "A3_inferred_concepts", "A4_auto_tags", "A5_embedding_selected",
        "A6_auto_hybrid", "manual_rich_ceiling",
    )
    metadata = {
        "A0_raw_bm25": (False, False, False, False),
        "A1_auto_keywords": (False, False, False, False),
        "A2_keywords_synonyms": (False, True, False, False),
        "A3_inferred_concepts": (False, True, True, False),
        "A4_auto_tags": (False, True, True, False),
        "A5_embedding_selected": (False, False, False, True),
        "A6_auto_hybrid": (False, True, True, True),
        "manual_rich_ceiling": (True, True, True, True),
    }
    overall = {(row["condition"], row["stratum"]): row for row in summaries}
    ablation_rows = []
    previous = None
    for condition in primary_order:
        row = overall[(condition, "all")]
        manual, synonyms, inferred, embedding = metadata[condition]
        ablation_rows.append({
            "policy": condition,
            "manual_metadata": int(manual),
            "synonyms": int(synonyms),
            "inferred_concepts": int(inferred),
            "embedding": int(embedding),
            "embedding_representation": selected_embedding if condition in {"A5_embedding_selected", "A6_auto_hybrid", "manual_rich_ceiling"} else "",
            "top1": row["top1"],
            "recall_at_3": row["recall_at_3"],
            "recall_at_5": row["recall_at_5"],
            "delta_top1_previous": "" if previous is None else float(row["top1"]) - float(previous["top1"]),
            "delta_recall_at_3_previous": "" if previous is None else float(row["recall_at_3"]) - float(previous["recall_at_3"]),
            "delta_recall_at_5_previous": "" if previous is None else float(row["recall_at_5"]) - float(previous["recall_at_5"]),
        })
        previous = row if condition not in {"A5_embedding_selected", "manual_rich_ceiling"} else previous
    _write_csv(args.output_dir / "auto_discovery_ablation.csv", ablation_rows)

    source_rows = _source_ablation_rows(views, queries, targets)
    _write_csv(args.output_dir / "auto_keyword_source_ablation.csv", source_rows)
    no_types = BASE_KEYWORD_SOURCES - {AutoEvidenceSource.TYPE_SCHEMA}
    type_rows = []
    for condition, sources in (("A1_without_type_schema", no_types), ("A1_with_type_schema", BASE_KEYWORD_SOURCES)):
        scores = _score_rows(views, queries, lambda text, language, view: weighted_keyword_score(text, view, sources=sources))
        type_rows.append({"condition": condition, "policy": "A1", "type_schema_terms": int(AutoEvidenceSource.TYPE_SCHEMA in sources), **_metrics(scores, targets, test_indices)})
    tag_without_types = _score_rows(
        views, queries,
        lambda text, language, view: len((set(concepts.concepts(text, language=language)["operation"]) | set(concepts.concepts(text, language=language)["object"])) & (set(view.operations) | set(view.objects))) / max(len(set(concepts.concepts(text, language=language)["operation"]) | set(concepts.concepts(text, language=language)["object"])), 1),
    )
    type_rows.extend((
        {"condition": "A4_without_type_schema", "policy": "A4", "type_schema_terms": 0, **_metrics(torch.maximum(a3, tag_without_types), targets, test_indices)},
        {"condition": "A4_with_type_schema", "policy": "A4", "type_schema_terms": 1, **_metrics(a4, targets, test_indices)},
    ))
    _write_csv(args.output_dir / "auto_type_schema_ablation.csv", type_rows)

    doc_variants = {
        "good": views,
        "minimal_one_line": tuple(
            _variant_view(
                view,
                remove_sources={
                    AutoEvidenceSource.DOCSTRING,
                    AutoEvidenceSource.PARAMETER_DESCRIPTION,
                    AutoEvidenceSource.RETURN_DESCRIPTION,
                },
                source_replacement=(
                    AutoEvidenceSource.DOCSTRING,
                    " ".join(dict(view.embedding_fields)["description"].split()[:4]),
                ),
                embedding_updates={"description": " ".join(dict(view.embedding_fields)["description"].split()[:4])},
            ) for view in views
        ),
        "none": tuple(
            _variant_view(
                view,
                remove_sources={AutoEvidenceSource.DOCSTRING, AutoEvidenceSource.PARAMETER_DESCRIPTION, AutoEvidenceSource.RETURN_DESCRIPTION},
                embedding_updates={"description": ""},
            ) for view in views
        ),
    }
    _write_csv(
        args.output_dir / "docstring_quality_ablation.csv",
        _quality_rows(doc_variants, queries, targets, encoder, query_vectors, selected_embedding, dimension="docstring"),
    )
    name_variants = {
        "descriptive": views,
        "abbreviated": tuple(
            _variant_view(
                view,
                remove_sources={AutoEvidenceSource.FUNCTION_NAME},
                source_replacement=(AutoEvidenceSource.FUNCTION_NAME, "_".join(part[:3] for part in view.name.split("_"))),
                embedding_updates={"name": "_".join(part[:3] for part in view.name.split("_"))},
            ) for view in views
        ),
        "opaque": tuple(
            _variant_view(
                view,
                remove_sources={AutoEvidenceSource.FUNCTION_NAME},
                source_replacement=(AutoEvidenceSource.FUNCTION_NAME, f"fn_{index:02d}"),
                embedding_updates={"name": f"fn_{index:02d}"},
            ) for index, view in enumerate(views, 1)
        ),
    }
    _write_csv(
        args.output_dir / "function_name_quality_ablation.csv",
        _quality_rows(name_variants, queries, targets, encoder, query_vectors, selected_embedding, dimension="function_name"),
    )

    baseline = overall[("A0_raw_bm25", "all")]
    ceiling = overall[("manual_rich_ceiling", "all")]
    retention_rows = []
    for condition in primary_order:
        row = overall[(condition, "all")]
        for metric in ("top1", "recall_at_3", "recall_at_5"):
            denominator = float(ceiling[metric]) - float(baseline[metric])
            retention_rows.append({
                "policy": condition,
                "metric": metric,
                "raw_lexical": baseline[metric],
                "manual_ceiling": ceiling[metric],
                "automatic_value": row[metric],
                "gain_retention": (float(row[metric]) - float(baseline[metric])) / denominator if denominator else math.nan,
            })
    _write_csv(args.output_dir / "auto_gain_retention.csv", retention_rows)

    union_channels = {
        "bm25": a0_bm25,
        "auto_keyword": a1,
        "keyword_synonym": a2,
        "auto_tag_concept": a4,
        "embedding": a5,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    candidate_rows, frontier, complementarity = _union_experiment(resources, queries, targets, union_channels, tokenizer)
    _write_csv(args.output_dir / "union_candidate_rows.csv", candidate_rows)
    _write_csv(args.output_dir / "union_vs_fusion_frontier.csv", frontier)
    _write_csv(args.output_dir / "union_channel_complementarity.csv", complementarity)

    validation_frontier = [row for row in frontier if row["split"] == "validation"]
    selected_union = max(
        (
            row for row in validation_frontier
            if row["strategy"] in {UnionStrategy.RAW_UNION.value, UnionStrategy.DIVERSITY_UNION.value}
        ),
        key=lambda row: (
            row["required_recall"], -row["unsafe_exposure"], row["useful_precision"],
            -row["mean_schema_tokens"], -row["max_candidates"], row["strategy"],
        ),
    )
    test_frontier = [row for row in frontier if row["split"] == "test"]
    selected_test = next(
        row for row in test_frontier
        if row["strategy"] == selected_union["strategy"] and row["max_candidates"] == selected_union["max_candidates"]
    )
    union_at_3 = next(row for row in test_frontier if row["strategy"] == selected_union["strategy"] and row["max_candidates"] == 3)
    union_at_5 = next(row for row in test_frontier if row["strategy"] == selected_union["strategy"] and row["max_candidates"] == 5)
    ablation_rows.insert(-1, {
        "policy": f"A7_{selected_union['strategy']}_k{selected_union['max_candidates']}",
        "manual_metadata": 0,
        "synonyms": 1,
        "inferred_concepts": 1,
        "embedding": 1,
        "embedding_representation": selected_embedding,
        "top1": "n/a",
        "recall_at_3": union_at_3["required_recall"],
        "recall_at_5": union_at_5["required_recall"],
        "union_selected_k": selected_union["max_candidates"],
        "union_selected_recall": selected_test["required_recall"],
        "delta_top1_previous": "n/a",
        "delta_recall_at_3_previous": "n/a",
        "delta_recall_at_5_previous": "n/a",
    })
    _write_csv(args.output_dir / "auto_discovery_ablation.csv", ablation_rows)

    for metric, value in (("recall_at_3", union_at_3["required_recall"]), ("recall_at_5", union_at_5["required_recall"])):
        denominator = float(ceiling[metric]) - float(baseline[metric])
        retention_rows.append({
            "policy": f"A7_{selected_union['strategy']}",
            "metric": metric,
            "raw_lexical": baseline[metric],
            "manual_ceiling": ceiling[metric],
            "automatic_value": value,
            "gain_retention": (float(value) - float(baseline[metric])) / denominator if denominator else "n/a",
        })
    _write_csv(args.output_dir / "auto_gain_retention.csv", retention_rows)

    gate_candidates = []
    for strategy in (UnionStrategy.RAW_UNION.value, UnionStrategy.DIVERSITY_UNION.value):
        for budget in K_VALUES:
            union = next(row for row in validation_frontier if row["strategy"] == strategy and row["max_candidates"] == budget)
            fused = next(row for row in validation_frontier if row["strategy"] == UnionStrategy.FUSED_SCORE.value and row["max_candidates"] == budget)
            if union["required_recall"] > fused["required_recall"] + 0.01 and union["unsafe_exposure"] <= fused["unsafe_exposure"] + 0.01:
                gate_candidates.append((union["required_recall"] - fused["required_recall"], -union["mean_schema_tokens"], strategy, budget))
    if gate_candidates:
        _, _, selected_strategy, selected_budget = max(gate_candidates)
        jit_rows = [{
            "status": "required_not_run_in_discovery_process",
            "gate": "open",
            "selected_strategy": selected_strategy,
            "selected_budget": selected_budget,
            "reason": "Run the frozen generative JIT harness in a separate low-memory process.",
        }]
    else:
        selected_strategy = ""
        selected_budget = ""
        jit_rows = [{
            "status": "not_run_gate_closed",
            "gate": "closed",
            "selected_strategy": "",
            "selected_budget": "",
            "reason": "Automatic union did not materially exceed matched-budget fused Top-K on validation with controlled unsafe exposure.",
        }]
    _write_csv(args.output_dir / "union_jit_ablation.csv", jit_rows)

    findings = {
        "schema_version": "1.0",
        "device": args.device,
        "selected_embedding_representation": selected_embedding,
        "embedding_validation": embedding_validation,
        "hybrid_weights": hybrid_weights,
        "hybrid_validation": hybrid_validation,
        "manual_selection": manual_selection,
        "union_jit_gate": jit_rows[0],
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "auto_discovery_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
