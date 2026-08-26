"""Run five-seed zero-configuration discovery scaling through 8K callables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.scaled_callable_catalog import ScaledCallableSpec, generate_scaled_callable_catalog
from data.semantic_concepts import canonical_concept_map, dictionary_sources_manifest
from data.semantic_hard_tools import SemanticHardQuery, semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.scaled_candidate_sets import candidate_orders, stable_descending_order
from pra_hf.agent_resources import AgentResource, DiscoveryRequest, PersistentResourceIndex, SideEffectClass
from pra_hf.auto_tool_discovery import (
    AutoEvidenceSource,
    AutoToolSemanticView,
    automatic_query_terms,
    automatic_semantic_view,
)
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder
from pra_hf.tool_records import PythonTypeSchemaCache, ToolRecord, tool_record_from_callable


SIZES = (32, 128, 512, 2048, 8192)
SEEDS = (11, 23, 37, 53, 71)
K_VALUES = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48)
BASE_KEYWORD_SOURCES = frozenset({
    AutoEvidenceSource.FUNCTION_NAME,
    AutoEvidenceSource.DOCSTRING,
    AutoEvidenceSource.PARAMETER_NAME,
    AutoEvidenceSource.PARAMETER_DESCRIPTION,
    AutoEvidenceSource.RETURN_DESCRIPTION,
    AutoEvidenceSource.TYPE_SCHEMA,
    AutoEvidenceSource.MODULE_NAMESPACE,
})
DICTIONARY_SOURCES = frozenset({*BASE_KEYWORD_SOURCES, AutoEvidenceSource.DICTIONARY_EXPANSION})
CHANNEL_ORDER = ("bm25", "auto_keyword", "keyword_synonym", "auto_tag_concept", "embedding")
A6_WEIGHTS = {"keyword_synonym": 0.25, "auto_tag_concept": 0.25, "embedding": 0.50}
PROTOCOL_VERSION = "paper6_5_scaled_auto_v2_final_k_curve"


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


def _raw_resource(record: ToolRecord, view: AutoToolSemanticView, spec: ScaledCallableSpec) -> AgentResource:
    source = record.to_agent_resource()
    return replace(
        source,
        description=view.raw_text,
        aliases=(),
        metadata={
            "zero_config_raw_callable": True,
            "scaled_target": spec.target,
            "anchor_tool": spec.anchor_tool,
            "generated_operation": spec.canonical_operation,
            "generated_object": spec.canonical_object,
            "confusion_axis": spec.confusion_axis,
        },
    )


def _build_catalog(
    size: int,
    seed: int,
    concepts,
) -> tuple[tuple[AgentResource, ...], tuple[AutoToolSemanticView, ...], tuple[ScaledCallableSpec, ...], dict[str, float]]:
    started = time.perf_counter()
    specs = generate_scaled_callable_catalog(size, seed=seed)
    generated_ms = (time.perf_counter() - started) * 1000.0
    cache = PythonTypeSchemaCache()
    started = time.perf_counter()
    items = []
    for spec in specs:
        record = tool_record_from_callable(
            spec.function,
            namespace="paper6_5_scaled",
            tenant_id="paper6_5",
            concept_map=concepts,
            type_cache=cache,
            metadata={"scaled_catalog": True},
        )
        view = automatic_semantic_view(record, concepts=concepts)
        resource = _raw_resource(record, view, spec)
        items.append((resource, view, spec))
    inspection_ms = (time.perf_counter() - started) * 1000.0
    # Stable URI order gives every tensorized tie the runtime's URI tie-break.
    items.sort(key=lambda row: row[0].uri)
    return (
        tuple(row[0] for row in items),
        tuple(row[1] for row in items),
        tuple(row[2] for row in items),
        {"callable_generation_ms": generated_ms, "inspection_and_view_ms": inspection_ms},
    )


def _score_weight_maps(query_terms: frozenset[str], maps: Sequence[Mapping[str, float]]) -> torch.Tensor:
    if not query_terms:
        return torch.zeros(len(maps), dtype=torch.float32)
    denominator = len(query_terms)
    return torch.tensor(
        [sum(values.get(term, 0.0) for term in query_terms) / denominator for values in maps],
        dtype=torch.float32,
    )


def _automatic_scores(
    views: Sequence[AutoToolSemanticView],
    queries: Sequence[SemanticHardQuery],
    concepts,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    base_maps = tuple(view.weighted_terms(sources=BASE_KEYWORD_SOURCES) for view in views)
    expanded_maps = tuple(view.weighted_terms(sources=DICTIONARY_SOURCES) for view in views)
    dictionary_maps = tuple(
        view.weighted_terms(sources={AutoEvidenceSource.DICTIONARY_EXPANSION}) for view in views
    )
    a1 = torch.zeros((len(queries), len(views)), dtype=torch.float32)
    a2 = torch.zeros_like(a1)
    a4 = torch.zeros_like(a1)
    a2_timings = []
    a4_timings = []
    for query_index, query in enumerate(queries):
        text = _full_query(query)
        raw_terms = automatic_query_terms(text)
        expanded_terms = automatic_query_terms(
            text,
            concepts=concepts,
            language=query.language,
            expand=True,
        )
        started = time.perf_counter_ns()
        base = _score_weight_maps(raw_terms, base_maps)
        expanded = _score_weight_maps(expanded_terms, expanded_maps)
        direct = _score_weight_maps(expanded_terms, dictionary_maps)
        a1[query_index] = base
        a2[query_index] = torch.maximum(base, torch.maximum(expanded, direct))
        a2_timings.append((time.perf_counter_ns() - started) / 1_000_000.0)

        started = time.perf_counter_ns()
        query_concepts = concepts.concepts(text, language=query.language)
        query_tags = set(query_concepts["operation"]) | set(query_concepts["object"])
        concept_scores = []
        tag_scores = []
        for view in views:
            operation = max(
                (query_concepts["operation"].get(value, 0.0) for value in view.operations),
                default=0.0,
            )
            objects = max(
                (query_concepts["object"].get(value, 0.0) for value in view.objects),
                default=0.0,
            )
            concept_scores.append(0.65 * operation + 0.35 * objects)
            tag_scores.append(
                len(query_tags & set(view.auto_tags)) / len(query_tags) if query_tags else 0.0
            )
        concepts_row = torch.tensor(concept_scores, dtype=torch.float32)
        tags_row = torch.tensor(tag_scores, dtype=torch.float32)
        a4[query_index] = torch.maximum(a2[query_index], torch.maximum(concepts_row, tags_row))
        a4_timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return (
        {"auto_keyword": a1, "keyword_synonym": a2, "auto_tag_concept": a4},
        {
            "a2_score_mean_ms": statistics.mean(a2_timings),
            "a2_score_p95_ms": _percentile(a2_timings, .95),
            "a4_score_mean_ms": statistics.mean(a4_timings),
            "a4_score_p95_ms": _percentile(a4_timings, .95),
        },
    )


def _bm25_scores(
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
) -> tuple[torch.Tensor, PersistentResourceIndex, dict[str, float]]:
    started = time.perf_counter()
    index = PersistentResourceIndex(resources)
    build_ms = (time.perf_counter() - started) * 1000.0
    by_uri = {resource.uri: idx for idx, resource in enumerate(resources)}
    scores = torch.zeros((len(queries), len(resources)), dtype=torch.float32)
    timings = []
    for query_index, query in enumerate(queries):
        request = DiscoveryRequest(
            query=_full_query(query),
            tenant_id="paper6_5",
            top_k=len(resources),
        )
        started = time.perf_counter_ns()
        rows = index.score(request, channels=("index",))
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        for row in rows:
            scores[query_index, by_uri[row.uri]] = row.index
    return scores, index, {
        "lexical_index_build_ms": build_ms,
        "bm25_score_mean_ms": statistics.mean(timings),
        "bm25_score_p95_ms": _percentile(timings, .95),
    }


def _embedding_scores(
    encoder: CompactEmbeddingEncoder,
    query_vectors: torch.Tensor,
    views: Sequence[AutoToolSemanticView],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    texts = [str(view.embedding_text("name_description")) for view in views]
    started = time.perf_counter()
    vectors = encoder.encode(texts)
    build_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter_ns()
    scores = ((query_vectors @ vectors.T) + 1.0).div(2.0).clamp(0.0, 1.0)
    score_ms = (time.perf_counter_ns() - started) / 1_000_000.0 / max(len(query_vectors), 1)
    return scores, vectors, {
        "embedding_build_ms": build_ms,
        "embedding_score_amortized_ms": score_ms,
    }


def _schema_token_lengths(tokenizer, resources: Sequence[AgentResource]) -> tuple[int, ...]:
    output = []
    values = [resource.content for resource in resources]
    for start in range(0, len(values), 512):
        encoded = tokenizer(
            values[start : start + 512],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        output.extend(int(value) for value in encoded["length"])
    return tuple(output)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _quality_rows(
    *,
    size: int,
    seed: int,
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
    policy_scores: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, object]], dict[str, tuple[tuple[int, ...], ...]], dict[str, float]]:
    rows = []
    orders = {}
    timings = {}
    for policy, scores in policy_scores.items():
        started = time.perf_counter_ns()
        policy_orders = tuple(stable_descending_order(scores[index]) for index in range(len(queries)))
        timings[f"{policy}_rank_amortized_ms"] = (
            (time.perf_counter_ns() - started) / 1_000_000.0 / max(len(queries), 1)
        )
        orders[policy] = policy_orders
        for query_index, query in enumerate(queries):
            rank = policy_orders[query_index].index(targets[query_index]) + 1
            rows.append({
                "catalog_size": size,
                "seed": seed,
                "query_id": query.query_id,
                "hardness_level": query.hardness_level,
                "language": query.language,
                "policy": policy,
                "rank": rank,
                "top1": int(rank == 1),
                "recall_at_3": int(rank <= 3),
                "recall_at_5": int(rank <= 5),
                "recall_at_10": int(rank <= 10),
            })
    return rows, orders, timings


def _candidate_rows(
    *,
    size: int,
    seed: int,
    resources: Sequence[AgentResource],
    queries: Sequence[SemanticHardQuery],
    targets: Sequence[int],
    channel_scores: Mapping[str, torch.Tensor],
    policy_orders: Mapping[str, tuple[tuple[int, ...], ...]],
    schema_tokens: Sequence[int],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    by_name = {resource.name: index for index, resource in enumerate(resources)}
    max_k = max(K_VALUES)
    rows = []
    timings = []
    for query_index, query in enumerate(queries):
        one_query_channels = {name: values[query_index] for name, values in channel_scores.items()}
        started = time.perf_counter_ns()
        unions = candidate_orders(one_query_channels, max_candidates=max_k)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        strategies = {
            "A2_topk": policy_orders["A2_keywords_synonyms"][query_index][:max_k],
            "A5_topk": policy_orders["A5_embedding"][query_index][:max_k],
            "A6_topk": policy_orders["A6_auto_hybrid"][query_index][:max_k],
            "A7_fused_score": unions["fused_score"],
            "A7_raw_union": unions["raw_union"],
            "A7_diversity_union": unions["diversity_union"],
        }
        useful = {by_name[name] for name in (query.required_tool, *query.useful_tools) if name in by_name}
        for strategy, order in strategies.items():
            for budget in K_VALUES:
                selected = order[: min(budget, len(order))]
                unsafe = {
                    index for index in selected
                    if index != targets[query_index]
                    and resources[index].side_effect_class == SideEffectClass.DESTRUCTIVE
                }
                rows.append({
                    "catalog_size": size,
                    "seed": seed,
                    "query_id": query.query_id,
                    "hardness_level": query.hardness_level,
                    "language": query.language,
                    "strategy": strategy,
                    "max_candidates": budget,
                    "candidate_count": len(selected),
                    "required_recall": int(targets[query_index] in selected),
                    "useful_precision": len(useful & set(selected)) / max(len(selected), 1),
                    "unsafe_exposure": int(bool(unsafe)),
                    "unsafe_candidate_count": len(unsafe),
                    "context_tokens": sum(schema_tokens[index] for index in selected),
                })
    return rows, {
        "a7_candidate_build_mean_ms": statistics.mean(timings),
        "a7_candidate_build_p95_ms": _percentile(timings, .95),
    }


def _semantic_bytes(views: Sequence[AutoToolSemanticView]) -> int:
    return sum(
        len(row.term.encode("utf-8"))
        + len(row.surface.encode("utf-8"))
        + len(row.source.value.encode("utf-8"))
        + 16
        for view in views
        for row in view.evidence
    )


def _run_shard(
    *,
    size: int,
    seed: int,
    queries: Sequence[SemanticHardQuery],
    concepts,
    encoder: CompactEmbeddingEncoder,
    query_vectors: torch.Tensor,
    tokenizer,
    protocol_hash: str,
) -> dict[str, object]:
    resources, views, specs, build_costs = _build_catalog(size, seed, concepts)
    target_by_name = {
        resource.name: index
        for index, (resource, spec) in enumerate(zip(resources, specs))
        if spec.target
    }
    targets = [target_by_name[query.required_tool] for query in queries]

    automatic, automatic_costs = _automatic_scores(views, queries, concepts)
    bm25, lexical_index, bm25_costs = _bm25_scores(resources, queries)
    embedding, embedding_vectors, embedding_costs = _embedding_scores(encoder, query_vectors, views)
    started = time.perf_counter_ns()
    a6 = (
        automatic["keyword_synonym"] * A6_WEIGHTS["keyword_synonym"]
        + automatic["auto_tag_concept"] * A6_WEIGHTS["auto_tag_concept"]
        + embedding * A6_WEIGHTS["embedding"]
    )
    a6_ms = (time.perf_counter_ns() - started) / 1_000_000.0 / max(len(queries), 1)
    quality, policy_orders, ranking_costs = _quality_rows(
        size=size,
        seed=seed,
        queries=queries,
        targets=targets,
        policy_scores={
            "A2_keywords_synonyms": automatic["keyword_synonym"],
            "A5_embedding": embedding,
            "A6_auto_hybrid": a6,
        },
    )
    schema_tokens = _schema_token_lengths(tokenizer, resources)
    channels = {
        "bm25": bm25,
        "auto_keyword": automatic["auto_keyword"],
        "keyword_synonym": automatic["keyword_synonym"],
        "auto_tag_concept": automatic["auto_tag_concept"],
        "embedding": embedding,
    }
    candidates, candidate_costs = _candidate_rows(
        size=size,
        seed=seed,
        resources=resources,
        queries=queries,
        targets=targets,
        channel_scores=channels,
        policy_orders=policy_orders,
        schema_tokens=schema_tokens,
    )
    cost = {
        "catalog_size": size,
        "seed": seed,
        **build_costs,
        **automatic_costs,
        **bm25_costs,
        **embedding_costs,
        **ranking_costs,
        **candidate_costs,
        "a6_fusion_amortized_ms": a6_ms,
        "lexical_index_bytes": lexical_index.estimated_bytes,
        "automatic_semantic_bytes": _semantic_bytes(views),
        "embedding_bytes": int(embedding_vectors.numel() * embedding_vectors.element_size()),
        "total_discovery_component_bytes": (
            lexical_index.estimated_bytes
            + _semantic_bytes(views)
            + int(embedding_vectors.numel() * embedding_vectors.element_size())
        ),
        "logical_catalog_schema_tokens": sum(schema_tokens),
        "mean_schema_tokens": statistics.mean(schema_tokens),
        "destructive_tools": sum(
            resource.side_effect_class == SideEffectClass.DESTRUCTIVE for resource in resources
        ),
        "target_tools": sum(spec.target for spec in specs),
    }
    return {
        "protocol_hash": protocol_hash,
        "catalog_size": size,
        "seed": seed,
        "quality_rows": quality,
        "candidate_rows": candidates,
        "cost": cost,
    }


def _shard_path(output: Path, size: int, seed: int) -> Path:
    return output / "shards" / f"size_{size}_seed_{seed}.json.gz"


def _write_shard(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)


def _read_shard(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    queries = tuple(row for row in semantic_hardness_queries() if row.split == "test")
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "catalog_sizes": list(args.sizes),
        "seeds": list(args.seeds),
        "candidate_budgets": list(K_VALUES),
        "target_tools": 18,
        "test_queries": len(queries),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "embedding_representation": "name_description",
        "a6_weights": A6_WEIGHTS,
        "channels": list(CHANNEL_ORDER),
        "dictionary": dictionary_sources_manifest(),
        "distractor_rule": "shares exactly one canonical operation/object facet with an anchor",
        "destructive_rule": "every fifth eligible distractor uses delete with the anchor object",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        **protocol,
        "protocol_hash": protocol_hash,
        "runtime": runtime_metadata(),
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    (args.output / "scaled_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    concepts = canonical_concept_map()
    encoder = CompactEmbeddingEncoder(
        str(args.model_root / "bge-small-en-v1.5"),
        revision=protocol["embedding_revision"],
        device=args.device,
        query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    started = time.perf_counter()
    query_vectors = encoder.encode([_full_query(row) for row in queries], query=True)
    query_encoding_ms = (time.perf_counter() - started) * 1000.0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)

    for size in args.sizes:
        for seed in args.seeds:
            shard = _shard_path(args.output, size, seed)
            if shard.exists() and not args.fresh:
                existing = _read_shard(shard)
                if existing.get("protocol_hash") == protocol_hash:
                    print(f"resume size={size} seed={seed}", flush=True)
                    continue
            started = time.perf_counter()
            payload = _run_shard(
                size=size,
                seed=seed,
                queries=queries,
                concepts=concepts,
                encoder=encoder,
                query_vectors=query_vectors,
                tokenizer=tokenizer,
                protocol_hash=protocol_hash,
            )
            payload["elapsed_seconds"] = time.perf_counter() - started
            _write_shard(shard, payload)
            print(
                f"completed size={size} seed={seed} elapsed={payload['elapsed_seconds']:.1f}s",
                flush=True,
            )

    quality_rows = []
    candidate_rows = []
    costs = []
    for size in args.sizes:
        for seed in args.seeds:
            payload = _read_shard(_shard_path(args.output, size, seed))
            if payload.get("protocol_hash") != protocol_hash:
                raise RuntimeError(f"Stale scaling shard for size={size}, seed={seed}.")
            quality_rows.extend(payload["quality_rows"])
            candidate_rows.extend(payload["candidate_rows"])
            costs.append({
                **payload["cost"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "query_encoding_batch_ms": query_encoding_ms,
                "query_encoding_amortized_ms": query_encoding_ms / len(queries),
            })
    _write_csv(args.output / "scaled_auto_quality_rows.csv", quality_rows)
    _write_csv(args.output / "scaled_union_candidate_rows.csv", candidate_rows)
    _write_csv(args.output / "scaled_discovery_costs.csv", costs)
    print(json.dumps({
        "output": str(args.output),
        "quality_rows": len(quality_rows),
        "candidate_rows": len(candidate_rows),
        "cost_rows": len(costs),
        "query_encoding_ms": query_encoding_ms,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_scaling",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
