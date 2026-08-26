"""Freeze large-catalog palettes and realistic disclosure stress records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.large_tool_schemas import LARGE_SCHEMA_CALLABLES
from data.semantic_concepts import canonical_concept_map
from data.semantic_hard_tools import semantic_hardness_queries
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.prepare_progressive_disclosure import TOOL_CASES
from experiments.paper6_5_tools.run_scaled_auto_discovery import (
    A6_WEIGHTS,
    _automatic_scores,
    _bm25_scores,
    _build_catalog,
    _embedding_scores,
    _full_query,
)
from experiments.paper6_5_tools.scaled_candidate_sets import candidate_orders, stable_descending_order
from pra_hf.agent_resources import SideEffectClass
from pra_hf.context_records import serialize_record, tool_definition_record
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder
from pra_hf.tool_records import tool_record_from_callable
from pra_hf.union_discovery import agreement_rerank


OUTPUT = ROOT / "docs/papers/shared/results/paper6_5_tools/missing_experiments"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SIZES = (32, 128, 512, 2048, 8192)
SEEDS = (11, 23, 37, 53, 71)
BUDGETS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48)
POLICIES = (
    "A0_bm25", "A1_fused", "A2_raw_union", "A3_diversity_union",
    "A4_agreement_union",
)
CHANNELS = ("bm25", "auto_keyword", "keyword_synonym", "auto_tag_concept", "embedding")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0


def _channel_rankings(scores: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {name: stable_descending_order(value) for name, value in scores.items()}


def _resource_view(resource, tokenizer) -> dict[str, object]:
    record = tool_definition_record(resource)
    selection = serialize_record(record, view="selection")
    full = serialize_record(record, view="full")
    return {
        "uri": resource.uri,
        "name": resource.name,
        "namespace": resource.namespace,
        "version": resource.version,
        "tenant_id": resource.tenant_id,
        "description": resource.description,
        "signature": str(resource.metadata.get("signature", "")),
        "side_effect": resource.side_effect_class.value,
        "selection_payload": selection,
        "full_payload": full,
        "provider_schema": json.loads(resource.content),
        "selection_tokens": len(tokenizer.encode(selection, add_special_tokens=False)),
        "full_tokens": len(tokenizer.encode(full, add_special_tokens=False)),
    }


def _large_schema_rows(tokenizer) -> list[dict[str, object]]:
    rows = []
    for function in LARGE_SCHEMA_CALLABLES:
        resource = tool_record_from_callable(
            function, namespace="paper6_5_large_schema", tenant_id="paper6_5"
        ).to_agent_resource()
        view = _resource_view(resource, tokenizer)
        schema = view["provider_schema"]
        properties = schema["parameters"].get("properties", {})
        rows.append({
            "tool_name": resource.name,
            "selection_tokens": view["selection_tokens"],
            "full_schema_tokens": view["full_tokens"],
            "selection_full_ratio": view["selection_tokens"] / max(view["full_tokens"], 1),
            "parameter_count": len(properties),
            "required_parameter_count": len(schema["parameters"].get("required", ())),
            "nested_schema": 1,
            "executable": 1,
        })
    return rows


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    concepts = canonical_concept_map()
    query_by_id = {row.query_id: row for row in semantic_hardness_queries()}
    cases = tuple(query_by_id[row.query_id] for row in TOOL_CASES)
    encoder = CompactEmbeddingEncoder(
        str(args.model_root / "bge-small-en-v1.5"), revision=BGE_REVISION,
        device=args.device, query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    query_vectors = encoder.encode([_full_query(row) for row in cases], query=True)
    palettes: list[dict[str, object]] = []
    view_rows: dict[tuple[int, int, str], dict[str, object]] = {}
    timing_rows = []

    for size in args.sizes:
        for seed in args.seeds:
            started = time.perf_counter()
            resources, views, specs, costs = _build_catalog(size, seed, concepts)
            automatic, auto_cost = _automatic_scores(views, cases, concepts)
            bm25, _, bm25_cost = _bm25_scores(resources, cases)
            embedding, _, embedding_cost = _embedding_scores(encoder, query_vectors, views)
            fused = (
                automatic["keyword_synonym"] * A6_WEIGHTS["keyword_synonym"]
                + automatic["auto_tag_concept"] * A6_WEIGHTS["auto_tag_concept"]
                + embedding * A6_WEIGHTS["embedding"]
            )
            names = [row.name for row in resources]
            uris = [row.uri for row in resources]
            target_index = {
                spec.anchor_tool: index
                for index, spec in enumerate(specs)
                if spec.target
            }
            for query_index, query in enumerate(cases):
                one_scores = {
                    "bm25": bm25[query_index],
                    "auto_keyword": automatic["auto_keyword"][query_index],
                    "keyword_synonym": automatic["keyword_synonym"][query_index],
                    "auto_tag_concept": automatic["auto_tag_concept"][query_index],
                    "embedding": embedding[query_index],
                }
                rankings = _channel_rankings(one_scores)
                unions = candidate_orders(one_scores, max_candidates=max(args.budgets))
                policy_orders = {
                    "A0_bm25": stable_descending_order(bm25[query_index]),
                    "A1_fused": stable_descending_order(fused[query_index]),
                    "A2_raw_union": unions["raw_union"],
                    "A3_diversity_union": unions["diversity_union"],
                }
                score_maps = {
                    channel: {uris[index]: float(values[index]) for index in range(len(resources))}
                    for channel, values in one_scores.items()
                }
                useful_names = {query.required_tool, *query.useful_tools}
                for policy in POLICIES:
                    for budget in args.budgets:
                        if policy == "A4_agreement_union":
                            raw = policy_orders["A2_raw_union"][:budget]
                            reranked_uris = agreement_rerank(
                                score_maps,
                                candidate_uris=tuple(uris[index] for index in raw),
                                support_depth=budget,
                                agreement_weight=0.1,
                            )
                            by_uri = {uri: index for index, uri in enumerate(uris)}
                            selected = tuple(by_uri[uri] for uri in reranked_uris)
                        else:
                            selected = policy_orders[policy][:budget]
                        candidate_rows = []
                        for admission_rank, index in enumerate(selected, start=1):
                            sources = [
                                {"channel": channel, "rank": rankings[channel].index(index) + 1,
                                 "score": float(one_scores[channel][index])}
                                for channel in CHANNELS
                            ]
                            supported = [row for row in sources if row["rank"] <= budget]
                            candidate_rows.append({
                                "name": names[index], "uri": uris[index],
                                "admission_rank": admission_rank,
                                "sources": supported,
                                "channel_agreement": len(supported),
                            })
                            key = (size, seed, uris[index])
                            if key not in view_rows:
                                view_rows[key] = {
                                    "catalog_size": size, "seed": seed,
                                    **_resource_view(resources[index], tokenizer),
                                }
                        target = target_index[query.required_tool]
                        unsafe = [index for index in selected if resources[index].side_effect_class == SideEffectClass.DESTRUCTIVE]
                        palettes.append({
                            "catalog_size": size,
                            "seed": seed,
                            "query_id": query.query_id,
                            "query": _full_query(query),
                            "target_name": query.required_tool,
                            "policy": policy,
                            "max_candidates": budget,
                            "candidate_count": len(selected),
                            "target_in_palette": int(target in selected),
                            "candidate_names": [names[index] for index in selected],
                            "candidates": candidate_rows,
                            "unsafe_candidates": [names[index] for index in unsafe],
                            "useful_candidates": [names[index] for index in selected if names[index] in useful_names],
                        })
                target = target_index[query.required_tool]
                target_key = (size, seed, uris[target])
                if target_key not in view_rows:
                    view_rows[target_key] = {
                        "catalog_size": size, "seed": seed,
                        **_resource_view(resources[target], tokenizer),
                    }
            timing_rows.append({
                "catalog_size": size, "seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
                **costs, **auto_cost, **bm25_cost, **embedding_cost,
            })
            print(f"prepared N={size} seed={seed} rows={len(palettes)}", flush=True)

    _write_jsonl(args.output / "large_catalog_palettes.jsonl", palettes)
    _write_jsonl(args.output / "progressive_tool_views.jsonl", list(view_rows.values()))
    _write_csv(args.output / "large_catalog_prepare_costs.csv", timing_rows)
    stress = _large_schema_rows(tokenizer)
    _write_csv(args.output / "large_schema_stress.csv", stress)
    full_lengths = [int(row["full_schema_tokens"]) for row in stress]
    manifest = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "catalog_sizes": list(args.sizes),
        "seeds": list(args.seeds),
        "budgets": list(args.budgets),
        "policies": list(POLICIES),
        "query_ids": [row.query_id for row in cases],
        "palette_rows": len(palettes),
        "unique_materializable_views": len(view_rows),
        "large_schema_tokens": {
            "median": _percentile(full_lengths, .5),
            "p90": _percentile(full_lengths, .9),
            "p95": _percentile(full_lengths, .95),
            "max": max(full_lengths),
        },
    }
    (args.output / "missing_experiments_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
