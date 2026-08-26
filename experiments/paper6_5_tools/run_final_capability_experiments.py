"""Run mixed tool/skill discovery and lazy capability lifecycle experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.declarative_skills import declarative_skill_catalog, skill_semantic_hard_queries
from data.semantic_concepts import canonical_concept_map
from data.semantic_hard_tools import semantic_hardness_queries
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.prepare_missing_experiments import _full_query
from experiments.paper6_5_tools.run_missing_experiments import (
    OllamaLabelModel,
    _batch_choose,
    _choice_prompt,
)
from experiments.paper6_5_tools.run_progressive_disclosure import FrozenCapabilityModel
from experiments.paper6_5_tools.run_scaled_auto_discovery import _build_catalog
from pra_hf.agent_resources import DiscoveryRequest, PersistentResourceIndex
from pra_hf.capability_runtime import CapabilityEncodingPolicy, LazyCapabilityRuntime
from pra_hf.context_records import RecordType, serialize_record, tool_definition_record
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder


OUTPUT = ROOT / "docs/papers/shared/results/paper6_5_tools/final_curves"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
K_VALUES = (4, 8, 12, 16, 24, 32)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _rank(scores: torch.Tensor) -> tuple[int, ...]:
    return tuple(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index)))


def _minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)


def _native_kv_bytes_per_token() -> int:
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return 2 * layers * kv_heads * head_dim * 2


def _mixed_registry(args, tokenizer, encoder):
    concepts = canonical_concept_map()
    tool_resources, _views, specs, build_costs = _build_catalog(
        args.catalog_size, args.catalog_seed, concepts
    )
    tool_records = tuple(tool_definition_record(resource) for resource in tool_resources)
    skills = declarative_skill_catalog()
    skill_resources = tuple(skill.to_agent_resource() for skill in skills)
    skill_records = tuple(skill.to_context_record() for skill in skills)
    records = (*tool_records, *skill_records)
    resources = (*tool_resources, *skill_resources)
    selection_text = tuple(serialize_record(record, view="selection") for record in records)
    discovery_resources = tuple(
        replace(resource, description=text, content="")
        for resource, text in zip(resources, selection_text)
    )
    lexical_started = time.perf_counter()
    lexical = PersistentResourceIndex(discovery_resources)
    lexical_build = time.perf_counter() - lexical_started
    embedding_started = time.perf_counter()
    vectors = encoder.encode(selection_text)
    embedding_build = time.perf_counter() - embedding_started
    return {
        "resources": resources,
        "records": records,
        "selection_text": selection_text,
        "selection_tokens": tuple(
            len(tokenizer.encode(text, add_special_tokens=False)) for text in selection_text
        ),
        "full_tokens": tuple(
            len(tokenizer.encode(serialize_record(record, view="full"), add_special_tokens=False))
            for record in records
        ),
        "lexical": lexical,
        "vectors": vectors,
        "tool_specs": specs,
        "build_costs": {
            **build_costs,
            "lexical_build_seconds": lexical_build,
            "embedding_build_seconds": embedding_build,
        },
    }


def _mixed_queries(registry) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target_tools = {
        spec.anchor_tool
        for spec in registry["tool_specs"]
        if spec.target
    }
    rows = []
    for query in semantic_hardness_queries():
        if query.required_tool in target_tools:
            rows.append({
                "query_id": query.query_id,
                "split": query.split,
                "query": _full_query(query),
                "target_name": query.required_tool,
                "target_type": "tool",
                "hardness_level": query.hardness_level,
                "family": query.canonical_object,
            })
    rows.extend({
        "query_id": query.query_id,
        "split": query.split,
        "query": query.query,
        "target_name": query.target_skill,
        "target_type": "skill",
        "hardness_level": query.hardness_level,
        "family": query.family,
    } for query in skill_semantic_hard_queries())
    return (
        [row for row in rows if row["split"] == "validation"],
        [row for row in rows if row["split"] == "test"],
    )


def _balanced_cases(rows: Sequence[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """Choose a deterministic, type-balanced model-facing cohort."""

    if limit <= 0 or limit >= len(rows):
        return list(rows)
    tools = [row for row in rows if row["target_type"] == "tool"]
    skills = [row for row in rows if row["target_type"] == "skill"]
    half = limit // 2
    selected = tools[:half] + skills[:half]
    remainder = limit - len(selected)
    if remainder:
        used = {row["query_id"] for row in selected}
        available = [row for row in rows if row["query_id"] not in used]
        selected.extend(available[:remainder])
    return selected


def _mixed_scores(registry, queries, encoder):
    resources = registry["resources"]
    by_uri = {resource.uri: index for index, resource in enumerate(resources)}
    lexical_rows = []
    for query in queries:
        values = torch.zeros(len(resources), dtype=torch.float32)
        for score in registry["lexical"].score(
            DiscoveryRequest(str(query["query"]), tenant_id="paper6_5", top_k=len(resources)),
            channels=("index",),
        ):
            values[by_uri[score.uri]] = score.index
        lexical_rows.append(_minmax(values))
    query_vectors = encoder.encode([str(row["query"]) for row in queries], query=True)
    semantic = (query_vectors @ registry["vectors"].T).clamp(-1, 1)
    semantic = torch.stack([_minmax(row) for row in semantic])
    return torch.stack(lexical_rows), semantic


def _target_indices(registry, queries) -> list[int]:
    by_key = {
        (resource.kind, resource.name): index
        for index, resource in enumerate(registry["resources"])
    }
    return [by_key[(str(row["target_type"]), str(row["target_name"]))] for row in queries]


def _choose_fusion_weight(registry, validation, encoder) -> float:
    lexical, semantic = _mixed_scores(registry, validation, encoder)
    targets = _target_indices(registry, validation)
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    return max(
        candidates,
        key=lambda weight: (
            statistics.fmean(
                float(_rank(lexical[index] * weight + semantic[index] * (1 - weight))[0] == target)
                for index, target in enumerate(targets)
            ),
            -abs(weight - 0.5),
        ),
    )


def run_mixed(args, tokenizer, encoder: CompactEmbeddingEncoder | None) -> dict[str, object]:
    checkpoint_path = args.output / "mixed_capability_prepared.json"
    checkpoint = None
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if candidate.get("protocol") == {
            "catalog_size": args.catalog_size,
            "catalog_seed": args.catalog_seed,
            "max_mixed_cases": args.max_mixed_cases,
            "k_values": list(K_VALUES),
        }:
            checkpoint = candidate

    if checkpoint is None:
        if encoder is None:
            raise RuntimeError("Mixed capability preparation requires an embedding encoder.")
        registry = _mixed_registry(args, tokenizer, encoder)
        validation, test = _mixed_queries(registry)
        weight = _choose_fusion_weight(registry, validation, encoder)
        model_test = _balanced_cases(test, args.max_mixed_cases)
        lexical, semantic = _mixed_scores(registry, model_test, encoder)
        targets = _target_indices(registry, model_test)
        resources = registry["resources"]
        prompts = []
        pending = []
        for query_index, (query, target) in enumerate(zip(model_test, targets)):
            order = _rank(lexical[query_index] * weight + semantic[query_index] * (1 - weight))
            for budget in K_VALUES:
                selected = order[:budget]
                labels = tuple(f"{resources[index].kind}:{resources[index].name}" for index in selected)
                payload = "\n".join(registry["selection_text"][index] for index in selected)
                target_label = f"{query['target_type']}:{query['target_name']}"
                base = {
                **query,
                "catalog_tools": args.catalog_size,
                "catalog_skills": len(declarative_skill_catalog()),
                "max_candidates": budget,
                "target_label": target_label,
                "target_in_palette": int(target in selected),
                "candidate_labels": "|".join(labels),
                "candidate_tool_count": sum(resources[index].kind == "tool" for index in selected),
                "candidate_skill_count": sum(resources[index].kind == "skill" for index in selected),
                "selection_tokens": sum(registry["selection_tokens"][index] for index in selected),
                "full_all_tokens": sum(registry["full_tokens"][index] for index in selected),
                "progressive_tokens_oracle": (
                    sum(registry["selection_tokens"][index] for index in selected)
                    + registry["full_tokens"][target]
                ),
                }
                pending.append((base, labels))
                prompts.append(_choice_prompt(str(query["query"]), payload, "typed capability"))
        checkpoint = {
            "protocol": {
                "catalog_size": args.catalog_size,
                "catalog_seed": args.catalog_seed,
                "max_mixed_cases": args.max_mixed_cases,
                "k_values": list(K_VALUES),
            },
            "prompts": prompts,
            "pending": [{"base": base, "labels": list(labels)} for base, labels in pending],
            "preparation": {
                "validation_rows": len(validation),
                "test_rows": len(test),
                "model_choice_rows": len(model_test),
                "fusion_lexical_weight": weight,
                "fusion_embedding_weight": 1 - weight,
                "catalog_tools": args.catalog_size,
                "catalog_skills": len(declarative_skill_catalog()),
                "build_costs": registry["build_costs"],
            },
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        del lexical, semantic, registry
    else:
        prompts = list(checkpoint["prompts"])
        pending = [
            (dict(row["base"]), tuple(row["labels"])) for row in checkpoint["pending"]
        ]

    # The compact encoder and frozen generator are never needed simultaneously.
    # Releasing it here avoids a multi-GB peak on small CPU/GPU workstations.
    if encoder is not None and hasattr(encoder, "model"):
        del encoder.model
    gc.collect()

    result_path = args.output / "mixed_capability_k_curve.csv"
    rows = _read_csv(result_path)
    completed = {
        (str(row["query_id"]), int(row["max_candidates"])) for row in rows
    }
    jobs = [
        (prompt, base, labels)
        for prompt, (base, labels) in zip(prompts, pending)
        if (str(base["query_id"]), int(base["max_candidates"])) not in completed
    ]
    model = (
        OllamaLabelModel(args.ollama_model)
        if args.choice_backend == "ollama"
        else FrozenCapabilityModel(torch.device(args.device))
    )
    for prompt, base, labels in jobs:
        if isinstance(model, OllamaLabelModel):
            choice, costs = model.choose(prompt, labels)
        else:
            choice, costs = _batch_choose(
                model, (prompt,), candidates=(labels,), batch_size=args.batch_size
            )[0]
        chosen_type = choice.split(":", 1)[0] if ":" in choice else ""
        rows.append({
            **base,
            "chosen_label": choice,
            "capability_type_correct": int(chosen_type == base["target_type"]),
            "resource_choice_correct": int(choice == base["target_label"]),
            "wrong_type_choice": int(bool(chosen_type) and chosen_type != base["target_type"]),
            "conditional_choice_denominator": base["target_in_palette"],
            **costs,
        })
        _write_csv(result_path, rows)
    return {
        **checkpoint["preparation"],
        "choice_rows": len(rows),
        "choice_backend": args.choice_backend,
        "resumed_from_prepared_checkpoint": checkpoint_path.exists(),
    }


def _runtime_lifecycle_rows(tokenizer) -> list[dict[str, object]]:
    tool_records = tuple(
        tool_definition_record(resource) for resource in realistic_tool_catalog()
    )
    skill_records = tuple(skill.to_context_record() for skill in declarative_skill_catalog())
    bytes_per_token = _native_kv_bytes_per_token()
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    encoder = lambda text: tokenizer.encode(text, add_special_tokens=False)
    rows = []
    for resource_type, records in (("tool", tool_records), ("skill", skill_records)):
        started = time.perf_counter()
        eager = LazyCapabilityRuntime(
            records,
            policy=CapabilityEncodingPolicy(lazy_selection=False, lazy_full=False),
            token_counter=counter,
            encoder=encoder,
            native_kv_bytes_per_token=bytes_per_token,
        )
        eager_seconds = time.perf_counter() - started
        eager_accounting = eager.accounting()

        started = time.perf_counter()
        lazy = LazyCapabilityRuntime(
            records,
            token_counter=counter,
            encoder=encoder,
            native_kv_bytes_per_token=bytes_per_token,
        )
        lazy_registration = time.perf_counter() - started
        started = time.perf_counter()
        first_palette = lazy.activate_selection_palette([record.record_id for record in records])
        first_selection = time.perf_counter() - started
        cold_full = []
        for record in records:
            started = time.perf_counter()
            value = lazy.activate_selected(record.record_id)
            cold_full.append(time.perf_counter() - started)
            assert value.semantic_rediscovery_calls == 0
            lazy.deactivate()
            lazy.activate_selection_palette([item.record_id for item in records])
        warm_full = []
        for record in records:
            started = time.perf_counter()
            value = lazy.activate_selected(record.record_id)
            warm_full.append(time.perf_counter() - started)
            assert value.cache_hit
            lazy.deactivate()
            lazy.activate_selection_palette([item.record_id for item in records])
        lazy_accounting = lazy.accounting()
        selection_tokens = sum(counter(serialize_record(record, view="selection")) for record in records)
        full_tokens = sum(counter(serialize_record(record, view="full")) for record in records)
        backing_bytes = sum(record.size_bytes for record in records)
        rows.append({
            "resource_type": resource_type,
            "records": len(records),
            "eager_startup_seconds": eager_seconds,
            "lazy_registration_seconds": lazy_registration,
            "lazy_first_selection_seconds": first_selection,
            "lazy_first_full_mean_seconds": statistics.fmean(cold_full),
            "lazy_warm_full_mean_seconds": statistics.fmean(warm_full),
            "selection_cache_hits_on_first_use": first_palette.cache_hits,
            "selection_cold_encodes_on_first_use": first_palette.cold_encodes,
            "selection_tokens": selection_tokens,
            "full_tokens": full_tokens,
            "backing_store_bytes": backing_bytes,
            "eager_resident_encoded_bytes": eager_accounting["resident_encoded_bytes"],
            "lazy_resident_after_reuse_bytes": lazy_accounting["resident_encoded_bytes"],
            "active_selection_bytes": selection_tokens * bytes_per_token,
            "active_full_mean_bytes": statistics.fmean(
                counter(serialize_record(record, view="full")) * bytes_per_token
                for record in records
            ),
            "native_kv_bytes_per_token": bytes_per_token,
            "semantic_rediscovery_calls": 0,
        })
    return rows


def run(args) -> None:
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    manifest_path = args.output / "final_capability_manifest.json"
    manifest: dict[str, object] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    manifest.update({
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": args.device,
        "phases": list(args.phases),
        "k_values": list(K_VALUES),
    })
    if "mixed" in args.phases:
        prepared = args.output / "mixed_capability_prepared.json"
        encoder = None if prepared.exists() else CompactEmbeddingEncoder(
                str(args.model_root / "bge-small-en-v1.5"),
                revision=BGE_REVISION,
                device=args.device,
                query_prefix="Represent this sentence for searching relevant passages: ",
                pooling="cls",
            )
        manifest["mixed"] = run_mixed(args, tokenizer, encoder)
    if "lazy" in args.phases:
        rows = _runtime_lifecycle_rows(tokenizer)
        _write_csv(args.output / "lazy_encoding_economics.csv", rows)
        manifest["lazy_rows"] = len(rows)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phases", nargs="+", choices=("mixed", "lazy"), default=("mixed", "lazy"))
    parser.add_argument("--catalog-size", type=int, default=8192)
    parser.add_argument("--catalog-seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-mixed-cases", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--choice-backend", choices=("ollama", "hf"), default="ollama")
    parser.add_argument("--ollama-model", default="qwen3:0.6b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
