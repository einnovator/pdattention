"""Run Paper 6.5 M6.5 external semantic-hard tool discovery.

Model and representation selection uses validation rows only. The frozen test
rows are scored after the English model, multilingual model, query scope,
representation, fusion weights, and staged confidence thresholds are fixed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.semantic_concepts import canonical_concept_map, dictionary_sources_manifest
from data.semantic_hard_tools import SemanticHardQuery, semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.agent_resources import ReliabilityCalibrator, normalize_text
from pra_hf.semantic_resource_discovery import (
    CompactEmbeddingEncoder,
    ExternalSemanticIndex,
    ToolSemanticCard,
    token_overlap,
)


@dataclass(frozen=True)
class EmbeddingModelSpec:
    key: str
    model_id: str
    revision: str
    local_name: str
    language_scope: str
    license: str
    pooling: str = "mean"
    query_prefix: str = ""


MODEL_SPECS = (
    EmbeddingModelSpec(
        "minilm_l6_en",
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "all-MiniLM-L6-v2",
        "en",
        "apache-2.0",
    ),
    EmbeddingModelSpec(
        "bge_small_en",
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "bge-small-en-v1.5",
        "en",
        "mit",
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    EmbeddingModelSpec(
        "minilm_l12_multi",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual",
        "apache-2.0",
    ),
)

POLICY_MODES = (
    "P0_token",
    "P1_bm25",
    "P2_dictionary",
    "P3_tags",
    "P4_lexical_dictionary_tags",
    "P5_english_embedding",
    "P6_multilingual_embedding",
    "P7_lexical_embedding",
    "P8_lexical_dictionary_embedding",
    "P9_oracle_identity",
    "P10_staged_external",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _full_text(row: SemanticHardQuery, *, scope: str) -> str:
    if scope == "context_query" and row.context:
        return f"Context: {row.context}\nQuery: {row.query}"
    return row.query


def _similarity(query: torch.Tensor, tools: torch.Tensor) -> torch.Tensor:
    if tools.ndim == 2:
        return query @ tools.T
    return torch.einsum("qd,tvd->qtv", query, tools).max(dim=-1).values


def _ranking_metrics(scores: torch.Tensor, targets: Sequence[int]) -> dict[str, float]:
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    target = torch.tensor(targets).reshape(-1, 1)
    ranks = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1": float((ranks == 1).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "recall_at_3": float((ranks <= 3).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "mean_rank": float(ranks.float().mean()),
    }


def _selection_key(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics["top1"]),
        float(metrics["mrr"]),
        float(metrics["recall_at_3"]),
        -float(metrics["mean_rank"]),
    )


def _embedding_card_texts(cards: Sequence[ToolSemanticCard], representation: str) -> list[str]:
    if representation == "E0_description":
        return [card.description_text for card in cards]
    if representation == "E1_name_description":
        return [card.name_description_text for card in cards]
    if representation == "E2_structured_card":
        return [card.structured_text for card in cards]
    raise ValueError(representation)


def _evaluate_embedding_models(
    rows: Sequence[SemanticHardQuery],
    cards: Sequence[ToolSemanticCard],
    target_indices: Sequence[int],
    *,
    model_root: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payloads: dict[str, dict[str, object]] = {}
    selection_rows = []
    model_manifest = []
    validation_indices = [index for index, row in enumerate(rows) if row.split == "validation"]
    for spec in MODEL_SPECS:
        local_path = model_root / spec.local_name
        encoder = CompactEmbeddingEncoder(
            str(local_path),
            revision=spec.revision,
            device=device,
            query_prefix=spec.query_prefix,
            pooling=spec.pooling,
            local_files_only=True,
        )
        representation_embeddings: dict[str, torch.Tensor] = {}
        registration_seconds = {}
        for representation in ("E0_description", "E1_name_description", "E2_structured_card"):
            started = time.perf_counter()
            representation_embeddings[representation] = encoder.encode(
                _embedding_card_texts(cards, representation), query=False
            )
            registration_seconds[representation] = time.perf_counter() - started
        started = time.perf_counter()
        flat_vectors = encoder.encode(
            [text for card in cards for text in card.vectors], query=False
        )
        representation_embeddings["E3_multi_vector"] = flat_vectors.reshape(len(cards), 3, -1)
        registration_seconds["E3_multi_vector"] = time.perf_counter() - started

        query_embeddings = {}
        query_encode_seconds = {}
        for scope in ("query_only", "context_query"):
            started = time.perf_counter()
            query_embeddings[scope] = encoder.encode(
                [_full_text(row, scope=scope) for row in rows], query=True
            )
            query_encode_seconds[scope] = time.perf_counter() - started

        best_key = None
        best_metrics = None
        all_scores = {}
        for representation, tool_vectors in representation_embeddings.items():
            for scope, query_vectors in query_embeddings.items():
                scores = _similarity(query_vectors, tool_vectors)
                config_key = f"{representation}|{scope}"
                all_scores[config_key] = scores
                if spec.language_scope == "en":
                    selected = [
                        index for index in validation_indices
                        if rows[index].language == "en" and rows[index].hardness_level in {"H1", "H2", "H3", "H4"}
                    ]
                else:
                    selected = [index for index in validation_indices if rows[index].hardness_level == "H5"]
                metrics = _ranking_metrics(
                    scores[selected],
                    [target_indices[index] for index in selected],
                )
                selection_rows.append({
                    "model": spec.key,
                    "model_id": spec.model_id,
                    "language_scope": spec.language_scope,
                    "representation": representation,
                    "query_scope": scope,
                    "split": "validation",
                    "selection_cohort": "english_h1_h4" if spec.language_scope == "en" else "multilingual_h5",
                    **metrics,
                    "registration_seconds": registration_seconds[representation],
                    "query_encoding_seconds": query_encode_seconds[scope],
                    "parameter_bytes": encoder.parameter_bytes,
                    "embedding_dimensions": encoder.dimensions,
                })
                if best_metrics is None or _selection_key(metrics) > _selection_key(best_metrics):
                    best_key, best_metrics = config_key, metrics
        assert best_key is not None and best_metrics is not None
        best_representation, best_scope = best_key.split("|")
        best_tools = representation_embeddings[best_representation]
        np.savez_compressed(
            output_dir / f"tool_embeddings_{spec.key}.npz",
            embeddings=best_tools.numpy(),
            uris=np.array([card.uri for card in cards]),
        )
        payloads[spec.key] = {
            "spec": spec,
            "representation": best_representation,
            "query_scope": best_scope,
            "scores": all_scores[best_key],
            "query_encode_seconds": query_encode_seconds[best_scope],
            "registration_seconds": registration_seconds[best_representation],
            "parameter_bytes": encoder.parameter_bytes,
            "embedding_bytes": int(best_tools.numel() * best_tools.element_size()),
            "dimensions": encoder.dimensions,
            "validation_metrics": best_metrics,
        }
        model_manifest.append({
            **asdict(spec),
            "cold_load_seconds": encoder.cold_load_seconds,
            "parameter_bytes": encoder.parameter_bytes,
            "embedding_dimensions": encoder.dimensions,
            "selected_representation": best_representation,
            "selected_query_scope": best_scope,
            "selection_metrics": best_metrics,
            "registration_seconds": registration_seconds[best_representation],
            "warm_query_encoding_seconds": query_encode_seconds[best_scope],
        })
        del encoder, representation_embeddings, query_embeddings, all_scores
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return payloads, selection_rows, model_manifest


def _rank_row(
    query: SemanticHardQuery,
    mode: str,
    scores: torch.Tensor,
    resources,
    *,
    latency_seconds: float,
    index_bytes: int,
    model_bytes: int,
    token_overlap_value: float,
    bm25_score: float,
    bm25_rank: int,
) -> dict[str, object]:
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    names = [resources[index].name for index in order]
    rank = names.index(query.required_tool) + 1
    top_score = float(scores[order[0]])
    second_score = float(scores[order[1]]) if len(order) > 1 else top_score
    normalized = normalize_text(" ".join((query.context, query.query)))
    resource = next(row for row in resources if row.name == query.required_tool)
    names_and_aliases = (resource.name, *resource.aliases)
    return {
        **query.to_dict(),
        "mode": mode,
        "top1_tool": names[0],
        "top1_correct": rank == 1,
        "required_rank": rank,
        "mrr": 1.0 / rank,
        "recall_at_1": float(rank <= 1),
        "recall_at_3": float(rank <= 3),
        "recall_at_5": float(rank <= 5),
        "top1_score": top_score,
        "margin": top_score - second_score,
        "top3_tools": "|".join(names[:3]),
        "useful_at_3": len(set(names[:3]) & set(query.useful_tools)),
        "related_at_3": len(set(names[:3]) & set(query.related_tools)),
        "unsafe_at_3": len(set(names[:3]) & set(query.unsafe_tools)),
        "false_action": rank != 1,
        "token_overlap": token_overlap_value,
        "bm25_required_score": bm25_score,
        "bm25_required_rank": bm25_rank,
        "canonical_operation_present": query.canonical_operation in normalized.split(),
        "canonical_object_present": query.canonical_object in normalized.split(),
        "tool_name_or_alias_present": any(normalize_text(value) in normalized for value in names_and_aliases),
        "language_match": query.language == "en",
        "routing_seconds": latency_seconds,
        "index_bytes": index_bytes,
        "model_bytes": model_bytes,
        "candidate_count": len(resources),
        "decision": "select",
        "selected_stage": mode,
        "calibrated_confidence": None,
    }


def _metric_key(scores: torch.Tensor, rows: Sequence[SemanticHardQuery], targets, split="validation"):
    selected = [index for index, row in enumerate(rows) if row.split == split]
    return _selection_key(_ranking_metrics(scores[selected], [targets[index] for index in selected]))


def _fit_staged_policy(policy_rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_identity = {(row["query_id"], row["mode"]): row for row in policy_rows}
    stages = ("P1_bm25", "P4_lexical_dictionary_tags", "P8_lexical_dictionary_embedding")
    calibrators = {}
    for stage in stages:
        calibrators[stage] = ReliabilityCalibrator.fit(
            (
                float(row["top1_score"]),
                bool(row["top1_correct"]),
            )
            for row in policy_rows
            if row["split"] == "validation" and row["mode"] == stage
        )
    candidates = []
    validation_ids = sorted({row["query_id"] for row in policy_rows if row["split"] == "validation"})
    for lexical_threshold in (0.70, 0.80, 0.90):
        for dictionary_threshold in (0.70, 0.80, 0.90):
            for embedding_threshold in (0.70, 0.80, 0.90):
                for ask_threshold in (0.35, 0.50):
                    selected = correct = wrong = asks = abstains = 0
                    for query_id in validation_ids:
                        decision = "abstain"
                        chosen = None
                        for stage, threshold in zip(stages, (lexical_threshold, dictionary_threshold, embedding_threshold)):
                            row = by_identity[(query_id, stage)]
                            confidence = calibrators[stage](float(row["top1_score"]))
                            if confidence >= threshold and float(row["margin"]) >= 0.02:
                                decision, chosen = "select", row
                                break
                        if chosen is None:
                            final = by_identity[(query_id, stages[-1])]
                            confidence = calibrators[stages[-1]](float(final["top1_score"]))
                            decision = "ask" if confidence >= ask_threshold else "abstain"
                        if decision == "select":
                            selected += 1
                            correct += int(bool(chosen["top1_correct"]))
                            wrong += int(not bool(chosen["top1_correct"]))
                        elif decision == "ask":
                            asks += 1
                        else:
                            abstains += 1
                    count = len(validation_ids)
                    utility = (correct - 2.0 * wrong + 0.05 * selected) / count
                    candidates.append({
                        "lexical_threshold": lexical_threshold,
                        "dictionary_threshold": dictionary_threshold,
                        "embedding_threshold": embedding_threshold,
                        "ask_threshold": ask_threshold,
                        "utility": utility,
                        "coverage": selected / count,
                        "selective_accuracy": correct / max(selected, 1),
                        "false_action_rate": wrong / count,
                        "ask_rate": asks / count,
                        "abstain_rate": abstains / count,
                    })
    selected_policy = max(
        candidates,
        key=lambda row: (row["utility"], row["selective_accuracy"], row["coverage"]),
    )
    staged_rows = []
    for query_id in sorted({row["query_id"] for row in policy_rows}):
        base = dict(by_identity[(query_id, stages[-1])])
        decision = "abstain"
        chosen = None
        confidence = 0.0
        thresholds = (
            selected_policy["lexical_threshold"],
            selected_policy["dictionary_threshold"],
            selected_policy["embedding_threshold"],
        )
        for stage, threshold in zip(stages, thresholds):
            row = by_identity[(query_id, stage)]
            confidence = calibrators[stage](float(row["top1_score"]))
            if confidence >= threshold and float(row["margin"]) >= 0.02:
                decision, chosen = "select", row
                break
        if chosen is None:
            chosen = by_identity[(query_id, stages[-1])]
            confidence = calibrators[stages[-1]](float(chosen["top1_score"]))
            decision = "ask" if confidence >= selected_policy["ask_threshold"] else "abstain"
        base.update(chosen)
        base["mode"] = "P10_staged_external"
        base["decision"] = decision
        base["selected_stage"] = chosen["mode"] if decision == "select" else decision
        base["calibrated_confidence"] = confidence
        base["false_action"] = decision == "select" and not bool(chosen["top1_correct"])
        staged_rows.append(base)
    return selected_policy, staged_rows


def run(args) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resources = realistic_tool_catalog()
    cards = tuple(ToolSemanticCard.from_resource(resource) for resource in resources)
    rows = list(semantic_hardness_queries())
    if args.max_queries is not None:
        rows = rows[: args.max_queries]
    by_name = {resource.name: index for index, resource in enumerate(resources)}
    target_indices = [by_name[row.required_tool] for row in rows]
    concepts = canonical_concept_map()
    semantic_index = ExternalSemanticIndex(resources, concepts, cards=cards)

    with (output_dir / "semantic_hardness_queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    (output_dir / "canonical_concepts.json").write_text(
        json.dumps(concepts.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "dictionary_sources.json").write_text(
        json.dumps(dictionary_sources_manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "tool_semantic_cards.json").write_text(
        json.dumps([asdict(card) for card in cards], indent=2, sort_keys=True), encoding="utf-8"
    )
    for split in ("validation", "test"):
        (output_dir / f"{split}_identity_manifest.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "query_ids": [row.query_id for row in rows if row.split == split],
                    "tool_identities": sorted({row.required_tool for row in rows if row.split == split}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    payloads, selection_rows, model_manifest = _evaluate_embedding_models(
        rows,
        cards,
        target_indices,
        model_root=args.model_root,
        output_dir=output_dir,
        device=torch.device(args.device),
    )
    english_keys = [spec.key for spec in MODEL_SPECS if spec.language_scope == "en"]
    best_english = max(
        english_keys,
        key=lambda key: _selection_key(payloads[key]["validation_metrics"]),
    )
    multilingual_keys = [spec.key for spec in MODEL_SPECS if spec.language_scope == "multilingual"]
    best_multilingual = max(
        multilingual_keys,
        key=lambda key: _selection_key(payloads[key]["validation_metrics"]),
    )
    generic_embedding = max(
        (best_english, best_multilingual),
        key=lambda key: _metric_key(payloads[key]["scores"], rows, target_indices),
    )

    lexical_rows = []
    channel_tensors = {name: torch.zeros((len(rows), len(resources))) for name in ("token", "bm25", "dictionary", "tags")}
    lexical_seconds = []
    bm25_required = []
    bm25_ranks = []
    overlap_values = []
    for query_index, query in enumerate(rows):
        started = time.perf_counter()
        scored = semantic_index.score(query.query, context=query.context, language=query.language)
        lexical_seconds.append(time.perf_counter() - started)
        for resource_index, score in enumerate(scored):
            channel_tensors["token"][query_index, resource_index] = score.token
            channel_tensors["bm25"][query_index, resource_index] = score.bm25
            channel_tensors["dictionary"][query_index, resource_index] = score.dictionary
            channel_tensors["tags"][query_index, resource_index] = score.tags
        target = target_indices[query_index]
        bm25_required.append(float(channel_tensors["bm25"][query_index, target]))
        bm25_order = torch.argsort(channel_tensors["bm25"][query_index], descending=True, stable=True).tolist()
        bm25_ranks.append(bm25_order.index(target) + 1)
        overlap_values.append(token_overlap(" ".join((query.context, query.query)), resources[target]))

    lexical = torch.maximum(channel_tensors["token"], channel_tensors["bm25"])
    dictionary = channel_tensors["dictionary"]
    tags = channel_tensors["tags"]
    english_embedding = (payloads[best_english]["scores"] + 1.0) / 2.0
    multilingual_embedding = (payloads[best_multilingual]["scores"] + 1.0) / 2.0
    generic_scores = (payloads[generic_embedding]["scores"] + 1.0) / 2.0
    p4 = 0.55 * lexical + 0.30 * dictionary + 0.15 * tags

    p7_candidates = {
        weight: weight * lexical + (1.0 - weight) * generic_scores
        for weight in (0.25, 0.50, 0.75)
    }
    p7_weight = max(p7_candidates, key=lambda value: _metric_key(p7_candidates[value], rows, target_indices))
    p7 = p7_candidates[p7_weight]
    p8_candidates = {
        weights: weights[0] * lexical + weights[1] * torch.maximum(dictionary, tags) + weights[2] * generic_scores
        for weights in ((0.25, 0.35, 0.40), (0.35, 0.25, 0.40), (0.20, 0.20, 0.60), (0.45, 0.25, 0.30))
    }
    p8_weights = max(p8_candidates, key=lambda value: _metric_key(p8_candidates[value], rows, target_indices))
    p8 = p8_candidates[p8_weights]
    oracle = torch.zeros_like(lexical)
    for query_index, target in enumerate(target_indices):
        oracle[query_index, target] = 1.0

    policy_scores = {
        "P0_token": channel_tensors["token"],
        "P1_bm25": channel_tensors["bm25"],
        "P2_dictionary": dictionary,
        "P3_tags": tags,
        "P4_lexical_dictionary_tags": p4,
        "P5_english_embedding": english_embedding,
        "P6_multilingual_embedding": multilingual_embedding,
        "P7_lexical_embedding": p7,
        "P8_lexical_dictionary_embedding": p8,
        "P9_oracle_identity": oracle,
    }
    concept_bytes = len(json.dumps(concepts.to_dict()).encode("utf-8"))
    card_bytes = len(json.dumps([asdict(card) for card in cards]).encode("utf-8"))
    index_costs = {
        "P0_token": semantic_index.estimated_lexical_bytes,
        "P1_bm25": semantic_index.estimated_lexical_bytes,
        "P2_dictionary": semantic_index.estimated_lexical_bytes + concept_bytes,
        "P3_tags": card_bytes,
        "P4_lexical_dictionary_tags": semantic_index.estimated_lexical_bytes + concept_bytes + card_bytes,
        "P5_english_embedding": payloads[best_english]["embedding_bytes"],
        "P6_multilingual_embedding": payloads[best_multilingual]["embedding_bytes"],
        "P7_lexical_embedding": semantic_index.estimated_lexical_bytes + payloads[generic_embedding]["embedding_bytes"],
        "P8_lexical_dictionary_embedding": semantic_index.estimated_lexical_bytes + concept_bytes + card_bytes + payloads[generic_embedding]["embedding_bytes"],
        "P9_oracle_identity": len(resources) * 8,
    }
    model_costs = {
        "P5_english_embedding": payloads[best_english]["parameter_bytes"],
        "P6_multilingual_embedding": payloads[best_multilingual]["parameter_bytes"],
        "P7_lexical_embedding": payloads[generic_embedding]["parameter_bytes"],
        "P8_lexical_dictionary_embedding": payloads[generic_embedding]["parameter_bytes"],
    }
    embedding_latency = {
        "P5_english_embedding": payloads[best_english]["query_encode_seconds"] / len(rows),
        "P6_multilingual_embedding": payloads[best_multilingual]["query_encode_seconds"] / len(rows),
        "P7_lexical_embedding": payloads[generic_embedding]["query_encode_seconds"] / len(rows),
        "P8_lexical_dictionary_embedding": payloads[generic_embedding]["query_encode_seconds"] / len(rows),
    }
    policy_rows = []
    for mode, scores in policy_scores.items():
        for query_index, query in enumerate(rows):
            latency = lexical_seconds[query_index]
            if mode in embedding_latency:
                latency += embedding_latency[mode]
            policy_rows.append(_rank_row(
                query,
                mode,
                scores[query_index],
                resources,
                latency_seconds=latency,
                index_bytes=int(index_costs[mode]),
                model_bytes=int(model_costs.get(mode, 0)),
                token_overlap_value=overlap_values[query_index],
                bm25_score=bm25_required[query_index],
                bm25_rank=bm25_ranks[query_index],
            ))
    staged_policy, staged_rows = _fit_staged_policy(policy_rows)
    policy_rows.extend(staged_rows)

    _write_csv(output_dir / "embedding_selection_rows.csv", selection_rows)
    _write_csv(output_dir / "semantic_hardness_rows.csv", policy_rows)
    _write_csv(
        output_dir / "external_semantic_latency.csv",
        [
            {
                "model": row["key"],
                "cold_load_seconds": row["cold_load_seconds"],
                "registration_seconds": row["registration_seconds"],
                "warm_query_encoding_seconds": row["warm_query_encoding_seconds"],
                "mean_warm_query_seconds": row["warm_query_encoding_seconds"] / len(rows),
            }
            for row in model_manifest
        ],
    )
    _write_csv(
        output_dir / "external_semantic_memory.csv",
        [
            {
                "model": row["key"],
                "parameter_bytes": row["parameter_bytes"],
                "tool_embedding_bytes": payloads[row["key"]]["embedding_bytes"],
                "dimensions": row["embedding_dimensions"],
            }
            for row in model_manifest
        ],
    )
    _write_csv(output_dir / "external_semantic_calibration.csv", [staged_policy])
    (output_dir / "embedding_model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    embedding_manifest = {
        key: {
            "file": f"tool_embeddings_{key}.npz",
            "representation": value["representation"],
            "query_scope": value["query_scope"],
            "shape": list(np.load(output_dir / f"tool_embeddings_{key}.npz")["embeddings"].shape),
            "bytes": value["embedding_bytes"],
        }
        for key, value in payloads.items()
    }
    (output_dir / "tool_embeddings_manifest.json").write_text(
        json.dumps(embedding_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    query_digest = hashlib.sha256(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "milestone": "M6.5",
        "query_count": len(rows),
        "resource_count": len(resources),
        "query_fingerprint": query_digest,
        "splits": {split: sum(row.split == split for row in rows) for split in ("audit", "validation", "test")},
        "hardness_levels": sorted({row.hardness_level for row in rows}),
        "languages": sorted({row.language for row in rows}),
        "best_english_embedding": best_english,
        "best_multilingual_embedding": best_multilingual,
        "generic_embedding_for_fusion": generic_embedding,
        "p7_lexical_weight": p7_weight,
        "p8_weights_lexical_dictionary_embedding": p8_weights,
        "staged_policy": staged_policy,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "main_model_hidden_state_used": False,
        "generation_model_call_used": False,
        "runtime_web_dependency": False,
        "model_weights_committed": False,
        "policy_modes": list(POLICY_MODES),
        "runtime": runtime_metadata(),
    }
    (output_dir / "semantic_hardness_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/m6_5_semantic_hard",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
