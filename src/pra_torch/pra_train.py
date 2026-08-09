"""PRA-specific adapters for the generic training engine."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from common.metrics import RunningAverages, cuda_memory_allocated, perplexity
from common.train import TrainingState, create_training_state, move_batch, train_model
from data.datamodules import PRADataModule
from .cache_services import build_cache_from_metadata
from .config import CacheServiceConfig, PRAConfig, ResolverServiceConfig, TrainConfig
from .metrics import (
    chunk_is_relevant,
    ranking_metrics,
)
from .memory import PRABatchedMemoryCache, PRASimpleMemoryCache
from .model import TinyPRAModel
from .prompt import IMPLICIT_PROMPT_HEAD_URI, prepare_prompt_batch_for_pra


def _expected_reference_uris(item: dict) -> set[str]:
    """Normalize URI- or legacy ID-based supervision to a URI identity set."""
    explicit = set(item.get("target_reference_uris") or [])
    if explicit:
        return explicit
    target_ids = set(item.get("target_reference_ids") or [])
    return {ref.uri for ref in item["references"] if ref.id in target_ids}


def _chunk_ranking_metrics(selected, item, cfg):
    """Score ranked chunks against optional ID/span ground truth."""
    target_ids = set(item.get("target_chunk_ids") or [])
    target_spans = list(item.get("target_chunk_spans") or [])
    if not target_ids and not target_spans:
        return None
    flags = [
        chunk_is_relevant(
            hit,
            target_ids,
            target_spans,
            cfg.chunk_match_mode,
            cfg.chunk_iou_threshold,
        )
        for hit in selected
    ]
    expected_count = len(target_ids) or len(target_spans)
    hits = sum(flags)
    reciprocal_rank = next((1.0 / rank for rank, flag in enumerate(flags, 1) if flag), 0.0)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(flags, 1) if flag)
    ideal_hits = min(expected_count, len(flags))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    recall = hits / max(expected_count, 1)
    precision = hits / max(len(flags), 1)
    return {
        "hit_at_1": float(bool(flags and flags[0])),
        "hit_at_k": float(hits > 0),
        "recall_at_k": recall,
        "precision_at_k": precision,
        "f1_at_k": 2 * precision * recall / max(precision + recall, 1e-12),
        "mrr": reciprocal_rank,
        "ndcg": dcg / max(ideal_dcg, 1e-12),
        "selected_count": float(len(selected)),
    }


def _deduplicated_reference_order(selected) -> list[str]:
    """Collapse chunk hits to URI order while preserving configured ranks."""
    ranked = sorted(selected, key=lambda hit: (hit.reference_rank, hit.rank_within_reference))
    return list(dict.fromkeys(hit.reference_uri for hit in ranked))


def _selections_by_row(by_layer: dict[int, list[list]], batch_size: int) -> list[dict[int, list]]:
    """Transpose ``layer -> row -> hits`` into ``row -> layer -> hits`` once."""
    selections = [dict() for _ in range(batch_size)]
    for layer_id, layer_rows in by_layer.items():
        if len(layer_rows) != batch_size:
            raise ValueError(
                f"Layer {layer_id} returned {len(layer_rows)} selection rows for batch {batch_size}."
            )
        for row_index, selected in enumerate(layer_rows):
            selections[row_index][layer_id] = list(selected)
    return selections


_RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)


def _complete_reference_rank_metrics(rankings_by_layer, target_uris: set[str]) -> dict:
    """Summarize pre-top-k target ranks while retaining layer-level evidence."""
    layer_rows = {}
    reciprocal_ranks = []
    score_margins = []
    gist_comparisons = 0
    union_by_k = {cutoff: set() for cutoff in _RANK_CUTOFFS}
    recall_by_k = {cutoff: [] for cutoff in _RANK_CUTOFFS}
    for layer_id, rankings in sorted(rankings_by_layer.items()):
        ranked_uris = [str(candidate["reference_uri"]) for candidate in rankings]
        rank_by_uri = {uri: rank for rank, uri in enumerate(ranked_uris, start=1)}
        target_ranks = sorted(rank_by_uri[uri] for uri in target_uris if uri in rank_by_uri)
        reciprocal_ranks.append(1.0 / target_ranks[0] if target_ranks else 0.0)
        target_scores = [
            float(candidate["reference_score"])
            for candidate in rankings
            if candidate["reference_uri"] in target_uris
        ]
        non_target_scores = [
            float(candidate["reference_score"])
            for candidate in rankings
            if candidate["reference_uri"] not in target_uris
        ]
        best_target = max(target_scores) if target_scores else None
        best_non_target = max(non_target_scores) if non_target_scores else None
        margin = (
            best_target - best_non_target
            if best_target is not None and best_non_target is not None
            else None
        )
        if margin is not None:
            score_margins.append(margin)
        for candidate in rankings:
            gist_comparisons += sum(
                int(chunk.get("gist_count", 1)) for chunk in candidate.get("chunks", [])
            )
        layer_row = {
            "ranked_reference_uris": ranked_uris,
            "target_reference_ranks": target_ranks,
            "mrr": reciprocal_ranks[-1],
            "best_target_score": best_target,
            "best_non_target_score": best_non_target,
            "score_margin": margin,
        }
        for cutoff in _RANK_CUTOFFS:
            selected = set(ranked_uris[:cutoff])
            union_by_k[cutoff].update(selected)
            recall = len(selected & target_uris) / max(len(target_uris), 1)
            recall_by_k[cutoff].append(recall)
            layer_row[f"recall_at_{cutoff}"] = recall
        layer_rows[str(layer_id)] = layer_row

    metrics = {
        "routing_mrr": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
        "routing_score_margin": sum(score_margins) / max(len(score_margins), 1)
        if score_margins
        else None,
        "gist_comparisons": gist_comparisons,
        "target_reference_count": len(target_uris),
        "rank_diagnostics_by_layer": layer_rows,
    }
    for cutoff in _RANK_CUTOFFS:
        covered = union_by_k[cutoff] & target_uris
        metrics[f"reference_recall_at_{cutoff}"] = sum(recall_by_k[cutoff]) / max(
            len(recall_by_k[cutoff]), 1
        )
        metrics[f"any_target_hit_at_{cutoff}"] = float(bool(covered))
        metrics[f"all_targets_hit_at_{cutoff}"] = float(
            bool(target_uris) and target_uris <= union_by_k[cutoff]
        )
        metrics[f"fraction_targets_covered_at_{cutoff}"] = len(covered) / max(
            len(target_uris), 1
        )
    return metrics


def _row_batching_metrics(layer_diagnostics: dict, row_index: int) -> dict[str, float]:
    """Extract genuine row-local memory packing values from layer diagnostics."""
    batching = layer_diagnostics.get("batching")
    if batching is None or row_index >= len(batching.selected_lengths):
        return {}
    valid = int(batching.selected_lengths[row_index])
    bucket_index = next(
        (
            index
            for index, members in enumerate(batching.bucket_membership)
            if row_index in members
        ),
        None,
    )
    allocated = int(batching.bucket_max_lengths[bucket_index]) if bucket_index is not None else 0
    padding = max(allocated - valid, 0)
    return {
        "memory_valid_positions": float(valid),
        "memory_allocated_positions": float(allocated),
        "memory_padding_positions": float(padding),
        "memory_padding_fraction": padding / max(allocated, 1),
        "memory_bucket_index": float(bucket_index) if bucket_index is not None else -1.0,
    }


def _synchronize_for_timing(device: str) -> None:
    """Synchronize CUDA so prompt-forward timing excludes queued execution."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _retrieval_metrics(caches, selections, metadata, cfg, diagnostics) -> dict[str, float]:
    """Aggregate routing quality, sparsity, recursion, and batching diagnostics.

    Language-model loss alone cannot show whether PRA found the intended source.
    These metrics separately evaluate URI/chunk selection and how much of the
    available memory was ultimately materialized into attention.
    """
    averages = RunningAverages()
    chunk_label_units = selection_units = 0
    if not (len(caches) == len(selections) == len(metadata)):
        raise ValueError("Caches, selections, and metadata must align by logical batch row.")
    for row_index, (cache, by_layer, item) in enumerate(zip(caches, selections, metadata)):
        expected_uris = _expected_reference_uris(item)
        for layer_id, selected in by_layer.items():
            selection_units += 1
            reference_metrics = ranking_metrics(_deduplicated_reference_order(selected), expected_uris)
            layer_metrics = {
                f"retrieval_reference_{key}": value for key, value in reference_metrics.items()
            }
            chunk_metrics = _chunk_ranking_metrics(selected, item, cfg)
            if chunk_metrics is not None:
                chunk_label_units += 1
                layer_metrics.update(
                    {f"retrieval_chunk_{key}": value for key, value in chunk_metrics.items()}
                )
            counts = cache.layer_counts(layer_id)
            selected_refs = len(set(hit.reference_uri for hit in selected))
            selected_chunks = len(selected)
            diagnostic = diagnostics.get(layer_id, {})
            row_batching = _row_batching_metrics(diagnostic, row_index)
            selected_tokens = int(
                row_batching.get(
                    "memory_valid_positions",
                    sum(hit.selected_token_count for hit in selected),
                )
            )
            selected_scores = [hit.chunk_score for hit in selected]
            chunk_gist_counts = [
                int(chunk.routing_gist.k.shape[0])
                for entry in cache.all_entries()
                for chunk in (
                    entry.layer_memory[layer_id].chunks
                    if layer_id in entry.layer_memory
                    else []
                )
            ]
            reference_gist_counts = [
                int(entry.reference_gists_by_layer[layer_id].k.shape[0])
                for entry in cache.all_entries()
                if layer_id in entry.reference_gists_by_layer
            ]
            winning_chunk_indices = [
                hit.winning_gist_index for hit in selected if hit.winning_gist_index is not None
            ]
            winning_chunk_scores = [
                hit.winning_gist_score for hit in selected if hit.winning_gist_score is not None
            ]
            winning_reference_indices = [
                hit.winning_reference_gist_index
                for hit in selected
                if hit.winning_reference_gist_index is not None
            ]
            winning_reference_scores = [
                hit.winning_reference_gist_score
                for hit in selected
                if hit.winning_reference_gist_score is not None
            ]
            recursive_depth = max(
                (int(entry.metadata.get("resolution_depth", 0)) for entry in cache.all_entries()),
                default=0,
            )
            recursion_refs = max(
                (
                    int(entry.metadata.get("resolution_budget_references_used", 0))
                    for entry in cache.all_entries()
                ),
                default=0,
            )
            recursion_tokens = max(
                (
                    int(entry.metadata.get("resolution_budget_tokens_used", 0))
                    for entry in cache.all_entries()
                ),
                default=0,
            )
            layer_metrics.update(
                {
                    "memory_available_reference_count": counts["references"],
                    "memory_available_chunk_count": counts["chunks"],
                    "cache_reference_token_count": counts["tokens"],
                    "retrieval_selected_reference_count": selected_refs,
                    "retrieval_selected_chunk_count": selected_chunks,
                    "memory_selected_token_count": selected_tokens,
                    "memory_selected_reference_fraction": selected_refs / max(counts["references"], 1),
                    "memory_selected_chunk_fraction": selected_chunks / max(counts["chunks"], 1),
                    "memory_selected_token_fraction": selected_tokens / max(counts["tokens"], 1),
                    "retrieval_routing_score_mean": sum(selected_scores) / max(len(selected_scores), 1),
                    "retrieval_routing_score_max": max(selected_scores, default=0.0),
                    "retrieval_zero_chunk_fraction": float(selected_chunks == 0),
                    "chunk_gists_requested": cfg.gists_per_chunk,
                    "chunk_gists_actual_mean": sum(chunk_gist_counts)
                    / max(len(chunk_gist_counts), 1),
                    "chunk_gists_actual_max": max(chunk_gist_counts, default=0),
                    "reference_gists_requested": cfg.reference_gists_per_reference,
                    "reference_gists_actual_mean": sum(reference_gist_counts)
                    / max(len(reference_gist_counts), 1),
                    "reference_gists_actual_max": max(reference_gist_counts, default=0),
                    "winning_chunk_gist_index": sum(winning_chunk_indices)
                    / max(len(winning_chunk_indices), 1),
                    "winning_reference_gist_index": sum(winning_reference_indices)
                    / max(len(winning_reference_indices), 1),
                    "chunk_best_gist_score": sum(winning_chunk_scores)
                    / max(len(winning_chunk_scores), 1),
                    "reference_best_gist_score": sum(winning_reference_scores)
                    / max(len(winning_reference_scores), 1),
                    "memory_retrieved_chunk_attended_fraction": float(selected_tokens > 0)
                    if selected_chunks
                    else 0.0,
                    "cache_recursive_expansion_depth": recursive_depth,
                    "cache_recursive_reference_budget_fraction": recursion_refs
                    / max(cfg.recursive_max_total_references, 1),
                    "cache_recursive_token_budget_fraction": recursion_tokens
                    / max(cfg.recursive_max_total_tokens, 1),
                }
            )
            averages.update(layer_metrics)
            averages.update(
                {f"layer_{layer_id}_{key.removeprefix('retrieval_')}": value for key, value in layer_metrics.items()}
            )
    # Layer diagnostics such as padding totals, output norms, and timing describe
    # the batched execution. Aggregate them once per layer rather than copying a
    # batch value into every row-level retrieval record.
    for layer_id, diagnostic in diagnostics.items():
        aggregate_metrics = {
            key: value for key, value in diagnostic.items() if isinstance(value, (int, float))
        }
        averages.update(aggregate_metrics)
        averages.update({f"layer_{layer_id}_{key}": value for key, value in aggregate_metrics.items()})

    result = averages.compute()
    result["retrieval_chunk_labels_available_fraction"] = chunk_label_units / max(selection_units, 1)
    result["reference_retrieval_accuracy"] = result.get("retrieval_reference_hit_at_k", 0.0)
    result["reference_selection_top1_accuracy"] = result.get("retrieval_reference_hit_at_1", 0.0)
    result["reference_selection_topk_accuracy"] = result.get("retrieval_reference_hit_at_k", 0.0)
    result["reference_selection_mrr"] = result.get("retrieval_reference_mrr", 0.0)
    result["selected_ref_count"] = result.get("retrieval_reference_selected_count", 0.0)
    return result


def _pra_batch_step(
    model,
    batch: dict,
    device: str,
    tokenizer,
    resolver_config: ResolverServiceConfig,
    cache_config: CacheServiceConfig,
) -> tuple[torch.Tensor, dict]:
    """Build row-local caches, then execute one logical-batch prompt forward.

    Reference encoding remains independent because each row owns a distinct URI
    namespace. Completed row caches are wrapped by ``PRABatchedMemoryCache``;
    ``input_ids [B,T]`` then traverses the Transformer once and returns logits
    ``[B,T,V]``. Routing/materialization remain row-local inside that forward.
    """
    batch = move_batch(batch, device)
    caches = []
    cache_build_duration = 0.0
    # Do not share a sample's URI table or selected memory with another batch row.
    for metadata in batch["metadata"]:
        cache_start = time.perf_counter()
        cache = build_cache_from_metadata(
            model,
            tokenizer,
            [metadata],
            device,
            resolver_config=resolver_config,
            cache_config=cache_config,
            attach_to_model=False,
        )
        cache_build_duration += time.perf_counter() - cache_start
        caches.append(cache)

    prepared = None
    should_prepare_prompt = (
        model.cfg.prompt_overflow_mode != "truncate"
        or model.cfg.max_prompt_direct_tokens is not None
    )
    if should_prepare_prompt:
        preparation_start = time.perf_counter()
        prepared = prepare_prompt_batch_for_pra(
            model,
            tokenizer,
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            metadata=batch.get("metadata"),
            caches=caches,
        )
        cache_build_duration += time.perf_counter() - preparation_start
        batch = {
            **batch,
            "input_ids": prepared.input_ids,
            "attention_mask": prepared.attention_mask,
            "labels": prepared.labels,
        }

    # URI strings may collide safely because the wrapper routes query row i only
    # through caches[i]. It never creates a flattened cross-row namespace.
    batch_cache = PRABatchedMemoryCache(caches)
    model.set_pra_cache(batch_cache)

    # This is the single expensive prompt Transformer execution for the logical batch.
    _synchronize_for_timing(device)
    prompt_start = time.perf_counter()
    logits = model(
        batch["input_ids"],
        attention_mask=prepared.attention_mask if prepared is not None else None,
    )
    _synchronize_for_timing(device)
    prompt_forward_duration = time.perf_counter() - prompt_start

    batch_size = int(batch["input_ids"].shape[0])
    selections = _selections_by_row(model.selected_chunks_by_layer(), batch_size)
    diagnostics = model.pra_diagnostics_by_layer()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=0)
    retrieval_metrics = _retrieval_metrics(
        caches, selections, batch["metadata"], model.cfg, diagnostics
    )
    retrieval_metrics["cache_build_duration_seconds"] = cache_build_duration / max(
        len(caches), 1
    )
    retrieval_metrics.update(
        {
            "cache_build_total_duration_seconds": cache_build_duration,
            "logical_batch_size": float(batch_size),
            "prompt_forward_calls": 1.0,
            "prompt_forward_duration_seconds": prompt_forward_duration,
        }
    )
    if prepared is not None:
        prompt_metric_names = tuple(prepared.stats[0].as_metrics()) if prepared.stats else ()
        for metric_name in prompt_metric_names:
            retrieval_metrics[metric_name] = sum(
                row.as_metrics()[metric_name] for row in prepared.stats
            ) / max(len(prepared.stats), 1)
        selected_head_chunks = [
            sum(
                1
                for layer_hits in row.values()
                for hit in layer_hits
                if hit.reference_uri == IMPLICIT_PROMPT_HEAD_URI
            )
            for row in selections
        ]
        retrieval_metrics["prompt_implicit_chunks_selected"] = sum(
            selected_head_chunks
        ) / max(len(selected_head_chunks), 1)
    return loss, {
        "tokens": int(batch["attention_mask"].sum().item()),
        "examples": batch_size,
        "batch": batch,
        "cache": caches[-1] if caches else None,
        "caches": caches,
        "batch_cache": batch_cache,
        "selections": selections,
        "diagnostics": diagnostics,
        "metrics": retrieval_metrics,
        "logits": logits,
    }


def evaluate_pra_model(
    *,
    model: TinyPRAModel,
    loader,
    tokenizer,
    train_config: TrainConfig,
    device: str,
    split: str,
    resolver_config: ResolverServiceConfig | dict | str | None = None,
    cache_config: CacheServiceConfig | dict | str | None = None,
    save_predictions: str | Path | None = None,
    save_traces: str | Path | None = None,
) -> dict:
    """Evaluate LM quality, routing behavior, recursion, cost, and throughput.

    Optional prediction files contain task outputs; trace files additionally
    preserve URI/chunk selections, bucket statistics, and recursive paths so a
    result can be causally inspected rather than inferred from loss alone.
    """
    resolver_config = ResolverServiceConfig.from_value(resolver_config or train_config.resolver_config)
    cache_config = CacheServiceConfig.from_value(cache_config or train_config.cache_config)
    was_training = model.training
    model.eval()
    averages = RunningAverages()
    exact = anchor_hits = cache_hits = total = 0
    retrieved_tokens = expanded_refs = expansion_depth = token_total = 0
    start = time.perf_counter()

    pred_path = Path(save_predictions) if save_predictions else None
    trace_path = Path(save_traces) if save_traces else None
    if pred_path:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    pred_f = pred_path.open("w", encoding="utf-8") if pred_path else None
    trace_f = trace_path.open("w", encoding="utf-8") if trace_path else None

    try:
        with torch.no_grad():
            for batch in loader:
                loss, batch_metrics = _pra_batch_step(
                    model,
                    batch,
                    device,
                    tokenizer,
                    resolver_config,
                    cache_config,
                )
                moved_batch = batch_metrics["batch"]
                predicted_ids = batch_metrics["logits"].argmax(dim=-1).detach().cpu()
                averages.update(
                    {
                        f"{split}_loss": float(loss.detach().cpu()),
                        **batch_metrics.get("metrics", {}),
                    },
                    weight=max(int(batch_metrics["examples"]), 1),
                )
                token_total += int(batch_metrics["tokens"])

                # Convert model/cache state into per-example task and audit records.
                for index, item in enumerate(moved_batch["metadata"]):
                    cache = batch_metrics["caches"][index]
                    retrieved_tokens += sum(
                        len(tokenizer.encode(entry.text)) for entry in cache.all_entries()
                    )
                    expanded_refs += len(cache.entries)
                    prediction = tokenizer.decode(predicted_ids[index])
                    answer = item["answer"].strip()
                    total += 1
                    if answer in prediction:
                        exact += 1
                    expected_uris = _expected_reference_uris(item)
                    cache_hit = not expected_uris or bool(expected_uris.intersection(cache.entries))
                    if cache_hit:
                        cache_hits += 1
                    layer_selections = batch_metrics["selections"][index]
                    anchors = item.get("expected_anchors", [])
                    resolution_events = getattr(cache, "resolution_events", [])
                    item_depth = max((event.depth for event in resolution_events), default=0)
                    expansion_depth += item_depth
                    cached_text = "\n".join(entry.text for entry in cache.all_entries())
                    anchor_hit = not anchors or any(
                        anchor in cached_text or anchor in " ".join(cache.entries) for anchor in anchors
                    )
                    if anchor_hit:
                        anchor_hits += 1
                    trace = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "predicted_answer": prediction,
                        "available_references": [asdict(ref) for ref in item["references"]],
                        "cached_references": list(cache.entries.keys()),
                        "selected_references_by_layer": {
                            str(layer_id): _deduplicated_reference_order(selected)
                            for layer_id, selected in layer_selections.items()
                        },
                        "selected_chunks_by_layer": {
                            str(layer_id): [hit.as_trace_dict() for hit in selected]
                            for layer_id, selected in layer_selections.items()
                        },
                        "expanded_anchors": anchors,
                        "expansion_depth": item_depth,
                        "cache_hits": cache_hit,
                        "retrieved_token_counts": [
                            len(tokenizer.encode(entry.text)) for entry in cache.all_entries()
                        ],
                        "memory_lengths_by_layer": {
                            str(layer_id): int(
                                _row_batching_metrics(
                                    batch_metrics["diagnostics"].get(layer_id, {}), index
                                ).get(
                                    "memory_valid_positions",
                                    sum(hit.selected_token_count for hit in selected),
                                )
                            )
                            for layer_id, selected in layer_selections.items()
                        },
                        "bucket_stats_by_layer": {
                            str(layer_id): _row_batching_metrics(
                                diagnostic,
                                index,
                            )
                            for layer_id, diagnostic in batch_metrics["diagnostics"].items()
                        },
                        "recursive_paths": [asdict(event) for event in resolution_events],
                        "routing_configuration": {
                            "search_strategy": model.cfg.search_strategy,
                            "top_k_references": model.cfg.top_k_references,
                            "top_k_chunks_per_reference": model.cfg.top_k_chunks_per_reference,
                            "gist_mode": model.cfg.gist_mode,
                            "chunking_mode": model.cfg.chunking_mode,
                            "detail_materialization": model.cfg.detail_materialization,
                        },
                    }
                    if pred_f:
                        pred_f.write(
                            json.dumps(
                                {"id": item["id"], "prediction": prediction, "answer": item["answer"]}
                            )
                            + "\n"
                        )
                    if trace_f:
                        trace_f.write(json.dumps(trace) + "\n")
    finally:
        if pred_f:
            pred_f.close()
        if trace_f:
            trace_f.close()
        if was_training:
            model.train()

    elapsed = max(time.perf_counter() - start, 1e-9)
    metrics = averages.compute()
    loss = metrics.get(f"{split}_loss", 0.0)
    metrics.update(
        {
            "loss": loss,
            "perplexity": perplexity(loss),
            "answer_accuracy": exact / max(total, 1),
            "expected_anchor_hit": anchor_hits / max(total, 1),
            "expansion_depth": expansion_depth / max(total, 1),
            "expanded_ref_count": expanded_refs / max(total, 1),
            "average_retrieved_tokens": retrieved_tokens / max(total, 1),
            "cache_hit_ratio": cache_hits / max(total, 1),
            "latency": elapsed,
            "examples_per_second": total / elapsed,
            "tokens_per_second": token_total / elapsed,
            "gpu_memory_allocated": cuda_memory_allocated(device),
        }
    )
    return metrics


def evaluate_reference_ablation(
    *,
    model: TinyPRAModel,
    loader,
    tokenizer,
    device: str,
    condition: str,
    resolver_config=None,
    cache_config=None,
    collect_per_example: bool = False,
    encoded_entry_cache: dict | None = None,
) -> dict:
    """Measure answer-token quality under controlled PRA memory interventions.

    ``valid`` uses normal routing; ``disabled`` bypasses the memory branch;
    ``empty`` keeps the branch enabled with no entries. ``shuffled`` and
    ``irrelevant`` replace URI sets, while ``oracle`` exposes only target URIs.
    The chunk variants preserve routing structure but filter or rotate token K/V.
    ``selected_chunks``, ``full_reference``, and ``gist_only`` compare the three
    detail-materialization modes without retraining the model.
    """
    supported = {
        "valid",
        "disabled",
        "shuffled",
        "irrelevant",
        "empty",
        "oracle",
        "full_reference",
        "selected_chunks",
        "oracle_chunks",
        "shuffled_chunks",
        "irrelevant_chunks",
        "gist_only",
        "native_all",
        "native_oracle",
        "native_disabled",
        "native_shuffled",
    }
    if condition not in supported:
        raise ValueError(f"Unsupported reference condition: {condition}")
    was_training = model.training
    model.eval()
    loss_sum = correct = token_count = 0
    per_example = []
    start = time.perf_counter()
    # Build deterministic cross-example pools only for URI-content interventions.
    reference_conditions = {"shuffled", "irrelevant", "native_shuffled"}
    reference_sets = [
        item["references"]
        for batch in loader
        for item in batch["metadata"]
    ] if condition in reference_conditions else []
    reference_pool = [reference for references in reference_sets for reference in references]
    shuffle_offset = max(len(reference_sets) // 2, 1)
    reference_index = 0
    previous_materialization = model.cfg.detail_materialization
    previous_routing = (
        model.cfg.top_k_references,
        model.cfg.top_k_chunks_per_reference,
        model.cfg.trigger_threshold,
    )
    if condition.startswith("native_") and model.cfg.memory_transport != "native_kv":
        raise ValueError(f"{condition} requires memory_transport='native_kv'")
    if condition in {"native_all", "native_oracle"}:
        model.cfg.top_k_references = 1_000_000
        model.cfg.top_k_chunks_per_reference = 1_000_000
        model.cfg.trigger_threshold = float("-inf")
        model.cfg.detail_materialization = "full_reference"
    # Materialization ablations temporarily alter policy and restore it in finally.
    if condition in {"full_reference", "selected_chunks", "gist_only"}:
        model.cfg.detail_materialization = condition
    ablation_materialization = model.cfg.detail_materialization
    try:
        with torch.no_grad():
            for batch in loader:
                moved = move_batch(batch, device)
                for index, item in enumerate(batch["metadata"]):
                    item_start = time.perf_counter()
                    if str(device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    # First intervene at URI/cache level, then optionally at chunk payload level.
                    if condition in {"disabled", "empty", "native_disabled"}:
                        model.clear_pra_cache()
                    else:
                        metadata = item
                        if condition in {"shuffled", "native_shuffled"}:
                            shuffled_index = (reference_index + shuffle_offset) % len(reference_sets)
                            metadata = {**item, "references": reference_sets[shuffled_index]}
                        elif condition == "irrelevant":
                            own_uris = {reference.uri for reference in item["references"]}
                            start_index = (reference_index + shuffle_offset) % max(len(reference_pool), 1)
                            candidates = reference_pool[start_index:] + reference_pool[:start_index]
                            irrelevant = [reference for reference in candidates if reference.uri not in own_uris]
                            metadata = {**item, "references": irrelevant[: len(item["references"])]}
                        elif condition in {"oracle", "native_oracle"}:
                            target_ids = set(item["target_reference_ids"])
                            if model.cfg.reference_encoding_strategy == "independent":
                                metadata = {
                                    **item,
                                    "references": [
                                        reference
                                        for reference in item["references"]
                                        if reference.id in target_ids
                                    ],
                                }
                        visible_uris = [reference.uri for reference in metadata["references"]]
                        if encoded_entry_cache is not None and all(
                            uri in encoded_entry_cache for uri in visible_uris
                        ):
                            cache = PRASimpleMemoryCache()
                            for uri in visible_uris:
                                cache.put(encoded_entry_cache[uri])
                            model.set_pra_cache(cache)
                        else:
                            cache = build_cache_from_metadata(
                                model,
                                tokenizer,
                                [metadata],
                                device,
                                resolver_config=resolver_config,
                                cache_config=cache_config,
                            )
                            if encoded_entry_cache is not None:
                                encoded_entry_cache.update(cache.entries)
                        if (
                            condition in {"oracle", "native_oracle"}
                            and model.cfg.reference_encoding_strategy != "independent"
                        ):
                            target_uris = _expected_reference_uris(item)
                            filtered_cache = PRASimpleMemoryCache()
                            for uri in target_uris:
                                entry = cache.get(uri)
                                if entry is not None:
                                    filtered_cache.put(entry)
                            cache = filtered_cache
                            model.set_pra_cache(cache)
                        if condition == "oracle_chunks":
                            target_ids = set(item.get("target_chunk_ids") or [])
                            if target_ids:
                                for entry in cache.all_entries():
                                    for memory in entry.layer_memory.values():
                                        memory.chunks[:] = [
                                            chunk for chunk in memory.chunks if chunk.chunk_id in target_ids
                                        ]
                        elif condition in {"shuffled_chunks", "irrelevant_chunks"}:
                            # Rotate K/V payloads while retaining chunk identities/gists. A
                            # routing hit now points to mismatched detail, testing causal use.
                            for layer_id in range(model.cfg.n_layers):
                                chunks = [
                                    chunk
                                    for entry in cache.all_entries()
                                    if layer_id in entry.layer_memory
                                    for chunk in entry.layer_memory[layer_id].chunks
                                ]
                                if len(chunks) > 1:
                                    payloads = [chunk.token_kv for chunk in chunks]
                                    rotated = payloads[1:] + payloads[:1]
                                    for chunk, payload in zip(chunks, rotated):
                                        chunk.token_kv = payload
                    position_offset = 0
                    if (
                        model.cfg.prompt_position_mode == "historical"
                        and condition not in {"disabled", "native_disabled"}
                    ):
                        position_offset = sum(
                            len(tokenizer.encode(str(reference.metadata.get("text", ""))))
                            for reference in item.get("references") or []
                        )
                    logits = model(
                        moved["input_ids"][index : index + 1],
                        use_pra_memory=condition not in {"disabled", "native_disabled"},
                        position_offset=position_offset,
                    )
                    labels = moved["labels"][index : index + 1]
                    valid = labels.ne(0)
                    count = int(valid.sum().item())
                    item_loss_sum = float(
                        F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=0,
                            reduction="sum",
                        ).detach().cpu()
                    )
                    item_correct = int((logits.argmax(dim=-1).eq(labels) & valid).sum().item())
                    loss_sum += item_loss_sum
                    correct += item_correct
                    token_count += count
                    if collect_per_example:
                        selections = model.selected_chunks_by_layer()
                        layer_hits = {
                            int(layer_id): (rows[0] if rows else [])
                            for layer_id, rows in selections.items()
                        }
                        rankings = model.routing_rankings_by_layer()
                        layer_rankings = {
                            int(layer_id): (rows[0] if rows else [])
                            for layer_id, rows in rankings.items()
                        }
                        selected_hits = [hit for hits in layer_hits.values() for hit in hits]
                        retrieved_by_layer = [
                            sum(hit.selected_token_count for hit in hits)
                            for hits in layer_hits.values()
                        ]
                        retrieved_tokens = (
                            sum(retrieved_by_layer) / len(retrieved_by_layer)
                            if retrieved_by_layer
                            else 0.0
                        )
                        local_tokens = int(
                            moved.get("attention_mask", moved["input_ids"].ne(0))[
                                index : index + 1
                            ].sum().item()
                        )
                        own_references = item.get("references") or []
                        displaced_tokens = sum(
                            len(tokenizer.encode(str(reference.metadata.get("text", ""))))
                            for reference in own_references
                        )
                        accessible_tokens = local_tokens + displaced_tokens
                        selected_uris = sorted({hit.reference_uri for hit in selected_hits})
                        target_uris = sorted(_expected_reference_uris(item))
                        target_uri_set = set(target_uris)
                        selected_uri_set = set(selected_uris)
                        complete_rank_metrics = _complete_reference_rank_metrics(
                            layer_rankings, target_uri_set
                        )
                        if model.cfg.collect_rank_diagnostics:
                            complete_rank_metrics["candidate_rankings_by_layer"] = {
                                str(layer_id): candidates
                                for layer_id, candidates in layer_rankings.items()
                            }
                        routing_scores = [float(hit.chunk_score) for hit in selected_hits]
                        reference_scores = [float(hit.reference_score) for hit in selected_hits]
                        diagnostics = model.pra_diagnostics_by_layer()
                        diagnostic_values = list(diagnostics.values())
                        routing_latency = sum(
                            float(value.get("routing_duration_seconds", 0.0))
                            for value in diagnostics.values()
                        )
                        materialization_latency = sum(
                            float(value.get("materialization_duration_seconds", 0.0))
                            for value in diagnostics.values()
                        )
                        attention_latency = sum(
                            float(value.get("memory_attention_duration_seconds", 0.0))
                            for value in diagnostics.values()
                        )
                        transfer_bytes = sum(
                            float(value.get("retrieved_kv_transfer_bytes", 0.0))
                            for value in diagnostics.values()
                        )
                        retrieved_physical = sum(
                            float(value.get("retrieved_physical_kv_tokens", 0.0))
                            for value in diagnostic_values
                        ) / max(len(diagnostic_values), 1)
                        retrieved_unique = sum(
                            float(value.get("retrieved_unique_source_tokens", 0.0))
                            for value in diagnostic_values
                        ) / max(len(diagnostic_values), 1)
                        visible_entries = model.pra_cache.all_entries()
                        unique_source_tokens = encoded_tokens = stored_tokens = 0
                        seen_encoding_runs = set()
                        for entry in visible_entries:
                            run_id = entry.metadata.get("encoding_run_id")
                            if run_id is not None:
                                if run_id in seen_encoding_runs:
                                    continue
                                seen_encoding_runs.add(run_id)
                                unique_source_tokens += int(
                                    entry.metadata["encoding_run_unique_source_tokens"]
                                )
                                encoded_tokens += int(
                                    entry.metadata[
                                        "encoding_run_encoded_tokens_including_overlap"
                                    ]
                                )
                                stored_tokens += int(
                                    entry.metadata["encoding_run_stored_kv_tokens"]
                                )
                            else:
                                unique_source_tokens += int(
                                    entry.metadata.get("unique_source_tokens", 0)
                                )
                                encoded_tokens += int(
                                    entry.metadata.get(
                                        "encoded_tokens_including_overlap", 0
                                    )
                                )
                                stored_tokens += int(
                                    entry.metadata.get(
                                        "stored_kv_tokens_including_overlap",
                                        entry.metadata.get(
                                            "encoded_tokens_including_overlap", 0
                                        ),
                                    )
                                )
                        row = item["sample"].metadata.get("row", {})
                        per_example.append(
                            {
                                "example_id": str(item["id"]),
                                "condition": condition,
                                "loss": item_loss_sum / max(count, 1),
                                "perplexity": perplexity(item_loss_sum / max(count, 1)),
                                "token_accuracy": item_correct / max(count, 1),
                                "target_tokens": count,
                                "local_tokens": local_tokens,
                                "displaced_tokens": displaced_tokens,
                                "accessible_tokens": accessible_tokens,
                                "retrieved_tokens": retrieved_tokens,
                                "retrieved_physical_kv_tokens": retrieved_physical,
                                "retrieved_unique_source_tokens": retrieved_unique,
                                "active_tokens": local_tokens + retrieved_tokens,
                                "active_fraction": (local_tokens + retrieved_tokens)
                                / max(accessible_tokens, 1),
                                "active_unique_fraction": (local_tokens + retrieved_unique)
                                / max(local_tokens + unique_source_tokens, 1),
                                "unique_source_tokens": unique_source_tokens,
                                "encoded_tokens_including_overlap": encoded_tokens,
                                "stored_kv_tokens_including_overlap": stored_tokens,
                                "duplication_factor": encoded_tokens
                                / max(unique_source_tokens, 1),
                                "chunk_overlap_fraction": model.cfg.chunk_overlap_fraction,
                                "chunk_overlap_tokens": model.cfg.resolved_chunk_overlap_tokens,
                                "overlap_materialization": model.cfg.overlap_materialization,
                                "reference_encoding_strategy": (
                                    model.cfg.reference_encoding_strategy
                                ),
                                "encoding_block_references": (
                                    model.cfg.encoding_block_references
                                ),
                                "encoding_overlap_fraction": (
                                    model.cfg.encoding_overlap_fraction
                                ),
                                "reference_position_mode": model.cfg.reference_position_mode,
                                "prompt_position_mode": model.cfg.prompt_position_mode,
                                "num_references": len(own_references),
                                "num_chunks": sum(
                                    len(entry.layer_memory.get(0).chunks)
                                    for entry in model.pra_cache.all_entries()
                                    if entry.layer_memory.get(0) is not None
                                ),
                                "num_selected_chunks": len(
                                    {hit.chunk_id for hit in selected_hits}
                                ),
                                "num_selected_references": len(selected_uris),
                                "selected_reference_uris": selected_uris,
                                "oracle_selected_reference_uris": target_uris,
                                "routed_selected_chunk_ids": sorted(
                                    {hit.chunk_id for hit in selected_hits}
                                ),
                                "selection_hit": bool(target_uri_set & selected_uri_set),
                                "recall_at_k": len(target_uri_set & selected_uri_set)
                                / max(len(target_uri_set), 1),
                                "reference_top1": float(
                                    bool(selected_uris and selected_uris[0] in target_uri_set)
                                ),
                                "routing_score_statistics": {
                                    "count": len(routing_scores),
                                    "min": min(routing_scores) if routing_scores else None,
                                    "max": max(routing_scores) if routing_scores else None,
                                    "mean": sum(routing_scores) / len(routing_scores)
                                    if routing_scores
                                    else None,
                                    "reference_mean": sum(reference_scores) / len(reference_scores)
                                    if reference_scores
                                    else None,
                                },
                                "attention_latency": attention_latency,
                                "routing_latency": routing_latency,
                                "kv_materialization_latency": materialization_latency,
                                "example_latency": time.perf_counter() - item_start,
                                "kv_transfer_bytes": transfer_bytes,
                                "peak_cuda_memory": (
                                    float(torch.cuda.max_memory_allocated(device))
                                    if str(device).startswith("cuda")
                                    and torch.cuda.is_available()
                                    else 0.0
                                ),
                                "fixed_target_id": row.get("fixed_target_id"),
                                "search_strategy": model.cfg.search_strategy,
                                "top_k_references": model.cfg.top_k_references,
                                "top_k_chunks_per_reference": model.cfg.top_k_chunks_per_reference,
                                "trigger_threshold": model.cfg.trigger_threshold,
                                "gist_mode": model.cfg.gist_mode,
                                "gists_per_chunk": model.cfg.gists_per_chunk,
                                "gist_score_aggregation": model.cfg.gist_score_aggregation,
                                "reference_level_gist_mode": model.cfg.reference_level_gist_mode,
                                "reference_gists_per_reference": (
                                    model.cfg.reference_gists_per_reference
                                ),
                                "reference_score_aggregation": (
                                    model.cfg.reference_score_aggregation
                                ),
                                **complete_rank_metrics,
                            }
                        )
                    reference_index += 1
    finally:
        model.cfg.detail_materialization = previous_materialization
        (
            model.cfg.top_k_references,
            model.cfg.top_k_chunks_per_reference,
            model.cfg.trigger_threshold,
        ) = previous_routing
    if was_training:
        model.train()
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = loss_sum / max(token_count, 1)
    result = {
        "condition": condition,
        "loss": loss,
        "perplexity": perplexity(loss),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
        "duration_seconds": elapsed,
        "detail_materialization": ablation_materialization,
        "ablation": condition,
        "memory_transport": model.cfg.memory_transport,
    }
    if collect_per_example:
        result["per_example"] = per_example
    return result


def native_kv_gap_metrics(results: list[dict]) -> dict[str, float]:
    """Compute the preregistered native-KV quality gaps from condition results."""
    from .native_metrics import derive_native_kv_metrics

    losses = {str(result["condition"]): float(result["loss"]) for result in results}
    aliases = {
        "valid": "native_routed",
        "native_disabled": "native_disabled",
    }
    for source, target in aliases.items():
        if source in losses:
            losses[target] = losses[source]
    return derive_native_kv_metrics(losses)


def _pra_checkpoint_extra(
    cfg: PRAConfig,
    train_config: TrainConfig,
    tokenizer,
    resolver_config: ResolverServiceConfig,
    cache_config: CacheServiceConfig,
) -> Callable[[], dict]:
    """Create a lazy serializer for PRA/tokenizer/service reproducibility data."""
    return lambda: {
        "cfg": cfg.__dict__,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "dataset_stage": train_config.dataset_stage,
        "resolver_config": resolver_config.__dict__,
        "cache_config": cache_config.__dict__,
        "reference_vocabulary": {k: v for k, v in tokenizer.stoi.items() if k.startswith("<REF_")},
        "tokenizer_type": type(tokenizer).__name__,
        "tokenizer_json": tokenizer.to_json() if hasattr(tokenizer, "to_json") else None,
    }


def create_pra_training_state(
    cfg: PRAConfig,
    train_config: TrainConfig,
    datamodule: PRADataModule,
    *,
    model: TinyPRAModel | None = None,
) -> TrainingState:
    """Create state for PRA training without starting the training loop."""
    model = model or TinyPRAModel(cfg)
    resolver_config = ResolverServiceConfig.from_value(train_config.resolver_config)
    cache_config = CacheServiceConfig.from_value(train_config.cache_config)
    return create_training_state(
        model,
        train_config,
        checkpoint_extra=_pra_checkpoint_extra(
            cfg,
            train_config,
            datamodule.tokenizer,
            resolver_config,
            cache_config,
        ),
    )


def train_pra_model(
    *,
    cfg: PRAConfig,
    train_config: TrainConfig,
    datamodule: PRADataModule,
    model: TinyPRAModel | None = None,
    resolver_config: ResolverServiceConfig | dict | str | None = None,
    cache_config: CacheServiceConfig | dict | str | None = None,
    state: TrainingState | None = None,
):
    """Bind PRA cache-aware batch/eval steps to the generic training engine.

    ``train_model`` owns optimization, scheduling, checkpoints, and logging;
    this adapter owns the PRA-specific requirement to resolve and encode each
    sample's references before its language-model forward pass.
    """
    tokenizer = datamodule.tokenizer
    resolver_config = ResolverServiceConfig.from_value(resolver_config or train_config.resolver_config)
    cache_config = CacheServiceConfig.from_value(cache_config or train_config.cache_config)
    if state is None:
        model = model or TinyPRAModel(cfg)
        state = create_training_state(
            model,
            train_config,
            checkpoint_extra=_pra_checkpoint_extra(
                cfg,
                train_config,
                tokenizer,
                resolver_config,
                cache_config,
            ),
        )
    else:
        state.checkpoint_extra = _pra_checkpoint_extra(
            cfg,
            train_config,
            tokenizer,
            resolver_config,
            cache_config,
        )

    def batch_step(current_model, batch: dict, device: str):
        return _pra_batch_step(
            current_model,
            batch,
            device,
            tokenizer,
            resolver_config,
            cache_config,
        )

    def eval_step(current_model, loader, device: str, split: str = "val"):
        return evaluate_pra_model(
            model=current_model,
            loader=loader,
            tokenizer=tokenizer,
            train_config=train_config,
            device=device,
            split=split,
            resolver_config=resolver_config,
            cache_config=cache_config,
        )

    return train_model(
        model=state.model,
        train_config=train_config,
        train_loader=datamodule.train_loader(),
        val_loader=datamodule.val_loader(),
        test_loader=datamodule.test_loader(),
        batch_step=batch_step,
        eval_step=eval_step,
        state=state,
    )
