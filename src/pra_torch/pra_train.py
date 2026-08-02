"""PRA-specific adapters for the generic training engine."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from data.datamodules import PRADataModule
from .cache_services import build_cache_from_metadata
from .config import CacheServiceConfig, PRAConfig, ResolverServiceConfig, TrainConfig
from .metrics import RunningAverages, cuda_memory_allocated, perplexity
from .model import TinyPRAModel
from .train import TrainingState, create_training_state, move_batch, train_model


def _pra_batch_step(
    model,
    batch: dict,
    device: str,
    tokenizer,
    resolver_config: ResolverServiceConfig,
    cache_config: CacheServiceConfig,
) -> tuple[torch.Tensor, dict]:
    batch = move_batch(batch, device)
    logits_by_example = []
    caches = []
    selections = []
    for index, metadata in enumerate(batch["metadata"]):
        cache = build_cache_from_metadata(
            model,
            tokenizer,
            [metadata],
            device,
            resolver_config=resolver_config,
            cache_config=cache_config,
        )
        logits_by_example.append(model(batch["input_ids"][index : index + 1]))
        caches.append(cache)
        selections.append(model.selected_references_by_layer())
    logits = torch.cat(logits_by_example, dim=0)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=0)
    return loss, {
        "tokens": int(batch["attention_mask"].sum().item()),
        "examples": int(batch["input_ids"].shape[0]),
        "batch": batch,
        "cache": caches[-1],
        "caches": caches,
        "selections": selections,
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
    """Evaluate one split and optionally emit JSONL predictions and PRA traces."""
    resolver_config = ResolverServiceConfig.from_value(resolver_config or train_config.resolver_config)
    cache_config = CacheServiceConfig.from_value(cache_config or train_config.cache_config)
    was_training = model.training
    model.eval()
    averages = RunningAverages()
    exact = anchor_hits = cache_hits = total = 0
    selection_units = selection_top1_hits = selection_topk_hits = 0
    selection_reciprocal_rank = selected_refs = 0
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
                averages.update({f"{split}_loss": float(loss.detach().cpu())})
                token_total += int(batch_metrics["tokens"])

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
                    expected_uris = {
                        ref.uri for ref in item["references"] if ref.id in item["target_reference_ids"]
                    }
                    cache_hit = not expected_uris or bool(expected_uris.intersection(cache.entries))
                    if cache_hit:
                        cache_hits += 1
                    layer_selections = batch_metrics["selections"][index]
                    for ranked in layer_selections.values():
                        selection_units += 1
                        selected_refs += len(ranked)
                        selected_uris = [uri for uri, _score in ranked]
                        if selected_uris and selected_uris[0] in expected_uris:
                            selection_top1_hits += 1
                        ranks = [rank for rank, uri in enumerate(selected_uris, start=1) if uri in expected_uris]
                        if ranks:
                            selection_topk_hits += 1
                            selection_reciprocal_rank += 1.0 / min(ranks)
                    anchors = item.get("expected_anchors", [])
                    expansion_depth += len(anchors)
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
                            str(layer_id): [uri for uri, _score in ranked]
                            for layer_id, ranked in layer_selections.items()
                        },
                        "expanded_anchors": anchors,
                        "expansion_depth": len(anchors),
                        "cache_hits": cache_hit,
                        "retrieved_token_counts": [
                            len(tokenizer.encode(entry.text)) for entry in cache.all_entries()
                        ],
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
            "reference_retrieval_accuracy": selection_topk_hits / max(selection_units, 1),
            "reference_selection_top1_accuracy": selection_top1_hits / max(selection_units, 1),
            "reference_selection_topk_accuracy": selection_topk_hits / max(selection_units, 1),
            "reference_selection_mrr": selection_reciprocal_rank / max(selection_units, 1),
            "selected_ref_count": selected_refs / max(selection_units, 1),
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
) -> dict:
    """Measure answer-token LM quality under reference-content ablations."""
    supported = {"valid", "disabled", "shuffled", "irrelevant", "empty", "oracle"}
    if condition not in supported:
        raise ValueError(f"Unsupported reference condition: {condition}")
    was_training = model.training
    model.eval()
    loss_sum = correct = token_count = 0
    start = time.perf_counter()
    batches = list(loader)
    reference_sets = [
        item["references"] for batch in batches for item in batch["metadata"]
    ]
    reference_pool = [reference for references in reference_sets for reference in references]
    shuffle_offset = max(len(reference_sets) // 2, 1)
    reference_index = 0
    with torch.no_grad():
        for batch in batches:
            moved = move_batch(batch, device)
            for index, item in enumerate(batch["metadata"]):
                if condition in {"disabled", "empty"}:
                    model.clear_pra_cache()
                else:
                    metadata = item
                    if condition == "shuffled":
                        shuffled_index = (reference_index + shuffle_offset) % len(reference_sets)
                        metadata = {**item, "references": reference_sets[shuffled_index]}
                    elif condition == "irrelevant":
                        own_uris = {reference.uri for reference in item["references"]}
                        start_index = (reference_index + shuffle_offset) % max(len(reference_pool), 1)
                        candidates = reference_pool[start_index:] + reference_pool[:start_index]
                        irrelevant = [reference for reference in candidates if reference.uri not in own_uris]
                        metadata = {**item, "references": irrelevant[: len(item["references"])]}
                    elif condition == "oracle":
                        target_ids = set(item["target_reference_ids"])
                        metadata = {
                            **item,
                            "references": [
                                reference for reference in item["references"] if reference.id in target_ids
                            ],
                        }
                    build_cache_from_metadata(
                        model,
                        tokenizer,
                        [metadata],
                        device,
                        resolver_config=resolver_config,
                        cache_config=cache_config,
                    )
                logits = model(
                    moved["input_ids"][index : index + 1],
                    use_pra_memory=condition != "disabled",
                )
                labels = moved["labels"][index : index + 1]
                valid = labels.ne(0)
                count = int(valid.sum().item())
                loss_sum += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=0,
                        reduction="sum",
                    ).detach().cpu()
                )
                correct += int((logits.argmax(dim=-1).eq(labels) & valid).sum().item())
                token_count += count
                reference_index += 1
    if was_training:
        model.train()
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = loss_sum / max(token_count, 1)
    return {
        "condition": condition,
        "loss": loss,
        "perplexity": perplexity(loss),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
        "duration_seconds": elapsed,
    }


def _pra_checkpoint_extra(
    cfg: PRAConfig,
    train_config: TrainConfig,
    tokenizer,
    resolver_config: ResolverServiceConfig,
    cache_config: CacheServiceConfig,
) -> Callable[[], dict]:
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
    """Train ``TinyPRAModel`` through the generic functional engine."""
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
