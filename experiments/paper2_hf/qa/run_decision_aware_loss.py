"""Test a decision-aware polarity loss on routed QASPER memory.

The experiment keeps the Paper 2 model, router, last-14 residual adapter,
identity split, optimizer, update count, and memory budget fixed. Validation
selects only the weight of a tokenizer-native binary yes/no loss; the selected
weight and the matched sequence-only baseline are then evaluated on test.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_last14_combo import (
    Variant,
    _configure_variant,
    _freeze_backbone,
    _load_checkpoint,
    _prepare_records,
    _save_checkpoint,
    last_band_layers,
)
from experiments.paper2_hf.qa.run_lm_head_control import (
    _evaluate_language_model,
    _load_wikitext_blocks,
)
from experiments.paper2_hf.qa.run_memory_gate import _activate, _sync
from experiments.paper2_hf.qa.run_qasper_diagnostic import (
    _answer_contained,
    _evaluate_one,
    _generate_with_finish,
    _normalize,
    _score_answer,
    _starts_with_polarity,
    polarity_token_ids,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection


ROOT = Path(__file__).resolve().parents[3]
SEEDS = (11, 23, 37, 53, 71)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
RESIDUAL = Variant("residual_16", residual_width=16)


def grouped_polarity_logits(
    first_token_logits: torch.Tensor,
    candidate_ids: dict[str, list[int]],
) -> torch.Tensor:
    """Return ``[B,2]`` grouped logits for all tokenizer-native yes/no forms."""
    if first_token_logits.ndim != 2:
        raise ValueError("Expected first-token logits with shape [batch, vocabulary].")
    if not candidate_ids["yes"] or not candidate_ids["no"]:
        raise ValueError("Both polarity candidate sets must be non-empty.")
    return torch.stack(
        tuple(
            torch.logsumexp(first_token_logits[:, candidate_ids[label]], dim=-1)
            for label in ("no", "yes")
        ),
        dim=-1,
    )


def decision_aware_objective(
    answer_logits: torch.Tensor,
    answer_ids: torch.Tensor,
    gold_polarity: str,
    candidate_ids: dict[str, list[int]],
    polarity_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine mean sequence CE with grouped binary polarity CE.

    ``answer_logits`` has shape ``[B,A,V]`` and predicts ``answer_ids`` with
    shape ``[B,A]``. The binary term uses only the first answer position, where
    QASPER's task-critical yes/no branch occurs.
    """
    if gold_polarity not in {"yes", "no"}:
        raise ValueError(f"Expected binary QASPER answer, got {gold_polarity!r}.")
    sequence_loss = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_ids.to(answer_logits.device).reshape(-1),
        reduction="mean",
    )
    binary_logits = grouped_polarity_logits(answer_logits[:, 0, :], candidate_ids)
    target = torch.full(
        (answer_logits.shape[0],),
        1 if gold_polarity == "yes" else 0,
        dtype=torch.long,
        device=answer_logits.device,
    )
    polarity_loss = F.cross_entropy(binary_logits, target)
    total = sequence_loss + float(polarity_weight) * polarity_loss
    return total, sequence_loss, polarity_loss


def compact_answer_logits(
    handle,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return only answer-position logits via the established compact path."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    positions = torch.arange(
        prompt_tokens - 1,
        full_ids.shape[1] - 1,
        device=device,
        dtype=torch.long,
    )
    return handle.model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
        logits_to_keep=positions,
    ).logits.float()


def class_balance(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report binary counts and the majority-class accuracy baseline."""
    counts = Counter(_normalize(row["answer"]) for row in examples)
    if set(counts) - {"yes", "no"}:
        raise ValueError(f"Non-binary answers in QASPER decision cohort: {counts}")
    total = sum(counts.values())
    majority_label, majority_count = max(sorted(counts.items()), key=lambda item: item[1])
    return {
        "examples": total,
        "yes": counts.get("yes", 0),
        "no": counts.get("no", 0),
        "majority_label": majority_label,
        "majority_accuracy": majority_count / total,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _release(handle, *groups: list[dict[str, Any]]) -> None:
    handle.cache.clear()
    for group in groups:
        group.clear()
    gc.collect()
    if handle.device.type == "cuda":
        torch.cuda.empty_cache()


def _checkpoint(directory: Path, polarity_weight: float, seed: int) -> Path:
    stable = str(polarity_weight).replace(".", "p")
    return directory / f"routed_residual_16_lambda_{stable}_seed{seed}.pt"


def _train_one(
    handle,
    records: list[dict[str, Any]],
    layers: tuple[int, ...],
    candidate_ids: dict[str, list[int]],
    polarity_weight: float,
    seed: int,
    steps: int,
    learning_rate: float,
    checkpoint: Path,
) -> tuple[dict[str, Any], int]:
    """Fit one seeded residual adapter with routed memory and a fixed loss."""
    _freeze_backbone(handle)
    torch.manual_seed(seed)
    parameters = _configure_variant(handle, RESIDUAL, reset=True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    total_losses: list[float] = []
    sequence_losses: list[float] = []
    polarity_losses: list[float] = []
    started = time.perf_counter()
    handle.model.eval()
    for step in range(steps):
        record = records[order[step % len(order)]]
        _activate(handle, record, layers, "routed")
        optimizer.zero_grad(set_to_none=True)
        logits = compact_answer_logits(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            handle.device,
        )
        total, sequence, polarity = decision_aware_objective(
            logits,
            record["answer_ids"],
            _normalize(record["example"]["answer"]),
            candidate_ids,
            polarity_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        total_losses.append(float(total.detach().cpu()))
        sequence_losses.append(float(sequence.detach().cpu()))
        polarity_losses.append(float(polarity.detach().cpu()))
    _sync(handle.device)
    window = min(len(records), len(total_losses))
    report = {
        "variant": RESIDUAL.name,
        "training_dataset": "qasper",
        "training_memory": "routed",
        "objective": "mean_sequence_ce_plus_lambda_grouped_binary_ce",
        "polarity_weight": polarity_weight,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_seconds": time.perf_counter() - started,
        "initial_total_loss": statistics.fmean(total_losses[:window]),
        "final_total_loss": statistics.fmean(total_losses[-window:]),
        "initial_sequence_loss": statistics.fmean(sequence_losses[:window]),
        "final_sequence_loss": statistics.fmean(sequence_losses[-window:]),
        "initial_polarity_loss": statistics.fmean(polarity_losses[:window]),
        "final_polarity_loss": statistics.fmean(polarity_losses[-window:]),
        "total_losses": total_losses,
        "sequence_losses": sequence_losses,
        "polarity_losses": polarity_losses,
    }
    return report, _save_checkpoint(checkpoint, handle, RESIDUAL, seed, report)


def _load_adapter(handle, checkpoint: Path) -> int:
    _freeze_backbone(handle)
    _configure_variant(handle, RESIDUAL, reset=True)
    _, size = _load_checkpoint(checkpoint, handle, RESIDUAL)
    return size


def _condition_row(
    handle,
    tokenizer,
    record: dict[str, Any],
    layers: tuple[int, ...],
    candidate_ids: dict[str, list[int]],
    condition: str,
    polarity_weight: float,
    seed: int,
    new_tokens: int,
    *,
    generate: bool,
) -> dict[str, Any]:
    _activate(handle, record, layers, condition)
    gold = _normalize(record["example"]["answer"])
    score = _score_answer(
        handle,
        record["prompt_ids"],
        record["prompt_mask"],
        record["answer_ids"],
        candidate_ids,
        gold,
        handle.device,
    )
    row = {
        "dataset": "qasper",
        "example_id": record["example"]["id"],
        "reference_answer": gold,
        "condition": condition,
        "polarity_weight": polarity_weight,
        "seed": seed,
        **score,
        "margin_correct": float(score["gold_polarity_margin"] > 0),
    }
    if generate:
        generated = _generate_with_finish(
            handle,
            tokenizer,
            record["prompt_ids"],
            record["prompt_mask"],
            handle.device,
            new_tokens,
        )
        observed = generated["generated_text"]
        decoded = _starts_with_polarity(observed)
        lexical = answer_metrics(observed, gold)
        row.update(
            {
                **generated,
                "decoded_polarity": decoded,
                "polarity_correct": float(decoded == gold),
                "format_correct": float(decoded is not None),
                "answer_contained": float(_answer_contained(observed, gold)),
                **lexical,
            }
        )
    return row


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "polarity_correct",
        "margin_correct",
        "gold_polarity_margin",
        "gold_sequence_logprob",
        "gold_first_token_rank",
        "answer_contained",
        "f1",
        "em",
        "format_correct",
        "eos_emitted",
        "hit_max_new_tokens",
    )
    output = []
    for key, values in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["rows"] = len(values)
        item["identities"] = len({row["example_id"] for row in values})
        item["seeds"] = len({row["seed"] for row in values})
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            item[f"{metric}_mean"] = statistics.fmean(samples) if samples else None
            item[f"{metric}_median"] = statistics.median(samples) if samples else None
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0 if samples else None
        output.append(item)
    return output


def select_polarity_weight(validation_aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose lambda by decoded polarity, then realization and margin, never test."""
    if not validation_aggregate:
        raise ValueError("No validation candidates supplied.")
    return max(
        validation_aggregate,
        key=lambda row: (
            row["polarity_correct_mean"],
            row["margin_correct_mean"],
            row["f1_mean"],
            row["eos_emitted_mean"],
            row["answer_contained_mean"],
            row["gold_polarity_margin_mean"],
            -float(row["polarity_weight"]),
        ),
    )


def aggregate_seed_means(
    seed_rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Aggregate per-seed cohort means and expose seed-level dispersion."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    suffix = "_mean"
    for key, values in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["seeds"] = len(values)
        metrics = sorted(
            {
                name[: -len(suffix)]
                for row in values
                for name, value in row.items()
                if name.endswith(suffix) and isinstance(value, (int, float))
            }
        )
        for metric in metrics:
            samples = [
                float(row[f"{metric}_mean"])
                for row in values
                if row.get(f"{metric}_mean") is not None
            ]
            item[f"{metric}_mean"] = statistics.fmean(samples) if samples else None
            item[f"{metric}_seed_std"] = (
                statistics.stdev(samples) if len(samples) > 1 else 0.0 if samples else None
            )
            item[f"{metric}_seed_values"] = samples
        output.append(item)
    return output


def _plot(validation: list[dict[str, Any]], test: list[dict[str, Any]], output: Path) -> None:
    lambdas = [float(row["polarity_weight"]) for row in validation]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(
        lambdas,
        [100 * row["polarity_correct_mean"] for row in validation],
        marker="o",
        color="#2878B5",
        label="Decoded polarity",
    )
    axes[0].plot(
        lambdas,
        [100 * row["margin_correct_mean"] for row in validation],
        marker="s",
        color="#5B8C3A",
        label="Margin > 0",
    )
    axes[0].axhline(75, color="#666666", linestyle="--", label="Validation majority")
    axes[0].set_xlabel("Polarity-loss weight")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8)

    lookup = {float(row["polarity_weight"]): row for row in test}
    baseline = lookup[0.0]
    selected = test[-1]
    labels = ["Sequence only", "Decision aware", "Existing R16"]
    values = [
        100 * baseline["polarity_correct_mean"],
        100 * selected["polarity_correct_mean"],
        72.5,
    ]
    axes[1].bar(labels, values, color=("#7F7F7F", "#2878B5", "#D95F02"))
    axes[1].axhline(62.5, color="#666666", linestyle="--", label="Test majority")
    axes[1].set_ylabel("Test polarity accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"decision_aware_polarity.{suffix}", dpi=190)
    plt.close(figure)


def audit_baseline_reproduction(
    test_rows: list[dict[str, Any]], previous_artifact: Path
) -> dict[str, Any]:
    """Require lambda zero to reproduce the prior routed-QASPER baseline exactly."""
    previous = json.loads(previous_artifact.read_text(encoding="utf-8"))
    old = {
        (row["example_id"], int(row["seed"])): row
        for row in previous["rows"]
        if row["condition"] == "pra_routed_residual_16_qasper_trained"
    }
    new = {
        (row["example_id"], int(row["seed"])): row
        for row in test_rows
        if row["condition"] == "routed" and float(row["polarity_weight"]) == 0.0
    }
    fields = (
        "generated_text",
        "polarity_correct",
        "answer_contained",
        "f1",
        "em",
        "eos_emitted",
        "hit_max_new_tokens",
        "gold_sequence_logprob",
        "gold_mean_token_logprob",
        "gold_first_token_rank",
        "gold_first_token_margin",
        "gold_polarity_margin",
    )
    keys_match = set(old) == set(new)
    common = old.keys() & new.keys()
    mismatches = {
        field: sum(old[key][field] != new[key][field] for key in common)
        for field in fields
    }
    exact = keys_match and not any(mismatches.values())
    try:
        provenance = str(previous_artifact.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        provenance = previous_artifact.name
    return {
        "previous_artifact": provenance,
        "old_rows": len(old),
        "new_rows": len(new),
        "keys_match": keys_match,
        "field_mismatches": mismatches,
        "exact": exact,
    }


def refresh_existing_artifact(path: Path, previous_artifact: Path) -> dict[str, Any]:
    """Add deterministic audit fields without repeating model inference."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["baseline_reproduction"] = audit_baseline_reproduction(
        artifact["test_rows"], previous_artifact
    )
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The decision-aware Paper 2 experiment requires CUDA.")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = load_hf_routing_projection(args.router_checkpoint, device=device)
    layers = last_band_layers(int(model.config.num_hidden_layers))
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=True,
            kv_cache_non_blocking=True,
        ),
        routing_projection=projection,
    )
    candidate_ids = polarity_token_ids(tokenizer)

    split_examples = {
        "train": [
            row for row in load_split_examples(args.cache_dir, 12, 0, args.data_seed)
            if row["dataset"] == "qasper"
        ],
        "validation": [
            row for row in load_split_examples(args.cache_dir, 4, 12, args.data_seed)
            if row["dataset"] == "qasper"
        ],
        "test": [
            row for row in load_split_examples(args.cache_dir, 8, 16, args.data_seed)
            if row["dataset"] == "qasper"
        ],
    }
    identities = {name: [row["id"] for row in values] for name, values in split_examples.items()}
    if any(
        set(identities[left]) & set(identities[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("Decision-loss train, validation, and test identities overlap.")
    balances = {name: class_balance(values) for name, values in split_examples.items()}
    print(f"class balance: {balances}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    print("preparing train and validation routed references", flush=True)
    train_records = _prepare_records(handle, tokenizer, split_examples["train"], layers, args, controls=False)
    validation_records = _prepare_records(
        handle, tokenizer, split_examples["validation"], layers, args, controls=False
    )
    training_reports: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for polarity_weight in args.polarity_weights:
        for seed in args.seeds:
            checkpoint = _checkpoint(checkpoint_dir, polarity_weight, seed)
            report, checkpoint_bytes = _train_one(
                handle,
                train_records,
                layers,
                candidate_ids,
                polarity_weight,
                seed,
                args.steps,
                args.learning_rate,
                checkpoint,
            )
            report["checkpoint_bytes"] = checkpoint_bytes
            training_reports.append(report)
            for record in validation_records:
                validation_rows.append(
                    _condition_row(
                        handle,
                        tokenizer,
                        record,
                        layers,
                        candidate_ids,
                        "routed",
                        polarity_weight,
                        seed,
                        args.new_tokens,
                        generate=True,
                    )
                )
            print(f"validation lambda={polarity_weight:g} seed={seed}", flush=True)

    validation_aggregate = aggregate(validation_rows, ("polarity_weight", "condition"))
    validation_by_seed = aggregate(
        validation_rows, ("polarity_weight", "condition", "seed")
    )
    validation_seed_aggregate = aggregate_seed_means(
        validation_by_seed, ("polarity_weight", "condition")
    )
    selection = select_polarity_weight(validation_aggregate)
    selected_weight = float(selection["polarity_weight"])
    print(f"selected polarity weight: {selected_weight:g}", flush=True)
    _release(handle, train_records, validation_records)

    print("preparing untouched QASPER test references", flush=True)
    test_records = _prepare_records(handle, tokenizer, split_examples["test"], layers, args, controls=False)
    frozen_no_memory: dict[str, dict[str, Any]] = {}
    _freeze_backbone(handle)
    _configure_variant(handle, Variant("fixed"), reset=True)
    for record in test_records:
        _activate(handle, record, layers, "none")
        score = _score_answer(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            candidate_ids,
            _normalize(record["example"]["answer"]),
            device,
        )
        generation = _generate_with_finish(
            handle,
            tokenizer,
            record["prompt_ids"],
            record["prompt_mask"],
            device,
            args.new_tokens,
        )
        frozen_no_memory[record["example"]["id"]] = {**score, **generation}

    test_rows: list[dict[str, Any]] = []
    tested_weights = tuple(dict.fromkeys((0.0, selected_weight)))
    no_memory_exact: list[dict[str, Any]] = []
    for polarity_weight in tested_weights:
        for seed in args.seeds:
            checkpoint_bytes = _load_adapter(
                handle, _checkpoint(checkpoint_dir, polarity_weight, seed)
            )
            conditions = ("routed", "oracle", "none") if polarity_weight == selected_weight else ("routed",)
            for record in test_records:
                for condition in conditions:
                    generate = condition in {"routed", "none"}
                    row = _condition_row(
                        handle,
                        tokenizer,
                        record,
                        layers,
                        candidate_ids,
                        condition,
                        polarity_weight,
                        seed,
                        args.new_tokens,
                        generate=generate,
                    )
                    row["checkpoint_bytes"] = checkpoint_bytes
                    test_rows.append(row)
                    if condition == "none":
                        reference = frozen_no_memory[row["example_id"]]
                        score_keys = (
                            "gold_sequence_logprob",
                            "gold_mean_token_logprob",
                            "gold_first_token_probability",
                            "gold_first_token_rank",
                            "gold_first_token_margin",
                            "yes_minus_no_logprob",
                        )
                        exact = all(row[key] == reference[key] for key in score_keys)
                        exact = exact and row["generated_text"] == reference["generated_text"]
                        no_memory_exact.append(
                            {
                                "seed": seed,
                                "example_id": row["example_id"],
                                "exact": exact,
                            }
                        )
            print(f"test lambda={polarity_weight:g} seed={seed}", flush=True)

    if no_memory_exact and not all(row["exact"] for row in no_memory_exact):
        raise AssertionError("Decision-aware residual changed the PRA-inactive path.")

    wikitext_blocks = _load_wikitext_blocks(
        tokenizer, args.cache_dir, args.wikitext_blocks, args.wikitext_tokens
    )
    _freeze_backbone(handle)
    _configure_variant(handle, Variant("fixed"), reset=True)
    frozen_wikitext = _evaluate_language_model(handle, wikitext_blocks, device)
    wikitext_rows = []
    for seed in args.seeds:
        _load_adapter(handle, _checkpoint(checkpoint_dir, selected_weight, seed))
        measured = _evaluate_language_model(handle, wikitext_blocks, device)
        handle.set_memory_enabled(True)
        wikitext_rows.append(
            {
                "seed": seed,
                "polarity_weight": selected_weight,
                **measured,
                "loss_delta": measured["loss"] - frozen_wikitext["loss"],
                "exact": measured["loss"] == frozen_wikitext["loss"],
            }
        )
    if not all(row["exact"] for row in wikitext_rows):
        raise AssertionError("Decision-aware residual changed no-memory WikiText loss.")

    test_seed_aggregate = aggregate(test_rows, ("polarity_weight", "condition", "seed"))
    test_aggregate = aggregate(test_rows, ("polarity_weight", "condition"))
    test_five_seed_aggregate = aggregate_seed_means(
        test_seed_aggregate, ("polarity_weight", "condition")
    )
    baseline_reproduction = audit_baseline_reproduction(
        test_rows,
        args.previous_diagnostic,
    )
    if not baseline_reproduction["exact"]:
        raise AssertionError(f"Sequence-only baseline did not reproduce: {baseline_reproduction}")
    routed_test = sorted(
        [
            row for row in test_aggregate
            if row["condition"] == "routed" and row["polarity_weight"] in tested_weights
        ],
        key=lambda row: float(row["polarity_weight"]),
    )
    _plot(validation_aggregate, routed_test, args.output_dir)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "Paper 2 decision-aware QASPER loss; test untouched until lambda selection",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layers": list(layers),
        "identity_disjoint": True,
        "identities": identities,
        "class_balance": balances,
        "polarity_candidate_ids": candidate_ids,
        "polarity_weights": list(args.polarity_weights),
        "selection_rule": "decoded polarity, margin accuracy, F1, EOS, containment, mean margin, lower lambda",
        "selected_polarity_weight": selected_weight,
        "selection": selection,
        "training_reports": training_reports,
        "validation_rows": validation_rows,
        "validation_by_seed": validation_by_seed,
        "validation_aggregate": validation_aggregate,
        "validation_seed_aggregate": validation_seed_aggregate,
        "test_rows": test_rows,
        "test_seed_aggregate": test_seed_aggregate,
        "test_aggregate": test_aggregate,
        "test_five_seed_aggregate": test_five_seed_aggregate,
        "baseline_reproduction": baseline_reproduction,
        "frozen_no_memory": frozen_no_memory,
        "no_memory_exact": no_memory_exact,
        "frozen_wikitext": frozen_wikitext,
        "wikitext_rows": wikitext_rows,
        "existing_residual_16_reference": {
            "source": "error_analysis/generation_error_analysis.json",
            "polarity_accuracy_mean": 0.725,
            "polarity_accuracy_seed_std": 0.05590169943749474,
            "answer_containment": 0.825,
            "f1": 0.15243284493284492,
            "eos_rate": 0.95,
        },
    }
    (args.output_dir / "decision_aware_loss.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "decision_aware_validation.csv", validation_aggregate)
    _write_csv(
        args.output_dir / "decision_aware_validation_by_seed.csv", validation_by_seed
    )
    _write_csv(
        args.output_dir / "decision_aware_validation_seed_aggregate.csv",
        validation_seed_aggregate,
    )
    _write_csv(args.output_dir / "decision_aware_test.csv", test_aggregate)
    _write_csv(args.output_dir / "decision_aware_test_by_seed.csv", test_seed_aggregate)
    _write_csv(
        args.output_dir / "decision_aware_test_five_seed.csv",
        test_five_seed_aggregate,
    )
    _write_csv(args.output_dir / "decision_aware_wikitext.csv", wikitext_rows)
    return artifact


def parse_args() -> argparse.Namespace:
    results = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--polarity-weights", nargs="+", type=float, default=list(LAMBDAS))
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--wikitext-blocks", type=int, default=2)
    parser.add_argument("--wikitext-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--router-checkpoint",
        type=Path,
        default=results / "routing" / "learned_adapter" / "checkpoints" / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=results / "decision_aware_loss",
    )
    parser.add_argument(
        "--previous-diagnostic",
        type=Path,
        default=results / "error_analysis" / "generation_error_analysis.json",
    )
    parser.add_argument(
        "--refresh-existing",
        type=Path,
        help="Refresh deterministic audit fields in a completed result without inference.",
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    args.polarity_weights = tuple(args.polarity_weights)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    result = (
        refresh_existing_artifact(parsed.refresh_existing, parsed.previous_diagnostic)
        if parsed.refresh_existing
        else run(parsed)
    )
    print(
        json.dumps(
            {
                "selected_polarity_weight": result["selected_polarity_weight"],
                "validation_candidates": len(result["validation_aggregate"]),
                "test_rows": len(result["test_rows"]),
            },
            indent=2,
        )
    )
