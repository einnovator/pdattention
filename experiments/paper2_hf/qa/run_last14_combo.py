"""Converge Paper 2 adaptation in Qwen's empirically best last-14 band.

The experiment freezes the base model and established semantic router, trains
only PRA-conditional residual and/or output-LoRA parameters, selects widths on
an identity-disjoint validation slice, and evaluates oracle and routed memory
against no-context, direct-evidence, and feasible full-context controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_memory_gate import _activate, _prepare_example, _sync
from experiments.paper2_hf.qa.run_oracle_memory_use import _generate, _prompt
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import (
    MEMORY_GATE_FIXED,
    PRAHFConfig,
    inject_pra,
    load_hf_routing_projection,
)


SEEDS = (11, 23, 37, 53, 71)
RESIDUAL_WIDTHS = (16, 32, 64)
LORA_RANKS = (4, 8)
INDIVIDUAL_VARIANTS = (
    "fixed",
    *(f"residual_{width}" for width in RESIDUAL_WIDTHS),
    *(f"lora_o_r{rank}" for rank in LORA_RANKS),
)


@dataclass(frozen=True)
class Variant:
    """One PRA-conditional memory-use parameterization."""

    name: str
    residual_width: int = 0
    lora_rank: int = 0


def variant_from_name(name: str) -> Variant:
    """Parse stable experiment names into independent adapter controls."""
    if name == "fixed":
        return Variant(name)
    if name.startswith("residual_"):
        return Variant(name, residual_width=int(name.rsplit("_", 1)[1]))
    if name.startswith("lora_o_r"):
        return Variant(name, lora_rank=int(name.rsplit("r", 1)[1]))
    if name.startswith("combo_residual_") and "_lora_r" in name:
        left, right = name.split("_lora_r", 1)
        return Variant(
            name,
            residual_width=int(left.rsplit("_", 1)[1]),
            lora_rank=int(right),
        )
    raise ValueError(f"Unsupported last-14 variant: {name}")


def _configure_variant(handle, variant: Variant, *, reset: bool) -> list[torch.nn.Parameter]:
    """Activate exactly the requested conditional banks and return their parameters."""
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    handle.configure_residual_adapter(0)
    handle.configure_late_band_lora(0)
    if variant.residual_width:
        handle.configure_residual_adapter(variant.residual_width, reset=reset)
    if variant.lora_rank:
        handle.configure_late_band_lora(
            variant.lora_rank,
            alpha=float(variant.lora_rank),
            dropout=0.0,
            reset=reset,
        )
    return handle.memory_use_parameters()


def _compact_gold_scores(handle, prompt_ids, prompt_mask, answer_ids, device):
    """Score only answer positions, avoiding dense full-context vocabulary logits."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    positions = torch.arange(
        prompt_tokens - 1,
        full_ids.shape[1] - 1,
        device=device,
        dtype=torch.long,
    )
    output = handle.model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
        logits_to_keep=positions,
    )
    logits = output.logits.float()
    targets = answer_ids.to(device)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    first_logits = logits[0, 0]
    first_target = int(targets[0, 0])
    wrong_logits = first_logits.clone()
    wrong_logits[first_target] = float("-inf")
    metrics = {
        "gold_sequence_logprob": float(token_log_probs.sum().detach().cpu()),
        "gold_mean_token_logprob": float(token_log_probs.mean().detach().cpu()),
        "gold_first_token_probability": float(
            first_logits.softmax(dim=-1)[first_target].detach().cpu()
        ),
        "gold_first_token_rank": int(
            (first_logits > first_logits[first_target]).sum().detach().cpu()
        )
        + 1,
        "gold_first_token_margin": float(
            (first_logits[first_target] - wrong_logits.max()).detach().cpu()
        ),
    }
    return -token_log_probs.mean(), metrics


def _add_context_controls(record, tokenizer, direct_tokens: int, full_tokens: int):
    """Attach evidence-only and complete-source prompts without silent truncation."""
    direct_ids, direct_mask, _ = _prompt(
        tokenizer,
        record["example"]["question"],
        context="\n".join(record["example"]["evidence"]),
        max_tokens=direct_tokens,
    )
    candidate_ids, candidate_mask, _ = _prompt(
        tokenizer,
        record["example"]["question"],
        context=record["example"]["source"],
        max_tokens=100_000,
    )
    record["direct_prompt_ids"] = direct_ids
    record["direct_prompt_mask"] = direct_mask
    record["full_context_tokens"] = int(candidate_ids.shape[1])
    record["full_context_complete"] = int(candidate_ids.shape[1]) <= full_tokens
    if record["full_context_complete"]:
        record["full_prompt_ids"] = candidate_ids
        record["full_prompt_mask"] = candidate_mask
    return record


def _freeze_backbone(handle) -> None:
    """Keep model, router, and inactive lazy banks frozen between variants."""
    for parameter in handle.model.parameters():
        parameter.requires_grad_(False)


def _active_parameter_names(handle) -> list[str]:
    return [name for name, parameter in handle.model.named_parameters() if parameter.requires_grad]


def _save_checkpoint(path: Path, handle, variant: Variant, seed: int, report: dict) -> int:
    """Persist only active PRA memory-use tensors plus protocol metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in handle.model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "variant": variant.__dict__,
            "seed": seed,
            "state_dict": state,
            "training": report,
        },
        path,
    )
    return path.stat().st_size


def _load_checkpoint(path: Path, handle, variant: Variant) -> tuple[dict, int]:
    """Restore a resumable variant after configuring the matching lazy banks."""
    payload = torch.load(path, map_location=handle.device, weights_only=False)
    expected = variant.__dict__
    if payload["variant"] != expected:
        raise ValueError(f"Checkpoint variant mismatch: {payload['variant']} != {expected}")
    named = dict(handle.model.named_parameters())
    for name, value in payload["state_dict"].items():
        named[name].data.copy_(value.to(named[name].device))
    report = dict(payload["training"])
    report["resumed_from_checkpoint"] = True
    return report, path.stat().st_size


def _train_variant(
    handle,
    records,
    layers,
    variant: Variant,
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    checkpoint: Path,
    *,
    optimizer_foreach: bool | None = None,
) -> tuple[dict, int]:
    """Optimize oracle-memory likelihood with exclusive conditional ownership."""
    _freeze_backbone(handle)
    torch.manual_seed(seed)
    parameters = _configure_variant(handle, variant, reset=True)
    if checkpoint.exists() and variant.name != "fixed":
        return _load_checkpoint(checkpoint, handle, variant)
    if variant.name == "fixed":
        return {
            "variant": variant.name,
            "seed": seed,
            "trainable_parameters": 0,
            "training_seconds": 0.0,
            "losses": [],
            "resumed_from_checkpoint": False,
        }, 0
    names = _active_parameter_names(handle)
    allowed = ("pra_residual_adapter", "pra_late_band_lora")
    if not names or any(not any(owner in name for owner in allowed) for name in names):
        raise RuntimeError(f"PRA adaptation leaked to the backbone: {names}")
    if variant.residual_width and not any("pra_residual_adapter" in name for name in names):
        raise RuntimeError("Residual parameters are absent from a residual variant.")
    if variant.lora_rank and not any("pra_late_band_lora" in name for name in names):
        raise RuntimeError("LoRA parameters are absent from a LoRA variant.")

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
        foreach=optimizer_foreach,
    )
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    losses = []
    handle.model.eval()
    started = time.perf_counter()
    for step in range(steps):
        record = records[order[step % len(order)]]
        _activate(handle, record, layers, "oracle")
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _compact_gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    _sync(device)
    report = {
        "variant": variant.name,
        "seed": seed,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": steps,
        "learning_rate": learning_rate,
        "optimizer_foreach": optimizer_foreach,
        "training_seconds": time.perf_counter() - started,
        "initial_loss_mean": statistics.fmean(losses[: min(len(losses), len(records))]),
        "final_loss_mean": statistics.fmean(losses[-min(len(losses), len(records)) :]),
        "losses": losses,
        "resumed_from_checkpoint": False,
    }
    return report, _save_checkpoint(checkpoint, handle, variant, seed, report)


@torch.no_grad()
def _score_prompt(
    handle,
    tokenizer,
    prompt_ids,
    prompt_mask,
    answer_ids,
    device,
    new_tokens,
    *,
    generate: bool = True,
):
    started = time.perf_counter()
    _, metrics = _compact_gold_scores(handle, prompt_ids, prompt_mask, answer_ids, device)
    _sync(device)
    teacher_seconds = time.perf_counter() - started
    if generate:
        prediction, generation_seconds = _generate(
            handle, tokenizer, prompt_ids, prompt_mask, device, new_tokens
        )
        generation_metrics = answer_metrics(
            prediction,
            tokenizer.decode(answer_ids[0], skip_special_tokens=True),
        )
    else:
        prediction = ""
        generation_seconds = 0.0
        generation_metrics = {"f1": None, "em": None}
    return {
        **metrics,
        **generation_metrics,
        "generated_answer": prediction,
        "teacher_forced_seconds": teacher_seconds,
        "generation_seconds": generation_seconds,
    }


@torch.no_grad()
def _context_controls(
    handle,
    tokenizer,
    records,
    layers,
    new_tokens,
    device,
    *,
    generate: bool = True,
    generate_full_context: bool = True,
):
    """Measure seed-independent no-context, direct-text, and feasible full-context controls."""
    controls = {}
    rows = []
    for record in records:
        example_id = record["example"]["id"]
        per_example = {}
        prompts = (
            ("none", record["prompt_ids"], record["prompt_mask"]),
            ("direct_text", record["direct_prompt_ids"], record["direct_prompt_mask"]),
        )
        if record["full_context_complete"]:
            prompts = (*prompts, ("full_context", record["full_prompt_ids"], record["full_prompt_mask"]))
        for condition, prompt_ids, prompt_mask in prompts:
            _activate(handle, record, layers, "none")
            metrics = _score_prompt(
                handle,
                tokenizer,
                prompt_ids,
                prompt_mask,
                record["answer_ids"],
                device,
                new_tokens,
                generate=(
                    generate
                    and (condition != "full_context" or generate_full_context)
                ),
            )
            per_example[condition] = metrics
            rows.append(
                {
                    "seed": None,
                    "variant": "context_baseline",
                    "dataset": record["example"]["dataset"],
                    "example_id": example_id,
                    "condition": condition,
                    "answer": record["example"]["answer"],
                    "prompt_tokens": int(prompt_ids.shape[1]),
                    "full_context_complete": record["full_context_complete"],
                    **metrics,
                }
            )
        controls[example_id] = per_example
    return controls, rows


def _entry_economics(record, layers, route_layer):
    entry = record["entry"]
    route_chunks = entry.layer_memory[route_layer].chunks
    routing_bytes = sum(int(chunk.metadata.get("routing_gist_bytes", 0)) for chunk in route_chunks)
    detail_bytes = sum(
        int(chunk.metadata.get("detail_kv_bytes", 0))
        for layer in layers
        for chunk in entry.layer_memory[layer].chunks
    )
    return {
        "candidate_chunks": len(route_chunks),
        "source_tokens": int(entry.metadata["source_tokens"]),
        "routing_index_bytes": routing_bytes,
        "cpu_detail_kv_bytes": detail_bytes,
        "routing_index_over_detail_kv": routing_bytes / max(detail_bytes, 1),
    }


def _valid_ratio(
    numerator: float | None,
    denominator: float | None,
    epsilon: float,
) -> float | None:
    """Return benefit recovery only for positive, non-negligible matched controls."""
    if (
        numerator is None
        or denominator is None
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= epsilon
    ):
        return None
    return numerator / denominator


@torch.no_grad()
def _evaluate_memory(
    handle,
    tokenizer,
    records,
    controls,
    layers,
    route_layer,
    variant: Variant,
    seed,
    new_tokens,
    device,
    base_parameters,
    router_parameters,
    checkpoint_bytes,
    recovery_epsilon,
    generate=True,
    conditions=("oracle", "routed"),
):
    """Evaluate oracle and routed memory with matched economics and recovery ratios."""
    rows = []
    parameter_count = sum(parameter.numel() for parameter in handle.memory_use_parameters())
    for record in records:
        baseline = controls[record["example"]["id"]]["none"]
        economics = _entry_economics(record, layers, route_layer)
        for condition in conditions:
            if condition not in {"oracle", "routed"}:
                raise ValueError(f"Unsupported memory evaluation condition: {condition}")
            _activate(handle, record, layers, condition)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            metrics = _score_prompt(
                handle,
                tokenizer,
                record["prompt_ids"],
                record["prompt_mask"],
                record["answer_ids"],
                device,
                new_tokens,
                generate=generate,
            )
            diagnostics = handle.diagnostics_by_layer()
            selected = handle.adapters[route_layer].last_selected_chunks[0]
            selected_tokens = sum(hit.selected_token_count for hit in selected)
            transfer_bytes = sum(
                int(values.get("retrieved_kv_transfer_bytes", 0))
                for layer, values in diagnostics.items()
                if layer in layers
            )
            active_tokens = sum(
                int(values.get("retrieved_physical_kv_tokens", 0))
                for layer, values in diagnostics.items()
                if layer in layers
            )
            sequence_delta = metrics["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
            mean_delta = metrics["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
            direct = controls[record["example"]["id"]]["direct_text"]
            direct_delta = direct["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
            full = controls[record["example"]["id"]].get("full_context")
            full_delta = (
                full["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
                if full is not None
                else None
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant.name,
                    "residual_width": variant.residual_width,
                    "lora_rank": variant.lora_rank,
                    "dataset": record["example"]["dataset"],
                    "example_id": record["example"]["id"],
                    "condition": condition,
                    "answer": record["example"]["answer"],
                    "pra_layers": len(layers),
                    "memory_use_parameters": parameter_count,
                    "router_parameters": router_parameters,
                    "total_pra_parameters": router_parameters + parameter_count,
                    "memory_use_parameter_percent": 100.0 * parameter_count / base_parameters,
                    "total_pra_parameter_percent": 100.0 * (router_parameters + parameter_count) / base_parameters,
                    "checkpoint_bytes": checkpoint_bytes,
                    "gold_sequence_logprob_delta_vs_none": sequence_delta,
                    "gold_mean_logprob_delta_vs_none": mean_delta,
                    "gold_first_token_rank_delta_vs_none": metrics["gold_first_token_rank"] - baseline["gold_first_token_rank"],
                    "gold_first_token_margin_delta_vs_none": metrics["gold_first_token_margin"] - baseline["gold_first_token_margin"],
                    "direct_sequence_benefit": direct_delta,
                    "full_sequence_benefit": full_delta,
                    "rho_direct": _valid_ratio(sequence_delta, direct_delta, recovery_epsilon),
                    "rho_full": _valid_ratio(sequence_delta, full_delta, recovery_epsilon),
                    "selected_chunks": len(selected),
                    "selected_chunk_fraction": len(selected) / max(economics["candidate_chunks"], 1),
                    "materialized_native_kv_tokens": selected_tokens,
                    "materialized_native_kv_fraction": selected_tokens / max(economics["source_tokens"], 1),
                    "active_kv_tokens_across_layers": active_tokens,
                    "gpu_materialized_kv_bytes": transfer_bytes,
                    "kv_transfer_bytes": transfer_bytes,
                    "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                    "pra_off_exact": True,
                    **economics,
                    **metrics,
                }
            )
    return rows


@torch.no_grad()
def _verify_off_exact(handle, tokenizer, record, layers, reference, device, new_tokens):
    """Require logits and greedy generation to equal the frozen no-PRA reference."""
    _activate(handle, record, layers, "none")
    _, metrics = _compact_gold_scores(
        handle,
        record["prompt_ids"],
        record["prompt_mask"],
        record["answer_ids"],
        device,
    )
    prediction, _ = _generate(
        handle,
        tokenizer,
        record["prompt_ids"],
        record["prompt_mask"],
        device,
        new_tokens,
    )
    return all(
        metrics[key] == reference["metrics"][key] for key in metrics
    ) and prediction == reference["generation"]


def _seed_aggregates(rows):
    metrics = (
        "gold_sequence_logprob",
        "gold_mean_token_logprob",
        "gold_sequence_logprob_delta_vs_none",
        "gold_mean_logprob_delta_vs_none",
        "gold_first_token_probability",
        "gold_first_token_rank",
        "gold_first_token_margin",
        "gold_first_token_rank_delta_vs_none",
        "gold_first_token_margin_delta_vs_none",
        "direct_sequence_benefit",
        "full_sequence_benefit",
        "f1",
        "em",
        "selected_chunk_fraction",
        "materialized_native_kv_tokens",
        "materialized_native_kv_fraction",
        "active_kv_tokens_across_layers",
        "routing_index_bytes",
        "routing_index_over_detail_kv",
        "gpu_materialized_kv_bytes",
        "cpu_detail_kv_bytes",
        "kv_transfer_bytes",
        "peak_gpu_bytes",
        "teacher_forced_seconds",
        "generation_seconds",
        "rho_direct",
        "rho_full",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["variant"], row["dataset"], row["condition"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = {
            "seed": key[0],
            "variant": key[1],
            "dataset": key[2],
            "condition": key[3],
            "examples": len(values),
            "memory_use_parameters": values[0]["memory_use_parameters"],
            "total_pra_parameters": values[0]["total_pra_parameters"],
            "memory_use_parameter_percent": values[0]["memory_use_parameter_percent"],
            "total_pra_parameter_percent": values[0]["total_pra_parameter_percent"],
            "checkpoint_bytes": values[0]["checkpoint_bytes"],
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            result[metric] = statistics.fmean(samples) if samples else None
            result[f"{metric}_valid_examples"] = len(samples)
        output.append(result)
    return output


def _aggregates(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["variant"], row["dataset"], row["condition"])].append(row)
    output = []
    identity = {
        "memory_use_parameters",
        "total_pra_parameters",
        "memory_use_parameter_percent",
        "total_pra_parameter_percent",
        "checkpoint_bytes",
        "examples",
    }
    for key, values in sorted(grouped.items()):
        result = {
            "variant": key[0],
            "dataset": key[1],
            "condition": key[2],
            "seeds": len(values),
            "examples_per_seed": values[0]["examples"],
        }
        numeric_keys = sorted(
            {
                name
                for row in values
                for name, value in row.items()
                if name not in {"seed", "variant", "dataset", "condition"}
                and not name.endswith("_valid_examples")
                and isinstance(value, (int, float))
            }
        )
        for metric in numeric_keys:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if not samples:
                result[f"{metric}_mean"] = None
                result[f"{metric}_std"] = None
                result[f"{metric}_ci95"] = None
                continue
            mean = statistics.fmean(samples)
            std = statistics.stdev(samples) if len(samples) > 1 else 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = 2.776 * std / math.sqrt(len(samples)) if len(samples) == 5 else None
        sequence_gain = result.get("gold_sequence_logprob_delta_vs_none_mean")
        direct_benefit = result.get("direct_sequence_benefit_mean")
        full_benefit = result.get("full_sequence_benefit_mean")
        result["rho_direct_cohort"] = _valid_ratio(
            sequence_gain, direct_benefit, 0.05
        )
        result["rho_full_cohort"] = _valid_ratio(
            sequence_gain, full_benefit, 0.05
        )
        output.append(result)
    return output


def _choose_best(validation_seed_rows, prefix: str) -> str:
    """Select by mean HotpotQA oracle sequence gain, never by test identities."""
    candidates = defaultdict(list)
    for row in validation_seed_rows:
        if (
            row["dataset"] == "hotpotqa"
            and row["condition"] == "oracle"
            and row["variant"].startswith(prefix)
        ):
            candidates[row["variant"]].append(row["gold_sequence_logprob_delta_vs_none"])
    if not candidates:
        raise ValueError(f"No validation candidates for prefix {prefix!r}.")
    return max(candidates, key=lambda name: statistics.fmean(candidates[name]))


def _paired_effects(seed_rows, combo_name, residual_name, lora_name):
    """Report seed-paired combo differences against each selected individual."""
    lookup = {
        (row["seed"], row["variant"], row["dataset"], row["condition"]): row
        for row in seed_rows
    }
    output = []
    seeds = sorted(
        {
            row["seed"]
            for row in seed_rows
            if row["variant"] == combo_name
        }
    )
    for dataset in ("hotpotqa", "qasper"):
        for condition in ("oracle", "routed"):
            for comparator in (residual_name, lora_name):
                differences = []
                for seed in seeds:
                    combo = lookup[(seed, combo_name, dataset, condition)]
                    other = lookup[(seed, comparator, dataset, condition)]
                    differences.append(
                        combo["gold_sequence_logprob_delta_vs_none"]
                        - other["gold_sequence_logprob_delta_vs_none"]
                    )
                mean = statistics.fmean(differences)
                std = statistics.stdev(differences) if len(differences) > 1 else 0.0
                output.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "combo": combo_name,
                        "comparator": comparator,
                        "paired_differences": differences,
                        "mean": mean,
                        "std": std,
                        "ci95": (
                            2.776 * std / math.sqrt(5)
                            if len(differences) == 5
                            else None
                        ),
                        "same_direction": all(value > 0 for value in differences)
                        or all(value < 0 for value in differences),
                    }
                )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _plot(aggregates, output_dir: Path) -> None:
    """Plot benefit recovery against active K/V and trainable adaptation size."""
    labels = {
        "fixed": "Frozen",
        "residual_16": "Residual 16",
        "residual_32": "Residual 32",
        "residual_64": "Residual 64",
        "lora_o_r4": "LoRA r4",
        "lora_o_r8": "LoRA r8",
    }
    short_labels = {
        "fixed": "F",
        "residual_16": "R16",
        "residual_32": "R32",
        "residual_64": "R64",
        "lora_o_r4": "L4",
        "lora_o_r8": "L8",
    }
    rows = [
        row for row in aggregates
        if row["dataset"] == "hotpotqa" and row["condition"] in {"oracle", "routed"}
    ]
    for row in rows:
        labels.setdefault(row["variant"], "Residual + LoRA")
        short_labels.setdefault(row["variant"], "R16+L4")
    colors = {"oracle": "#245A8D", "routed": "#A34832"}
    label_offsets = {
        "oracle": {
            "fixed": (6, 12),
            "residual_16": (6, 7),
            "residual_32": (6, -3),
            "residual_64": (6, -10),
            "lora_o_r4": (6, 8),
            "lora_o_r8": (6, -15),
            "combo_residual_16_lora_r4": (6, -1),
        },
        "routed": {
            "fixed": (6, 11),
            "residual_16": (6, -9),
            "residual_32": (6, 5),
            "residual_64": (6, 8),
            "lora_o_r4": (6, -2),
            "lora_o_r8": (6, 5),
            "combo_residual_16_lora_r4": (6, -8),
        },
    }
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["oracle"],
               markeredgecolor=colors["oracle"], label="Oracle selection"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["routed"],
               markeredgecolor=colors["routed"], label="Learned routing"),
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for row in rows:
        if row.get("rho_direct_cohort") is None:
            continue
        axis.scatter(
            100 * row["materialized_native_kv_fraction_mean"],
            100 * row["rho_direct_cohort"],
            color=colors[row["condition"]],
            marker="o" if row["condition"] == "oracle" else "s",
            s=45,
        )
        if row["variant"] in {"fixed", "residual_16", "lora_o_r4"} or row["variant"].startswith("combo_"):
            axis.annotate(
                short_labels[row["variant"]],
                (100 * row["materialized_native_kv_fraction_mean"], 100 * row["rho_direct_cohort"]),
                xytext=label_offsets[row["condition"]][row["variant"]],
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(100, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Materialized native-K/V tokens (% of source)")
    axis.set_ylabel("Direct-evidence benefit recovered (%)")
    axis.grid(alpha=0.25)
    axis.margins(x=0.10, y=0.08)
    axis.legend(handles=legend, loc="center left", frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"recovery_vs_materialized_kv.{suffix}", dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    for row in rows:
        if row.get("rho_direct_cohort") is None:
            continue
        axis.scatter(
            row["memory_use_parameter_percent_mean"],
            100 * row["rho_direct_cohort"],
            color=colors[row["condition"]],
            marker="o" if row["condition"] == "oracle" else "s",
            s=45,
        )
        axis.annotate(
            labels[row["variant"]],
            (row["memory_use_parameter_percent_mean"], 100 * row["rho_direct_cohort"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("PRA-specific memory-use parameters (% of base model)")
    axis.set_ylabel("Direct-evidence benefit recovered (%)")
    axis.grid(alpha=0.25)
    axis.margins(x=0.08, y=0.08)
    axis.legend(handles=legend, loc="lower right", frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"recovery_vs_adaptation_size.{suffix}", dpi=190)
    plt.close(figure)


def _prepare_records(handle, tokenizer, examples, layers, args, *, controls: bool):
    records = [
        _prepare_example(handle, tokenizer, example, layers, args.prompt_tokens, handle.device)
        for example in examples
    ]
    if controls:
        records = [
            _add_context_controls(record, tokenizer, args.direct_text_tokens, args.full_context_tokens)
            for record in records
        ]
    return records


def _load_examples(args, count, offset):
    return load_split_examples(args.cache_dir, count, offset, args.data_seed)


def last_band_layers(layer_count: int, width: int = 14) -> tuple[int, ...]:
    """Return the final contiguous PRA compatibility band."""
    if layer_count < width or width <= 0:
        raise ValueError(f"Cannot select a last-{width} band from {layer_count} layers.")
    return tuple(range(layer_count - width, layer_count))


def run(args):
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = load_hf_routing_projection(args.checkpoint, device=device)
    layer_count = int(model.config.num_hidden_layers)
    layers = last_band_layers(layer_count)
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
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    route_layer = layers[-1]
    base_parameters = sum(
        parameter.numel()
        for name, parameter in handle.model.named_parameters()
        if not name.startswith("pra_")
    )
    router_parameters = int(projection.parameter_count)
    train_examples = [
        example for example in _load_examples(args, args.train_examples, 0)
        if example["dataset"] == "hotpotqa"
    ]
    validation_examples = _load_examples(
        args, args.validation_examples, args.validation_offset
    )
    test_examples = _load_examples(args, args.test_examples, args.test_offset)
    identities = {
        "train": [row["id"] for row in train_examples],
        "validation": [row["id"] for row in validation_examples],
        "test": [row["id"] for row in test_examples],
    }
    if set(identities["train"]) & set(identities["validation"]):
        raise AssertionError("Train and validation identities overlap.")
    if set(identities["train"]) & set(identities["test"]):
        raise AssertionError("Train and test identities overlap.")
    if set(identities["validation"]) & set(identities["test"]):
        raise AssertionError("Validation and test identities overlap.")

    print("preparing last-14 training and validation references", flush=True)
    train_records = _prepare_records(
        handle, tokenizer, train_examples, layers, args, controls=False
    )
    validation_records = _prepare_records(
        handle, tokenizer, validation_examples, layers, args, controls=True
    )
    _configure_variant(handle, Variant("fixed"), reset=True)
    validation_controls, validation_control_rows = _context_controls(
        handle, tokenizer, validation_records, layers, args.new_tokens, device
    )
    reference_record = validation_records[0]
    off_reference = {
        "metrics": validation_controls[reference_record["example"]["id"]]["none"],
        "generation": validation_controls[reference_record["example"]["id"]]["none"]["generated_answer"],
    }

    training_reports = []
    validation_rows = []
    checkpoint_dir = args.output_dir / "checkpoints"
    for seed in args.seeds:
        for name in args.individual_variants:
            variant = variant_from_name(name)
            report, checkpoint_bytes = _train_variant(
                handle,
                train_records,
                layers,
                variant,
                seed,
                args.steps,
                args.learning_rate,
                device,
                checkpoint_dir / f"{variant.name}_seed{seed}.pt",
            )
            report["checkpoint_bytes"] = checkpoint_bytes
            report["pra_off_exact"] = _verify_off_exact(
                handle,
                tokenizer,
                reference_record,
                layers,
                off_reference,
                device,
                args.new_tokens,
            )
            if not report["pra_off_exact"]:
                raise AssertionError(f"PRA-off exactness failed for {variant.name}, seed {seed}.")
            training_reports.append(report)
            validation_rows.extend(
                _evaluate_memory(
                    handle,
                    tokenizer,
                    validation_records,
                    validation_controls,
                    layers,
                    route_layer,
                    variant,
                    seed,
                    args.new_tokens,
                    device,
                    base_parameters,
                    router_parameters,
                    checkpoint_bytes,
                    args.recovery_epsilon,
                )
            )
            print(f"validation seed={seed} variant={variant.name}", flush=True)
    validation_seed_rows = _seed_aggregates(validation_rows)
    best_residual = _choose_best(validation_seed_rows, "residual_")
    best_lora = _choose_best(validation_seed_rows, "lora_o_r")
    residual = variant_from_name(best_residual)
    lora = variant_from_name(best_lora)
    combo = Variant(
        f"combo_residual_{residual.residual_width}_lora_r{lora.lora_rank}",
        residual_width=residual.residual_width,
        lora_rank=lora.lora_rank,
    )
    for seed in args.seeds:
        report, checkpoint_bytes = _train_variant(
            handle,
            train_records,
            layers,
            combo,
            seed,
            args.steps,
            args.learning_rate,
            device,
            checkpoint_dir / f"{combo.name}_seed{seed}.pt",
        )
        report["checkpoint_bytes"] = checkpoint_bytes
        report["pra_off_exact"] = _verify_off_exact(
            handle,
            tokenizer,
            reference_record,
            layers,
            off_reference,
            device,
            args.new_tokens,
        )
        if not report["pra_off_exact"]:
            raise AssertionError(f"PRA-off exactness failed for {combo.name}, seed {seed}.")
        training_reports.append(report)
        validation_rows.extend(
            _evaluate_memory(
                handle,
                tokenizer,
                validation_records,
                validation_controls,
                layers,
                route_layer,
                combo,
                seed,
                args.new_tokens,
                device,
                base_parameters,
                router_parameters,
                checkpoint_bytes,
                args.recovery_epsilon,
            )
        )
        print(f"validation seed={seed} variant={combo.name}", flush=True)

    del validation_records, train_records
    handle.cache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("preparing larger identity-disjoint test references", flush=True)
    test_records = _prepare_records(
        handle, tokenizer, test_examples, layers, args, controls=True
    )
    _configure_variant(handle, Variant("fixed"), reset=True)
    test_controls, test_control_rows = _context_controls(
        handle, tokenizer, test_records, layers, args.new_tokens, device
    )
    test_rows = []
    test_variants = (*args.individual_variants, combo.name)
    for seed in args.seeds:
        for name in test_variants:
            variant = variant_from_name(name)
            _freeze_backbone(handle)
            _configure_variant(handle, variant, reset=True)
            checkpoint = checkpoint_dir / f"{variant.name}_seed{seed}.pt"
            checkpoint_bytes = 0
            if variant.name != "fixed":
                _, checkpoint_bytes = _load_checkpoint(checkpoint, handle, variant)
            test_rows.extend(
                _evaluate_memory(
                    handle,
                    tokenizer,
                    test_records,
                    test_controls,
                    layers,
                    route_layer,
                    variant,
                    seed,
                    args.new_tokens,
                    device,
                    base_parameters,
                    router_parameters,
                    checkpoint_bytes,
                    args.recovery_epsilon,
                )
            )
            print(f"test seed={seed} variant={variant.name}", flush=True)

    seed_rows = _seed_aggregates(test_rows)
    aggregates = _aggregates(seed_rows)
    validation_seed_rows = _seed_aggregates(validation_rows)
    validation_aggregates = _aggregates(validation_seed_rows)
    paired = _paired_effects(seed_rows, combo.name, best_residual, best_lora)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "last-14 PRA memory-use convergence with identity-disjoint selection and test",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_checkpoint": str(args.checkpoint),
        "routing_parameters": router_parameters,
        "base_model_parameters": base_parameters,
        "layer_ids": list(layers),
        "verified_depth_artifact": "multilayer_pra/oracle_layer_depth.json",
        "data_seed": args.data_seed,
        "optimization_seeds": list(args.seeds),
        "identities": identities,
        "identity_disjoint": True,
        "train_hotpotqa_examples": len(train_examples),
        "validation_examples_per_dataset": args.validation_examples,
        "test_examples_per_dataset": args.test_examples,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "selected_best_residual": best_residual,
        "selected_best_lora": best_lora,
        "combination": combo.name,
        "selection_rule": "maximum five-seed mean HotpotQA validation oracle gold-sequence log-probability gain",
        "recovery_denominator_rule": f"positive gold-sequence benefit greater than {args.recovery_epsilon}",
        "full_context_token_limit": args.full_context_tokens,
        "full_context_eligible_test_examples": sum(record["full_context_complete"] for record in test_records),
        "training": training_reports,
        "validation_control_rows": validation_control_rows,
        "validation_rows": validation_rows,
        "validation_seed_aggregates": validation_seed_rows,
        "validation_aggregates": validation_aggregates,
        "test_control_rows": test_control_rows,
        "test_rows": test_rows,
        "seed_aggregates": seed_rows,
        "aggregates": aggregates,
        "paired_effects": paired,
        "pra_off_all_exact": all(report["pra_off_exact"] for report in training_reports),
        "native_limit_violations": handle.native_limit_violations,
        "scope_note": "HotpotQA remains a one-shot multi-hop stress and causal-use probe; iterative evidence gathering is deferred to Paper 2.5.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "last14_combo.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "last14_combo_test.csv", test_rows)
    _write_csv(args.output_dir / "last14_combo_seed_aggregate.csv", seed_rows)
    _write_csv(args.output_dir / "last14_combo_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "last14_combo_controls.csv", test_control_rows)
    _write_csv(args.output_dir / "last14_combo_paired.csv", paired)
    _plot(aggregates, args.output_dir)
    return artifact


def parse_args():
    results = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int, default=12)
    parser.add_argument("--validation-examples", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=12)
    parser.add_argument("--test-examples", type=int, default=8)
    parser.add_argument("--test-offset", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--recovery-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--individual-variants",
        nargs="+",
        choices=INDIVIDUAL_VARIANTS,
        default=list(INDIVIDUAL_VARIANTS),
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=results / "routing" / "learned_adapter" / "checkpoints" / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=results / "last14_combo")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    args.individual_variants = tuple(args.individual_variants)
    return args


if __name__ == "__main__":
    options = parse_args()
    if options.plot_only:
        existing = json.loads(
            (options.output_dir / "last14_combo.json").read_text(encoding="utf-8")
        )
        _plot(existing["aggregates"], options.output_dir)
        result = existing
    else:
        result = run(options)
    print(
        json.dumps(
            {
                "selected_best_residual": result["selected_best_residual"],
                "selected_best_lora": result["selected_best_lora"],
                "combination": result["combination"],
                "pra_off_all_exact": result["pra_off_all_exact"],
                "native_limit_violations": result["native_limit_violations"],
            },
            indent=2,
        )
    )
