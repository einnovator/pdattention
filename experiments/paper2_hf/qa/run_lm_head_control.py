"""Measure global LM-head adaptation against PRA-conditional calibration."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import load_wikitext_splits, wikitext_documents
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_late_band_lora import (
    _add_direct_prompt,
    _prediction_logits,
)
from experiments.paper2_hf.qa.run_memory_gate import (
    _activate,
    _gold_scores,
    _prepare_example,
    _sync,
    _write_csv,
)
from experiments.paper2_hf.qa.run_oracle_memory_use import _generate
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import MEMORY_GATE_FIXED, PRAHFConfig, inject_pra, load_hf_routing_projection


HEAD_LORA_RANK = 8
VARIANTS = ("fixed", "lm_head_lora_r8", "lm_head", "final_norm_lm_head")


class LMHeadLoRA(nn.Module):
    """Add a global low-rank delta to a frozen decoder output head.

    The frozen head remains tied to the input embedding table. Only ``down`` and
    ``up`` are registered here, so optimization cannot mutate token embeddings.
    """

    def __init__(self, base_head: nn.Linear, rank: int, *, alpha: float) -> None:
        super().__init__()
        self.__dict__["base_head"] = base_head
        self.input_width = int(base_head.in_features)
        self.output_width = int(base_head.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(self.input_width, self.rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(self.rank, self.output_width, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return frozen logits plus an FP32 ``[B,T,V]`` low-rank delta."""
        base_logits = self.base_head(hidden_states).float()
        delta = self.up(self.down(hidden_states.float()))
        return base_logits + self.scaling * delta


def _clone_output_head(base_head: nn.Linear) -> nn.Linear:
    """Create an untied trainable head with the native dtype and exact weights."""
    cloned = nn.Linear(
        base_head.in_features,
        base_head.out_features,
        bias=False,
        device=base_head.weight.device,
        dtype=base_head.weight.dtype,
    )
    with torch.no_grad():
        cloned.weight.copy_(base_head.weight)
    return cloned


def _configure_variant(
    handle,
    original_head: nn.Linear,
    original_norm_weight: torch.Tensor,
    variant: str,
):
    """Restore the frozen decoder and activate exactly one global readout control."""
    handle.model.set_output_embeddings(original_head)
    with torch.no_grad():
        handle.model.model.norm.weight.copy_(original_norm_weight)
    for parameter in handle.model.parameters():
        parameter.requires_grad_(False)
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    handle.configure_residual_adapter(0)
    handle.configure_late_band_lora(0)

    if variant == "fixed":
        return []
    if variant == "lm_head_lora_r8":
        adapted = LMHeadLoRA(original_head, HEAD_LORA_RANK, alpha=HEAD_LORA_RANK)
        adapted.to(original_head.weight.device)
        handle.model.set_output_embeddings(adapted)
        return list(adapted.parameters())
    if variant in {"lm_head", "final_norm_lm_head"}:
        adapted = _clone_output_head(original_head)
        handle.model.set_output_embeddings(adapted)
        parameters = list(adapted.parameters())
        if variant == "final_norm_lm_head":
            handle.model.model.norm.weight.requires_grad_(True)
            parameters.append(handle.model.model.norm.weight)
        return parameters
    raise ValueError(f"Unsupported LM-head control: {variant}")


def _load_wikitext_blocks(tokenizer, cache_dir: Path, count: int, tokens: int):
    """Build deterministic ordinary-language blocks from cached WikiText-2 validation text."""
    split = load_wikitext_splits(cache_dir=cache_dir)["validation"]
    texts = wikitext_documents(split, max_documents=256)
    encoded = tokenizer(
        f"\n\n{tokenizer.eos_token}\n\n".join(texts),
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    needed = count * (tokens + 1)
    if len(encoded) < needed:
        raise RuntimeError(f"WikiText validation supplied {len(encoded)} tokens; need {needed}.")
    return [
        torch.tensor(encoded[offset : offset + tokens + 1], dtype=torch.long).unsqueeze(0)
        for offset in range(0, needed, tokens + 1)
    ]


@torch.no_grad()
def _evaluate_language_model(handle, blocks, device):
    """Return token-weighted no-PRA WikiText loss and perplexity."""
    handle.set_memory_enabled(False)
    loss_sum = 0.0
    token_count = 0
    started = time.perf_counter()
    for block in blocks:
        ids = block.to(device)
        logits = handle.model(input_ids=ids, use_cache=False).logits.float()
        targets = ids[:, 1:]
        loss_sum += float(
            F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).cpu()
        )
        token_count += int(targets.numel())
    _sync(device)
    loss = loss_sum / token_count
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "tokens": token_count,
        "seconds": time.perf_counter() - started,
    }


def _train_variant(
    handle,
    original_head,
    original_norm_weight,
    records,
    layers,
    variant,
    seed,
    steps,
    learning_rate,
    no_pra_weight,
    device,
):
    """Optimize one readout control on oracle-memory QA plus a no-PRA constraint."""
    parameters = _configure_variant(
        handle, original_head, original_norm_weight, variant
    )
    if not parameters:
        return {
            "variant": variant,
            "trainable_parameters": 0,
            "losses": [],
            "memory_losses": [],
            "no_pra_control_losses": [],
            "no_pra_control_all_exact": True,
            "training_seconds": 0.0,
            "trainable_memory_bytes": 0,
            "optimizer": "none",
        }
    allowed = ("lm_head", "model.norm.weight")
    trainable_names = [
        name for name, parameter in handle.model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not any(owner in name for owner in allowed) for name in trainable_names
    ):
        raise RuntimeError(f"Readout adaptation leaked to the backbone: {trainable_names}")

    if variant in {"lm_head", "final_norm_lm_head"}:
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.0)
        optimizer_name = "SGD"
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=0.0,
            foreach=False,
        )
        optimizer_name = "AdamW"
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    losses = []
    memory_losses = []
    control_losses = []
    control_exact = []
    handle.model.eval()
    started = time.perf_counter()
    for step in range(steps):
        record = records[order[step % len(order)]]
        _activate(handle, record, layers, "oracle")
        optimizer.zero_grad(set_to_none=True)
        memory_loss, _ = _gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        if not torch.isfinite(memory_loss):
            raise FloatingPointError(
                f"Non-finite {variant} memory loss at step {step}: {memory_loss}"
            )
        memory_loss.backward()

        _activate(handle, record, layers, "none")
        no_pra_logits = _prediction_logits(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        target = record["no_pra_control_logits"].to(device)
        control_loss = F.kl_div(
            F.log_softmax(no_pra_logits, dim=-1),
            F.softmax(target, dim=-1),
            reduction="batchmean",
        ) / max(int(no_pra_logits.shape[1]), 1)
        weighted_control = float(no_pra_weight) * control_loss
        if not torch.isfinite(weighted_control):
            raise FloatingPointError(
                f"Non-finite {variant} control loss at step {step}: {control_loss}"
            )
        weighted_control.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
        optimizer.step()

        losses.append(float(memory_loss.detach().cpu() + weighted_control.detach().cpu()))
        memory_losses.append(float(memory_loss.detach().cpu()))
        control_losses.append(float(control_loss.detach().cpu()))
        control_exact.append(torch.equal(no_pra_logits.detach(), target))
    _sync(device)
    return {
        "variant": variant,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_memory_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "steps": steps,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "no_pra_regularization_weight": float(no_pra_weight),
        "no_pra_control_loss": "KL distillation over answer-position vocabulary distributions",
        "training_seconds": time.perf_counter() - started,
        "losses": losses,
        "memory_losses": memory_losses,
        "no_pra_control_losses": control_losses,
        "initial_memory_loss_mean": statistics.fmean(
            memory_losses[: min(len(memory_losses), len(records))]
        ),
        "final_memory_loss_mean": statistics.fmean(
            memory_losses[-min(len(memory_losses), len(records)) :]
        ),
        "no_pra_control_loss_max": max(control_losses, default=0.0),
        "no_pra_control_all_exact": all(control_exact),
    }


@torch.no_grad()
def _evaluate_variant(
    handle,
    tokenizer,
    records,
    layers,
    variant,
    seed,
    new_tokens,
    device,
    frozen_no_pra,
    trainable_parameter_count,
    trainable_memory_bytes,
):
    """Measure memory utility and global no-PRA drift for one readout."""
    rows = []
    current_no_pra = {}
    for record in records:
        _activate(handle, record, layers, "none")
        _, metrics = _gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        current_no_pra[record["example"]["id"]] = metrics

    for record in records:
        example_id = record["example"]["id"]
        baseline = current_no_pra[example_id]
        frozen = frozen_no_pra[example_id]
        for condition in ("none", "oracle", "routed", "direct_text"):
            if condition == "direct_text":
                _activate(handle, record, layers, "none")
                prompt_ids = record["direct_prompt_ids"]
                prompt_mask = record["direct_prompt_mask"]
            else:
                _activate(handle, record, layers, condition)
                prompt_ids = record["prompt_ids"]
                prompt_mask = record["prompt_mask"]
            started = time.perf_counter()
            _, metrics = _gold_scores(
                handle, prompt_ids, prompt_mask, record["answer_ids"], device
            )
            _sync(device)
            teacher_forced_seconds = time.perf_counter() - started
            prediction, generation_seconds = _generate(
                handle,
                tokenizer,
                prompt_ids,
                prompt_mask,
                device,
                new_tokens,
            )
            diagnostics = handle.diagnostics_by_layer()
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "dataset": record["example"]["dataset"],
                    "example_id": example_id,
                    "condition": condition,
                    "answer": record["example"]["answer"],
                    "generated_answer": prediction,
                    "trainable_parameter_count": trainable_parameter_count,
                    "trainable_memory_bytes": trainable_memory_bytes,
                    "gold_mean_logprob_delta_vs_none": (
                        metrics["gold_mean_token_logprob"]
                        - baseline["gold_mean_token_logprob"]
                    ),
                    "gold_first_token_rank_delta_vs_none": (
                        metrics["gold_first_token_rank"]
                        - baseline["gold_first_token_rank"]
                    ),
                    "gold_first_token_margin_delta_vs_none": (
                        metrics["gold_first_token_margin"]
                        - baseline["gold_first_token_margin"]
                    ),
                    "no_pra_gold_logprob_delta_vs_frozen": (
                        baseline["gold_mean_token_logprob"]
                        - frozen["gold_mean_token_logprob"]
                    ),
                    "no_pra_first_token_rank_delta_vs_frozen": (
                        baseline["gold_first_token_rank"]
                        - frozen["gold_first_token_rank"]
                    ),
                    "no_pra_first_token_margin_delta_vs_frozen": (
                        baseline["gold_first_token_margin"]
                        - frozen["gold_first_token_margin"]
                    ),
                    "pra_off_gold_metrics_exact": baseline == frozen,
                    "teacher_forced_seconds": teacher_forced_seconds,
                    "generation_seconds": generation_seconds,
                    "active_memory_tokens": sum(
                        int(values.get("retrieved_physical_kv_tokens", 0))
                        for layer, values in diagnostics.items()
                        if layer in layers
                    ),
                    **metrics,
                    **answer_metrics(prediction, record["example"]["answer"]),
                }
            )
    return rows


def _aggregate_language_rows(rows):
    """Aggregate WikiText measurements across optimization seeds."""
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        output.append(
            {
                "variant": variant,
                "seeds": len(selected),
                **{
                    f"{metric}_{suffix}": value
                    for metric in ("loss", "perplexity", "loss_delta_vs_frozen", "seconds")
                    for suffix, value in (
                        ("mean", statistics.fmean(float(row[metric]) for row in selected)),
                        (
                            "std",
                            statistics.pstdev(float(row[metric]) for row in selected)
                            if len(selected) > 1
                            else 0.0,
                        ),
                    )
                },
            }
        )
    return output


def _aggregate_control_rows(rows):
    """Aggregate QA rows, including the control-specific ordinary-path metrics."""
    metrics = (
        "gold_mean_token_logprob",
        "gold_mean_logprob_delta_vs_none",
        "gold_first_token_rank",
        "gold_first_token_rank_delta_vs_none",
        "gold_first_token_margin",
        "gold_first_token_margin_delta_vs_none",
        "no_pra_gold_logprob_delta_vs_frozen",
        "no_pra_first_token_rank_delta_vs_frozen",
        "no_pra_first_token_margin_delta_vs_frozen",
        "f1",
        "em",
        "answer_contained",
        "teacher_forced_seconds",
        "generation_seconds",
        "active_memory_tokens",
        "trainable_parameter_count",
        "trainable_parameter_percent",
        "trainable_memory_bytes",
        "trainable_memory_bytes_fp32",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["variant"], row["dataset"], row["condition"])].append(
            row
        )
    seed_rows = []
    for key, values in sorted(grouped.items()):
        seed_rows.append(
            {
                "seed": key[0],
                "variant": key[1],
                "dataset": key[2],
                "condition": key[3],
                "examples": len(values),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in values)
                    for metric in metrics
                },
            }
        )
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["variant"], row["dataset"], row["condition"])].append(row)
    aggregates = []
    for key, values in sorted(grouped.items()):
        record = {
            "variant": key[0],
            "dataset": key[1],
            "condition": key[2],
            "seeds": len(values),
            "examples_per_seed": values[0]["examples"],
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            record[f"{metric}_mean"] = statistics.fmean(samples)
            record[f"{metric}_std"] = statistics.pstdev(samples)
        aggregates.append(record)
    return seed_rows, aggregates


def _comparison_rows(aggregates, language_aggregates, result_root: Path):
    """Join this control to the matched gate, residual, and conditional-LoRA results."""
    gate = list(
        csv.DictReader(
            (result_root / "memory_residual_adapter" / "memory_gate_aggregate.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    lora = list(
        csv.DictReader(
            (result_root / "late_band_lora" / "late_band_lora_aggregate.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    sources = {
        "frozen": (gate, "fixed"),
        "memory_gate": (gate, "single"),
        "residual_32": (gate, "residual_32"),
        "late_o_lora_r8": (lora, "lora_o_r8"),
        "head_lora_r8": (aggregates, "lm_head_lora_r8"),
        "lm_head": (aggregates, "lm_head"),
        "norm_lm_head": (aggregates, "final_norm_lm_head"),
    }
    language = {row["variant"]: row for row in language_aggregates}
    available_variants = {row["variant"] for row in aggregates}
    rows = []
    for method, (source, variant) in sources.items():
        if source is aggregates and variant not in available_variants:
            continue
        for dataset in ("hotpotqa", "qasper"):
            routed = next(
                row
                for row in source
                if row["variant"] == variant
                and row["dataset"] == dataset
                and row["condition"] == "routed"
            )
            none = next(
                row
                for row in source
                if row["variant"] == variant
                and row["dataset"] == dataset
                and row["condition"] == "none"
            )
            conditional = method in {
                "frozen",
                "memory_gate",
                "residual_32",
                "late_o_lora_r8",
            }
            own_language = language.get(variant)
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "trainable_parameter_count": float(
                        routed["trainable_parameter_count_mean"]
                    ),
                    "routed_logprob_delta_mean": float(
                        routed["gold_mean_logprob_delta_vs_none_mean"]
                    ),
                    "routed_logprob_delta_std": float(
                        routed["gold_mean_logprob_delta_vs_none_std"]
                    ),
                    "routed_f1_mean": float(routed["f1_mean"]),
                    "routed_f1_std": float(routed["f1_std"]),
                    "no_pra_qa_logprob_delta_mean": (
                        0.0
                        if conditional
                        else float(none["no_pra_gold_logprob_delta_vs_frozen_mean"])
                    ),
                    "no_pra_qa_logprob_delta_std": (
                        0.0
                        if conditional
                        else float(none["no_pra_gold_logprob_delta_vs_frozen_std"])
                    ),
                    "wikitext_loss_delta_mean": (
                        0.0 if conditional else own_language["loss_delta_vs_frozen_mean"]
                    ),
                    "wikitext_loss_delta_std": (
                        0.0 if conditional else own_language["loss_delta_vs_frozen_std"]
                    ),
                }
            )
    return rows


def _plot(comparison, output_dir):
    """Plot memory gains beside ordinary-behavior regression."""
    preferred_methods = [
        "frozen",
        "memory_gate",
        "residual_32",
        "late_o_lora_r8",
        "head_lora_r8",
        "lm_head",
        "norm_lm_head",
    ]
    label_by_method = {
        "frozen": "Frozen",
        "memory_gate": "Gate",
        "residual_32": "Residual 32",
        "late_o_lora_r8": "Late O-LoRA",
        "head_lora_r8": "Head LoRA",
        "lm_head": "LM head",
        "norm_lm_head": "Norm+head",
    }
    methods = [
        method
        for method in preferred_methods
        if any(row["method"] == method for row in comparison)
    ]
    labels = [label_by_method[method] for method in methods]
    colors = {"hotpotqa": "#245A8D", "qasper": "#A34832"}
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)
    metrics = (
        ("routed_logprob_delta", "Routed gold $\\Delta\\log p$"),
        ("routed_f1", "Routed generated F1"),
        ("no_pra_qa_logprob_delta", "No-PRA QA $\\Delta\\log p$"),
    )
    width = 0.36
    for axis, (metric, title) in zip(axes.flat[:3], metrics):
        for offset, dataset in ((-width / 2, "hotpotqa"), (width / 2, "qasper")):
            selected = [
                next(
                    row
                    for row in comparison
                    if row["method"] == method and row["dataset"] == dataset
                )
                for method in methods
            ]
            axis.bar(
                [index + offset for index in range(len(methods))],
                [row[f"{metric}_mean"] for row in selected],
                yerr=[row[f"{metric}_std"] for row in selected],
                width=width,
                color=colors[dataset],
                label=dataset,
                capsize=2,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(range(len(methods)), labels, rotation=22, ha="right")
    language_axis = axes[1, 1]
    language_values = [
        next(row for row in comparison if row["method"] == method)[
            "wikitext_loss_delta_mean"
        ]
        for method in methods
    ]
    language_errors = [
        next(row for row in comparison if row["method"] == method)[
            "wikitext_loss_delta_std"
        ]
        for method in methods
    ]
    language_axis.bar(
        range(len(methods)), language_values, yerr=language_errors, color="#6B665E", capsize=2
    )
    language_axis.axhline(0.0, color="black", linewidth=0.7)
    language_axis.set_title("No-PRA WikiText-2 loss change")
    language_axis.grid(axis="y", alpha=0.22)
    language_axis.set_xticks(range(len(methods)), labels, rotation=22, ha="right")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside upper center", ncol=2, frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"lm_head_control.{suffix}", dpi=180)
    plt.close(figure)


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
    original_head = model.get_output_embeddings()
    original_norm_weight = model.model.norm.weight.detach().clone()
    tied_before_control = (
        original_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    )
    projection = load_hf_routing_projection(args.checkpoint, device=device)
    layer_count = int(model.config.num_hidden_layers)
    layers = tuple(range(layer_count - 4, layer_count))
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
            collect_detailed_timing=False,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    train_examples = [
        example
        for example in load_split_examples(
            args.cache_dir, args.train_examples, 0, args.data_seed
        )
        if example["dataset"] == "hotpotqa"
    ]
    heldout_examples = load_split_examples(
        args.cache_dir, args.heldout_examples, args.heldout_offset, args.data_seed
    )
    print("preparing frozen references, controls, and WikiText blocks", flush=True)
    train_records = [
        _prepare_example(handle, tokenizer, example, layers, args.prompt_tokens, device)
        for example in train_examples
    ]
    heldout_records = [
        _add_direct_prompt(
            _prepare_example(handle, tokenizer, example, layers, args.prompt_tokens, device),
            tokenizer,
            args.direct_text_tokens,
        )
        for example in heldout_examples
    ]
    wikitext_blocks = _load_wikitext_blocks(
        tokenizer, args.cache_dir, args.wikitext_blocks, args.wikitext_tokens
    )
    _configure_variant(handle, original_head, original_norm_weight, "fixed")
    with torch.no_grad():
        for record in train_records:
            _activate(handle, record, layers, "none")
            record["no_pra_control_logits"] = _prediction_logits(
                handle,
                record["prompt_ids"],
                record["prompt_mask"],
                record["answer_ids"],
                device,
            ).detach().cpu()
    frozen_no_pra = {}
    with torch.no_grad():
        for record in heldout_records:
            _activate(handle, record, layers, "none")
            _, metrics = _gold_scores(
                handle,
                record["prompt_ids"],
                record["prompt_mask"],
                record["answer_ids"],
                device,
            )
            frozen_no_pra[record["example"]["id"]] = metrics
    frozen_language = _evaluate_language_model(handle, wikitext_blocks, device)

    base_model_parameters = sum(
        parameter.numel()
        for name, parameter in handle.model.named_parameters()
        if not name.startswith("pra_memory_gate")
        and not name.startswith("pra_residual_adapter")
        and not name.startswith("pra_late_band_lora")
    )
    rows = []
    training = []
    language_rows = []
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            learning_rate = (
                args.head_lora_learning_rate
                if variant == "lm_head_lora_r8"
                else args.full_head_learning_rate
            )
            report = _train_variant(
                handle,
                original_head,
                original_norm_weight,
                train_records,
                layers,
                variant,
                seed,
                args.steps,
                learning_rate,
                args.no_pra_weight,
                device,
            )
            report["seed"] = seed
            training.append(report)
            rows.extend(
                _evaluate_variant(
                    handle,
                    tokenizer,
                    heldout_records,
                    layers,
                    variant,
                    seed,
                    args.new_tokens,
                    device,
                    frozen_no_pra,
                    report["trainable_parameters"],
                    report["trainable_memory_bytes"],
                )
            )
            language = _evaluate_language_model(handle, wikitext_blocks, device)
            language_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    **language,
                    "loss_delta_vs_frozen": language["loss"] - frozen_language["loss"],
                }
            )
            print(
                f"seed={seed} variant={variant} params={report['trainable_parameters']} "
                f"wiki_delta={language['loss'] - frozen_language['loss']:+.6f}",
                flush=True,
            )
            if variant != "fixed":
                handle.model.set_output_embeddings(original_head)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for row in rows:
        parameter_count = int(row["trainable_parameter_count"])
        row["trainable_parameter_percent"] = 100.0 * parameter_count / base_model_parameters
        row["trainable_memory_bytes_fp32"] = 4 * parameter_count
    seed_rows, aggregates = _aggregate_control_rows(rows)
    language_aggregates = _aggregate_language_rows(language_rows)
    result_root = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    comparison = _comparison_rows(aggregates, language_aggregates, result_root)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "global LM-head readout control after PRA-conditional calibration",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_checkpoint": str(args.checkpoint),
        "data_seed": args.data_seed,
        "optimization_seeds": args.seeds,
        "train_dataset": "hotpotqa",
        "train_examples": len(train_records),
        "heldout_examples_per_dataset": args.heldout_examples,
        "heldout_offset": args.heldout_offset,
        "layer_ids": layers,
        "steps": args.steps,
        "full_head_learning_rate": args.full_head_learning_rate,
        "head_lora_learning_rate": args.head_lora_learning_rate,
        "no_pra_regularization_weight": args.no_pra_weight,
        "variants": args.variants,
        "base_model_parameters": base_model_parameters,
        "native_head_was_tied_to_embeddings": tied_before_control,
        "full_head_control_explicitly_untied": True,
        "wikitext_protocol": {
            "dataset": "wikitext-2-raw-v1 validation",
            "blocks": args.wikitext_blocks,
            "tokens_per_block": args.wikitext_tokens,
            "frozen": frozen_language,
        },
        "training": training,
        "rows": rows,
        "seed_aggregates": seed_rows,
        "aggregates": aggregates,
        "language_rows": language_rows,
        "language_aggregates": language_aggregates,
        "comparison": comparison,
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lm_head_control.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "lm_head_control.csv", rows)
    _write_csv(args.output_dir / "lm_head_control_seed_aggregate.csv", seed_rows)
    _write_csv(args.output_dir / "lm_head_control_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "lm_head_control_language.csv", language_rows)
    _write_csv(
        args.output_dir / "lm_head_control_language_aggregate.csv", language_aggregates
    )
    _write_csv(args.output_dir / "lm_head_control_comparison.csv", comparison)
    _plot(comparison, args.output_dir)
    return artifact


def parse_args():
    result_root = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int, default=8)
    parser.add_argument("--heldout-examples", type=int, default=4)
    parser.add_argument("--heldout-offset", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--head-lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--full-head-learning-rate", type=float, default=5e-2)
    parser.add_argument("--no-pra-weight", type=float, default=1.0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--wikitext-blocks", type=int, default=8)
    parser.add_argument("--wikitext-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            result_root
            / "routing"
            / "learned_adapter"
            / "checkpoints"
            / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=result_root / "lm_head_control")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.plot_only:
        result = json.loads(
            (arguments.output_dir / "lm_head_control.json").read_text(encoding="utf-8")
        )
        _plot(result["comparison"], arguments.output_dir)
    else:
        result = run(arguments)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
