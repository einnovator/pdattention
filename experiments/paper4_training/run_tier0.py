"""Run the controlled Paper 4 adaptation-capacity ladder.

This runner deliberately fixes oracle memory identities.  It tests whether
increasing transformer plasticity improves consumption of correct native K/V;
router and materializer learning belong to later gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pra_torch.config import PRAConfig
from pra_torch.controlled_local_sa import (
    ControlledExample,
    ControlledTokenizer,
    collate_controlled,
    controlled_examples,
    last_valid_logits,
)
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra
from pra_torch.pra_aware_training import (
    AdaptationRegime,
    MemoryCondition,
    build_controlled_memory_batch,
    encode_differentiable_memory_kv,
    forward_with_differentiable_memory,
    install_adaptation_regime,
    parameter_summary,
    trainable_parameters,
)


SEEDS = (17, 29, 41, 53, 67)
PRA_REGIMES: tuple[AdaptationRegime, ...] = (
    "frozen",
    "consumer_lora",
    "interface_lora",
    "broad_lora",
    "full_weight",
    "native_scratch",
)
MEMORY_CONDITIONS: tuple[MemoryCondition, ...] = (
    "none",
    "matched_distractor",
    "evidence_only",
    "whole_parent",
)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch without changing deterministic kernel policy."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(
    tokenizer: ControlledTokenizer,
    *,
    window: int | None,
    variant: str,
    d_model: int,
    layers: int,
    pra_layers: tuple[int, ...],
    device: str,
) -> PRAConfig:
    """Construct matched RoPE architectures; only topology/window varies."""
    return PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=layers,
        d_ff=4 * d_model,
        max_seq_len=256,
        model_max_context_tokens=256,
        position_encoding="rope",
        self_attention_window=window,
        dropout=0.0,
        model_variant=variant,
        pra_layer_ids=pra_layers,
        memory_transport="native_kv",
        collect_per_head_metrics=True,
        device=device,
    )


def batches(count: int, batch_size: int, *, seed: int, epoch: int):
    """Yield deterministic shuffled index batches for one epoch."""
    indices = list(range(count))
    random.Random(seed + epoch * 7_919).shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def answer_metrics(logits: torch.Tensor, answers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row NLL and gold-versus-best-alternative logit margin."""
    losses = F.cross_entropy(logits, answers, reduction="none")
    gold = logits.gather(1, answers[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, answers[:, None], float("-inf"))
    return losses, gold - masked.max(dim=1).values


def train_sa(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict:
    """Answer-supervise a dense or local self-attention baseline."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    completed = tokens = 0
    losses: list[float] = []
    started = time.perf_counter()
    epoch = 0
    while completed < steps:
        for indices in batches(len(examples), batch_size, seed=seed, epoch=epoch):
            if completed >= steps:
                break
            batch = [examples[index] for index in indices]
            input_ids, mask, answers = collate_controlled(
                batch, pad_token_id=tokenizer.pad_token_id, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = last_valid_logits(
                model(input_ids, use_pra_memory=False, attention_mask=mask), mask
            )
            loss = F.cross_entropy(logits, answers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            tokens += int(mask.sum())
            losses.append(float(loss.detach()))
        epoch += 1
    elapsed = time.perf_counter() - started
    return {
        "training_steps": completed,
        "training_tokens": tokens,
        "training_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "final_train_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
    }


def train_pra(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict:
    """Train oracle-selected whole-parent memory through differentiable K/V."""
    parameters = list(trainable_parameters(model))
    if not parameters or steps == 0:
        return {
            "training_steps": 0,
            "training_tokens": 0,
            "training_seconds": 0.0,
            "tokens_per_second": 0.0,
            "final_train_loss": float("nan"),
        }
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    completed = tokens = 0
    losses: list[float] = []
    started = time.perf_counter()
    epoch = 0
    while completed < steps:
        for indices in batches(len(examples), batch_size, seed=seed, epoch=epoch):
            if completed >= steps:
                break
            batch = [examples[index] for index in indices]
            query_ids, query_mask, answers = collate_controlled(
                batch,
                pad_token_id=tokenizer.pad_token_id,
                query_only=True,
                device=device,
            )
            memory = build_controlled_memory_batch(
                batch,
                condition="whole_parent",
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = forward_with_differentiable_memory(
                model, query_ids, memory, attention_mask=query_mask
            )
            logits = last_valid_logits(output.logits, query_mask)
            loss = F.cross_entropy(logits, answers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            completed += 1
            tokens += int(query_mask.sum() + memory.attention_mask.sum())
            losses.append(float(loss.detach()))
        epoch += 1
    elapsed = time.perf_counter() - started
    return {
        "training_steps": completed,
        "training_tokens": tokens,
        "training_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "final_train_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
    }


@torch.no_grad()
def evaluate_sa(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    batch_size: int,
    device: str,
) -> dict:
    """Evaluate baseline prediction from the complete physical parent context."""
    model.eval()
    loss_rows: list[torch.Tensor] = []
    margin_rows: list[torch.Tensor] = []
    correct = total = 0
    by_depth: dict[int, list[int]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, mask, answers = collate_controlled(
            batch, pad_token_id=tokenizer.pad_token_id, device=device
        )
        logits = last_valid_logits(
            model(input_ids, use_pra_memory=False, attention_mask=mask), mask
        )
        losses, margins = answer_metrics(logits, answers)
        predictions = logits.argmax(dim=-1)
        loss_rows.append(losses.cpu())
        margin_rows.append(margins.cpu())
        correct += int((predictions == answers).sum())
        total += len(batch)
        for example, prediction, answer in zip(batch, predictions, answers):
            counts = by_depth.setdefault(example.depth, [0, 0])
            counts[0] += int(prediction == answer)
            counts[1] += 1
    losses = torch.cat(loss_rows)
    margins = torch.cat(margin_rows)
    result = {
        "condition": "full_context",
        "nll": float(losses.mean()),
        "perplexity": float(losses.mean().exp()),
        "accuracy": correct / max(total, 1),
        "answer_margin": float(margins.mean()),
        "evidence_attention_mass": float("nan"),
        "distractor_attention_mass": float("nan"),
        "memory_attention_mass": float("nan"),
        "residual_divergence": float("nan"),
    }
    for depth, (depth_correct, depth_total) in sorted(by_depth.items()):
        result[f"accuracy_depth_{depth}"] = depth_correct / depth_total
    return result


@torch.no_grad()
def evaluate_pra_condition(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    condition: MemoryCondition,
    batch_size: int,
    device: str,
) -> dict:
    """Evaluate one memory intervention and its final-layer consumption metrics."""
    model.eval()
    totals = {
        "loss": 0.0,
        "margin": 0.0,
        "correct": 0.0,
        "evidence_mass": 0.0,
        "distractor_mass": 0.0,
        "memory_mass": 0.0,
        "update": 0.0,
        "count": 0,
    }
    hidden_rows = []
    layer_totals: dict[int, dict[str, float]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        query_ids, query_mask, answers = collate_controlled(
            batch,
            pad_token_id=tokenizer.pad_token_id,
            query_only=True,
            device=device,
        )
        memory = build_controlled_memory_batch(
            batch,
            condition=condition,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        output = forward_with_differentiable_memory(
            model, query_ids, memory, attention_mask=query_mask
        )
        logits = last_valid_logits(output.logits, query_mask)
        losses, margins = answer_metrics(logits, answers)
        final_indices = query_mask.sum(dim=1) - 1
        hidden_rows.append(
            output.hidden_states[torch.arange(len(batch), device=device), final_indices].cpu()
        )
        final_layer = output.layer_metrics[max(output.layer_metrics)]
        count = len(batch)
        totals["loss"] += float(losses.sum())
        totals["margin"] += float(margins.sum())
        totals["correct"] += float((logits.argmax(dim=-1) == answers).sum())
        totals["evidence_mass"] += final_layer["evidence_attention_mass"] * count
        totals["distractor_mass"] += final_layer["distractor_attention_mass"] * count
        totals["memory_mass"] += final_layer["final_token_memory_attention_mass"] * count
        totals["update"] += final_layer["memory_update_per_token"] * count
        totals["count"] += count
        for layer_id, metrics in output.layer_metrics.items():
            layer = layer_totals.setdefault(
                layer_id,
                {
                    "evidence_attention_mass": 0.0,
                    "distractor_attention_mass": 0.0,
                    "memory_attention_mass": 0.0,
                    "memory_update_per_token": 0.0,
                    "count": 0.0,
                },
            )
            for name in (
                "evidence_attention_mass",
                "distractor_attention_mass",
                "memory_update_per_token",
            ):
                layer[name] += float(metrics[name]) * count
            layer["memory_attention_mass"] += float(
                metrics["final_token_memory_attention_mass"]
            ) * count
            layer["count"] += count
    count = max(totals["count"], 1)
    return {
        "condition": condition,
        "nll": totals["loss"] / count,
        "perplexity": math.exp(min(totals["loss"] / count, 20.0)),
        "accuracy": totals["correct"] / count,
        "answer_margin": totals["margin"] / count,
        "evidence_attention_mass": totals["evidence_mass"] / count,
        "distractor_attention_mass": totals["distractor_mass"] / count,
        "memory_attention_mass": totals["memory_mass"] / count,
        "memory_update_per_token": totals["update"] / count,
        "hidden_states": torch.cat(hidden_rows),
        "layer_profiles": [
            {
                "layer": layer_id,
                **{
                    name: value / max(metrics["count"], 1.0)
                    for name, value in metrics.items()
                    if name != "count"
                },
            }
            for layer_id, metrics in sorted(layer_totals.items())
        ],
    }


def evaluate_pra(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    batch_size: int,
    device: str,
) -> tuple[list[dict], list[dict]]:
    """Run all causal memory interventions and add no-memory divergence."""
    raw = [
        evaluate_pra_condition(
            model,
            examples,
            tokenizer,
            condition=condition,
            batch_size=batch_size,
            device=device,
        )
        for condition in MEMORY_CONDITIONS
    ]
    baseline_hidden = raw[0]["hidden_states"]
    rows = []
    profiles = []
    for row in raw:
        hidden = row.pop("hidden_states")
        condition = row["condition"]
        profiles.extend(
            {"condition": condition, **profile}
            for profile in row.pop("layer_profiles")
        )
        row["residual_divergence"] = float(
            (1.0 - F.cosine_similarity(hidden, baseline_hidden, dim=-1)).mean()
        )
        rows.append(row)
    return rows, profiles


@torch.no_grad()
def evaluate_representation_portability(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Compare evidence K/V encoded alone versus inside the whole parent."""
    model.eval()
    totals: dict[int, dict[str, float]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        evidence = build_controlled_memory_batch(
            batch,
            condition="evidence_only",
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        parent = build_controlled_memory_batch(
            batch,
            condition="whole_parent",
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        evidence_kv = encode_differentiable_memory_kv(model, evidence)
        parent_kv = encode_differentiable_memory_kv(model, parent)
        for layer_id in evidence_kv:
            layer = totals.setdefault(layer_id, {"key": 0.0, "value": 0.0, "count": 0.0})
            evidence_keys, evidence_values = evidence_kv[layer_id]
            parent_keys, parent_values = parent_kv[layer_id]
            for row in range(len(batch)):
                mask = parent.evidence_mask[row, : parent.lengths[row]]
                selected_key = parent_keys[row][:, :, mask, :]
                selected_value = parent_values[row][:, :, mask, :]
                if selected_key.shape != evidence_keys[row].shape:
                    raise RuntimeError("Evidence token alignment changed between memory conditions.")
                layer["key"] += float(
                    (1.0 - F.cosine_similarity(evidence_keys[row], selected_key, dim=-1)).mean()
                )
                layer["value"] += float(
                    (1.0 - F.cosine_similarity(evidence_values[row], selected_value, dim=-1)).mean()
                )
                layer["count"] += 1
    return [
        {
            "layer": layer_id,
            "key_context_divergence": values["key"] / max(values["count"], 1.0),
            "value_context_divergence": values["value"] / max(values["count"], 1.0),
            "examples": int(values["count"]),
        }
        for layer_id, values in sorted(totals.items())
    ]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write union-key dictionaries as a stable CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[dict]) -> list[dict]:
    """Aggregate seed-level metric rows by model and intervention."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["condition"]), []).append(row)
    output = []
    metric_names = (
        "nll",
        "perplexity",
        "accuracy",
        "answer_margin",
        "evidence_attention_mass",
        "distractor_attention_mass",
        "memory_attention_mass",
        "memory_update_per_token",
        "residual_divergence",
    )
    for (model, condition), members in groups.items():
        summary = {"model": model, "condition": condition, "seeds": len(members)}
        for metric in metric_names:
            values = [float(row[metric]) for row in members if metric in row and math.isfinite(float(row[metric]))]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
                summary[metric] = mean
                summary[f"{metric}_sd"] = math.sqrt(variance)
        output.append(summary)
    return output


def derive_ladder(rows: Sequence[dict], parameter_rows: Sequence[dict]) -> list[dict]:
    """Compute paired causal effects for each PRA adaptation rung and seed."""
    lookup = {(row["model"], row["seed"], row["condition"]): row for row in rows}
    fractions = {(row["model"], row["seed"]): row["trainable_fraction"] for row in parameter_rows}
    output = []
    for model in PRA_REGIMES:
        seeds = sorted({row["seed"] for row in rows if row["model"] == model})
        for seed in seeds:
            none = lookup[(model, seed, "none")]
            distractor = lookup[(model, seed, "matched_distractor")]
            evidence = lookup[(model, seed, "evidence_only")]
            parent = lookup[(model, seed, "whole_parent")]
            output.append(
                {
                    "model": model,
                    "seed": seed,
                    "trainable_fraction": fractions[(model, seed)],
                    "correct_memory_margin_gain_vs_none": evidence["answer_margin"] - none["answer_margin"],
                    "correct_memory_margin_gain_vs_distractor": evidence["answer_margin"] - distractor["answer_margin"],
                    "evidence_only_minus_parent_nll": evidence["nll"] - parent["nll"],
                    "evidence_only_minus_parent_accuracy": evidence["accuracy"] - parent["accuracy"],
                    "evidence_only_minus_parent_margin": evidence["answer_margin"] - parent["answer_margin"],
                    "whole_parent_evidence_selectivity": parent["evidence_attention_mass"] / max(parent["memory_attention_mass"], 1e-12),
                    "useful_memory_residual_divergence": evidence["residual_divergence"],
                    "distractor_residual_divergence": distractor["residual_divergence"],
                }
            )
    return output


def plot_ladder(ladder: Sequence[dict], output_dir: Path) -> None:
    """Plot adaptation fraction against causal memory usefulness."""
    order = list(PRA_REGIMES)
    grouped = {model: [row for row in ladder if row["model"] == model] for model in order}
    x = [sum(row["trainable_fraction"] for row in grouped[model]) / len(grouped[model]) for model in order]
    margin = [sum(row["correct_memory_margin_gain_vs_distractor"] for row in grouped[model]) / len(grouped[model]) for model in order]
    selectivity = [sum(row["whole_parent_evidence_selectivity"] for row in grouped[model]) / len(grouped[model]) for model in order]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(x, margin, marker="o", color="#006d77")
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].set(xlabel="Trainable parameter fraction", ylabel="Evidence margin gain vs distractor")
    axes[1].plot(x, selectivity, marker="s", color="#b44724")
    axes[1].set(xlabel="Trainable parameter fraction", ylabel="Evidence share of memory attention", ylim=(0, 1))
    for axis, values in zip(axes, (margin, selectivity)):
        for x_value, y_value, label in zip(x, values, order):
            axis.annotate(label.replace("_", " "), (x_value, y_value), xytext=(3, 4), textcoords="offset points", fontsize=7)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "adaptation_capacity_vs_memory_usefulness.png", dpi=180)
    figure.savefig(figures / "adaptation_capacity_vs_memory_usefulness.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/papers/shared/results/paper4_training/tier0"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--base-steps", type=int, default=800)
    parser.add_argument("--adapt-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--pra-layers", default="3,7")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tokenizer = ControlledTokenizer()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    pra_layers = tuple(int(value) for value in args.pra_layers.split(",") if value)
    train_data = controlled_examples(tokenizer, count=args.train_examples, seed=100_001)
    test_data = controlled_examples(tokenizer, count=args.test_examples, seed=900_001)
    architecture = {
        "protocol": "oracle-fixed differentiable native-KV consumer learning",
        "seeds": seeds,
        "base_steps": args.base_steps,
        "adapt_steps": args.adapt_steps,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "local_window": args.window,
        "pra_layers": pra_layers,
        "lora_rank": args.lora_rank,
        "memory_conditions": MEMORY_CONDITIONS,
        "router_trained": False,
        "materializer_trained": False,
    }
    (args.output_dir / "paper4_architecture_configs.json").write_text(
        json.dumps(architecture, indent=2), encoding="utf-8"
    )
    metric_rows: list[dict] = []
    parameter_rows: list[dict] = []
    timing_rows: list[dict] = []
    consumer_rows: list[dict] = []
    portability_rows: list[dict] = []

    for seed in seeds:
        set_seed(seed)
        global_cfg = model_config(tokenizer, window=None, variant="td_sa", d_model=args.d_model, layers=args.layers, pra_layers=pra_layers, device=args.device)
        local_cfg = model_config(tokenizer, window=args.window, variant="td_sa", d_model=args.d_model, layers=args.layers, pra_layers=pra_layers, device=args.device)
        target_cfg = model_config(tokenizer, window=args.window, variant="td_layered_pra", d_model=args.d_model, layers=args.layers, pra_layers=pra_layers, device=args.device)
        base_models = {}
        for label, cfg in (("global_sa", global_cfg), ("local_sa", local_cfg)):
            # Matched SA baselines differ only in the attention window.
            set_seed(seed)
            checkpoint = checkpoint_dir / f"{label}_seed{seed}.pt"
            model = TinyPRAModel(cfg).to(args.device)
            if args.resume and checkpoint.exists():
                payload = torch.load(checkpoint, map_location=args.device, weights_only=True)
                model.load_state_dict(payload["model"])
                training = payload["training"]
            else:
                training = train_sa(model, train_data, tokenizer, steps=args.base_steps, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=seed, device=args.device)
                torch.save({"model": model.state_dict(), "training": training}, checkpoint)
            base_models[label] = model
            metrics = evaluate_sa(model, test_data, tokenizer, batch_size=args.batch_size, device=args.device)
            metric_rows.append({"model": label, "seed": seed, **metrics})
            parameter_rows.append({"model": label, "seed": seed, **parameter_summary(model)})
            timing_rows.append({"model": label, "seed": seed, **training})

        for regime in PRA_REGIMES:
            set_seed(seed + 1_003)
            if regime == "native_scratch":
                model = TinyPRAModel(target_cfg).to(args.device)
            else:
                model = convert_sa_model_to_pra(base_models["local_sa"], target_cfg).to(args.device)
            targets = install_adaptation_regime(
                model,
                regime,
                lora_rank=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
            )
            steps = args.base_steps + args.adapt_steps if regime == "native_scratch" else args.adapt_steps
            learning_rate = args.lora_learning_rate if "lora" in regime else args.learning_rate
            checkpoint = checkpoint_dir / f"{regime}_seed{seed}.pt"
            if args.resume and checkpoint.exists():
                payload = torch.load(checkpoint, map_location=args.device, weights_only=True)
                model.load_state_dict(payload["model"])
                training = payload["training"]
            else:
                training = train_pra(model, train_data, tokenizer, steps=steps, batch_size=args.batch_size, learning_rate=learning_rate, seed=seed + 31, device=args.device)
                torch.save({"model": model.state_dict(), "training": training}, checkpoint)
            timing_rows.append({"model": regime, "seed": seed, **training})
            parameter_rows.append({"model": regime, "seed": seed, "lora_targets": len(targets), **parameter_summary(model)})
            metrics_by_condition, profiles = evaluate_pra(
                model,
                test_data,
                tokenizer,
                batch_size=args.batch_size,
                device=args.device,
            )
            for metrics in metrics_by_condition:
                metric_rows.append({"model": regime, "seed": seed, **metrics})
            consumer_rows.extend(
                {"model": regime, "seed": seed, **profile} for profile in profiles
            )
            portability_rows.extend(
                {"model": regime, "seed": seed, **row}
                for row in evaluate_representation_portability(
                    model,
                    test_data[: min(64, len(test_data))],
                    tokenizer,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            )
        write_csv(args.output_dir / "adaptation_ladder_seed_results.csv", metric_rows)
        write_csv(args.output_dir / "lora_configs.csv", parameter_rows)
        write_csv(args.output_dir / "paper4_hardware_benchmarks.csv", timing_rows)
        write_csv(args.output_dir / "consumer_layer_profiles.csv", consumer_rows)
        write_csv(args.output_dir / "representation_portability.csv", portability_rows)
        if args.benchmark_only:
            break

    summaries = aggregate(metric_rows)
    ladder = derive_ladder(metric_rows, parameter_rows)
    write_csv(args.output_dir / "adaptation_ladder_results.csv", ladder)
    write_csv(args.output_dir / "memory_modularity.csv", ladder)
    write_csv(args.output_dir / "scaling_summary.csv", summaries)
    write_csv(
        args.output_dir / "full_weight_training_runs.csv",
        [row for row in timing_rows if row["model"] == "full_weight"],
    )
    write_csv(
        args.output_dir / "topology_after_training.csv",
        [
            {
                "status": "not_measured_at_gate_0",
                "reason": "router and materializer are fixed to isolate consumer learning",
                "next_gate": "repeat R@K, MRR, recovery depth, shortcut, branching, and contraction after a positive consumer gate",
            }
        ],
    )
    (args.output_dir / "claude_usage_budget.md").write_text(
        "# Claude usage budget\n\nNo Claude API calls were used for Tier 0.\n",
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "native_pra_training_runs.csv",
        [row for row in timing_rows if row["model"] == "native_scratch"],
    )
    plot_ladder(ladder, args.output_dir)
    findings = {
        "status": "tier0_controlled_complete" if len(seeds) == len(SEEDS) and not args.benchmark_only else "tier0_pilot",
        "seed_count": len({row["seed"] for row in metric_rows}),
        "gate_0": "pending_summary",
        "scope": "consumer learning under fixed oracle memory; routing/materialization held constant",
    }
    (args.output_dir / "paper4_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **findings}, indent=2))


if __name__ == "__main__":
    main()
