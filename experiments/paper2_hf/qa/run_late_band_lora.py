"""Compare late-band conditional LoRA with frozen PRA and residual adaptation."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_memory_gate import (
    _activate,
    _aggregate,
    _gold_scores,
    _prepare_example,
    _sync,
    _write_csv,
)
from experiments.paper2_hf.qa.run_oracle_memory_use import _generate, _prompt
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import MEMORY_GATE_FIXED, PRAHFConfig, inject_pra, load_hf_routing_projection


LORA_RANKS = (2, 4, 8)
VARIANTS = (
    "fixed",
    "residual_32",
    *(f"lora_o_r{rank}" for rank in LORA_RANKS),
)


def _prediction_logits(handle, prompt_ids, prompt_mask, answer_ids, device):
    """Return answer-position logits while retaining gradients when available."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    output = handle.model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
    )
    positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    return output.logits[:, positions, :].float()


def _add_direct_prompt(record, tokenizer, max_tokens):
    """Attach the evidence-only direct-text control prompt to one record."""
    prompt_ids, prompt_mask, _ = _prompt(
        tokenizer,
        record["example"]["question"],
        context="\n".join(record["example"]["evidence"]),
        max_tokens=max_tokens,
    )
    record["direct_prompt_ids"] = prompt_ids
    record["direct_prompt_mask"] = prompt_mask
    return record


def _configure_variant(handle, variant):
    """Enable exactly one calibration owner and return its parameters."""
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    handle.configure_residual_adapter(0)
    handle.configure_late_band_lora(0)
    if variant == "fixed":
        return [], 0, 0
    if variant == "residual_32":
        handle.configure_residual_adapter(32, reset=True)
        return handle.residual_adapter_parameters(), 32, 0
    if variant.startswith("lora_o_r"):
        rank = int(variant.rsplit("r", 1)[1])
        handle.configure_late_band_lora(
            rank,
            alpha=float(rank),
            dropout=0.0,
            reset=True,
        )
        return handle.late_band_lora_parameters(), 0, rank
    raise ValueError(f"Unsupported late-band variant: {variant}")


def _train_variant(
    handle,
    records,
    layers,
    variant,
    seed,
    steps,
    learning_rate,
    no_pra_weight,
    device,
):
    """Train one conditional adapter with an explicit no-PRA control batch."""
    parameters, residual_width, lora_rank = _configure_variant(handle, variant)
    if not parameters:
        return {
            "variant": variant,
            "trainable_parameters": 0,
            "losses": [],
            "memory_losses": [],
            "no_pra_control_losses": [],
            "no_pra_control_all_exact": True,
            "training_seconds": 0.0,
            "residual_bottleneck": residual_width,
            "lora_rank": lora_rank,
        }
    expected_owner = (
        "pra_residual_adapter" if residual_width else "pra_late_band_lora"
    )
    trainable_names = [
        name for name, parameter in handle.model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(expected_owner not in name for name in trainable_names):
        raise RuntimeError(f"Late-band adaptation leaked to base parameters: {trainable_names}")

    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
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

        _activate(handle, record, layers, "none")
        no_pra_logits = _prediction_logits(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        target = record["no_pra_control_logits"].to(device)
        control_loss = F.mse_loss(no_pra_logits, target)
        loss = memory_loss + float(no_pra_weight) * control_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        memory_losses.append(float(memory_loss.detach().cpu()))
        control_losses.append(float(control_loss.detach().cpu()))
        control_exact.append(torch.equal(no_pra_logits.detach(), target))
    _sync(device)
    return {
        "variant": variant,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": steps,
        "learning_rate": learning_rate,
        "no_pra_regularization_weight": float(no_pra_weight),
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
        "residual_bottleneck": residual_width,
        "lora_rank": lora_rank,
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
    pra_off_reference,
):
    """Measure no-memory, oracle, routed, and direct-text controls."""
    rows = []
    baselines = {}
    for record in records:
        _activate(handle, record, layers, "none")
        _, metrics = _gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        baselines[record["example"]["id"]] = metrics

    parameter_count = (
        handle.residual_adapter.trainable_parameter_count
        + handle.late_band_lora.trainable_parameter_count
    )
    for record in records:
        baseline = baselines[record["example"]["id"]]
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
                handle,
                prompt_ids,
                prompt_mask,
                record["answer_ids"],
                device,
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
                    "example_id": record["example"]["id"],
                    "condition": condition,
                    "answer": record["example"]["answer"],
                    "generated_answer": prediction,
                    "trainable_parameter_count": parameter_count,
                    "residual_bottleneck": handle.residual_adapter.bottleneck,
                    "lora_rank": handle.late_band_lora.rank,
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
                    "pra_off_gold_logprob_exact": (
                        condition != "none"
                        or metrics == pra_off_reference[record["example"]["id"]]
                    ),
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


def _plot(aggregates, output_dir):
    """Plot likelihood and decoded F1 with direct text as a horizontal control."""
    labels = ["Frozen", "Residual 32", "O-LoRA r2", "O-LoRA r4", "O-LoRA r8"]
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    colors = {"oracle": "#245A8D", "routed": "#A34832"}
    markers = {"oracle": "o", "routed": "s"}
    for row_index, dataset in enumerate(("hotpotqa", "qasper")):
        dataset_rows = [row for row in aggregates if row["dataset"] == dataset]
        for column, metric, ylabel in (
            (0, "gold_mean_logprob_delta_vs_none", r"Gold mean-token $\Delta\log p$"),
            (1, "f1", "Generated token F1"),
        ):
            axis = axes[row_index, column]
            for condition in ("oracle", "routed"):
                selected = [
                    next(
                        row for row in dataset_rows
                        if row["variant"] == variant and row["condition"] == condition
                    )
                    for variant in VARIANTS
                ]
                axis.errorbar(
                    range(len(VARIANTS)),
                    [row[f"{metric}_mean"] for row in selected],
                    yerr=[row[f"{metric}_std"] for row in selected],
                    color=colors[condition],
                    marker=markers[condition],
                    capsize=2,
                    label=condition,
                )
            direct = next(
                row for row in dataset_rows
                if row["variant"] == "fixed" and row["condition"] == "direct_text"
            )
            axis.axhline(
                direct[f"{metric}_mean"],
                color="#3D7A57",
                linestyle=":",
                label="direct text",
            )
            if metric.startswith("gold_"):
                axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_ylabel(f"{dataset}: {ylabel}")
            axis.grid(alpha=0.22)
            axis.set_xticks(range(len(VARIANTS)), labels, rotation=18, ha="right")
    axes[0, 0].set_title("Gold likelihood")
    axes[0, 1].set_title("Greedy answer quality")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"late_band_lora.{suffix}", dpi=180)
    plt.close(figure)


def run(args):
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
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
        args.cache_dir,
        args.heldout_examples,
        args.heldout_offset,
        args.data_seed,
    )
    print("preparing frozen references and controls", flush=True)
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
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    handle.configure_residual_adapter(0)
    handle.configure_late_band_lora(0)
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
    pra_off_reference = {}
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
            pra_off_reference[record["example"]["id"]] = metrics

    base_model_parameters = sum(
        parameter.numel()
        for name, parameter in handle.model.named_parameters()
        if not name.startswith("pra_memory_gate")
        and not name.startswith("pra_residual_adapter")
        and not name.startswith("pra_late_band_lora")
    )
    rows = []
    training = []
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            learning_rate = (
                args.residual_learning_rate
                if variant.startswith("residual_")
                else args.lora_learning_rate
            )
            report = _train_variant(
                handle,
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
                    pra_off_reference,
                )
            )
            print(
                f"seed={seed} variant={variant} params={report['trainable_parameters']} "
                f"control_exact={report['no_pra_control_all_exact']}",
                flush=True,
            )
    for row in rows:
        parameter_count = int(row["trainable_parameter_count"])
        row["trainable_parameter_percent"] = 100.0 * parameter_count / base_model_parameters
        row["trainable_memory_bytes_fp32"] = 4 * parameter_count
    seed_rows, aggregates = _aggregate(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "conditional output-projection LoRA in Qwen final four PRA layers",
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
        "target_module": "attention output projection",
        "conditional_on_materialized_pra_memory": True,
        "lora_alpha_policy": "alpha equals rank",
        "lora_dropout": 0.0,
        "steps": args.steps,
        "lora_learning_rate": args.lora_learning_rate,
        "residual_learning_rate": args.residual_learning_rate,
        "no_pra_regularization_weight": args.no_pra_weight,
        "variants": args.variants,
        "base_model_parameters": base_model_parameters,
        "training": training,
        "rows": rows,
        "seed_aggregates": seed_rows,
        "aggregates": aggregates,
        "pra_off_all_exact": all(
            row["pra_off_gold_logprob_exact"]
            for row in rows
            if row["condition"] == "none"
        ),
        "training_no_pra_controls_all_exact": all(
            report["no_pra_control_all_exact"] for report in training
        ),
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "late_band_lora.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "late_band_lora.csv", rows)
    _write_csv(args.output_dir / "late_band_lora_seed_aggregate.csv", seed_rows)
    _write_csv(args.output_dir / "late_band_lora_aggregate.csv", aggregates)
    _plot(aggregates, args.output_dir)
    return artifact


def parse_args():
    result_dir = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int, default=8)
    parser.add_argument("--heldout-examples", type=int, default=4)
    parser.add_argument("--heldout-offset", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-pra-weight", type=float, default=1.0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            result_dir
            / "routing"
            / "learned_adapter"
            / "checkpoints"
            / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=result_dir / "late_band_lora")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.plot_only:
        result = json.loads(
            (arguments.output_dir / "late_band_lora.json").read_text(encoding="utf-8")
        )
        _plot(result["aggregates"], arguments.output_dir)
    else:
        result = run(arguments)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
