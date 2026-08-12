"""Train and evaluate minimal frozen-backbone PRA memory calibration."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_oracle_memory_use import (
    _answer_ids,
    _generate,
    _oracle_selections,
    _prompt,
    _route_once,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import (
    MEMORY_GATE_FIXED,
    MEMORY_GATE_PER_LAYER,
    MEMORY_GATE_SINGLE,
    PRAHFConfig,
    inject_pra,
    load_hf_routing_projection,
)


RESIDUAL_WIDTHS = (16, 32, 64)
VARIANTS = (
    MEMORY_GATE_FIXED,
    MEMORY_GATE_SINGLE,
    MEMORY_GATE_PER_LAYER,
    *(f"residual_{width}" for width in RESIDUAL_WIDTHS),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gold_scores(
    handle,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Return differentiable negative log-likelihood and detached rank metrics."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    prediction_positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    output = handle.model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
    )
    logits = output.logits[:, prediction_positions, :].float()
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


def _prepare_example(handle, tokenizer, example, layers, prompt_tokens, device):
    """Publish one frozen reference and cache its oracle and routed identities."""
    source_ids = tokenizer(
        example["source"], return_tensors="pt", add_special_tokens=False
    ).input_ids
    spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
    handle.cache.clear()
    entry = handle.add_reference(
        f"gate://{example['dataset']}/{example['id']}",
        source_ids,
        text=example["source"],
    )
    prompt_ids, prompt_mask, _ = _prompt(
        tokenizer,
        example["question"],
        max_tokens=prompt_tokens,
    )
    answer_ids = _answer_ids(tokenizer, example["answer"])
    oracle = {
        layer: [_oracle_selections(entry, layer, spans)]
        for layer in layers
    }
    route = _route_once(
        handle,
        tokenizer,
        example["question"],
        max(layers),
        prompt_tokens,
        device,
    )
    return {
        "example": example,
        "entry": entry,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "answer_ids": answer_ids,
        "oracle": oracle,
        "routed": route["selected"],
    }


def _activate(handle, record, layers, condition):
    """Install one record's cache entry and select oracle, routed, or no memory."""
    handle.cache.clear()
    handle.cache.put(record["entry"])
    if condition == "none":
        handle.configure_memory_layers(set())
        return
    if condition == "oracle":
        fixed = record["oracle"]
    elif condition == "routed":
        fixed = handle.map_chunk_identities_to_layers(record["routed"], layers)
    else:
        raise ValueError(f"Unsupported memory condition: {condition}")
    handle.configure_memory_layers(set(layers), fixed_selections=fixed)


def _train_variant(handle, records, layers, variant, seed, steps, learning_rate, device):
    """Optimize one minimal memory calibration on oracle HotpotQA examples."""
    if variant.startswith("residual_"):
        width = int(variant.rsplit("_", 1)[1])
        handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
        handle.configure_residual_adapter(width, reset=True)
        parameters = handle.residual_adapter_parameters()
    else:
        handle.configure_residual_adapter(0)
        handle.configure_memory_gate(variant, initial_value=1.0)
        parameters = handle.memory_gate_parameters()
    if variant == MEMORY_GATE_FIXED:
        return {
            "variant": variant,
            "trainable_parameters": 0,
            "losses": [],
            "training_seconds": 0.0,
            "gate_values": handle.memory_gate_values(),
            "residual_bottleneck": 0,
        }
    trainable_names = [
        name for name, parameter in handle.model.named_parameters() if parameter.requires_grad
    ]
    expected_owner = (
        "pra_residual_adapter"
        if variant.startswith("residual_")
        else "pra_memory_gate"
    )
    if not trainable_names or any(expected_owner not in name for name in trainable_names):
        raise RuntimeError(
            f"Memory calibration leaked to base parameters: {trainable_names}"
        )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    order = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(order)
    losses = []
    started = time.perf_counter()
    # Keep dropout and all other frozen-model behavior in inference mode while
    # retaining autograd only for the active calibration parameters.
    handle.model.eval()
    for step in range(steps):
        record = records[order[step % len(order)]]
        _activate(handle, record, layers, "oracle")
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        loss.backward()
        max_grad_norm = 1.0 if variant.startswith("residual_") else 10.0
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        if not variant.startswith("residual_"):
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(-2.0, 4.0)
        losses.append(float(loss.detach().cpu()))
    _sync(device)
    handle.model.eval()
    return {
        "variant": variant,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": steps,
        "learning_rate": learning_rate,
        "training_seconds": time.perf_counter() - started,
        "losses": losses,
        "initial_loss_mean": statistics.fmean(losses[: min(len(losses), len(records))]),
        "final_loss_mean": statistics.fmean(losses[-min(len(losses), len(records)) :]),
        "gate_values": handle.memory_gate_values(),
        "residual_bottleneck": handle.residual_adapter.bottleneck,
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
    """Evaluate no-memory parity plus oracle and routed memory on held-out QA."""
    rows = []
    no_memory_before = {}
    for record in records:
        _activate(handle, record, layers, "none")
        _, metrics = _gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        no_memory_before[record["example"]["id"]] = metrics
    for record in records:
        baseline = no_memory_before[record["example"]["id"]]
        for condition in ("none", "oracle", "routed"):
            _activate(handle, record, layers, condition)
            started = time.perf_counter()
            _, metrics = _gold_scores(
                handle,
                record["prompt_ids"],
                record["prompt_mask"],
                record["answer_ids"],
                device,
            )
            _sync(device)
            teacher_forced_seconds = time.perf_counter() - started
            prediction, generation_seconds = _generate(
                handle,
                tokenizer,
                record["prompt_ids"],
                record["prompt_mask"],
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
                    "trainable_parameter_count": (
                        sum(
                            parameter.numel()
                            for parameter in handle.memory_gate_parameters()
                        )
                        + handle.residual_adapter.trainable_parameter_count
                    ),
                    "residual_bottleneck": handle.residual_adapter.bottleneck,
                    "gate_values": handle.memory_gate_values(),
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
                        or metrics
                        == pra_off_reference[record["example"]["id"]]
                    ),
                    "teacher_forced_seconds": teacher_forced_seconds,
                    "generation_seconds": generation_seconds,
                    "active_memory_tokens": sum(
                        int(row.get("retrieved_physical_kv_tokens", 0))
                        for layer, row in diagnostics.items()
                        if layer in layers
                    ),
                    **metrics,
                    **answer_metrics(prediction, record["example"]["answer"]),
                }
            )
    return rows


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["seed"], row["variant"], row["dataset"], row["condition"])
        ].append(row)
    seed_rows = []
    metrics = (
        "gold_mean_token_logprob",
        "gold_mean_logprob_delta_vs_none",
        "gold_first_token_rank",
        "gold_first_token_rank_delta_vs_none",
        "gold_first_token_margin",
        "gold_first_token_margin_delta_vs_none",
        "f1",
        "em",
        "answer_contained",
        "teacher_forced_seconds",
        "generation_seconds",
        "active_memory_tokens",
        "trainable_parameter_count",
        "trainable_parameter_percent",
        "trainable_memory_bytes_fp32",
    )
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
    aggregate = []
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
        aggregate.append(record)
    return seed_rows, aggregate


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _plot(aggregate, output_dir):
    rows = [
        row for row in aggregate
        if row["condition"] in {"oracle", "routed"}
    ]
    variants = ["fixed", "single", "residual_16", "residual_32", "residual_64"]
    labels = ["Frozen PRA", "Scalar gate", "Residual 16", "Residual 32", "Residual 64"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    colors = {"hotpotqa": "#245A8D", "qasper": "#A34832"}
    offsets = {
        ("hotpotqa", "oracle"): -0.18,
        ("hotpotqa", "routed"): -0.06,
        ("qasper", "oracle"): 0.06,
        ("qasper", "routed"): 0.18,
    }
    for dataset, color in colors.items():
        for condition, marker in (("oracle", "o"), ("routed", "s")):
            selected = [
                next(
                    row for row in rows
                    if row["variant"] == variant
                    and row["dataset"] == dataset
                    and row["condition"] == condition
                )
                for variant in variants
            ]
            x_positions = [
                index + offsets[(dataset, condition)]
                for index in range(len(variants))
            ]
            axes[0].errorbar(
                x_positions,
                [row["gold_mean_logprob_delta_vs_none_mean"] for row in selected],
                yerr=[row["gold_mean_logprob_delta_vs_none_std"] for row in selected],
                marker=marker,
                linestyle="none",
                color=color,
                capsize=2,
                label=f"{dataset}, {condition}",
            )
            axes[1].errorbar(
                x_positions,
                [row["f1_mean"] for row in selected],
                yerr=[row["f1_std"] for row in selected],
                marker=marker,
                linestyle="none",
                color=color,
                capsize=2,
                label=f"{dataset}, {condition}",
            )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Gold mean-token log-probability delta")
    axes[1].set_ylabel("Generated token F1")
    for axis in axes:
        axis.set_xticks(range(len(variants)), labels, rotation=18, ha="right")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"memory_residual_adapter.{suffix}", dpi=180)
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
            memory_gate_mode=MEMORY_GATE_FIXED,
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
    print("preparing frozen references", flush=True)
    train_records = [
        _prepare_example(handle, tokenizer, example, layers, args.prompt_tokens, device)
        for example in train_examples
    ]
    heldout_records = [
        _prepare_example(handle, tokenizer, example, layers, args.prompt_tokens, device)
        for example in heldout_examples
    ]
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    handle.configure_residual_adapter(0)
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
    rows = []
    training = []
    base_model_parameters = sum(
        parameter.numel()
        for name, parameter in handle.model.named_parameters()
        if not name.startswith("pra_memory_gate")
        and not name.startswith("pra_residual_adapter")
    )
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            report = _train_variant(
                handle,
                train_records,
                layers,
                variant,
                seed,
                args.steps,
                (
                    args.residual_learning_rate
                    if variant.startswith("residual_")
                    else args.gate_learning_rate
                ),
                device,
            )
            report["seed"] = seed
            training.append(report)
            torch.save(
                {
                    "variant": variant,
                    "seed": seed,
                    "layer_ids": layers,
                    "gate_values": handle.memory_gate_values(),
                    "trainable_state_dict": {
                        name: parameter.detach().cpu()
                        for name, parameter in handle.model.named_parameters()
                        if parameter.requires_grad
                    },
                    "residual_bottleneck": handle.residual_adapter.bottleneck,
                },
                checkpoints / f"{variant}_seed{seed}.pt",
            )
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
                f"seed={seed} variant={variant} gates={handle.memory_gate_values()}",
                flush=True,
            )
    for row in rows:
        parameter_count = int(row["trainable_parameter_count"])
        row["trainable_parameter_percent"] = (
            100.0 * parameter_count / base_model_parameters
        )
        row["trainable_memory_bytes_fp32"] = 4 * parameter_count
    seed_rows, aggregate = _aggregate(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "frozen Qwen final-four-layer PRA memory calibration",
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
        "gate_learning_rate": args.gate_learning_rate,
        "residual_learning_rate": args.residual_learning_rate,
        "variants": args.variants,
        "base_model_parameters": base_model_parameters,
        "training": training,
        "rows": rows,
        "seed_aggregates": seed_rows,
        "aggregates": aggregate,
        "pra_off_all_exact": all(
            row["pra_off_gold_logprob_exact"] for row in rows if row["condition"] == "none"
        ),
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "memory_gate.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "memory_gate.csv", rows)
    _write_csv(args.output_dir / "memory_gate_seed_aggregate.csv", seed_rows)
    _write_csv(args.output_dir / "memory_gate_aggregate.csv", aggregate)
    _plot(aggregate, args.output_dir)
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
    parser.add_argument("--gate-learning-rate", type=float, default=0.1)
    parser.add_argument("--residual-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
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
    parser.add_argument("--output-dir", type=Path, default=result_dir / "memory_gate")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from the existing memory_gate.json artifact.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.plot_only:
        result = json.loads(
            (arguments.output_dir / "memory_gate.json").read_text(encoding="utf-8")
        )
        _plot(result["aggregates"], arguments.output_dir)
    else:
        result = run(arguments)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
