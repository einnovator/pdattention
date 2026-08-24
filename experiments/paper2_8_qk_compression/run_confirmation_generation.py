"""Replay fresh-cohort Paper 2.8 selections through native-K/V generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_confirmation import (
    CONFIRMATION_OFFSET,
    CONFIRMATION_PER_DATASET,
    RESULT_ROOT,
)
from experiments.paper2_8_qk_compression.run_gated_study import (
    MODEL_ID,
    MODEL_REVISION,
    _bootstrap,
    _write_csv,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_oracle_memory_use import (
    _answer_ids,
    _generate,
    _prompt,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import PRAHFConfig, inject_pra
from pra_torch.memory import SelectedChunk


MEMORY_CONDITIONS = (
    "native_mean",
    "exact",
    "bm25",
    "hybrid",
    "lowrank_r16_ensemble",
    "lowrank_r8_kmeans_ensemble",
    "shuffled_selection",
    "irrelevant_bottom",
    "oracle_evidence",
)
ALL_CONDITIONS = (
    "no_memory",
    "empty_memory",
    *MEMORY_CONDITIONS,
    "direct_evidence",
    "full_context",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _selected_lookup(rows: list[dict]) -> dict[tuple[str, str, str], list[int]]:
    output = {}
    for row in rows:
        if int(row["seed"]) != -1 or row["condition"] not in MEMORY_CONDITIONS:
            continue
        output[(row["dataset"], row["example_id"], row["condition"])] = [
            int(value) for value in row["selected_chunks"].split()
        ]
    return output


def _fixed_selection(entry, layer: int, indices: list[int], spans) -> list[list[SelectedChunk]]:
    chunks = sorted(entry.layer_memory[layer].chunks, key=lambda chunk: chunk.logical_start)
    actual_spans = [(chunk.logical_start, chunk.logical_end) for chunk in chunks]
    if list(map(tuple, spans)) != actual_spans:
        raise AssertionError("Feature-cache chunks do not align with native-K/V cache chunks.")
    selected = []
    for rank, index in enumerate(indices, start=1):
        chunk = chunks[index]
        selected.append(
            SelectedChunk(
                entry=entry,
                chunk=chunk,
                reference_score=1.0,
                chunk_score=1.0 - rank * 1e-6,
                layer_id=layer,
                reference_rank=1,
                rank_within_reference=rank,
                metadata={"selection_source": "paper2_8_confirmation_replay"},
            )
        )
    return [selected]


def _teacher_forced(model, prompt_ids, prompt_mask, answer_ids, device) -> dict:
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
        ).logits[:, positions, :].float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    targets = answer_ids.to(device)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[0]
    first_logits = logits[0, 0]
    first_target = int(targets[0, 0])
    return {
        "gold_sequence_logprob": float(token_log_probs.sum()),
        "gold_mean_token_logprob": float(token_log_probs.mean()),
        "gold_first_token_probability": float(first_logits.softmax(dim=-1)[first_target]),
        "gold_first_token_rank": int((first_logits > first_logits[first_target]).sum()) + 1,
        "teacher_forced_seconds": time.perf_counter() - started,
    }


def _prompt_for_condition(tokenizer, example: dict, condition: str, args):
    if condition == "direct_evidence":
        context = "\n".join(example["evidence"])
        return (*_prompt(tokenizer, example["question"], context=context, max_tokens=args.direct_text_tokens)[:2], True)
    if condition == "full_context":
        ids, mask, _ = _prompt(
            tokenizer,
            example["question"],
            context=example["source"],
            max_tokens=args.full_context_tokens,
        )
        source_tokens = len(tokenizer(example["source"], add_special_tokens=False).input_ids)
        complete = source_tokens + args.prompt_tokens <= args.full_context_tokens
        return ids, mask, complete
    ids, mask, _ = _prompt(
        tokenizer, example["question"], max_tokens=args.prompt_tokens
    )
    return ids, mask, True


def _condition_row(
    handle,
    tokenizer,
    example: dict,
    feature: dict,
    entry,
    condition: str,
    selected: list[int] | None,
    args,
    device: torch.device,
) -> dict:
    if condition in MEMORY_CONDITIONS:
        fixed = _fixed_selection(entry, args.layer, selected or [], feature["local_spans"])
        handle.configure_memory_layers({args.layer}, fixed_selections={args.layer: fixed})
    elif condition == "empty_memory":
        handle.configure_memory_layers({args.layer}, fixed_selections={args.layer: [[]]})
    else:
        handle.configure_memory_layers(set())
    prompt_ids, prompt_mask, eligible = _prompt_for_condition(
        tokenizer, example, condition, args
    )
    base = {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "condition": condition,
        "eligible": eligible,
        "question": example["question"],
        "answer": example["answer"],
        "selected_chunks": "" if selected is None else " ".join(map(str, selected)),
        "requested_chunks": 0 if selected is None else len(selected),
    }
    if not eligible:
        return base
    answer_ids = _answer_ids(tokenizer, example["answer"])
    scored = _teacher_forced(
        handle.model, prompt_ids, prompt_mask, answer_ids, device
    )
    prediction, generation_seconds = _generate(
        handle, tokenizer, prompt_ids, prompt_mask, device, args.new_tokens
    )
    diagnostics = handle.diagnostics_by_layer().get(args.layer, {})
    materialized = int(diagnostics.get("retrieved_physical_kv_tokens", 0))
    expected = (
        sum(int(feature["local_token_mask"][index].sum()) for index in (selected or []))
        if condition in MEMORY_CONDITIONS
        else 0
    )
    if condition in MEMORY_CONDITIONS and materialized != expected:
        raise AssertionError(
            f"Materialized {materialized} native K/V tokens; expected {expected}."
        )
    return {
        **base,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "answer_tokens": int(answer_ids.shape[1]),
        "generated_answer": prediction,
        "generation_seconds": generation_seconds,
        "materialized_native_kv_tokens": materialized,
        "active_memory_fraction": materialized / max(int(feature["source_tokens"]), 1),
        "native_limit_violations": handle.native_limit_violations,
        **scored,
        **answer_metrics(prediction, example["answer"]),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in ("combined", "hotpotqa", "qasper"):
        for condition in ALL_CONDITIONS:
            group = [
                row
                for row in rows
                if row["condition"] == condition
                and row.get("eligible")
                and (dataset == "combined" or row["dataset"] == dataset)
            ]
            if not group:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "examples": len(group),
                    **{
                        metric: statistics.fmean(float(row.get(metric, 0.0)) for row in group)
                        for metric in (
                            "gold_sequence_logprob",
                            "gold_mean_token_logprob",
                            "gold_logprob_delta_vs_none",
                            "gold_mean_logprob_delta_vs_none",
                            "gold_first_token_probability",
                            "gold_first_token_rank",
                            "f1",
                            "em",
                            "answer_contained",
                            "materialized_native_kv_tokens",
                            "active_memory_fraction",
                            "teacher_forced_seconds",
                            "generation_seconds",
                            "native_limit_violations",
                        )
                    },
                }
            )
    return output


def _paired(rows: list[dict], seed: int) -> list[dict]:
    lookup = {
        (row["dataset"], row["example_id"], row["condition"]): row
        for row in rows
        if row.get("eligible")
    }
    output = []
    comparisons = [
        (condition, "no_memory")
        for condition in ALL_CONDITIONS
        if condition != "no_memory"
    ]
    comparisons.extend(
        (condition, baseline)
        for condition in (
            "native_mean",
            "exact",
            "bm25",
            "hybrid",
            "lowrank_r16_ensemble",
            "lowrank_r8_kmeans_ensemble",
            "oracle_evidence",
        )
        for baseline in ("shuffled_selection", "irrelevant_bottom")
    )
    comparisons.extend(
        (condition, baseline)
        for condition in ("lowrank_r16_ensemble", "lowrank_r8_kmeans_ensemble")
        for baseline in ("native_mean", "exact", "hybrid")
    )
    for dataset in ("hotpotqa", "qasper"):
        identities = sorted(
            {key[1] for key in lookup if key[0] == dataset and key[2] == "no_memory"}
        )
        for condition, baseline_name in comparisons:
            for metric in ("gold_mean_token_logprob", "f1", "em"):
                differences = []
                for example_id in identities:
                    current = lookup.get((dataset, example_id, condition))
                    baseline = lookup.get((dataset, example_id, baseline_name))
                    if current is not None and baseline is not None:
                        differences.append(float(current[metric]) - float(baseline[metric]))
                if not differences:
                    continue
                low, high = _bootstrap(
                    differences,
                    seed
                    + sum(map(ord, dataset + condition + baseline_name + metric)),
                )
                output.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "baseline": baseline_name,
                        "metric": metric,
                        "pairs": len(differences),
                        "mean_delta": statistics.fmean(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": sum(value > 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                        "losses": sum(value < 0 for value in differences),
                    }
                )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    shown = (
        "no_memory",
        "native_mean",
        "exact",
        "bm25",
        "hybrid",
        "lowrank_r16_ensemble",
        "lowrank_r8_kmeans_ensemble",
        "shuffled_selection",
        "irrelevant_bottom",
        "oracle_evidence",
        "direct_evidence",
    )
    labels = ("None", "Mean", "Exact", "BM25", "Hybrid", "LR16", "LR8", "Random", "Bottom", "Oracle", "Evidence")
    figure, axes = plt.subplots(2, 2, figsize=(13, 7.2), constrained_layout=True)
    for column, dataset in enumerate(("hotpotqa", "qasper")):
        lookup = {
            row["condition"]: row
            for row in summary
            if row["dataset"] == dataset and row["condition"] in shown
        }
        axes[0, column].bar(labels, [lookup[name]["gold_mean_logprob_delta_vs_none"] for name in shown])
        axes[1, column].bar(labels, [lookup[name]["f1"] for name in shown])
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].set_title(dataset.upper())
        axes[0, column].set_ylabel("Gold mean-token logP delta")
        axes[1, column].set_ylabel("Generated token F1")
        for axis in axes[:, column]:
            axis.tick_params(axis="x", rotation=38)
            axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"confirmation_generation.{suffix}", dpi=190)
    plt.close(figure)


def _is_eligible(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _finalize(args: argparse.Namespace, rows: list[dict]) -> dict:
    """Rebuild paired deltas and aggregates from the durable row checkpoint."""
    baselines = {
        (row["dataset"], row["example_id"]): row
        for row in rows
        if row["condition"] == "no_memory" and _is_eligible(row.get("eligible"))
    }
    for row in rows:
        if not _is_eligible(row.get("eligible")):
            continue
        baseline = baselines[(row["dataset"], row["example_id"])]
        row["gold_logprob_delta_vs_none"] = float(row["gold_sequence_logprob"]) - float(
            baseline["gold_sequence_logprob"]
        )
        row["gold_mean_logprob_delta_vs_none"] = float(
            row["gold_mean_token_logprob"]
        ) - float(baseline["gold_mean_token_logprob"])
    _write_csv(args.output_dir / "per_example.csv", rows)
    typed = [
        {
            key: (
                value
                if key
                in {
                    "dataset",
                    "example_id",
                    "condition",
                    "question",
                    "answer",
                    "generated_answer",
                    "selected_chunks",
                    "skip_reason",
                }
                else (_is_eligible(value) if key == "eligible" else value)
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    summary = _aggregate(typed)
    paired = _paired(typed, args.bootstrap_seed)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "paired_effects.csv", paired)
    _plot(summary, args.output_dir)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": args.layer,
        "examples_per_dataset": args.examples_per_dataset,
        "conditions": list(ALL_CONDITIONS),
        "selection_artifact": str(args.selection_rows.resolve().relative_to(ROOT)),
        "exact_four_chunk_budget": True,
        "materialized_native_kv_unchanged": True,
        "teacher_forced_gold_logprob_primary": True,
        "generated_em_f1_secondary": True,
        "full_context_token_limit": args.full_context_tokens,
        "command": (
            "python experiments/paper2_8_qk_compression/"
            "run_confirmation_generation.py --device cuda"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired)}


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.output_dir / "per_example.csv"
    if args.finalize_only:
        if not row_path.exists():
            raise FileNotFoundError(f"No row checkpoint to finalize: {row_path}")
        return _finalize(args, _read_csv(row_path))
    print(f"[startup] loading tokenizer {MODEL_ID}@{MODEL_REVISION}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print(f"[startup] loading frozen model on {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7:
        model.generation_config.disable_compile = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"[startup] injecting PRA at layer {args.layer}", flush=True)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(args.layer,),
            model_max_context_tokens=args.full_context_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=256,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=128,
            top_k_references=1,
            top_k_chunks_per_reference=4,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
    )
    print(f"[startup] loading frozen selections from {args.selection_rows}", flush=True)
    selections = _selected_lookup(_read_csv(args.selection_rows))
    print(f"[startup] loading native-QK features from {args.features}", flush=True)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    features_by_id = {(row["dataset"], row["example_id"]): row for row in features}
    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.offset, args.dataset_seed
    )
    existing = _read_csv(row_path) if row_path.exists() and not args.overwrite else []
    completed = {(row["dataset"], row["example_id"], row["condition"]) for row in existing}
    rows = existing
    print(
        f"[startup] resuming with {len(completed)} completed condition rows",
        flush=True,
    )
    for index, example in enumerate(examples, start=1):
        identity = (example["dataset"], example["id"])
        if all((*identity, condition) in completed for condition in ALL_CONDITIONS):
            print(
                f"[generation {index}/{len(examples)}] {example['dataset']} "
                f"{example['id']} [resume: complete]",
                flush=True,
            )
            continue
        feature = features_by_id[(example["dataset"], example["id"])]
        handle.cache.clear()
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        entry = handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source_ids,
            text=example["source"],
        )
        example_rows = []
        for condition in ALL_CONDITIONS:
            key = (example["dataset"], example["id"], condition)
            if key in completed:
                continue
            selected = selections.get(key) if condition in MEMORY_CONDITIONS else None
            if condition in MEMORY_CONDITIONS and selected is None:
                raise KeyError(f"Missing frozen selection for {key}")
            try:
                row = _condition_row(
                    handle,
                    tokenizer,
                    example,
                    feature,
                    entry,
                    condition,
                    selected,
                    args,
                    device,
                )
            except torch.OutOfMemoryError:
                if condition != "full_context":
                    raise
                # Full-context is an upper-bound control, not a PRA condition.
                # Preserve the bounded run when eager O(T^2) attention exceeds
                # the evaluation GPU's physical capacity.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                row = {
                    "dataset": example["dataset"],
                    "example_id": example["id"],
                    "condition": condition,
                    "eligible": False,
                    "skip_reason": "cuda_oom_full_context_control",
                    "question": example["question"],
                    "answer": example["answer"],
                    "selected_chunks": "",
                    "requested_chunks": 0,
                }
                print(
                    f"[generation {index}/{len(examples)}] full-context OOM; "
                    "recording the control as ineligible",
                    flush=True,
                )
            example_rows.append(row)
            rows.append(row)
            _write_csv(row_path, rows)
        eligible = [row for row in example_rows if row.get("eligible")]
        baseline = next(
            (row for row in rows if row["dataset"] == example["dataset"] and row["example_id"] == example["id"] and row["condition"] == "no_memory"),
            None,
        )
        if baseline is None:
            raise RuntimeError("No-memory baseline is required for paired deltas.")
        for row in eligible:
            row["gold_logprob_delta_vs_none"] = float(row["gold_sequence_logprob"]) - float(baseline["gold_sequence_logprob"])
            row["gold_mean_logprob_delta_vs_none"] = float(row["gold_mean_token_logprob"]) - float(baseline["gold_mean_token_logprob"])
        _write_csv(row_path, rows)
        print(
            f"[generation {index}/{len(examples)}] {example['dataset']} {example['id']} "
            + " ".join(
                f"{row['condition']}={row.get('gold_mean_logprob_delta_vs_none', float('nan')):+.3f}"
                for row in eligible
            ),
            flush=True,
        )
    return _finalize(args, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--offset", type=int, default=CONFIRMATION_OFFSET)
    parser.add_argument("--examples-per-dataset", type=int, default=CONFIRMATION_PER_DATASET)
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=1024)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--features",
        type=Path,
        default=RESULT_ROOT / "native_qk_features_confirmation.pt",
    )
    parser.add_argument(
        "--selection-rows",
        type=Path,
        default=RESULT_ROOT / "confirmation/per_example.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULT_ROOT / "confirmation_generation"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
