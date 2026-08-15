"""Evaluate one-shot and evolved-state PRA on trained LocalSA controls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from pra_torch.controlled_local_sa import (
    ControlledExample,
    ControlledTokenizer,
    controlled_examples,
)
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra

from experiments.paper2_5_iterative_pra.run_controlled_local_sa import (
    DEFAULT_OUTPUT,
    SEEDS,
    _append_or_replace,
    _read_csv,
    model_config,
    parse_windows,
    window_name,
)


def _pra_patterns(n_layers: int) -> dict[str, tuple[int, ...]]:
    """Return reproducible consumer placements for one-shot and spacing controls."""
    matched = tuple(
        sorted({0, n_layers // 3, (2 * n_layers) // 3, n_layers - 1})
    )
    return {
        "one_shot": (0,),
        "iterative_matched": matched,
        "late_only": (n_layers - 1,),
        "spacing_1": tuple(range(n_layers)),
        "spacing_2": tuple(range(0, n_layers, 2)),
        "spacing_4": tuple(range(0, n_layers, 4)),
        "spacing_8": (0,),
    }


def _put_references(
    model: TinyPRAModel,
    example: ControlledExample,
    tokenizer: ControlledTokenizer,
    device: str,
) -> None:
    """Build cache entries from observable fact content only.

    Gold path annotations are deliberately absent from cache metadata and from
    every routing call; they are consulted only after inference for scoring.
    """
    model.clear_pra_cache()
    for reference in example.references:
        entry = model.encode_reference_tokens_to_cache(
            reference.uri,
            reference.token_ids,
            tokenizer,
            device,
            metadata={"controlled_fact": True},
        )
        model.pra_cache.put(entry)


@torch.no_grad()
def precompute_reference_entries(
    source: TinyPRAModel,
    examples: list[ControlledExample],
    tokenizer: ControlledTokenizer,
    device: str,
) -> dict[str, list]:
    """Encode exact layer-native K/V once for every later PRA placement.

    Disabled-memory SA/PRA conversion is logit-equivalent, and all consumer
    placements share the same converted projections. Extra layer payloads in a
    cache entry are inert when a condition has no PRA consumer at that layer.
    """
    encoder_cfg = replace(
        source.cfg,
        model_variant="td_layered_pra",
        pra_layer_ids=tuple(range(source.cfg.n_layers)),
        trigger_threshold=-1.0,
    )
    encoder = convert_sa_model_to_pra(source, encoder_cfg).to(device).eval()
    encoded = {}
    for example in examples:
        entries = []
        for reference in example.references:
            entries.append(
                encoder.encode_reference_tokens_to_cache(
                    reference.uri,
                    reference.token_ids,
                    tokenizer,
                    device,
                    metadata={"controlled_fact": True},
                )
            )
        encoded[example.example_id] = entries
    return encoded


def _put_precomputed_references(model: TinyPRAModel, entries: list) -> None:
    """Publish immutable precomputed entries into one condition-local cache."""
    model.clear_pra_cache()
    for entry in entries:
        model.pra_cache.put(entry)


@torch.no_grad()
def _sa_prediction(
    model: TinyPRAModel,
    token_ids: tuple[int, ...],
    answer_id: int,
    device: str,
) -> tuple[int, float]:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    logits = model(ids, use_pra_memory=False)[0, -1]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return int(logits.argmax()) == answer_id, elapsed


@torch.no_grad()
def evaluate_condition(
    source: TinyPRAModel,
    examples: list[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    condition: str,
    layer_ids: tuple[int, ...],
    device: str,
    top_k_references: int,
    materialization: str,
    entries_by_example: dict[str, list] | None = None,
) -> list[dict]:
    """Convert exact SA weights, route content, and score after unblinding."""
    target_cfg = replace(
        source.cfg,
        model_variant="td_layered_pra",
        pra_layer_ids=layer_ids,
        top_k_references=top_k_references,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        detail_materialization=materialization,
        max_materialized_memory_tokens=20,
        collect_routing_metrics=True,
        collect_rank_diagnostics=True,
    )
    model = convert_sa_model_to_pra(source, target_cfg).to(device).eval()
    rows = []
    for example in examples:
        if entries_by_example is None:
            _put_references(model, example, tokenizer, device)
        else:
            _put_precomputed_references(model, entries_by_example[example.example_id])
        ids = torch.tensor([example.query_input_ids], dtype=torch.long, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits, trace = model.forward_progressive_pra(ids, prevent_reference_replay=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        selected = [
            uri
            for layer in trace
            for uri in layer["selected_reference_uris"]
        ]
        selected_unique = list(dict.fromkeys(selected))
        gold = list(example.target_reference_uris)
        selected_set = set(selected_unique)
        prefix = 0
        for selected_uri, target_uri in zip(selected_unique, gold):
            if selected_uri != target_uri:
                break
            prefix += 1
        correct = int(int(logits[0, -1].argmax()) == example.answer_id)
        materialized = sum(int(layer["materialized_tokens"]) for layer in trace)
        active_memory_fractions = [
            float(layer["active_memory_fraction"]) for layer in trace
        ]
        memory_local_ratios = [
            float(layer["memory_to_local_output_norm_ratio"]) for layer in trace
        ]
        final_memory_masses = [
            float(layer["final_token_memory_attention_mass"]) for layer in trace
        ]
        output_divergence_ratios = [
            float(layer["pra_output_divergence_ratio"]) for layer in trace
        ]
        state_changes = []
        for previous, current in zip(trace, trace[1:]):
            state_changes.append(
                float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        previous["query_state"], current["query_state"], dim=-1
                    ).mean()
                )
            )
        rows.append(
            {
                "condition": condition,
                "materialization": materialization,
                "example_id": example.example_id,
                "depth": example.depth,
                "correct": correct,
                "selected_reference_count": len(selected_unique),
                "target_reference_count": len(gold),
                "reference_recall": len(selected_set & set(gold)) / max(len(gold), 1),
                "complete_path_recovery": int(set(gold) <= selected_set),
                "ordered_prefix_recovery": prefix / max(len(gold), 1),
                "first_hop_hit": int(bool(gold) and gold[0] in selected_set),
                "pra_consumer_layers": len(layer_ids),
                "search_layers": len(trace),
                "materialized_native_kv_tokens": materialized,
                "layer_token_kv_states": materialized,
                "latency_seconds": elapsed,
                "mean_intervention_state_displacement": (
                    statistics.fmean(state_changes) if state_changes else 0.0
                ),
                "mean_active_memory_fraction": (
                    statistics.fmean(active_memory_fractions)
                    if active_memory_fractions
                    else 0.0
                ),
                "mean_memory_to_local_output_norm_ratio": (
                    statistics.fmean(memory_local_ratios)
                    if memory_local_ratios
                    else 0.0
                ),
                "mean_final_token_memory_attention_mass": (
                    statistics.fmean(final_memory_masses)
                    if final_memory_masses
                    else 0.0
                ),
                "mean_pra_output_divergence_ratio": (
                    statistics.fmean(output_divergence_ratios)
                    if output_divergence_ratios
                    else 0.0
                ),
                "replay_count": sum(
                    len(layer["replayed_reference_uris"]) for layer in trace
                ),
                "selected_reference_uris": json.dumps(selected_unique),
            }
        )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Aggregate per-example PRA outcomes without collapsing the five seeds."""
    def number(value) -> float:
        if isinstance(value, bool):
            return float(value)
        text = str(value).strip().lower()
        if text in {"true", "false"}:
            return float(text == "true")
        return float(value)

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["window"],
            int(row["seed"]),
            row["condition"],
            row["materialization"],
            int(row["depth"]),
        )
        groups.setdefault(key, []).append(row)
    metrics = (
        "correct",
        "reference_recall",
        "complete_path_recovery",
        "ordered_prefix_recovery",
        "first_hop_hit",
        "selected_reference_count",
        "materialized_native_kv_tokens",
        "layer_token_kv_states",
        "latency_seconds",
        "mean_intervention_state_displacement",
        "mean_active_memory_fraction",
        "mean_memory_to_local_output_norm_ratio",
        "mean_final_token_memory_attention_mass",
        "mean_pra_output_divergence_ratio",
        "replay_count",
    )
    output = []
    for (window, seed, condition, materialization, depth), group in sorted(
        groups.items(), key=str
    ):
        output.append(
            {
                "window": window,
                "seed": seed,
                "condition": condition,
                "materialization": materialization,
                "depth": depth,
                "examples": len(group),
                **{
                    metric: statistics.fmean(number(row[metric]) for row in group)
                    for metric in metrics
                },
            }
        )
    return output


def _baseline_rows(
    model: TinyPRAModel,
    examples: list[ControlledExample],
    *,
    device: str,
) -> list[dict]:
    rows = []
    for example in examples:
        full_correct, full_latency = _sa_prediction(
            model, example.full_input_ids, example.answer_id, device
        )
        query_correct, query_latency = _sa_prediction(
            model, example.query_input_ids, example.answer_id, device
        )
        for condition, correct, latency in (
            ("local_sa_full_context", full_correct, full_latency),
            ("local_sa_query_only", query_correct, query_latency),
        ):
            rows.append(
                {
                    "condition": condition,
                    "materialization": "none",
                    "example_id": example.example_id,
                    "depth": example.depth,
                    "correct": correct,
                    "selected_reference_count": 0,
                    "target_reference_count": example.depth,
                    "reference_recall": 0.0,
                    "complete_path_recovery": 0,
                    "ordered_prefix_recovery": 0.0,
                    "first_hop_hit": 0,
                    "pra_consumer_layers": 0,
                    "search_layers": 0,
                    "materialized_native_kv_tokens": 0,
                    "layer_token_kv_states": 0,
                    "latency_seconds": latency,
                    "mean_intervention_state_displacement": 0.0,
                    "mean_active_memory_fraction": 0.0,
                    "mean_memory_to_local_output_norm_ratio": 0.0,
                    "mean_final_token_memory_attention_mass": 0.0,
                    "mean_pra_output_divergence_ratio": 0.0,
                    "replay_count": 0,
                    "selected_reference_uris": "[]",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", default="16,32,64,128,global")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--examples", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    tokenizer = ControlledTokenizer()
    patterns = _pra_patterns(args.layers)
    raw_path = args.output_dir / "local_pra_one_shot_iterative_rows.csv"

    for window in parse_windows(args.windows):
        for seed in [int(value) for value in args.seeds.split(",")]:
            checkpoint = args.output_dir / "checkpoints" / f"{window_name(window)}_seed{seed}.pt"
            if not checkpoint.exists():
                print(f"skip missing {checkpoint}")
                continue
            cfg = model_config(
                tokenizer,
                window=window,
                device=args.device,
                d_model=args.d_model,
                n_layers=args.layers,
            )
            source = TinyPRAModel(cfg).to(args.device).eval()
            source.load_state_dict(
                torch.load(checkpoint, map_location=args.device, weights_only=False)["model"]
            )
            examples = controlled_examples(
                tokenizer,
                count=args.examples,
                seed=100_004,
                depths=(1, 2, 3, 4, 8),
                distractors=(4, 8),
                evidence_gaps=(0, 2, 6),
                lexical_overlaps=(0.0, 0.5, 1.0),
                relation_types=(4, 8, 15),
                branchings=(0, 1, 2),
            )
            rows = _baseline_rows(source, examples, device=args.device)
            entries_by_example = precompute_reference_entries(
                source, examples, tokenizer, args.device
            )
            for condition, layer_ids in patterns.items():
                top_k = 4 if condition in {"one_shot", "late_only"} else 1
                rows.extend(
                    evaluate_condition(
                        source,
                        examples,
                        tokenizer,
                        condition=condition,
                        layer_ids=layer_ids,
                        device=args.device,
                        top_k_references=top_k,
                        materialization="selected_chunks",
                        entries_by_example=entries_by_example,
                    )
                )
            # Whole-parent is an identity control here: each controlled URI is
            # exactly one indivisible fact, making payload equality auditable.
            rows.extend(
                evaluate_condition(
                    source,
                    examples,
                    tokenizer,
                    condition="one_shot_whole_parent",
                    layer_ids=patterns["one_shot"],
                    device=args.device,
                    top_k_references=4,
                    materialization="full_reference",
                    entries_by_example=entries_by_example,
                )
            )
            rows = [
                {"window": window_name(window), "seed": seed, **row}
                for row in rows
            ]
            _append_or_replace(
                raw_path,
                rows,
                ("window", "seed", "condition", "materialization", "example_id"),
            )
            print(f"completed PRA {window_name(window)} seed={seed}")

    raw = _read_csv(raw_path)
    aggregates = aggregate_rows(raw)
    _write_targets(args.output_dir, aggregates)


def _write_targets(output_dir: Path, aggregates: list[dict]) -> None:
    """Emit the guide's required projections from one auditable result table."""
    fieldnames = sorted({key for row in aggregates for key in row})

    def write(name: str, selected: list[dict]) -> None:
        path = output_dir / name
        if not selected:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    write("local_pra_one_shot_iterative.csv", aggregates)
    spacing = [row for row in aggregates if row["condition"].startswith("spacing_")]
    write("pra_spacing_results.csv", spacing)
    write("window_spacing_interaction.csv", spacing)
    write(
        "local_vs_global_graph_metrics.csv",
        [
            row
            for row in aggregates
            if row["condition"]
            in {
                "local_sa_full_context",
                "one_shot",
                "iterative_matched",
                "spacing_1",
                "spacing_2",
            }
        ],
    )


if __name__ == "__main__":
    main()
