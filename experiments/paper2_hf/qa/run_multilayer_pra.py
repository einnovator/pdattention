"""Measure frozen-Qwen PRA consumption as a function of transformer depth."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_oracle_memory_use import (
    _answer_ids,
    _hidden_deltas,
    _mean,
    _oracle_selections,
    _prompt,
    _teacher_forced,
    _write_csv,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from common.recall_sparsity import recall_sparsity_curve
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is present in the experiment environment.
    psutil = None


@dataclass(frozen=True)
class Condition:
    """One memory intervention with an explicit consumption-layer schedule."""

    name: str
    kind: str
    schedule: str
    layers: tuple[int, ...]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes() -> int | None:
    return psutil.Process().memory_info().rss if psutil is not None else None


def layer_schedules(layer_count: int) -> dict[str, tuple[int, ...]]:
    """Resolve canonical depth and placement schedules for one decoder stack."""
    if layer_count < 8:
        raise ValueError("The Paper 2 layer sweep expects at least eight decoder layers.")

    def last(count: int) -> tuple[int, ...]:
        return tuple(range(layer_count - count, layer_count))

    quarter = max(1, math.ceil(layer_count * 0.25))
    half = max(1, math.ceil(layer_count * 0.50))
    middle_start = (layer_count - 4) // 2
    evenly_spaced = tuple(
        sorted({round(index * (layer_count - 1) / 3) for index in range(4)})
    )
    every_four = tuple(range(0, layer_count, 4))

    def evenly(count: int) -> tuple[int, ...]:
        return tuple(
            sorted({round(index * (layer_count - 1) / (count - 1)) for index in range(count)})
        )
    return {
        "last_1": last(1),
        "last_2": last(2),
        "last_4": last(4),
        "last_8": last(8),
        "last_12": last(min(12, layer_count)),
        "last_14": last(min(14, layer_count)),
        "last_16": last(min(16, layer_count)),
        "last_20": last(min(20, layer_count)),
        "last_24": last(min(24, layer_count)),
        "last_quarter": last(quarter),
        "last_half": last(half),
        "all": tuple(range(layer_count)),
        "early_4": tuple(range(4)),
        "middle_4": tuple(range(middle_start, middle_start + 4)),
        "even_4": evenly_spaced,
        "even_8": evenly(8),
        "every_4": every_four,
    }


class _FirstTokenClock(StoppingCriteria):
    """Record synchronized time when greedy generation emits its first token."""

    def __init__(self, device: torch.device, started: float) -> None:
        self.device = device
        self.started = started
        self.first_token_seconds: float | None = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.first_token_seconds is None:
            _sync(self.device)
            self.first_token_seconds = time.perf_counter() - self.started
        return False


def _generate_timed(
    handle,
    tokenizer,
    prompt_ids,
    prompt_mask,
    device,
    new_tokens,
    *,
    position_offset: int = 0,
):
    encoded = {
        "input_ids": prompt_ids.to(device),
        "attention_mask": prompt_mask.to(device),
        "position_ids": torch.arange(
            position_offset,
            position_offset + prompt_ids.shape[1],
            device=device,
        ).unsqueeze(0),
    }
    _sync(device)
    started = time.perf_counter()
    clock = _FirstTokenClock(device, started)
    with torch.no_grad():
        output = handle.model.generate(
            **encoded,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=[clock],
        )
    _sync(device)
    duration = time.perf_counter() - started
    continuation = output[0, prompt_ids.shape[1] :]
    return {
        "generated_answer": tokenizer.decode(
            continuation, skip_special_tokens=True
        ).strip(),
        "ttft_seconds": clock.first_token_seconds,
        "generation_seconds": duration,
    }


def _cache_bytes(entry, layers: tuple[int, ...]) -> dict:
    per_layer = {}
    for layer in layers:
        chunks = entry.layer_memory[layer].chunks
        per_layer[str(layer)] = {
            "detail_kv_bytes": sum(
                int(chunk.metadata["detail_kv_bytes"]) for chunk in chunks
            ),
            "routing_index_bytes": sum(
                int(chunk.metadata["routing_gist_bytes"]) for chunk in chunks
            ),
            "detail_tokens": sum(chunk.token_count for chunk in chunks),
            "chunks": len(chunks),
        }
    return {
        "per_layer": per_layer,
        "detail_kv_bytes": sum(row["detail_kv_bytes"] for row in per_layer.values()),
        "routing_index_bytes": sum(
            row["routing_index_bytes"] for row in per_layer.values()
        ),
        "detail_tokens_across_layers": sum(
            row["detail_tokens"] for row in per_layer.values()
        ),
    }


def _route_once(
    handle,
    tokenizer,
    question,
    route_layer,
    prompt_tokens,
    device,
    *,
    position_offset: int = 0,
):
    """Route at one validated layer, returning IDs that can be replayed elsewhere."""
    prompt_ids, prompt_mask, _ = _prompt(
        tokenizer, question, max_tokens=prompt_tokens
    )
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    positions = torch.arange(
        position_offset,
        position_offset + prompt_ids.shape[1],
        device=device,
    ).unsqueeze(0)
    adapter = handle.adapters[route_layer]
    handle.configure_memory_layers(set())
    adapter.begin_capture(positions)
    _sync(device)
    query_started = time.perf_counter()
    with torch.no_grad():
        handle.model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            position_ids=positions,
            use_cache=False,
        )
    _sync(device)
    query_seconds = time.perf_counter() - query_started
    captured = adapter.consume_capture()
    query = adapter._routing_query_states(
        captured.hidden_states,
        captured.pre_query,
        captured.post_query,
    )
    _sync(device)
    routing_started = time.perf_counter()
    selected, rankings = adapter.pra_core.route_memory(query)
    _sync(device)
    return {
        "selected": selected,
        "rankings": rankings,
        "query_encoding_seconds": query_seconds,
        "routing_seconds": time.perf_counter() - routing_started,
        "route_layer": route_layer,
    }


def _selection_overlap(selected, evidence_spans) -> dict:
    spans = [(hit.logical_start, hit.logical_end) for hit in selected[0]]
    covered = [
        any(max(start, left) < min(end, right) for left, right in spans)
        for start, end in evidence_spans
    ]
    return {
        "routing_any_evidence_recall": float(any(covered)),
        "routing_all_evidence_recall": float(bool(covered) and all(covered)),
        "routing_target_coverage": sum(covered) / max(len(covered), 1),
        "routed_chunk_ids": [hit.chunk_id for hit in selected[0]],
        "routed_spans": spans,
    }


def _routing_ranking_trace(rankings, selected, evidence_spans) -> dict:
    """Normalize the route-once ranking for exact parent-identity recall curves."""
    references = rankings[0] if rankings else []
    ranked_chunks = [
        chunk
        for reference in references
        for chunk in reference.get("chunks", [])
    ]
    ranked_ids = [str(chunk["chunk_id"]) for chunk in ranked_chunks]
    evidence_ids = {
        str(chunk["chunk_id"])
        for chunk in ranked_chunks
        if any(
            max(int(chunk["token_start"]), start)
            < min(int(chunk["token_end"]), end)
            for start, end in evidence_spans
        )
    }
    token_lengths = [
        int(chunk["token_end"]) - int(chunk["token_start"])
        for chunk in ranked_chunks
    ]
    ranks = [index + 1 for index, chunk_id in enumerate(ranked_ids) if chunk_id in evidence_ids]
    return {
        "routing_candidate_chunks": len(ranked_ids),
        "routing_selected_parent_chunks": len(selected[0]),
        "routing_selected_parent_fraction": len(selected[0]) / max(len(ranked_ids), 1),
        "routing_best_evidence_rank": min(ranks, default=None),
        "routing_ranked_chunk_ids": ranked_ids,
        "routing_evidence_chunk_ids": sorted(evidence_ids),
        "routing_ranked_chunk_token_lengths": token_lengths,
        "routing_complete_rankings": references,
    }


def _diagnostic_sums(diagnostics: dict, active_layers: tuple[int, ...]) -> dict:
    rows = [diagnostics[layer] for layer in active_layers]

    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows)

    return {
        "aggregate_materialized_kv_tokens": total(
            "retrieved_physical_kv_tokens"
        ),
        "aggregate_transferred_kv_bytes": total("retrieved_kv_transfer_bytes"),
        "aggregate_routing_seconds": total("routing_duration_seconds"),
        "aggregate_materialization_seconds": total(
            "materialization_duration_seconds"
        ),
        "aggregate_transfer_seconds": total(
            "selected_kv_transfer_duration_seconds"
        ),
        "aggregate_memory_attention_seconds": total(
            "memory_attention_duration_seconds"
        ),
        "chunks_budget_rejected": total("chunks_budget_rejected"),
    }


def _evaluate_example(
    handle,
    tokenizer,
    example,
    entry,
    evidence_spans,
    conditions,
    route_once,
    cache_bytes,
    args,
    device,
    source_tokens,
):
    rows = []
    baseline_hidden = None
    oracle = {
        layer: [_oracle_selections(entry, layer, evidence_spans)]
        for layer in handle.adapters
    }
    route_overlap = (
        _selection_overlap(route_once["selected"], evidence_spans)
        if route_once is not None
        else {}
    )
    route_ranking = (
        _routing_ranking_trace(
            route_once["rankings"], route_once["selected"], evidence_spans
        )
        if route_once is not None
        else {}
    )
    for condition in conditions:
        context = "\n".join(example["evidence"]) if condition.kind == "direct" else None
        prompt_ids, prompt_mask, context_tokens = _prompt(
            tokenizer,
            example["question"],
            context=context,
            max_tokens=args.direct_text_tokens if context else args.prompt_tokens,
        )
        answer_ids = _answer_ids(tokenizer, example["answer"])
        position_offset = (
            source_tokens
            if args.position_mode == "corrected" and condition.kind != "direct"
            else 0
        )
        if condition.kind == "oracle":
            fixed = {layer: oracle[layer] for layer in condition.layers}
        elif condition.kind == "routed":
            if route_once is None:
                raise RuntimeError("Routed conditions require a route-once selection.")
            fixed = handle.map_chunk_identities_to_layers(
                route_once["selected"], condition.layers
            )
        else:
            fixed = None
        handle.configure_memory_layers(set(condition.layers), fixed_selections=fixed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        scored, hidden = _teacher_forced(
            handle,
            tokenizer,
            prompt_ids,
            prompt_mask,
            answer_ids,
            context_tokens,
            evidence_spans,
            device,
            position_offset=position_offset,
        )
        if condition.kind == "none":
            baseline_hidden = hidden
        generated = _generate_timed(
            handle,
            tokenizer,
            prompt_ids,
            prompt_mask,
            device,
            args.new_tokens,
            position_offset=position_offset,
        )
        diagnostics = handle.diagnostics_by_layer()
        sums = _diagnostic_sums(diagnostics, condition.layers)
        attention_rows = scored["attention_by_layer"]
        memory_mass = _mean(
            [row["memory_attention_mass"] for row in attention_rows.values()]
        )
        evidence_mass = _mean(
            [row["evidence_attention_mass"] for row in attention_rows.values()]
        )
        per_layer_attention = [
            {
                "layer": int(layer),
                "memory_attention_mass": trace["memory_attention_mass"],
                "evidence_attention_mass": trace["evidence_attention_mass"],
                "local_attention_mass": (
                    None
                    if trace["memory_attention_mass"] is None
                    else 1.0 - trace["memory_attention_mass"]
                ),
            }
            for layer, trace in attention_rows.items()
        ]
        selected_by_layer = {
            str(layer): [hit.as_trace_dict() for hit in adapter.last_selected_chunks[0]]
            for layer, adapter in handle.adapters.items()
            if adapter.memory_enabled
        }
        if condition.kind == "oracle":
            evidence_guaranteed = all(
                {hit.chunk_id for hit in oracle[layer][0]}.issubset(
                    {hit["chunk_id"] for hit in selected_by_layer[str(layer)]}
                )
                for layer in condition.layers
            )
        else:
            evidence_guaranteed = condition.kind == "direct"
        peak_gpu = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        row = {
            "dataset": example["dataset"],
            "example_id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "condition": condition.name,
            "kind": condition.kind,
            "schedule": condition.schedule,
            "pra_layers": list(condition.layers),
            "pra_layer_count": len(condition.layers),
            "pra_layer_fraction": len(condition.layers) / len(handle.adapters),
            "first_pra_layer": min(condition.layers, default=None),
            "last_pra_layer": max(condition.layers, default=None),
            "position_mode": args.position_mode,
            "query_position_offset": position_offset,
            "reference_position_policy": "source_relative",
            "memory_lifetime_policy": "request_prefill_and_decode",
            "prompt_tokens": int(prompt_ids.shape[1]),
            "answer_tokens": int(answer_ids.shape[1]),
            "evidence_guaranteed": evidence_guaranteed,
            "evidence_parent_chunks": max(
                (len(oracle[layer][0]) for layer in condition.layers), default=0
            ),
            "memory_attention_mass": memory_mass,
            "evidence_attention_mass": evidence_mass,
            "direct_or_local_attention_mass": scored[
                "direct_context_attention_mass"
            ],
            "attention_by_layer": per_layer_attention,
            "hidden_state_delta_by_layer": (
                []
                if baseline_hidden is None or condition.kind in {"none", "direct"}
                else _hidden_deltas(hidden, baseline_hidden)
            ),
            "selected_by_layer": selected_by_layer,
            "diagnostics_by_layer": diagnostics,
            "resident_reference_detail_kv_bytes": cache_bytes["detail_kv_bytes"],
            "resident_routing_index_bytes": cache_bytes["routing_index_bytes"],
            "route_layer_routing_index_bytes": cache_bytes["per_layer"].get(
                str(route_once["route_layer"]), {}
            ).get("routing_index_bytes", 0) if route_once is not None else 0,
            "process_rss_bytes": _rss_bytes(),
            "peak_gpu_memory_bytes": peak_gpu,
            "route_once_query_encoding_seconds": (
                route_once["query_encoding_seconds"]
                if route_once is not None and condition.kind == "routed"
                else 0.0
            ),
            "route_once_routing_seconds": (
                route_once["routing_seconds"]
                if route_once is not None and condition.kind == "routed"
                else 0.0
            ),
            "native_limit_violations": handle.native_limit_violations,
            **sums,
            **scored,
            **generated,
            **answer_metrics(generated["generated_answer"], example["answer"]),
            **(route_overlap if condition.kind == "routed" else {}),
            **(route_ranking if condition.kind == "routed" else {}),
        }
        # Keep the normalized layer rows rather than the helper's string-keyed trace map.
        row["attention_by_layer"] = per_layer_attention
        rows.append(row)
    baseline = next(row for row in rows if row["condition"] == "no_memory")
    for row in rows:
        row["gold_logprob_delta_vs_none"] = (
            row["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
        )
        row["gold_mean_logprob_delta_vs_none"] = (
            row["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "gold_logprob_delta_vs_none",
        "gold_mean_logprob_delta_vs_none",
        "gold_first_token_probability",
        "gold_first_token_rank",
        "f1",
        "em",
        "answer_contained",
        "memory_attention_mass",
        "evidence_attention_mass",
        "aggregate_materialized_kv_tokens",
        "aggregate_transferred_kv_bytes",
        "aggregate_routing_seconds",
        "aggregate_materialization_seconds",
        "aggregate_transfer_seconds",
        "aggregate_memory_attention_seconds",
        "route_once_query_encoding_seconds",
        "route_once_routing_seconds",
        "ttft_seconds",
        "teacher_forced_seconds",
        "generation_seconds",
        "resident_reference_detail_kv_bytes",
        "resident_routing_index_bytes",
        "peak_gpu_memory_bytes",
        "process_rss_bytes",
        "chunks_budget_rejected",
        "native_limit_violations",
        "routing_any_evidence_recall",
        "routing_all_evidence_recall",
        "routing_target_coverage",
        "routing_candidate_chunks",
        "routing_selected_parent_chunks",
        "routing_selected_parent_fraction",
        "route_layer_routing_index_bytes",
    )
    output = []
    condition_order = list(dict.fromkeys(row["condition"] for row in rows))
    for dataset in ("combined", "hotpotqa", "qasper"):
        for condition in condition_order:
            values = [
                row
                for row in rows
                if row["condition"] == condition
                and (dataset == "combined" or row["dataset"] == dataset)
            ]
            first = values[0]
            record = {
                "dataset": dataset,
                "condition": condition,
                "examples": len(values),
                "kind": first["kind"],
                "schedule": first["schedule"],
                "pra_layers": json.dumps(first["pra_layers"]),
                "pra_layer_count": first["pra_layer_count"],
                "pra_layer_fraction": first["pra_layer_fraction"],
                "first_pra_layer": first["first_pra_layer"],
                "last_pra_layer": first["last_pra_layer"],
                "evidence_guaranteed": _mean(
                    [float(row["evidence_guaranteed"]) for row in values]
                ),
            }
            record.update(
                {metric: _mean([row.get(metric) for row in values]) for metric in metrics}
            )
            ranking_rows = [
                row
                for row in values
                if row.get("routing_ranked_chunk_ids")
                and row.get("routing_evidence_chunk_ids")
            ]
            if ranking_rows:
                curve = recall_sparsity_curve(
                    [row["routing_ranked_chunk_ids"] for row in ranking_rows],
                    [set(row["routing_evidence_chunk_ids"]) for row in ranking_rows],
                    candidate_sizes=[
                        int(row["routing_candidate_chunks"]) for row in ranking_rows
                    ],
                    candidate_token_lengths=[
                        row["routing_ranked_chunk_token_lengths"] for row in ranking_rows
                    ],
                    require_complete_endpoint=True,
                )
                by_fraction = {
                    float(point["fraction"]): point for point in curve["curve"]
                }
                record.update(
                    {
                        "routing_r_at_5pct": by_fraction[0.05]["recall"],
                        "routing_r_at_10pct": by_fraction[0.10]["recall"],
                        "routing_r_at_20pct": by_fraction[0.20]["recall"],
                        "routing_r_at_30pct": by_fraction[0.30]["recall"],
                        "routing_f80": curve["inverse"]["f80"],
                        "routing_f90": curve["inverse"]["f90"],
                        "routing_auc_0_30": curve["auc_0_30"],
                        "routing_kv_fraction_at_5pct": by_fraction[0.05][
                            "selected_kv_token_fraction"
                        ],
                        "routing_kv_fraction_at_10pct": by_fraction[0.10][
                            "selected_kv_token_fraction"
                        ],
                        "routing_kv_fraction_at_20pct": by_fraction[0.20][
                            "selected_kv_token_fraction"
                        ],
                        "routing_kv_fraction_at_30pct": by_fraction[0.30][
                            "selected_kv_token_fraction"
                        ],
                    }
                )
            output.append(record)
    return output


def _attention_aggregates(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        for layer in row["attention_by_layer"]:
            key = (row["dataset"], row["condition"], layer["layer"])
            grouped.setdefault(key, []).append(layer)
    return [
        {
            "dataset": key[0],
            "condition": key[1],
            "layer": key[2],
            "examples": len(values),
            "memory_attention_mass": _mean(
                [value["memory_attention_mass"] for value in values]
            ),
            "evidence_attention_mass": _mean(
                [value["evidence_attention_mass"] for value in values]
            ),
            "local_attention_mass": _mean(
                [value["local_attention_mass"] for value in values]
            ),
        }
        for key, values in sorted(grouped.items())
    ]


def _rectangular_rows(rows: list[dict]) -> list[dict]:
    """Fill optional metric columns so heterogeneous experiment rows serialize."""
    fields = sorted({key for row in rows for key in row})

    def scalar(value):
        if isinstance(value, str):
            return value.strip().replace("\r\n", "\\n").replace("\n", "\\n")
        return value

    return [
        {field: scalar(row.get(field)) for field in fields}
        for row in rows
    ]


def _plots(aggregates, attention, output_dir: Path, stem: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    datasets = (("hotpotqa", "#2f6b9a"), ("qasper", "#d9782d"))
    for dataset, color in datasets:
        rows = [
            row
            for row in aggregates
            if row["dataset"] == dataset and row["kind"] in {"oracle", "routed"}
        ]
        rows.sort(key=lambda row: row["pra_layer_count"])
        layer_counts = [row["pra_layer_count"] for row in rows]
        quality = [row["gold_mean_logprob_delta_vs_none"] for row in rows]
        if len(set(layer_counts)) == len(layer_counts):
            axes[0, 0].plot(
                layer_counts, quality, marker="o", label=dataset, color=color
            )
        else:
            axes[0, 0].scatter(layer_counts, quality, label=dataset, color=color)
        axes[0, 1].scatter(
            [row["first_pra_layer"] for row in rows],
            [row["gold_mean_logprob_delta_vs_none"] for row in rows],
            label=dataset,
            color=color,
        )
        materialized = [row["aggregate_materialized_kv_tokens"] for row in rows]
        if len(set(materialized)) == len(materialized):
            axes[1, 0].plot(
                materialized, quality, marker="o", label=dataset, color=color
            )
        else:
            axes[1, 0].scatter(materialized, quality, label=dataset, color=color)
    axes[0, 0].set_xlabel("PRA-enabled layers")
    axes[0, 0].set_ylabel("Gold mean-token log-probability delta")
    axes[0, 1].set_xlabel("First PRA layer")
    axes[0, 1].set_ylabel("Gold mean-token log-probability delta")
    axes[1, 0].set_xlabel("Aggregate materialized K/V tokens")
    axes[1, 0].set_ylabel("Gold mean-token log-probability delta")

    preferred_conditions = {
        "oracle_last_4",
        "oracle_last_8",
        "oracle_last_half",
        "oracle_all",
    }
    available_conditions = list(
        dict.fromkeys(
            row["condition"]
            for row in attention
            if row["dataset"] == "hotpotqa"
        )
    )
    selected_conditions = [
        condition for condition in available_conditions if condition in preferred_conditions
    ]
    if len(selected_conditions) < 2:
        selected_conditions = available_conditions[:4]
    for condition in selected_conditions:
        rows = [
            row
            for row in attention
            if row["dataset"] == "hotpotqa" and row["condition"] == condition
        ]
        if rows:
            axes[1, 1].plot(
                [row["layer"] for row in rows],
                [row["memory_attention_mass"] for row in rows],
                marker=".",
                label=condition.replace("oracle_", ""),
            )
    axes[1, 1].set_xlabel("Decoder layer")
    axes[1, 1].set_ylabel("HotpotQA memory attention mass")
    for axis in axes.flat:
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=180)
    plt.close(figure)


def _audit(model, handle, entry, prior_oracle_layers) -> dict:
    injected = tuple(sorted(handle.adapters))
    stored = tuple(sorted(entry.layer_memory))
    return {
        "total_qwen_transformer_layers": int(model.config.num_hidden_layers),
        "previous_canonical_pra_layer_ids": [int(model.config.num_hidden_layers) - 1],
        "previous_canonical_pra_layer_count": 1,
        "previous_oracle_injected_layer_ids": list(prior_oracle_layers),
        "previous_oracle_max_active_layer_ids": list(prior_oracle_layers[-4:]),
        "experiment_injected_layer_ids": list(injected),
        "experiment_injected_layer_count": len(injected),
        "first_experiment_pra_layer": min(injected),
        "last_experiment_pra_layer": max(injected),
        "reference_stored_layer_ids": list(stored),
        "reference_stores_kv_for_every_injected_layer": stored == injected,
        "implicit_head_uses_same_reference_publication": True,
        "default_routing_policy": "independent_per_active_layer",
        "route_once_identity_reuse_supported": True,
        "route_once_payload_policy": "same chunk ID; target layer native post-RoPE K/V",
    }


def run(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = None
    if args.phase == "routed" and args.router == "learned":
        projection = load_hf_routing_projection(args.checkpoint, device=device)
    layer_count = int(model.config.num_hidden_layers)
    schedules = layer_schedules(layer_count)
    unknown = set(args.schedules).difference(schedules)
    if unknown:
        raise ValueError(f"Unknown layer schedules: {sorted(unknown)}")
    all_layers = schedules["all"]
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=all_layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=args.encoding_block_tokens,
            routing_chunk_tokens=args.routing_chunk_tokens,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=args.top_k,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    conditions = [Condition("no_memory", "none", "none", ())]
    prefix = "oracle" if args.phase == "oracle" else args.router
    kind = "oracle" if args.phase == "oracle" else "routed"
    conditions.extend(
        Condition(f"{prefix}_{name}", kind, name, schedules[name])
        for name in args.schedules
    )
    conditions.append(Condition("direct_text_oracle", "direct", "direct", ()))
    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.example_offset, args.seed
    )
    rows = []
    audit = None
    publication_rows = []
    for index, example in enumerate(examples, start=1):
        handle.cache.clear()
        source = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        evidence_spans = evidence_token_spans(
            tokenizer, example["source"], example["evidence"]
        )
        _sync(device)
        publication_started = time.perf_counter()
        entry = handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source,
            text=example["source"],
        )
        _sync(device)
        cache_bytes = _cache_bytes(entry, all_layers)
        publication_rows.append(
            {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "source_tokens": int(source.shape[1]),
                "publication_seconds": time.perf_counter() - publication_started,
                **{key: value for key, value in cache_bytes.items() if key != "per_layer"},
                "process_rss_bytes": _rss_bytes(),
            }
        )
        if audit is None:
            prior = (layer_count // 3, *range(layer_count - 4, layer_count))
            audit = _audit(model, handle, entry, prior)
        route_once = None
        if args.phase == "routed":
            route_once = _route_once(
                handle,
                tokenizer,
                example["question"],
                layer_count - 1,
                args.prompt_tokens,
                device,
                position_offset=(
                    int(source.shape[1]) if args.position_mode == "corrected" else 0
                ),
            )
        example_rows = _evaluate_example(
            handle,
            tokenizer,
            example,
            entry,
            evidence_spans,
            conditions,
            route_once,
            cache_bytes,
            args,
            device,
            int(source.shape[1]),
        )
        rows.extend(example_rows)
        print(
            f"[{index}/{len(examples)}] {example['dataset']} "
            + " ".join(
                f"{row['condition']}={row['gold_mean_logprob_delta_vs_none']:+.3f}"
                for row in example_rows
            ),
            flush=True,
        )
    aggregates = _aggregate(rows)
    attention = _attention_aggregates(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "frozen Qwen route-once, layer-native multi-layer PRA consumption sweep",
        "position_mode": args.position_mode,
        "transport_invariants": {
            "external_kv": "native post-RoPE per destination layer",
            "query_offset": (
                "after logical source extent"
                if args.position_mode == "corrected"
                else "historical restart at zero"
            ),
            "deduplication": "chunk identity then overlap-aware materialization",
            "memory_lifetime": "request prefill and decode",
            "attention_composition": "shared softmax over local and memory keys",
            "physical_kv_heads": "native GQA/MQA heads without query-head expansion",
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "phase": args.phase,
        "router": args.router if args.phase == "routed" else None,
        "routing_checkpoint": (
            str(args.checkpoint.resolve().relative_to(ROOT))
            if projection is not None
            else None
        ),
        "routing_adapter_parameters": (
            projection.parameter_count if projection is not None else 0
        ),
        "seed": args.seed,
        "example_offset": args.example_offset,
        "examples_per_dataset": args.examples_per_dataset,
        "audit": audit,
        "resolved_schedules": {
            name: list(schedules[name]) for name in args.schedules
        },
        "settings": {
            "native_context_tokens": args.native_tokens,
            "prompt_tokens": args.prompt_tokens,
            "direct_text_tokens": args.direct_text_tokens,
            "encoding_block_tokens": args.encoding_block_tokens,
            "routing_chunk_tokens": args.routing_chunk_tokens,
            "memory_budget_tokens_per_layer": args.memory_tokens,
            "top_k_parent_chunks": args.top_k,
            "new_tokens": args.new_tokens,
        },
        "publication_rows": publication_rows,
        "rows": rows,
        "aggregates": aggregates,
        "attention_aggregates": attention,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.stem}.json"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    _write_csv(args.output_dir / f"{args.stem}.csv", _rectangular_rows(flat))
    _write_csv(
        args.output_dir / f"{args.stem}_aggregate.csv",
        _rectangular_rows(aggregates),
    )
    _write_csv(
        args.output_dir / f"{args.stem}_attention.csv",
        _rectangular_rows(attention),
    )
    _write_csv(
        args.output_dir / f"{args.stem}_publication.csv",
        _rectangular_rows(publication_rows),
    )
    _plots(aggregates, attention, args.output_dir, args.stem)
    return artifact


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    result_dir = (
        ROOT
        / "docs"
        / "papers"
        / "shared"
        / "results"
        / "paper2_hf"
        / "multilayer_pra"
    )
    checkpoint = (
        ROOT
        / "docs"
        / "papers"
        / "shared"
        / "results"
        / "paper2_hf"
        / "routing"
        / "learned_adapter"
        / "checkpoints"
        / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", choices=("oracle", "routed"), default="oracle")
    parser.add_argument("--router", choices=("learned", "zero_shot"), default="learned")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--encoding-block-tokens", type=int, default=128)
    parser.add_argument("--routing-chunk-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--position-mode",
        choices=("original", "corrected"),
        default="corrected",
    )
    parser.add_argument(
        "--schedules",
        default="last_1,last_2,last_4,last_8,last_quarter,last_half,all",
    )
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument("--output-dir", type=Path, default=result_dir)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()
    args.schedules = _csv_tuple(args.schedules)
    if args.stem is None:
        args.stem = (
            "oracle_layer_depth"
            if args.phase == "oracle"
            else f"{args.router}_route_once"
        )
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
