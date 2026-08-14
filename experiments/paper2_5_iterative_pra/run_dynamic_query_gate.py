"""Gate 1: compare static Q0 with frozen-model Q1 after integrating active A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_grounded_facet_gate import (
    FacetConfig,
    _support_span,
    build_facets,
)
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    competition_rank,
    evidence_parent_groups,
    native_qk_parent_scores,
    validation_partition,
)
from pra_hf.dynamic_query import (
    RECONSTRUCTION_MODES,
    SUPPORT_MODES,
    build_dynamic_query_facets,
)
from pra_hf.grounded_propagation import (
    generate_associative_candidates,
    query_validate_candidates,
    rank_grounded_candidates,
)
from pra_torch.hf import load_hf_routing_projection


CANDIDATE_KS = (1, 2, 3, 4, 5, 6, 8, 11)
PRIMARY_K = 4
MATERIAL_R1_GAIN = 0.10


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tensor_bytes(value) -> int:
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0


def _native_cache_bytes(feature: dict) -> int:
    return sum(
        _tensor_bytes(feature[key])
        for key in (
            "local_pre_query",
            "local_pre_key",
            "local_token_mask",
            "local_parent_indices",
        )
    )


def _facet_text(tokenizer, ids: torch.Tensor, start: int, end: int) -> str:
    if start < 0:
        return "[global contextual query]"
    return tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()


def _best_target_record(ranking, targets: set[int]):
    return next(
        (row for row in sorted(ranking.candidates, key=lambda item: item.query_rank) if row.parent_index in targets),
        None,
    )


def _score_state(
    *,
    feature: dict,
    state: dict,
    facets,
    projection,
    parent_memory: torch.Tensor,
    proposals: dict,
    native_rank: dict,
    association_seconds: float,
    memory_projection_seconds: float,
    native_dots: int,
    tokenizer,
    seed: int,
    device: torch.device,
) -> list[dict]:
    if device.type == "cuda":
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    else:
        baseline_allocated = 0
    projection_started = time.perf_counter()
    projected_facets = projection.project_query(facets.hidden.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    projection_seconds = time.perf_counter() - projection_started
    rows = []
    for candidate_k in CANDIDATE_KS:
        candidates = proposals[candidate_k]
        candidate_ids = torch.tensor(candidates.parent_indices, dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        score_started = time.perf_counter()
        packed = projected_facets.float() @ parent_memory[candidate_ids].float().T
        score_matrix = torch.full(
            (projected_facets.shape[0], len(feature["parent_spans"])),
            float("-inf"),
            device=device,
        )
        score_matrix[:, candidate_ids] = packed
        validation = query_validate_candidates(score_matrix, candidates)
        ranking = rank_grounded_candidates(
            candidates,
            validation,
            mode="query_rerank",
            final_k=max(1, len(candidates.parent_indices)),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        score_seconds = time.perf_counter() - score_started
        selected = ranking.selected
        target_rank = next(
            (rank for rank, parent in enumerate(selected, start=1) if parent in state["targets"]),
            candidate_k + 1,
        )
        target_record = _best_target_record(ranking, state["targets"])
        winning_facet = target_record.validating_facet if target_record is not None else None
        provenance = facets.provenance[winning_facet] if winning_facet is not None else None
        target_scores = [
            row.query_score for row in ranking.candidates if row.parent_index in state["targets"]
        ]
        distractor_scores = [
            row.query_score for row in ranking.candidates if row.parent_index not in state["targets"]
        ]
        query_margin = (
            max(target_scores) - max(distractor_scores)
            if target_scores and distractor_scores
            else None
        )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        )
        native_transfer_bytes = _native_cache_bytes(feature) + _tensor_bytes(
            feature["local_token_mask"]
        )
        prompt_transfer_events = 2 if state["h2d_prompt_bytes"] else 0
        rows.append(
            {
                "partition": validation_partition(feature["example_id"]),
                "dataset": "hotpotqa",
                "example_id": feature["example_id"],
                "seed": seed,
                "hop": 1,
                "transition": state["transition"],
                "query_state_id": state["query_state_id"],
                "query_reconstruction_mode": state["mode"],
                "query_support_mode": state["support_mode"],
                "facet_type": state["facet_type"],
                "facet_count": len(facets.provenance),
                "K_search": candidate_k,
                "candidate_parent_count": len(candidates.parent_indices),
                "source_parent": json.dumps(sorted(state["source_group"])),
                "oracle_parent": json.dumps(sorted(state["targets"])),
                "candidate_parent_ids": json.dumps(candidates.parent_indices),
                "candidate_native_rank": native_rank["target_rank"],
                "candidate_query_rank": target_rank,
                "candidate_score": target_record.query_score if target_record else None,
                "successor_candidate_recall": float(
                    bool(set(candidates.parent_indices) & state["targets"])
                ),
                "mrr": 1.0 / target_rank,
                **{
                    f"recall_at_{cutoff}": float(target_rank <= cutoff)
                    for cutoff in CANDIDATE_KS
                },
                "oracle_margin": query_margin,
                "winning_facet": winning_facet,
                "winning_facet_family": provenance.family if provenance else None,
                "winning_facet_span": (
                    json.dumps([provenance.token_start, provenance.token_end])
                    if provenance
                    else None
                ),
                "winning_facet_text": (
                    _facet_text(
                        tokenizer,
                        state["prompt_ids"],
                        provenance.token_start,
                        provenance.token_end,
                    )
                    if provenance
                    else None
                ),
                "winning_facet_includes_a": bool(
                    provenance and provenance.family.startswith("memory_a_")
                ),
                "question": state["question"],
                "source_a_text": state["a_text"],
                "oracle_b_text": json.dumps(
                    [state["parent_texts"][index] for index in sorted(state["targets"])]
                ),
                "B_ceiling": 1,
                "theta": None,
                "activated": json.dumps(selected[:1]),
                "active_parent_count": 1,
                "logical_reference_tokens": feature["source_tokens"],
                "logical_parent_count": len(feature["parent_spans"]),
                "conceptual_active_parent_count": len(state["source_group"]) + 1,
                "conceptual_active_parent_fraction": (
                    (len(state["source_group"]) + 1) / len(feature["parent_spans"])
                ),
                "materialized_parent_count": 0,
                "materialized_native_kv_tokens": 0,
                "active_native_kv_fraction": 0.0,
                "native_kv_bytes": 0,
                "peak_native_kv_tokens": 0,
                "peak_active_parent_count": len(state["source_group"]) + 1,
                "query_support_tokens": state["query_support_tokens"],
                "native_k_candidate_comparisons": native_dots,
                "semantic_comparisons": validation.comparisons,
                "search_comparisons": native_dots + validation.comparisons,
                "reencoding_time": state["reencoding_time"],
                "memory_projection_time": memory_projection_seconds,
                "query_projection_time": projection_seconds,
                "candidate_scoring_time": score_seconds,
                "native_search_time": association_seconds,
                "prefill_to_ready_time": (
                    state["reencoding_time"]
                    + memory_projection_seconds
                    + projection_seconds
                    + score_seconds
                    + association_seconds
                ),
                "model_baseline_gpu_allocated_bytes": state["model_baseline_allocated"],
                "total_reencoding_peak_gpu_allocated_bytes": state["capture_peak_allocated"],
                "total_reencoding_peak_gpu_reserved_bytes": state["capture_peak_reserved"],
                "incremental_reencoding_peak_allocated_bytes": state["capture_incremental"],
                "routing_peak_gpu_allocated_bytes": peak_allocated,
                "routing_peak_gpu_reserved_bytes": peak_reserved,
                "incremental_routing_peak_allocated_bytes": max(
                    0, peak_allocated - baseline_allocated
                ),
                "cpu_reference_cache_bytes": state["source_feature_file_bytes"],
                "gpu_reference_cache_bytes": (
                    _native_cache_bytes(feature) + _tensor_bytes(parent_memory)
                ),
                "gist_cache_bytes": _tensor_bytes(feature["parent_hidden"]),
                "projected_parent_cache_bytes": _tensor_bytes(parent_memory),
                "query_facet_tensor_bytes": _tensor_bytes(facets.hidden),
                "candidate_score_tensor_bytes": _tensor_bytes(packed),
                "h2d_transfer_bytes": (
                    state["h2d_prompt_bytes"]
                    + native_transfer_bytes
                    + _tensor_bytes(feature["parent_hidden"])
                    + _tensor_bytes(facets.hidden)
                ),
                # Five native-QK tensor copies (the token mask is used twice),
                # one parent-memory copy, one facet copy, and optionally the
                # prompt ids plus attention mask used to reconstruct Q1.
                "h2d_transfer_events": 7 + prompt_transfer_events,
                "h2d_transfer_time": None,
                "currently_materialized_parent_count": 0,
                "tpot": None,
                "generation_performed": False,
            }
        )
    return rows


def _aggregate(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    metrics = (
        "facet_count", "candidate_parent_count", "successor_candidate_recall", "mrr",
        *(f"recall_at_{cutoff}" for cutoff in CANDIDATE_KS),
        "candidate_native_rank", "candidate_query_rank", "oracle_margin",
        "winning_facet_includes_a", "query_support_tokens",
        "conceptual_active_parent_count", "conceptual_active_parent_fraction",
        "materialized_parent_count", "materialized_native_kv_tokens", "native_kv_bytes",
        "native_k_candidate_comparisons", "semantic_comparisons", "search_comparisons",
        "reencoding_time", "query_projection_time", "candidate_scoring_time",
        "native_search_time", "prefill_to_ready_time",
        "memory_projection_time",
        "total_reencoding_peak_gpu_allocated_bytes",
        "total_reencoding_peak_gpu_reserved_bytes",
        "incremental_reencoding_peak_allocated_bytes",
        "routing_peak_gpu_allocated_bytes", "routing_peak_gpu_reserved_bytes",
        "incremental_routing_peak_allocated_bytes", "cpu_reference_cache_bytes",
        "gpu_reference_cache_bytes", "gist_cache_bytes", "query_facet_tensor_bytes",
        "projected_parent_cache_bytes", "candidate_score_tensor_bytes", "h2d_transfer_bytes",
    )
    output = []
    for key, values in grouped.items():
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        record["identities"] = len({row["example_id"] for row in values})
        record["edges"] = len({(row["example_id"], row["transition"]) for row in values})
        record["seeds"] = len({row["seed"] for row in values})
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if samples:
                record[metric] = statistics.fmean(samples)
                if metric in {
                    "materialized_native_kv_tokens",
                    "conceptual_active_parent_count",
                    "prefill_to_ready_time",
                }:
                    ordered = sorted(samples)
                    record[f"{metric}_p95"] = ordered[
                        min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
                    ]
                    record[f"{metric}_max"] = max(ordered)
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row[key]) for key in dimensions))


def _select_dynamic(summary: list[dict]) -> tuple[dict, list[dict]]:
    candidates = [
        row
        for row in summary
        if row["partition"] == "validation"
        and row["K_search"] == PRIMARY_K
        and row["query_reconstruction_mode"] != "static_q0"
    ]
    winner = max(
        candidates,
        key=lambda row: (
            row["recall_at_1"],
            row["mrr"],
            -row["prefill_to_ready_time"],
            -row["facet_count"],
            row["query_reconstruction_mode"],
            row["query_support_mode"],
        ),
    )
    return winner, [
        {
            "candidate": f"{row['query_reconstruction_mode']}:{row['query_support_mode']}",
            "recall_at_1": row["recall_at_1"],
            "mrr": row["mrr"],
            "recall_at_4": row["recall_at_4"],
            "facet_count": row["facet_count"],
            "prefill_to_ready_time": row["prefill_to_ready_time"],
        }
        for row in candidates
    ]


def _plot(summary: list[dict], selected: dict, output_dir: Path) -> None:
    test = [row for row in summary if row["partition"] == "test"]
    static = [row for row in test if row["query_reconstruction_mode"] == "static_q0"]
    dynamic = [
        row
        for row in test
        if row["query_reconstruction_mode"] == selected["query_reconstruction_mode"]
        and row["query_support_mode"] == selected["query_support_mode"]
    ]
    lookup = {
        (label, int(row["K_search"])): row
        for label, values in (("Static Q0", static), ("Dynamic Q1", dynamic))
        for row in values
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colors = {"Static Q0": "#4c78a8", "Dynamic Q1": "#f58518"}
    for label in ("Static Q0", "Dynamic Q1"):
        axes[0].plot(
            CANDIDATE_KS,
            [lookup[label, k]["recall_at_1"] for k in CANDIDATE_KS],
            marker="o", label=label, color=colors[label],
        )
        axes[1].plot(
            CANDIDATE_KS,
            [lookup[label, k]["mrr"] for k in CANDIDATE_KS],
            marker="o", label=label, color=colors[label],
        )
    axes[0].set_ylabel("Conditional successor R@1")
    axes[1].set_ylabel("Conditional successor MRR")
    for axis in axes:
        axis.set_xlabel("Native candidate breadth K")
        axis.set_xticks(CANDIDATE_KS)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"dynamic_query_k_gate.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    source_features = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    static_features = torch.load(args.static_query_file, map_location="cpu", weights_only=False)
    dynamic_features = torch.load(args.dynamic_query_file, map_location="cpu", weights_only=False)
    static_by_id = {row["example_id"]: row for row in static_features}
    dynamic_by_edge = defaultdict(list)
    for row in dynamic_features:
        dynamic_by_edge[row["example_id"], int(row["transition"])].append(row)
    gate_a = json.loads(args.gate_a_file.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_gate_file.read_text(encoding="utf-8"))
    manifest = json.loads(args.dynamic_manifest_file.read_text(encoding="utf-8"))
    facet_config = FacetConfig(**gate_a["selected_facet_config"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    projections = {
        seed: load_hf_routing_projection(
            args.projection_dir / "checkpoints" /
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt",
            device=device,
        )
        for seed in args.seeds
    }

    rows = []
    for feature in source_features:
        groups = evidence_parent_groups(feature)
        if feature["dataset"] != "hotpotqa" or len(groups) < 2:
            continue
        static_feature = static_by_id[feature["example_id"]]
        static_enriched = {
            **static_feature,
            **{key: feature[key] for key in ("parent_spans", "source_tokens", "parent_positive_mask", "evidence_spans")},
        }
        static_facets = build_facets(
            static_enriched, facet_config, gate_a["selected_query_support"], tokenizer
        )
        static_support = _support_span(static_feature, gate_a["selected_query_support"])
        for transition, (source_group, targets) in enumerate(zip(groups, groups[1:])):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            association_started = time.perf_counter()
            association_scores, native_dots = native_qk_parent_scores(
                feature,
                source_group,
                device,
                token_reduction="top_m_mean",
                head_reduction="top_m_mean",
                top_m=4,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            association_seconds = time.perf_counter() - association_started
            native_rank = competition_rank(association_scores, targets, source_group)
            proposals = {
                candidate_k: generate_associative_candidates(
                    association_scores,
                    source_parents=source_group,
                    candidate_k=candidate_k,
                    comparisons=native_dots,
                )
                for candidate_k in CANDIDATE_KS
            }
            dynamic_states = dynamic_by_edge[feature["example_id"], transition]
            if {row["query_reconstruction_mode"] for row in dynamic_states} != set(RECONSTRUCTION_MODES):
                raise ValueError("Dynamic reconstruction variants are incomplete.")
            for seed, projection in projections.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                memory_projection_started = time.perf_counter()
                parent_memory = projection.project_memory(feature["parent_hidden"].to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                memory_projection_seconds = time.perf_counter() - memory_projection_started
                static_state = {
                    "transition": transition,
                    "query_state_id": f"{feature['example_id']}:t0:static_q0",
                    "mode": "static_q0",
                    "support_mode": gate_a["selected_query_support"],
                    "facet_type": facet_config.name,
                    "source_group": source_group,
                    "targets": targets,
                    "prompt_ids": static_feature["prompt_input_ids"],
                    "query_support_tokens": static_support[1] - static_support[0],
                    "question": static_feature["question"],
                    "a_text": dynamic_states[0]["memory_text"],
                    "parent_texts": dynamic_states[0]["parent_texts"],
                    "reencoding_time": 0.0,
                    "model_baseline_allocated": manifest["model_baseline_gpu_allocated_bytes"],
                    "capture_peak_allocated": manifest["model_baseline_gpu_allocated_bytes"],
                    "capture_peak_reserved": manifest["model_baseline_gpu_reserved_bytes"],
                    "capture_incremental": 0,
                    "source_feature_file_bytes": manifest["source_feature_file_bytes"],
                    "h2d_prompt_bytes": 0,
                }
                rows.extend(_score_state(
                    feature=feature, state=static_state, facets=static_facets,
                    projection=projection, parent_memory=parent_memory, proposals=proposals,
                    native_rank=native_rank, association_seconds=association_seconds,
                    memory_projection_seconds=memory_projection_seconds,
                    native_dots=native_dots, tokenizer=tokenizer, seed=seed, device=device,
                ))
                for dynamic in dynamic_states:
                    for support_mode in SUPPORT_MODES:
                        dynamic_facets = build_dynamic_query_facets(
                            dynamic["query_hidden_states"].float(),
                            question_span=tuple(dynamic["question_span"]),
                            memory_span=tuple(dynamic["memory_span"]),
                            support_mode=support_mode,
                            window=2,
                            stride=1,
                            include_global=True,
                            native_query=dynamic["query_pre_query"].float(),
                        )
                        support_tokens = (
                            dynamic["question_span"][1] - dynamic["question_span"][0]
                            + (
                                dynamic["memory_span"][1] - dynamic["memory_span"][0]
                                if support_mode == "question_and_memory"
                                else 0
                            )
                        )
                        dynamic_state = {
                            "transition": transition,
                            "query_state_id": dynamic["query_state_id"],
                            "mode": dynamic["query_reconstruction_mode"],
                            "support_mode": support_mode,
                            "facet_type": "w2_s1_dynamic",
                            "source_group": source_group,
                            "targets": targets,
                            "prompt_ids": dynamic["prompt_input_ids"],
                            "query_support_tokens": support_tokens,
                            "question": dynamic["question"],
                            "a_text": dynamic["memory_text"],
                            "parent_texts": dynamic["parent_texts"],
                            "reencoding_time": dynamic["reencoding_time"],
                            "model_baseline_allocated": manifest["model_baseline_gpu_allocated_bytes"],
                            "capture_peak_allocated": dynamic["total_peak_gpu_allocated_bytes"],
                            "capture_peak_reserved": dynamic["total_peak_gpu_reserved_bytes"],
                            "capture_incremental": dynamic["incremental_reencoding_peak_allocated_bytes"],
                            "source_feature_file_bytes": manifest["source_feature_file_bytes"],
                            "h2d_prompt_bytes": dynamic["h2d_prompt_bytes"],
                        }
                        rows.extend(_score_state(
                            feature=feature, state=dynamic_state, facets=dynamic_facets,
                            projection=projection, parent_memory=parent_memory, proposals=proposals,
                            native_rank=native_rank, association_seconds=association_seconds,
                            memory_projection_seconds=memory_projection_seconds,
                            native_dots=native_dots, tokenizer=tokenizer, seed=seed, device=device,
                        ))
        print(f"[dynamic-gate] {feature['example_id']} transitions={len(groups)-1}", flush=True)

    rank_lookup = {
        (row["example_id"], row["seed"], row["transition"], row["K_search"]): row["candidate_query_rank"]
        for row in rows if row["query_reconstruction_mode"] == "static_q0"
    }
    for row in rows:
        static_rank = rank_lookup[row["example_id"], row["seed"], row["transition"], row["K_search"]]
        row["static_query_rank"] = static_rank
        row["rank_movement_from_q0"] = static_rank - row["candidate_query_rank"]

    dimensions = (
        "partition", "query_reconstruction_mode", "query_support_mode", "facet_type", "K_search"
    )
    summary = _aggregate(rows, dimensions)
    selected, selection_audit = _select_dynamic(summary)

    def find(partition: str, mode: str, support: str, k: int):
        return next(
            row for row in summary
            if row["partition"] == partition
            and row["query_reconstruction_mode"] == mode
            and row["query_support_mode"] == support
            and row["K_search"] == k
        )

    static_validation = find("validation", "static_q0", gate_a["selected_query_support"], PRIMARY_K)
    static_test = find("test", "static_q0", gate_a["selected_query_support"], PRIMARY_K)
    selected_test = find(
        "test", selected["query_reconstruction_mode"], selected["query_support_mode"], PRIMARY_K
    )
    prior_validation = prior["validation_selection"]
    prior_test = prior["heldout_query_grounded"]
    for observed, expected, label in (
        (static_validation["recall_at_1"], prior_validation["recall_at_1"], "validation R@1"),
        (static_validation["mrr"], prior_validation["mrr"], "validation MRR"),
        (static_test["recall_at_1"], prior_test["recall_at_1"], "held-out R@1"),
        (static_test["mrr"], prior_test["mrr"], "held-out MRR"),
    ):
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise AssertionError(f"Static {label} baseline changed: {observed} != {expected}")

    # The dynamic-query experiment is downstream of candidate discovery.  Pin
    # native-QK K=4 recovery to the prior gate so a scorer drift cannot be
    # mistaken for a benefit from reconstructing Q1.
    for observed, expected, label in (
        (
            static_validation["successor_candidate_recall"],
            prior_validation["recall_at_4"],
            "validation candidate recall",
        ),
        (
            static_test["successor_candidate_recall"],
            prior["heldout_association"]["target_in_associative_candidates"],
            "held-out candidate recall",
        ),
    ):
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise AssertionError(f"Native {label} changed: {observed} != {expected}")

    r1_gain = selected_test["recall_at_1"] - static_test["recall_at_1"]
    mrr_gain = selected_test["mrr"] - static_test["mrr"]
    gate_passed = r1_gain >= MATERIAL_R1_GAIN and mrr_gain > 0
    diagnostics = [
        row for row in rows
        if row["K_search"] == max(CANDIDATE_KS)
        and row["query_reconstruction_mode"] == selected["query_reconstruction_mode"]
        and row["query_support_mode"] == selected["query_support_mode"]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_default_changed": False,
        "training_performed": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seeds": list(args.seeds),
        "candidate_ks": list(CANDIDATE_KS),
        "primary_k": PRIMARY_K,
        "reconstruction_candidates": list(RECONSTRUCTION_MODES),
        "support_candidates": list(SUPPORT_MODES),
        "selected_dynamic_query": {
            "query_reconstruction_mode": selected["query_reconstruction_mode"],
            "query_support_mode": selected["query_support_mode"],
            "facet_type": selected["facet_type"],
        },
        "selection_audit": selection_audit,
        "static_validation": static_validation,
        "selected_dynamic_validation": selected,
        "static_heldout": static_test,
        "selected_dynamic_heldout": selected_test,
        "heldout_r1_gain": r1_gain,
        "heldout_mrr_gain": mrr_gain,
        "gate_1_success_rule": {"minimum_r1_gain": MATERIAL_R1_GAIN, "mrr_must_improve": True},
        "gate_1_passed": gate_passed,
        "gate_2_run": False,
        "gate_2_reason": (
            "requires_discovery_surface_implementation"
            if gate_passed
            else "stopped_by_predeclared_dynamic_query_gate"
        ),
        "native_candidate_generation_changed": False,
        "raw_native_semantic_scores_mixed": False,
        "native_kv_materialization_performed": False,
        "conceptual_active_and_native_kv_separated": True,
        "serving_metric_deferrals": {
            "tpot": "no generation in conditional routing diagnostic",
            "throughput": "conditional diagnostic is not an end-to-end serving path",
            "concurrency": "deferred to Paper 3.5 and larger GPU",
            "dollar_per_million_tokens": "no cloud hardware/pricing run",
        },
    }
    (args.output_dir / "dynamic_query_gate_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "dynamic_query_rows.csv", rows)
    _write_csv(args.output_dir / "dynamic_query_summary.csv", summary)
    _write_csv(args.output_dir / "dynamic_query_selection.csv", selection_audit)
    _write_csv(args.output_dir / "dynamic_query_bridge_diagnostics.csv", diagnostics)
    _plot(summary, selected, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument("--source-feature-file", type=Path, default=result_root / "native_qk_closure/native_qk_features_test.pt")
    parser.add_argument("--static-query-file", type=Path, default=result_root / "query_entry_facets/query_entry_features.pt")
    parser.add_argument("--dynamic-query-file", type=Path, default=result_root / "dynamic_query_discovery/dynamic_query_features.pt")
    parser.add_argument("--dynamic-manifest-file", type=Path, default=result_root / "dynamic_query_discovery/dynamic_query_feature_manifest.json")
    parser.add_argument("--gate-a-file", type=Path, default=result_root / "grounded_query_facets/grounded_facet_gate_results.json")
    parser.add_argument("--prior-gate-file", type=Path, default=result_root / "grounded_query_facets/grounded_propagation_results.json")
    parser.add_argument("--projection-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--output-dir", type=Path, default=result_root / "dynamic_query_discovery")
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "selected_dynamic_query": result["selected_dynamic_query"],
        "static_heldout": result["static_heldout"],
        "selected_dynamic_heldout": result["selected_dynamic_heldout"],
        "gate_1_passed": result["gate_1_passed"],
    }, indent=2))
