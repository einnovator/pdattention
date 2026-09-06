"""Run the Paper 3.3 packed-teacher oracle sparsity gate on MLX.

Retrieval, reranking, selected records, record order, and prompt semantics are
frozen once per question. The only intervention is which causal
document-to-document token pairs remain visible at each transformer layer.
Every selected pair executes the frozen host model's original attention across
all heads; no synthetic K/V is appended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
    _token_segments,
)
from experiments.paper3_2_rag.run_prerope_causal_decomposition import (
    DEFAULT_RERANKER,
)
from experiments.rag_vs_pra.datasets import (
    controlled_fixture,
    load_multihop_rag,
    select_cohort,
)
from experiments.rag_vs_pra.run_powered_decomposition import PersistentMLXBackend
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.rag_causal_decomposition import (
    DocumentAttentionPolicy,
    build_document_attention_mask,
)
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    PositionBindingMode,
    encode_native_memory,
    encode_native_memory_with_mask,
    native_memory_diagnostics,
    rebind_native_memories_to_receipt,
    make_native_prompt_cache,
)
from pra_hf.sparse_crossdoc import (
    CrossDocumentAttentionCollector,
    InteractionGroupKind,
    InteractionGroupKey,
    cumulative_attention_mass_plan,
    full_interaction_plan,
    interaction_group_ablation_plan,
    interaction_group_keys,
    interaction_localization,
    ranked_edge_plan,
    ranked_physical_indices,
    ranked_physical_prefix,
    ranked_physical_prefix_by_group_utility,
    selected_interaction_localization,
    top_attention_edge_plan,
)


SCHEMA_VERSION = "paper3.3-oracle-sparsity-run-v2"
DEFAULT_EDGE_PERCENTAGES = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
DEFAULT_MASS_PERCENTAGES = (50.0, 75.0, 90.0, 95.0, 99.0)
DEFAULT_SPLIT_MANIFEST = Path(
    "docs/papers/shared/results/paper3_3_sparse_crossdoc/splits.json"
)
SUPPORTED_RANKING_TARGETS = {
    "attention",
    "pair_nll",
    "pair_js",
    "pair_nll_x_attention",
    "layer_nll",
    "layer_js",
    "layer_head_nll",
    "layer_head_js",
}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _percentages(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(item < 0.0 or item > 100.0 for item in values):
        raise argparse.ArgumentTypeError("percentages must lie in [0, 100]")
    return tuple(dict.fromkeys(values))


def _ranking_targets(value: str) -> tuple[str, ...]:
    targets = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    unknown = set(targets) - SUPPORTED_RANKING_TARGETS
    if not targets or unknown:
        raise argparse.ArgumentTypeError(
            f"ranking targets must be selected from {sorted(SUPPORTED_RANKING_TARGETS)}; "
            f"got {sorted(unknown)}"
        )
    return targets


def _resolve_reranker_device(requested: str) -> str:
    """Resolve accelerator preference once and persist it in run provenance."""

    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _atomic_json(path: Path, value: object) -> None:
    """Write resumable experiment state without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _select_split_cohort(
    questions: Sequence[object],
    *,
    split_manifest: Path | None,
    split_name: str,
    max_examples: int,
    seed: int,
) -> tuple[tuple[object, ...], dict[str, object] | None]:
    """Restrict sampling to a frozen split before selecting a seeded cohort."""

    split_metadata = None
    eligible = tuple(questions)
    if split_manifest is not None:
        split_metadata = json.loads(split_manifest.read_text(encoding="utf-8"))
        key = f"{split_name}_ids"
        if key not in split_metadata:
            raise ValueError(f"split manifest does not contain {key}")
        allowed = set(str(value) for value in split_metadata[key])
        eligible = tuple(
            question
            for question in questions
            if str(getattr(question, "example_id")) in allowed
        )
        if len(eligible) != len(allowed):
            present = {str(getattr(question, "example_id")) for question in eligible}
            missing = sorted(allowed - present)
            raise ValueError(
                f"dataset is missing frozen {split_name} IDs: {missing[:4]}"
            )
    return select_cohort(eligible, max_examples=max_examples, seed=seed), split_metadata


def _ranking_group(target: str) -> InteractionGroupKind | None:
    if target.startswith("pair_"):
        return "document_pair"
    if target.startswith("layer_head_"):
        return "layer_head"
    if target.startswith("layer_"):
        return "layer"
    return None


def _ranking_condition(target: str) -> str:
    return {
        "attention": "ORACLE_TOP_ATTENTION",
        "pair_nll": "ORACLE_PAIR_NLL",
        "pair_js": "ORACLE_PAIR_JS",
        "pair_nll_x_attention": "ORACLE_PAIR_NLL_X_ATTENTION",
        "layer_nll": "ORACLE_LAYER_NLL",
        "layer_js": "ORACLE_LAYER_JS",
        "layer_head_nll": "ORACLE_LAYER_HEAD_NLL",
        "layer_head_js": "ORACLE_LAYER_HEAD_JS",
    }[target]


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate conditions without treating questions as independent seeds."""

    grouped: dict[tuple[str, float | None], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), row.get("target_percentage"))
        grouped.setdefault(key, []).append(row)
    result = []
    for (condition, target), values in grouped.items():
        result.append(
            {
                "condition": condition,
                "target_percentage": target,
                "examples": len(values),
                "token_f1": _mean(values, "token_f1"),
                "official_score": _mean(values, "official_multihop_rag_score"),
                "exact_match": _mean(values, "exact_match"),
                "gold_answer_mean_nll": _mean(values, "gold_answer_mean_nll"),
                "first_step_js_vs_reference": _mean(values, "first_step_js_divergence"),
                "selected_logical_edge_fraction": _mean(
                    values, "selected_logical_edge_fraction"
                ),
                "selected_physical_edge_fraction": _mean(
                    values, "selected_physical_edge_fraction"
                ),
                "retained_attention_mass": _mean(values, "retained_attention_mass"),
                "selected_physical_head_edges": _mean(
                    values, "selected_physical_head_edges"
                ),
                "encode_ms": _mean(values, "encode_ms"),
                "ttft_ms": _mean(values, "ttft_ms"),
                "total_latency_ms": _mean(values, "total_latency_ms"),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            str(row["condition"]),
            float(row["target_percentage"] or -1.0),
        ),
    )


def paired_bootstrap_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    reference_condition: str = "PACKED_RAG_INSTRUMENTED",
    bootstrap_replicates: int = 2000,
    seed: int = 3303,
) -> list[dict[str, object]]:
    """Bootstrap paired condition-minus-teacher effects over question IDs."""

    import numpy as np

    reference = {
        str(row["example_id"]): row
        for row in rows
        if row["condition"] == reference_condition
    }
    grouped: dict[tuple[str, float | None], list[Mapping[str, object]]] = {}
    for row in rows:
        if row["condition"] == reference_condition:
            continue
        grouped.setdefault(
            (str(row["condition"]), row.get("target_percentage")), []
        ).append(row)
    result = []
    for group_index, ((condition, target), values) in enumerate(
        sorted(grouped.items())
    ):
        pairs = [
            (row, reference[str(row["example_id"])])
            for row in values
            if str(row["example_id"]) in reference
        ]
        metrics: dict[str, object] = {}
        for metric in (
            "token_f1",
            "official_multihop_rag_score",
            "gold_answer_mean_nll",
        ):
            differences = np.asarray(
                [
                    float(condition_row[metric]) - float(reference_row[metric])
                    for condition_row, reference_row in pairs
                    if condition_row.get(metric) is not None
                    and reference_row.get(metric) is not None
                ],
                dtype=np.float64,
            )
            if not differences.size:
                metrics[metric] = None
                continue
            rng = np.random.default_rng(seed + group_index)
            samples = rng.choice(
                differences,
                size=(bootstrap_replicates, differences.size),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(samples, (0.025, 0.975))
            metrics[metric] = {
                "mean_difference": float(differences.mean()),
                "ci95": [float(low), float(high)],
            }
        result.append(
            {
                "condition": condition,
                "target_percentage": target,
                "paired_examples": len(pairs),
                "reference_condition": reference_condition,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": seed + group_index,
                "effects": metrics,
            }
        )
    return result


def summarize_selected_localization(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate where each 0.1% ranking spends its physical-edge budget."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ranking_target"]), []).append(row)
    summaries = []
    for target, values in sorted(grouped.items()):
        layer_counts: dict[int, int] = {}
        layer_head_counts: dict[tuple[int, int], int] = {}
        pair_counts: dict[tuple[int, int], int] = {}
        total = 0
        for value in values:
            total += int(value["selected_physical_head_edges"])
            for row in value["top_layers"]:
                layer = int(row["layer"])
                layer_counts[layer] = layer_counts.get(layer, 0) + int(
                    row["selected_physical_head_edges"]
                )
            for row in value.get("layer_heads", value["top_layer_heads"]):
                key = (int(row["layer"]), int(row["head"]))
                layer_head_counts[key] = layer_head_counts.get(key, 0) + int(
                    row["selected_physical_head_edges"]
                )
            for row in value["record_pairs"]:
                key = (
                    int(row["source_record_index"]),
                    int(row["target_record_index"]),
                )
                pair_counts[key] = pair_counts.get(key, 0) + int(
                    row["selected_physical_head_edges"]
                )

        def top_rows(counts: Mapping[tuple[int, ...] | int, int], count: int):
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            return [
                {
                    "key": list(key) if isinstance(key, tuple) else [key],
                    "selected_physical_head_edges": value,
                    "selected_fraction": value / max(total, 1),
                }
                for key, value in ordered[:count]
            ]

        summaries.append(
            {
                "ranking_target": target,
                "examples": len(values),
                "target_percentage": 0.1,
                "selected_physical_head_edges": total,
                "top_layers": top_rows(layer_counts, 8),
                "top_layer_heads": top_rows(layer_head_counts, 16),
                "top_record_index_pairs": top_rows(pair_counts, 8),
            }
        )
    return summaries


def oracle_gate(summary: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the prespecified inception gate to top-edge conditions only."""

    eligible = [
        row
        for row in summary
        if row["condition"] == "ORACLE_TOP_ATTENTION"
        and float(row["target_percentage"] or 0.0) <= 5.0
        and row["token_f1"] is not None
        and row["official_score"] is not None
    ]
    passing = [
        row
        for row in eligible
        if float(row["token_f1"]) >= 0.19 and float(row["official_score"]) >= 0.67
    ]
    return {
        "schema_version": "paper3.3-oracle-headroom-gate-v1",
        "status": "PASS_SMOKE" if passing else "FAIL_SMOKE",
        "criteria": {
            "maximum_dense_edge_fraction": 0.05,
            "minimum_token_f1": 0.19,
            "minimum_official_score": 0.67,
        },
        "qualifier": (
            "A small natural cohort can establish mechanism headroom only; it "
            "cannot qualify the learned-selector claim."
        ),
        "passing_conditions": passing,
    }


def ranking_frontier_diagnostics(
    summary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Report monotonicity explicitly instead of inferring it from one point."""

    conditions = sorted(
        {
            str(row["condition"])
            for row in summary
            if str(row["condition"]).startswith("ORACLE_")
            and row["condition"] != "ORACLE_CUMULATIVE_MASS"
        }
    )
    result = []
    for condition in conditions:
        all_points = sorted(
            (row for row in summary if row["condition"] == condition),
            key=lambda row: float(row["target_percentage"] or 0.0),
        )
        points = [
            row for row in all_points if float(row["target_percentage"] or 0.0) <= 1.0
        ]
        f1_violations = []
        official_violations = []
        for left, right in zip(points, points[1:]):
            if float(right["token_f1"] or 0.0) + 1e-12 < float(left["token_f1"] or 0.0):
                f1_violations.append(
                    [left["target_percentage"], right["target_percentage"]]
                )
            if float(right["official_score"] or 0.0) + 1e-12 < float(
                left["official_score"] or 0.0
            ):
                official_violations.append(
                    [left["target_percentage"], right["target_percentage"]]
                )
        best = (
            max(points, key=lambda row: float(row["token_f1"] or 0.0))
            if points
            else None
        )
        result.append(
            {
                "condition": condition,
                "points": len(all_points),
                "monotonicity_scope_maximum_percentage": 1.0,
                "token_f1_monotonic_non_decreasing": not f1_violations,
                "official_score_monotonic_non_decreasing": not official_violations,
                "token_f1_violation_intervals": f1_violations,
                "official_score_violation_intervals": official_violations,
                "best_at_or_below_one_percent": (
                    {
                        "target_percentage": best["target_percentage"],
                        "token_f1": best["token_f1"],
                        "official_score": best["official_score"],
                    }
                    if best is not None
                    else None
                ),
            }
        )
    return result


def interventional_oracle_gate(
    summary: Sequence[Mapping[str, object]],
    frontiers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Gate learning on powered quality and a non-erratic sparse frontier."""

    eligible = [
        row
        for row in summary
        if str(row["condition"]).startswith("ORACLE_")
        and row["condition"] != "ORACLE_CUMULATIVE_MASS"
        and float(row["target_percentage"] or 0.0) <= 5.0
        and row["token_f1"] is not None
        and row["official_score"] is not None
    ]
    powered = bool(eligible) and min(int(row["examples"]) for row in eligible) >= 100
    monotonic = {
        str(row["condition"]): bool(row["token_f1_monotonic_non_decreasing"])
        for row in frontiers
    }
    passing = [
        row
        for row in eligible
        if float(row["token_f1"]) >= 0.19
        and float(row["official_score"]) >= 0.67
        and monotonic.get(str(row["condition"]), False)
    ]
    unlock = powered and bool(passing)
    return {
        "schema_version": "paper3.3-interventional-oracle-gate-v1",
        "status": "PASS_POWERED" if unlock else "LOCKED",
        "powered_cohort_required": 100,
        "criteria": {
            "maximum_dense_edge_fraction": 0.05,
            "minimum_token_f1": 0.19,
            "minimum_official_score": 0.67,
            "token_f1_frontier_monotonic_non_decreasing": True,
        },
        "powered_cohort_observed": powered,
        "learned_selector_training_unlocked": unlock,
        "passing_conditions": passing,
    }


def _condition_row(
    *,
    condition: str,
    question: object,
    backend: PersistentMLXBackend,
    memory: object,
    encode_ms: float,
    selection_receipt_id: str,
    reference_logits: object | None,
    reference_condition: str,
    plan: object | None = None,
    execution: tuple[str, dict[str, object], object | None] | None = None,
) -> dict[str, object]:
    prediction, metrics, logits = execution or _execute(backend, question, memory)
    distribution = (
        {
            "first_step_logit_max_abs_delta": 0.0,
            "first_step_logit_mean_abs_delta": 0.0,
            "first_step_js_divergence": 0.0,
            "first_step_kl_reference_to_condition": 0.0,
        }
        if reference_logits is None
        else _distribution_diagnostics(reference_logits, logits)
    )
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "example_id": getattr(question, "example_id"),
        "condition": condition,
        "selection_receipt_id": selection_receipt_id,
        "distribution_reference_condition": reference_condition,
        "prediction": prediction,
        "encode_ms": encode_ms,
        **metrics,
        **distribution,
    }
    if plan is not None:
        receipt = plan.to_dict()
        row.update(receipt)
        row["target_percentage"] = float(receipt["target"]) * 100.0
    return row


def _encode_sparse_plan(
    *,
    backend: PersistentMLXBackend,
    packed_tokens: Sequence[int],
    blocked_mask: Sequence[Sequence[bool]],
    revision: str,
    graph: object,
    plan: object,
) -> tuple[object, float]:
    """Encode one replay plan through the unchanged host attention path."""

    started = time.perf_counter()
    memory = encode_native_memory_with_mask(
        backend.model,
        packed_tokens,
        blocked_mask,
        model_revision=revision,
        sparse_mask_provider=lambda layer, _heads: plan.mask_for_layer(
            layer,
            base_mask=blocked_mask,
            source_tokens=graph.source_tokens,
            target_tokens=graph.target_tokens,
        ),
    )
    return memory, (time.perf_counter() - started) * 1000.0


def _score_only(
    backend: PersistentMLXBackend, question: object, memory: object
) -> tuple[dict[str, object], object | None]:
    """Score the gold answer without decoding, for inexpensive interventions."""

    query_tokens = list(
        backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
    )
    scoring = backend._score_gold_answer(
        question,
        query_tokens,
        make_native_prompt_cache(backend.model, memory),
        include_first_step_logits=True,
    )
    logits = scoring.pop("_first_step_logits_f32", None)
    return scoring, logits


def _measure_group_interventions(
    *,
    graph: object,
    kinds: Sequence[InteractionGroupKind],
    backend: PersistentMLXBackend,
    question: object,
    packed_tokens: Sequence[int],
    blocked_mask: Sequence[Sequence[bool]],
    revision: str,
    teacher_nll: float,
    teacher_logits: object,
) -> tuple[
    list[dict[str, object]],
    dict[InteractionGroupKind, dict[InteractionGroupKey, float]],
    dict[InteractionGroupKind, dict[InteractionGroupKey, float]],
]:
    """Estimate leave-one-group-out utility using NLL and first-step JS."""

    rows: list[dict[str, object]] = []
    nll_utilities: dict[InteractionGroupKind, dict[InteractionGroupKey, float]] = {}
    js_utilities: dict[InteractionGroupKind, dict[InteractionGroupKey, float]] = {}
    for kind in kinds:
        nll_utilities[kind] = {}
        js_utilities[kind] = {}
        keys = interaction_group_keys(graph, kind)
        print(f"  measuring {len(keys)} {kind} ablations", flush=True)
        for group_index, key in enumerate(keys, 1):
            if group_index == 1 or group_index % 8 == 0 or group_index == len(keys):
                print(f"    {kind} {group_index}/{len(keys)}", flush=True)
            plan = interaction_group_ablation_plan(graph, kind, key)
            memory, encode_ms = _encode_sparse_plan(
                backend=backend,
                packed_tokens=packed_tokens,
                blocked_mask=blocked_mask,
                revision=revision,
                graph=graph,
                plan=plan,
            )
            scoring, logits = _score_only(backend, question, memory)
            ablated_nll = float(scoring["gold_answer_mean_nll"])
            nll_delta = ablated_nll - teacher_nll
            distribution = _distribution_diagnostics(teacher_logits, logits)
            js_delta = float(distribution["first_step_js_divergence"] or 0.0)
            nll_utilities[kind][key] = nll_delta
            js_utilities[kind][key] = js_delta
            rows.append(
                {
                    "schema_version": "paper3.3-group-intervention-v1",
                    "example_id": getattr(question, "example_id"),
                    "graph_digest": graph.graph_digest,
                    "group_kind": kind,
                    "group_key": list(key),
                    "ablated_physical_edge_fraction": 1.0
                    - plan.selected_physical_edge_fraction,
                    "teacher_gold_answer_mean_nll": teacher_nll,
                    "ablated_gold_answer_mean_nll": ablated_nll,
                    "gold_answer_nll_delta": nll_delta,
                    "first_step_js_divergence": js_delta,
                    "first_step_logit_max_abs_delta": distribution[
                        "first_step_logit_max_abs_delta"
                    ],
                    "encode_ms": encode_ms,
                    "gold_scoring_ms": scoring.get("gold_scoring_ms"),
                }
            )
    return rows, nll_utilities, js_utilities


def _ranking_indices(
    *,
    graph: object,
    target: str,
    attention_ranking: object,
    maximum_count: int,
    nll_utilities: Mapping[InteractionGroupKind, Mapping[InteractionGroupKey, float]],
    js_utilities: Mapping[InteractionGroupKind, Mapping[InteractionGroupKey, float]],
) -> object:
    """Construct one observational or interventional physical-edge ranking."""

    if target == "attention":
        return attention_ranking
    kind = _ranking_group(target)
    if kind is None:
        raise ValueError(f"unsupported ranking target: {target}")
    utilities = js_utilities[kind] if target.endswith("_js") else nll_utilities[kind]
    combination = (
        "utility_x_attention" if target.endswith("_x_attention") else "lexicographic"
    )
    return ranked_physical_prefix_by_group_utility(
        graph, kind, utilities, maximum_count, combination=combination
    )


def _mask_cache_key(plan: object) -> str:
    return hashlib.sha256(plan.selected_mask.tobytes()).hexdigest()


def _plot(
    summary: Sequence[Mapping[str, object]],
    localization: Sequence[Mapping[str, object]],
    selected_localization: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    top = [row for row in summary if row["condition"] == "ORACLE_TOP_ATTENTION"]
    if top:
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        percentages = [float(row["target_percentage"]) for row in top]
        positions = list(range(len(percentages)))
        labels = [f"{value:g}" for value in percentages]
        axes[0].plot(
            positions,
            [float(row["token_f1"] or 0.0) for row in top],
            marker="o",
            label="Token F1",
        )
        axes[0].plot(
            positions,
            [float(row["official_score"] or 0.0) for row in top],
            marker="s",
            label="Official",
        )
        axes[0].axhline(
            0.19, color="black", linestyle="--", linewidth=1, label="F1 gate"
        )
        axes[0].set_xlabel("Dense cross-document edges retained (%)")
        axes[0].set_ylabel("Task score")
        axes[0].set_xticks(positions, labels, rotation=35, ha="right")
        axes[0].legend(fontsize=8)
        axes[1].plot(
            positions,
            [100.0 * float(row["retained_attention_mass"]) for row in top],
            marker="o",
        )
        axes[1].set_xlabel("Dense cross-document edges retained (%)")
        axes[1].set_ylabel("Teacher attention mass retained (%)")
        axes[1].set_xticks(positions, labels, rotation=35, ha="right")
        figure.tight_layout()
        figure.savefig(output / "oracle_quality_frontier.pdf", bbox_inches="tight")
        figure.savefig(
            output / "oracle_quality_frontier.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

    interventional = [
        row
        for row in summary
        if str(row["condition"]).startswith("ORACLE_")
        and row["condition"] != "ORACLE_CUMULATIVE_MASS"
    ]
    conditions = tuple(dict.fromkeys(str(row["condition"]) for row in interventional))
    if len(conditions) > 1:
        figure, axis = plt.subplots(figsize=(8.4, 4.2))
        for condition in conditions:
            values = [row for row in interventional if row["condition"] == condition]
            values.sort(key=lambda row: float(row["target_percentage"] or 0.0))
            axis.plot(
                [float(row["target_percentage"] or 0.0) for row in values],
                [float(row["token_f1"] or 0.0) for row in values],
                marker="o",
                label=condition.removeprefix("ORACLE_").replace("_", " ").title(),
            )
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("Physical cross-document edges retained (%)")
        axis.set_ylabel("Token F1")
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(output / "ranking_target_frontiers.pdf", bbox_inches="tight")
        figure.savefig(
            output / "ranking_target_frontiers.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

    if localization:
        layer_count = (
            max(int(row["layer"]) for item in localization for row in item["layers"])
            + 1
        )
        layer_mass = [0.0] * layer_count
        for item in localization:
            for row in item["layers"]:
                layer_mass[int(row["layer"])] += float(row["attention_mass_fraction"])
        total = sum(layer_mass) or 1.0
        layer_mass = [value / total for value in layer_mass]
        figure, axis = plt.subplots(figsize=(7.2, 3.4))
        axis.bar(range(layer_count), layer_mass, color="#2f6f9f")
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Cross-document attention mass fraction")
        figure.tight_layout()
        figure.savefig(output / "oracle_layer_localization.pdf", bbox_inches="tight")
        figure.savefig(
            output / "oracle_layer_localization.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

    selected_targets = tuple(
        dict.fromkeys(str(item["ranking_target"]) for item in selected_localization)
    )
    if selected_targets:
        layer_count = (
            max(
                int(row["layer"])
                for item in selected_localization
                for row in item.get("layer_heads", item["top_layer_heads"])
            )
            + 1
        )
        head_count = (
            max(
                int(row["head"])
                for item in selected_localization
                for row in item.get("layer_heads", item["top_layer_heads"])
            )
            + 1
        )
        figure, axes = plt.subplots(
            len(selected_targets),
            1,
            figsize=(8.5, max(2.8 * len(selected_targets), 3.2)),
            squeeze=False,
        )
        for axis, target in zip(axes[:, 0], selected_targets):
            counts = np.zeros((layer_count, head_count), dtype=float)
            for item in selected_localization:
                if item["ranking_target"] != target:
                    continue
                for row in item.get("layer_heads", item["top_layer_heads"]):
                    counts[int(row["layer"]), int(row["head"])] += float(
                        row["selected_physical_head_edges"]
                    )
            total = counts.sum() or 1.0
            image = axis.imshow(
                counts.T / total,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap="viridis",
            )
            axis.set_title(target.replace("_", " ").title())
            axis.set_xlabel("Decoder layer")
            axis.set_ylabel("Query head")
            figure.colorbar(image, ax=axis, label="Selected-edge share")
        figure.tight_layout()
        figure.savefig(output / "selected_edge_layer_head.pdf", bbox_inches="tight")
        figure.savefig(
            output / "selected_edge_layer_head.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("fixture", "multihoprag"), default="multihoprag"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--reranker-device", default="auto")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--edge-percentages", type=_percentages, default=DEFAULT_EDGE_PERCENTAGES
    )
    parser.add_argument(
        "--mass-percentages", type=_percentages, default=DEFAULT_MASS_PERCENTAGES
    )
    parser.add_argument(
        "--ranking-targets", type=_ranking_targets, default=("attention",)
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--split-name", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--no-frozen-split", action="store_true")
    parser.add_argument("--skip-mass-frontier", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "fixture":
        documents, questions, dataset_metadata = controlled_fixture(seed=args.seed)
    else:
        documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    split_manifest = (
        None
        if args.no_frozen_split or args.dataset == "fixture"
        else args.split_manifest
    )
    questions, split_metadata = _select_split_cohort(
        questions,
        split_manifest=split_manifest,
        split_name=args.split_name,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    revision = _resolve_hf_revision(args.model, args.revision)
    reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
    reranker_device = _resolve_reranker_device(args.reranker_device)
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        device=reranker_device,
        name_prefix="paper3_3_oracle",
    )
    backend = PersistentMLXBackend(args.model, revision, args.max_new_tokens)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    args.output.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_configuration = {
        "dataset": args.dataset,
        "dataset_revision": dataset_metadata["dataset_revision"],
        "model": args.model,
        "model_revision": revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "reranker_device": reranker_device,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "max_resources": args.max_resources,
        "max_new_tokens": args.max_new_tokens,
        "edge_percentages": list(args.edge_percentages),
        "mass_percentages": []
        if args.skip_mass_frontier
        else list(args.mass_percentages),
        "ranking_targets": list(args.ranking_targets),
        "split_name": args.split_name if split_manifest is not None else None,
        "split_digest": split_metadata.get("split_digest") if split_metadata else None,
    }
    run_configuration_digest = hashlib.sha256(
        json.dumps(run_configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    rows: list[dict[str, object]] = []
    graph_summaries: list[dict[str, object]] = []
    localizations: list[dict[str, object]] = []
    selected_localizations: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    started = time.time()
    for question_index, question in enumerate(questions, 1):
        print(f"[{question_index}/{len(questions)}] {question.example_id}", flush=True)
        checkpoint_path = checkpoint_dir / (
            hashlib.sha256(question.example_id.encode()).hexdigest()[:16] + ".json"
        )
        if args.resume and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("run_configuration_digest") != run_configuration_digest:
                raise RuntimeError(
                    f"checkpoint configuration mismatch for {question.example_id}; "
                    "use a different output directory"
                )
            rows.extend(checkpoint["rows"])
            graph_summaries.append(checkpoint["graph_summary"])
            localizations.append(checkpoint["localization"])
            selected_localizations.extend(checkpoint.get("selected_localizations", []))
            intervention_rows.extend(checkpoint.get("interventions", []))
            receipts.append(checkpoint["receipt"])
            print("  resumed completed question", flush=True)
            continue
        row_start = len(rows)
        selected_localization_start = len(selected_localizations)
        intervention_start = len(intervention_rows)
        candidate = make_candidate_receipt(
            dataset=args.dataset,
            dataset_revision=dataset_metadata["dataset_revision"],
            corpus_revision=dataset_metadata["corpus_revision"],
            corpus_sha256=dataset_metadata["corpus_sha256"],
            question=question,
            retriever=retriever,
            candidate_count=args.candidate_count,
            chunker=chunker,
            ensure_gold=False,
            seed=args.seed,
        )
        prepared = prepare_candidate_context(
            candidate, by_id, token_count=backend.token_count
        )
        ranking_started = time.perf_counter()
        ranking = selector.rank(question.question, prepared.chunks)
        ranking_ms = (time.perf_counter() - ranking_started) * 1000.0
        context = packed_context_from_ranking(
            condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
            selector_name=selector.name,
            ranked=ranking,
            prepared=prepared,
            token_budget=args.token_budget,
            selector_latency_ms=ranking_ms,
        )
        selected = tuple(context.chunks[: args.max_resources])
        if len(selected) < 2:
            continue
        context = replace(
            context,
            chunks=selected,
            packed_tokens=sum(row.chunk.token_count for row in selected),
            candidate_chunks=prepared.chunks,
        )
        selection = SelectionReceipt.from_context(
            candidate_receipt_id=candidate.receipt_id,
            example_id=question.example_id,
            context=context,
            selector_revision=selector.name,
        )
        texts = tuple(row.chunk.text for row in selected)
        document_ids = tuple(row.chunk.document_id for row in selected)
        record_ids = tuple(row.chunk.chunk_id for row in selected)
        segments = _token_segments(backend.tokenizer, texts)
        lengths = tuple(len(segment) for segment in segments)
        packed_tokens = tuple(token for segment in segments for token in segment)
        records = tuple(
            ContextRecord(
                record_id="pra://multihoprag/chunk/" + quote(record_id, safe=""),
                record_type=RecordType.RAG_CHUNK,
                payload=text,
                selection_provenance={
                    "selection_receipt_id": selection.receipt_id,
                    "rank": row.rank,
                    "score": row.score,
                    "document_id": row.chunk.document_id,
                },
                version=dataset_metadata["dataset_revision"],
            )
            for row, record_id, text in zip(selected, record_ids, texts)
        )
        resources = tuple(
            SelectedResource(
                resource_id=row.chunk.chunk_id,
                chunk_id=row.chunk.chunk_id,
                source_sha256=hashlib.sha256(
                    row.chunk.text.encode("utf-8")
                ).hexdigest(),
                source_positions=tuple(range(len(segment))),
                rank=row.rank,
                score=row.score,
            )
            for row, segment in zip(selected, segments)
        )
        composition = compose_resources(
            resources,
            selection_receipt_id=selection.receipt_id,
            profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
            position_policy=PositionPolicy.GLOBAL_PACKED,
            near_gap=0,
        )
        full_mask, full_receipt = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.FULL_CAUSAL
        )
        blocked_mask, blocked_receipt = build_document_attention_mask(
            lengths, policy=DocumentAttentionPolicy.NO_CROSS_DOC
        )

        collector = CrossDocumentAttentionCollector(
            lengths,
            record_ids=record_ids,
            selection_receipt_id=selection.receipt_id,
            model_revision=revision,
        )
        encode_started = time.perf_counter()
        instrumented = encode_native_memory_with_mask(
            backend.model,
            packed_tokens,
            full_mask,
            model_revision=revision,
            attention_observer=collector.observe,
        )
        instrumented_ms = (time.perf_counter() - encode_started) * 1000.0
        graph = collector.finalize()
        graph_path = (
            graph_dir
            / f"{hashlib.sha256(question.example_id.encode()).hexdigest()[:16]}.npz"
        )
        graph.save(graph_path)
        localization = interaction_localization(graph)
        localization["example_id"] = question.example_id
        localizations.append(localization)
        graph_summary = graph.summary()
        graph_summary.update(
            {
                "example_id": question.example_id,
                "path": str(graph_path.relative_to(args.output)),
            }
        )
        graph_summaries.append(graph_summary)
        sparse_percentages = [
            percentage for percentage in args.edge_percentages if percentage < 100.0
        ]
        maximum_sparse_fraction = max(sparse_percentages, default=0.0) / 100.0
        maximum_ranked_count = int(
            math.ceil(graph.physical_edge_count * maximum_sparse_fraction)
        )
        ranked_edges = (
            ranked_physical_indices(graph)
            if not args.skip_mass_frontier
            else ranked_physical_prefix(graph, maximum_ranked_count)
        )

        encode_started = time.perf_counter()
        packed = encode_native_memory(
            backend.model, packed_tokens, model_revision=revision
        )
        packed_ms = (time.perf_counter() - encode_started) * 1000.0
        host_diagnostic = native_memory_diagnostics(packed, instrumented)
        encode_started = time.perf_counter()
        blocked = encode_native_memory_with_mask(
            backend.model, packed_tokens, blocked_mask, model_revision=revision
        )
        blocked_ms = (time.perf_counter() - encode_started) * 1000.0
        encode_started = time.perf_counter()
        independent_pre = tuple(
            encode_native_memory(
                backend.model,
                segment,
                position_binding_mode=PositionBindingMode.PRE_ROPE,
                model_revision=revision,
            )
            for segment in segments
        )
        independent = rebind_native_memories_to_receipt(
            backend.model, independent_pre, composition
        )
        independent_ms = (time.perf_counter() - encode_started) * 1000.0

        packed_execution = _execute(backend, question, packed)
        packed_logits = packed_execution[2]
        instrumented_execution = _execute(backend, question, instrumented)
        explicit_teacher_logits = instrumented_execution[2]
        packed_row = _condition_row(
            condition="PACKED_RAG_HOST",
            question=question,
            backend=backend,
            memory=packed,
            encode_ms=packed_ms,
            selection_receipt_id=selection.receipt_id,
            reference_logits=None,
            reference_condition="SELF",
            execution=packed_execution,
        )
        rows.append(packed_row)
        rows.append(
            _condition_row(
                condition="PACKED_RAG_INSTRUMENTED",
                question=question,
                backend=backend,
                memory=instrumented,
                encode_ms=instrumented_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=packed_logits,
                reference_condition="PACKED_RAG_HOST",
                execution=instrumented_execution,
            )
        )
        rows.append(
            _condition_row(
                condition="NO_CROSS_DOC_PACKED",
                question=question,
                backend=backend,
                memory=blocked,
                encode_ms=blocked_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=explicit_teacher_logits,
                reference_condition="PACKED_RAG_INSTRUMENTED",
            )
        )
        rows.append(
            _condition_row(
                condition="INDEPENDENT_PRA",
                question=question,
                backend=backend,
                memory=independent,
                encode_ms=independent_ms,
                selection_receipt_id=selection.receipt_id,
                reference_logits=packed_logits,
                reference_condition="PACKED_RAG_HOST",
            )
        )

        teacher_nll = float(instrumented_execution[1]["gold_answer_mean_nll"])
        required_kinds = tuple(
            kind
            for kind in ("document_pair", "layer", "layer_head")
            if any(_ranking_group(target) == kind for target in args.ranking_targets)
        )
        current_interventions, nll_utilities, js_utilities = (
            _measure_group_interventions(
                graph=graph,
                kinds=required_kinds,
                backend=backend,
                question=question,
                packed_tokens=packed_tokens,
                blocked_mask=blocked_mask,
                revision=revision,
                teacher_nll=teacher_nll,
                teacher_logits=explicit_teacher_logits,
            )
        )
        intervention_rows.extend(current_interventions)

        full_oracle_diagnostic: dict[str, object] | None = None
        execution_cache: dict[
            str, tuple[object, float, tuple[str, dict[str, object], object | None]]
        ] = {}
        for target in args.ranking_targets:
            print(f"  replaying {target} frontier", flush=True)
            target_ranking = _ranking_indices(
                graph=graph,
                target=target,
                attention_ranking=ranked_edges,
                maximum_count=maximum_ranked_count,
                nll_utilities=nll_utilities,
                js_utilities=js_utilities,
            )
            condition = _ranking_condition(target)
            for percentage in args.edge_percentages:
                if percentage == 100.0:
                    plan = full_interaction_plan(
                        graph, mode=condition.removeprefix("ORACLE_")
                    )
                elif target == "attention":
                    plan = top_attention_edge_plan(
                        graph, percentage / 100.0, ranked=target_ranking
                    )
                else:
                    plan = ranked_edge_plan(
                        graph,
                        percentage / 100.0,
                        ranked=target_ranking,
                        mode=condition.removeprefix("ORACLE_"),
                    )
                cache_key = _mask_cache_key(plan)
                cached = execution_cache.get(cache_key)
                if cached is None:
                    memory, encode_ms = _encode_sparse_plan(
                        backend=backend,
                        packed_tokens=packed_tokens,
                        blocked_mask=blocked_mask,
                        revision=revision,
                        graph=graph,
                        plan=plan,
                    )
                    execution = _execute(backend, question, memory)
                    if percentage in (0.0, 100.0):
                        execution_cache[cache_key] = (memory, encode_ms, execution)
                else:
                    memory, encode_ms, execution = cached
                if percentage == 100.0 and full_oracle_diagnostic is None:
                    full_oracle_diagnostic = native_memory_diagnostics(
                        instrumented, memory
                    )
                rows.append(
                    _condition_row(
                        condition=condition,
                        question=question,
                        backend=backend,
                        memory=memory,
                        encode_ms=encode_ms,
                        selection_receipt_id=selection.receipt_id,
                        reference_logits=explicit_teacher_logits,
                        reference_condition="PACKED_RAG_INSTRUMENTED",
                        plan=plan,
                        execution=execution,
                    )
                )
                if percentage == 0.1:
                    selected_localization = selected_interaction_localization(
                        graph, plan
                    )
                    selected_localization.update(
                        {
                            "example_id": question.example_id,
                            "ranking_target": target,
                        }
                    )
                    selected_localizations.append(selected_localization)
        if not args.skip_mass_frontier:
            for percentage in args.mass_percentages:
                plan = cumulative_attention_mass_plan(
                    graph, percentage / 100.0, ranked=ranked_edges
                )
                memory, encode_ms = _encode_sparse_plan(
                    backend=backend,
                    packed_tokens=packed_tokens,
                    blocked_mask=blocked_mask,
                    revision=revision,
                    graph=graph,
                    plan=plan,
                )
                rows.append(
                    _condition_row(
                        condition="ORACLE_CUMULATIVE_MASS",
                        question=question,
                        backend=backend,
                        memory=memory,
                        encode_ms=encode_ms,
                        selection_receipt_id=selection.receipt_id,
                        reference_logits=explicit_teacher_logits,
                        reference_condition="PACKED_RAG_INSTRUMENTED",
                        plan=plan,
                    )
                )
        receipt = {
            "example_id": question.example_id,
            "candidate_receipt": candidate.to_dict(),
            "selection_receipt": selection.to_dict(),
            "record_contracts": [
                {
                    "record_id": record.record_id,
                    "record_type": record.record_type.value,
                    "version": record.version,
                    "source_fingerprint": record.source_fingerprint,
                }
                for record in records
            ],
            "document_ids": list(document_ids),
            "record_ids": list(record_ids),
            "document_lengths": list(lengths),
            "full_mask_receipt": full_receipt.to_dict(),
            "blocked_mask_receipt": blocked_receipt.to_dict(),
            "instrumented_vs_host_diagnostic": host_diagnostic,
            "instrumented_host_parity": (
                float(host_diagnostic["max_key_abs_delta"]) < 1e-5
                and float(host_diagnostic["max_value_abs_delta"]) < 1e-5
            ),
            "full_oracle_vs_instrumented_diagnostic": full_oracle_diagnostic,
            "full_oracle_replay_parity": bool(
                full_oracle_diagnostic
                and float(full_oracle_diagnostic["max_key_abs_delta"]) < 1e-5
                and float(full_oracle_diagnostic["max_value_abs_delta"]) < 1e-5
            ),
        }
        receipts.append(receipt)
        _atomic_json(
            checkpoint_path,
            {
                "schema_version": "paper3.3-oracle-question-checkpoint-v1",
                "example_id": question.example_id,
                "run_configuration_digest": run_configuration_digest,
                "ranking_targets": list(args.ranking_targets),
                "edge_percentages": list(args.edge_percentages),
                "rows": rows[row_start:],
                "graph_summary": graph_summary,
                "localization": localization,
                "selected_localizations": selected_localizations[
                    selected_localization_start:
                ],
                "interventions": intervention_rows[intervention_start:],
                "receipt": receipt,
            },
        )

    summary = summarize_rows(rows)
    paired_effects = paired_bootstrap_effects(rows)
    selected_localization_summary = summarize_selected_localization(
        selected_localizations
    )
    gate = oracle_gate(summary)
    frontier_diagnostics = ranking_frontier_diagnostics(summary)
    causal_gate = interventional_oracle_gate(summary, frontier_diagnostics)
    observed_questions = len({str(row["example_id"]) for row in rows})
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper3.3_oracle_cross_document_sparsity",
        "scope": (
            "powered_natural_interventional_gate"
            if args.dataset == "multihoprag" and observed_questions >= 100
            else "small_natural_mechanism_gate"
            if args.dataset == "multihoprag"
            else "fixture_smoke"
        ),
        "evidence_tier": (
            "MEASURED_POWERED"
            if args.dataset == "multihoprag" and observed_questions >= 100
            else "MEASURED_SMOKE"
        ),
        "dataset": dataset_metadata,
        "model": args.model,
        "model_revision": revision,
        "reranker": args.reranker,
        "reranker_revision": reranker_revision,
        "reranker_device": reranker_device,
        "seed": args.seed,
        "questions": observed_questions,
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "edge_percentages": list(args.edge_percentages),
        "mass_percentages": (
            [] if args.skip_mass_frontier else list(args.mass_percentages)
        ),
        "ranking_targets": list(args.ranking_targets),
        "run_configuration": run_configuration,
        "run_configuration_digest": run_configuration_digest,
        "frozen_split": (
            {
                "manifest": str(split_manifest),
                "name": args.split_name,
                "split_digest": split_metadata["split_digest"],
            }
            if split_manifest is not None and split_metadata is not None
            else None
        ),
        "selector_frozen": True,
        "oracle_granularity": "layer_head_token_pair",
        "base_model_weights_frozen": True,
        "persistent_records_mutated": False,
        "git_commit": _git_commit(),
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "elapsed_seconds": time.time() - started,
        "inherited_residual_baseline": {
            "source": "docs/papers/shared/results/paper3_2_rag/crossdoc_adapter/qwen3_1_7b_rank8_five_seed/manifest.json",
            "token_f1": 0.153518,
            "official_score": 0.533333,
            "status": "INHERITED_NOT_RERUN",
        },
        "gate": gate,
        "interventional_gate": causal_gate,
        "ranking_frontier_diagnostics": frontier_diagnostics,
        "conditions": summary,
        "paired_bootstrap_effects": paired_effects,
        "graphs": graph_summaries,
        "localization": localizations,
        "selected_edge_localization": selected_localizations,
        "selected_edge_localization_summary": selected_localization_summary,
        "group_interventions": intervention_rows,
        "receipts": receipts,
        "rows": rows,
    }
    _atomic_json(args.output / "manifest.json", result)
    _atomic_json(args.output / "group_interventions.json", intervention_rows)
    _atomic_json(
        args.output / "selected_edge_localization.json", selected_localizations
    )
    _plot(summary, localizations, selected_localizations, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate": gate["status"],
                "questions": result["questions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
