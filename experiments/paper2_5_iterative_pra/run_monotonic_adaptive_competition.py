"""Validation-first study of monotonic roots and adaptive transition breadth.

The runner replays the frozen Paper-2.5 feature cache.  It first compares root
locking rules with fixed Top-4 propagation, then freezes one root rule per
geometry before comparing fixed Top-1, fixed Top-4, and adaptive Top-1/2/4.
Only validation examples select policies; final claims use held-out examples.
No model, routing projection, or SDK default is trained or modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    oracle_set_metrics,
    validation_partition,
)
from pra_hf.adaptive_competition import (
    AdaptiveCompetitionConfig,
    AdaptiveCompetitionRouter,
    RootLockConfig,
    TransitionPolicyConfig,
    TransitionScores,
    deterministic_topk,
    root_seed_agreement,
)
from pra_hf.native_closure import native_local_qk_scores
from pra_torch.hf import load_hf_routing_projection


FRACTIONS = (0.10, 0.20, 0.30, 0.40)
ROOT_DROP_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15)
ROOT_Z_THRESHOLDS = (-0.5, 0.0, 0.5, 1.0)
ROOT_AGREEMENT_THRESHOLDS = (0.4, 0.6, 0.8)
ADAPTIVE_MODERATE = (0.25, 0.35, 0.45)
ADAPTIVE_HIGH = (0.45, 0.55, 0.65)
GEOMETRIES = ("semantic", "native_rank")


@dataclass
class FrozenTrace:
    """One seed/example's frozen root and source-conditioned transitions."""

    root_scores: torch.Tensor
    transitions: dict[int, TransitionScores]
    transition_traces: dict[int, dict]


def _best_root_local(
    direct_local: torch.Tensor, parent_indices: torch.Tensor, parent: int
) -> int:
    rows = torch.nonzero(parent_indices == parent, as_tuple=False).flatten()
    return int(rows[torch.argmax(direct_local[rows])])


def _parent_max(
    local_scores: torch.Tensor, parent_indices: torch.Tensor, parent_count: int
) -> torch.Tensor:
    output = local_scores.new_full((parent_count,), float("-inf"))
    for parent in range(parent_count):
        rows = parent_indices == parent
        if bool(rows.any()):
            output[parent] = local_scores[rows].max()
    return output


def _transition_for_source(
    feature: dict,
    source: int,
    direct_local: torch.Tensor,
    local_memory: torch.Tensor,
    local_query: torch.Tensor,
    *,
    device: torch.device,
    candidate_fraction: float,
) -> tuple[TransitionScores, dict]:
    """Replay projection-correct local semantics and frozen Top-4 native QK."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    parent_count = len(feature["parent_spans"])
    parent_indices = feature["local_parent_indices"].to(device)
    source_local = _best_root_local(direct_local, parent_indices, source)
    local_scores = local_memory @ local_query[source_local]
    semantic = _parent_max(local_scores, parent_indices, parent_count)
    semantic[source] = float("-inf")

    pool_size = max(1, math.ceil(parent_count * candidate_fraction))
    narrowed = deterministic_topk(semantic, pool_size)
    local_mask = torch.zeros_like(parent_indices, dtype=torch.bool)
    for parent in narrowed:
        local_mask |= parent_indices == parent
    candidate_locals = torch.nonzero(local_mask, as_tuple=False).flatten()
    native = semantic.new_full((parent_count,), float("-inf"))
    native_dots = 0
    if candidate_locals.numel():
        pre_q = feature["local_pre_query"].to(device)
        pre_k = feature["local_pre_key"].to(device)
        masks = feature["local_token_mask"].to(device)
        scored = native_local_qk_scores(
            pre_q[source_local : source_local + 1],
            pre_k[candidate_locals],
            masks[source_local : source_local + 1],
            masks[candidate_locals],
            token_reduction="top_m_mean",
            head_reduction="top_m_mean",
            top_m=4,
        )
        native_dots = scored.dot_products
        for parent in narrowed:
            packed = parent_indices[candidate_locals] == parent
            if bool(packed.any()):
                native[parent] = scored.scores[0, packed].max()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - started
    semantic_order = deterministic_topk(semantic, 4)
    native_order = deterministic_topk(native, 4)
    trace = {
        "source_parent": source,
        "source_local": source_local,
        "semantic_top4": semantic_order,
        "semantic_top4_scores": [float(semantic[parent]) for parent in semantic_order],
        "native_top4": native_order,
        "native_top4_raw_scores": [float(native[parent]) for parent in native_order],
        "native_candidate_parents": narrowed,
    }
    return (
        TransitionScores(
            semantic=semantic.detach().cpu(),
            native_raw=native.detach().cpu(),
            semantic_comparisons=int(local_scores.numel()),
            native_qk_comparisons=native_dots,
            scoring_seconds=scoring_seconds,
        ),
        trace,
    )


def build_frozen_traces(
    features: list[dict],
    seeds: tuple[int, ...],
    fractions: tuple[float, ...],
    projection_dir: Path,
    device: torch.device,
    candidate_fraction: float,
) -> dict[tuple[str, int], FrozenTrace]:
    """Project each seed once and cache every source reachable from root Top-B."""
    traces: dict[tuple[str, int], FrozenTrace] = {}
    for seed in seeds:
        checkpoint = projection_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projection = load_hf_routing_projection(checkpoint, device=device)
        for feature_index, feature in enumerate(features):
            with torch.no_grad():
                root = projection.project_query(
                    feature["query_hidden"].to(device).unsqueeze(0)
                )[0]
                parent_hidden = feature["parent_hidden"].to(device)
                local_hidden = feature["local_hidden"].to(device)
                parent_memory = F.normalize(
                    projection.project_memory(parent_hidden).float(), dim=-1
                )
                local_memory = F.normalize(
                    projection.project_memory(local_hidden).float(), dim=-1
                )
                local_query = F.normalize(
                    projection.project_query(local_hidden).float(), dim=-1
                )
                root = F.normalize(root.float(), dim=-1)
                root_scores = parent_memory @ root
                direct_local = local_memory @ root
                max_budget = max(
                    1,
                    math.ceil(len(feature["parent_spans"]) * max(fractions)),
                )
                sources = deterministic_topk(root_scores, max_budget)
                transitions = {}
                transition_traces = {}
                for source in sources:
                    scores, trace = _transition_for_source(
                        feature,
                        source,
                        direct_local,
                        local_memory,
                        local_query,
                        device=device,
                        candidate_fraction=candidate_fraction,
                    )
                    transitions[source] = scores
                    transition_traces[source] = trace
            traces[(feature["example_id"], seed)] = FrozenTrace(
                root_scores.detach().cpu(), transitions, transition_traces
            )
            del parent_hidden, local_hidden, parent_memory, local_memory, local_query
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"adaptive competition features: seed {seed} complete", flush=True)
    return traces


def _root_policy_candidates() -> list[tuple[str, RootLockConfig]]:
    values = [
        (f"fixed_prefix_{count}", RootLockConfig("fixed", fixed_count=count))
        for count in (1, 2, 4)
    ]
    values.extend(
        (
            f"score_drop_{threshold:g}",
            RootLockConfig("score_drop", threshold=threshold),
        )
        for threshold in ROOT_DROP_THRESHOLDS
    )
    values.extend(
        (f"zscore_{threshold:g}", RootLockConfig("zscore", threshold=threshold))
        for threshold in ROOT_Z_THRESHOLDS
    )
    values.extend(
        (
            f"seed_agreement_{threshold:g}",
            RootLockConfig("seed_agreement", threshold=threshold),
        )
        for threshold in ROOT_AGREEMENT_THRESHOLDS
    )
    return values


def _transition_policy_candidates() -> list[tuple[str, TransitionPolicyConfig]]:
    values = [
        ("fixed_k1", TransitionPolicyConfig("fixed", fixed_k=1)),
        ("fixed_k4", TransitionPolicyConfig("fixed", fixed_k=4)),
    ]
    for moderate in ADAPTIVE_MODERATE:
        for high in ADAPTIVE_HIGH:
            if moderate <= high:
                values.append(
                    (
                        f"adaptive_m{moderate:g}_h{high:g}",
                        TransitionPolicyConfig(
                            "adaptive",
                            moderate_confidence=moderate,
                            high_confidence=high,
                        ),
                    )
                )
    return values


def _agreements(
    feature: dict,
    traces: dict[tuple[str, int], FrozenTrace],
    seeds: tuple[int, ...],
    budget: int,
) -> dict[int, float]:
    rankings = [
        deterministic_topk(traces[(feature["example_id"], seed)].root_scores, budget)
        for seed in seeds
    ]
    return root_seed_agreement(rankings, budget)


def evaluate_policy(
    feature: dict,
    seed: int,
    fraction: float,
    trace: FrozenTrace,
    *,
    root_policy_name: str,
    root_policy: RootLockConfig,
    transition_policy_name: str,
    transition_policy: TransitionPolicyConfig,
    geometry: str,
    agreement: dict[int, float],
    stage: str,
) -> dict:
    """Run the public policy API and attach labels only after selection."""
    budget = max(1, math.ceil(len(feature["parent_spans"]) * fraction))

    def provider(source: int) -> TransitionScores:
        return trace.transitions[source]

    started = time.perf_counter()
    result = AdaptiveCompetitionRouter().route(
        trace.root_scores,
        provider,
        AdaptiveCompetitionConfig(
            total_budget=budget,
            root_lock=root_policy,
            transition=transition_policy,
            transition_geometry=geometry,
        ),
        agreement=agreement,
    )
    wall_time = time.perf_counter() - started
    selected, root_top_b, locked = (
        set(result.selected),
        set(result.root_top_b),
        set(result.locked_roots),
    )
    oracle = canonical_oracle_parent_indices(feature)
    groups = evidence_parent_groups(feature)
    first_group = groups[0] if groups else set()
    later_groups = groups[1:]
    first_present = bool(first_group & root_top_b)
    first_locked = bool(first_group & locked)
    later_recovered = bool(later_groups) and all(
        bool(group & selected) for group in later_groups
    )
    metrics = oracle_set_metrics(selected, oracle)
    selected_tokens = sum(
        int(feature["parent_spans"][parent][1])
        - int(feature["parent_spans"][parent][0])
        for parent in selected
    )
    confidence_rows = [asdict(value) for value in result.transition_confidences]
    transition_traces = [trace.transition_traces[source] for source in result.locked_roots]
    row = {
        "stage": stage,
        "partition": validation_partition(feature["example_id"]),
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "fraction": fraction,
        "budget": budget,
        "root_topB_ids": json.dumps(result.root_top_b),
        "root_scores": json.dumps([float(value) for value in trace.root_scores]),
        "locked_root_ids": json.dumps(result.locked_roots),
        "lock_policy": root_policy_name,
        "root_confidence": json.dumps(
            {
                **result.root_confidence,
                "agreement": {
                    str(parent): agreement.get(parent, 0.0)
                    for parent in result.root_top_b
                }
            },
            sort_keys=True,
        ),
        "oracle_root_state": (
            "locked"
            if bool(oracle & locked)
            else "present_not_locked"
            if bool(oracle & root_top_b)
            else "absent_from_root_topB"
        ),
        "oracle_root_present": float(bool(oracle & root_top_b)),
        "oracle_root_locked": float(bool(oracle & locked)),
        "first_oracle_group_present": float(first_present),
        "first_oracle_group_locked": float(first_locked),
        "second_oracle_recovered_given_first_locked": (
            float(later_recovered) if first_locked and later_groups else None
        ),
        "locked_root_count": len(result.locked_roots),
        "propagation_budget": budget - len(result.locked_roots),
        "propagation_activation": float(bool(result.propagated)),
        "transition_geometry": geometry,
        "transition_policy": transition_policy_name,
        "transition_k": json.dumps(result.transition_ks),
        "mean_transition_k": (
            statistics.fmean(result.transition_ks) if result.transition_ks else 0.0
        ),
        "transition_confidence": json.dumps(confidence_rows, sort_keys=True),
        "transition_score_traces": json.dumps(transition_traces, sort_keys=True),
        "propagated_ids": json.dumps(result.propagated),
        "final_ids": json.dumps(result.selected),
        "chain_complete": float(
            bool(groups) and all(bool(selected & group) for group in groups)
        ),
        **metrics,
        "root_comparisons": result.root_comparisons,
        "transition_comparisons": (
            result.semantic_comparisons + result.native_qk_comparisons
        ),
        "total_routing_comparisons": (
            result.root_comparisons
            + result.semantic_comparisons
            + result.native_qk_comparisons
        ),
        "semantic_comparisons": result.semantic_comparisons,
        "native_qk_comparisons": result.native_qk_comparisons,
        "propagation_hops": int(bool(result.transition_ks)),
        "unique_parents_selected": len(selected),
        "active_final_kv_tokens": selected_tokens,
        "active_final_kv_fraction": selected_tokens / max(feature["source_tokens"], 1),
        "wall_time": wall_time,
        "frozen_scorer_seconds": result.scoring_seconds,
        "estimated_full_routing_seconds": wall_time + result.scoring_seconds,
    }
    if not locked <= selected:
        raise AssertionError("Locked root persistence failed.")
    if len(selected) != min(budget, len(feature["parent_spans"])):
        raise AssertionError("Matched final parent budget was not conserved.")
    return row


def _aggregate(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    metrics = (
        "oracle_recall",
        "complete_oracle",
        "chain_complete",
        "oracle_root_present",
        "oracle_root_locked",
        "first_oracle_group_present",
        "first_oracle_group_locked",
        "second_oracle_recovered_given_first_locked",
        "locked_root_count",
        "propagation_budget",
        "propagation_activation",
        "mean_transition_k",
        "root_comparisons",
        "semantic_comparisons",
        "native_qk_comparisons",
        "total_routing_comparisons",
        "active_final_kv_tokens",
        "active_final_kv_fraction",
        "wall_time",
        "frozen_scorer_seconds",
        "estimated_full_routing_seconds",
    )
    output = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        for metric in metrics:
            samples = [
                float(row[metric])
                for row in values
                if row.get(metric) is not None
            ]
            if samples:
                record[metric] = statistics.fmean(samples)
                record[f"{metric}_n"] = len(samples)
        output.append(record)
    return output


def _validation_objective(rows: list[dict]) -> float:
    """Favor Hotpot chain progress, QASPER preservation, then lower expansion."""
    primary = [row for row in rows if row["fraction"] in {0.20, 0.30, 0.40}]
    hotpot = [row for row in primary if row["dataset"] == "hotpotqa"]
    qasper = [row for row in primary if row["dataset"] == "qasper"]
    return (
        statistics.fmean(row["chain_complete"] for row in hotpot)
        + 0.5 * statistics.fmean(row["oracle_recall"] for row in hotpot)
        + statistics.fmean(row["oracle_recall"] for row in qasper)
        - 0.05 * statistics.fmean(row["propagation_activation"] for row in qasper)
    )


def _select_per_geometry(
    rows: list[dict], policy_field: str, *, minimum_propagation: float = 0.0
) -> tuple[dict[str, str], list[dict]]:
    selections, audit = {}, []
    for geometry in GEOMETRIES:
        candidates = sorted({row[policy_field] for row in rows if row["transition_geometry"] == geometry})
        scored = []
        for candidate in candidates:
            values = [
                row
                for row in rows
                if row["transition_geometry"] == geometry
                and row[policy_field] == candidate
                and row["partition"] == "validation"
            ]
            score = _validation_objective(values)
            propagation = statistics.fmean(
                float(row["propagation_activation"]) for row in values
            )
            record = {
                "geometry": geometry,
                "policy": candidate,
                "validation_objective": score,
                "validation_propagation_activation": propagation,
                "validation_rows": len(values),
            }
            audit.append(record)
            if propagation >= minimum_propagation:
                scored.append(record)
        if not scored:
            raise ValueError(
                f"No {geometry} policy meets propagation floor {minimum_propagation}."
            )
        winner = max(
            scored,
            key=lambda row: (row["validation_objective"], -len(row["policy"]), row["policy"]),
        )
        selections[geometry] = winner["policy"]
    return selections, audit


def _baseline_rows(path: Path) -> list[dict]:
    keep = {"one_shot_parent", "local_gist_closure", "native_qk_top4_topk_p20"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] in keep]
    for row in rows:
        row["partition"] = validation_partition(row["example_id"])
        for field in (
            "fraction", "oracle_recall", "complete_oracle", "chain_completion",
            "semantic_gist_comparisons", "native_qk_dot_products", "routing_seconds",
        ):
            row[field] = float(row[field])
    return rows


def _baseline_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["partition"], row["dataset"], row["fraction"], row["method"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                "partition": key[0], "dataset": key[1], "fraction": key[2],
                "method": key[3], "rows": len(values),
                "oracle_recall": statistics.fmean(row["oracle_recall"] for row in values),
                "complete_oracle": statistics.fmean(row["complete_oracle"] for row in values),
                "chain_complete": statistics.fmean(row["chain_completion"] for row in values),
                "semantic_comparisons": statistics.fmean(row["semantic_gist_comparisons"] for row in values),
                "native_qk_comparisons": statistics.fmean(row["native_qk_dot_products"] for row in values),
                "wall_time": statistics.fmean(row["routing_seconds"] for row in values),
            }
        )
    return output


def _synthetic_controls(
    frozen: list[tuple[str, RootLockConfig, str, TransitionPolicyConfig, str]]
) -> list[dict]:
    """Exercise direct, ambiguous, and distractor-heavy bounded decisions."""
    cases = {
        "obvious_direct": {
            "root": torch.tensor([1.0, 0.1, 0.0, -0.1, -0.2]),
            "budget": 1,
            "semantic": torch.tensor([-torch.inf, 0.8, 0.2, 0.1, 0.0]),
            "native": torch.tensor([-torch.inf, 8.0, 4.0, 3.0, 2.0]),
            "oracle": {0},
        },
        "ambiguous_two_hop": {
            "root": torch.tensor([0.70, 0.69, 0.68, 0.67, 0.66, 0.10]),
            "budget": 3,
            "semantic": torch.tensor([-torch.inf, 1.0, 0.7, 0.6, 0.5, 0.99]),
            "native": torch.tensor([-torch.inf, 1.0, 0.7, 0.6, 0.5, 0.99]),
            "oracle": {0, 5},
        },
        "distractor_true_edge_rank4": {
            "root": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.0]),
            "budget": 5,
            "semantic": torch.tensor([-torch.inf, 1.0, 1.0, 1.0, -torch.inf, 1.0]),
            "native": torch.tensor([-torch.inf, 1.0, 1.0, 1.0, -torch.inf, 1.0]),
            "oracle": {0, 5},
        },
    }
    rows = []
    for policy_name, root_policy, transition_name, transition_policy, geometry in frozen:
        for case_name, case in cases.items():
            def provider(_: int, values=case) -> TransitionScores:
                return TransitionScores(values["semantic"], values["native"])

            result = AdaptiveCompetitionRouter().route(
                case["root"], provider,
                AdaptiveCompetitionConfig(
                    case["budget"], root_policy, transition_policy, geometry
                ),
                agreement={index: 1.0 for index in range(case["root"].numel())},
            )
            rows.append(
                {
                    "case": case_name,
                    "policy": f"{geometry}:{policy_name}:{transition_name}",
                    "budget": case["budget"],
                    "locked": json.dumps(result.locked_roots),
                    "propagated": json.dumps(result.propagated),
                    "selected": json.dumps(result.selected),
                    "transition_k": json.dumps(result.transition_ks),
                    "oracle_recall": len(set(result.selected) & case["oracle"]) / len(case["oracle"]),
                    "budget_conserved": float(len(result.selected) == case["budget"]),
                    "locked_persisted": float(set(result.locked_roots) <= set(result.selected)),
                }
            )
    diagnostic_root = RootLockConfig("fixed", fixed_count=1)
    diagnostic_transitions = (
        ("fixed_k1", TransitionPolicyConfig("fixed", fixed_k=1)),
        ("fixed_k4", TransitionPolicyConfig("fixed", fixed_k=4)),
        (
            "adaptive_m0.35_h0.55",
            TransitionPolicyConfig(
                "adaptive", moderate_confidence=0.35, high_confidence=0.55
            ),
        ),
    )
    for geometry in GEOMETRIES:
        for transition_name, transition_policy in diagnostic_transitions:
            for case_name, case in cases.items():
                def provider(_: int, values=case) -> TransitionScores:
                    return TransitionScores(values["semantic"], values["native"])

                result = AdaptiveCompetitionRouter().route(
                    case["root"], provider,
                    AdaptiveCompetitionConfig(
                        case["budget"], diagnostic_root, transition_policy, geometry
                    ),
                )
                rows.append(
                    {
                        "case": case_name,
                        "policy": f"diagnostic:{geometry}:fixed_prefix_1:{transition_name}",
                        "budget": case["budget"],
                        "locked": json.dumps(result.locked_roots),
                        "propagated": json.dumps(result.propagated),
                        "selected": json.dumps(result.selected),
                        "transition_k": json.dumps(result.transition_ks),
                        "oracle_recall": len(set(result.selected) & case["oracle"]) / len(case["oracle"]),
                        "budget_conserved": float(len(result.selected) == case["budget"]),
                        "locked_persisted": float(set(result.locked_roots) <= set(result.selected)),
                    }
                )
    return rows


def _heldout_effects(
    final_summary: list[dict], baseline_summary: list[dict]
) -> list[dict]:
    one_shot = {
        (row["dataset"], row["fraction"]): row
        for row in baseline_summary
        if row["partition"] == "test" and row["method"] == "one_shot_parent"
    }
    output = []
    for row in final_summary:
        if row["partition"] != "test":
            continue
        baseline = one_shot[(row["dataset"], row["fraction"])]
        extra_comparisons = (
            row["root_comparisons"]
            + row["semantic_comparisons"]
            + row["native_qk_comparisons"]
            - baseline["semantic_comparisons"]
            - baseline["native_qk_comparisons"]
        )
        chain_gain = row["chain_complete"] - baseline["chain_complete"]
        oracle_gain = row["oracle_recall"] - baseline["oracle_recall"]
        output.append(
            {
                "dataset": row["dataset"],
                "fraction": row["fraction"],
                "policy_role": row["policy_role"],
                "transition_geometry": row["transition_geometry"],
                "lock_policy": row["lock_policy"],
                "transition_policy": row["transition_policy"],
                "delta_chain_complete_vs_one_shot": chain_gain,
                "delta_oracle_recall_vs_one_shot": oracle_gain,
                "extra_search_comparisons_vs_one_shot": extra_comparisons,
                "chain_gain_per_extra_million_comparisons": (
                    chain_gain * 1_000_000 / extra_comparisons
                    if extra_comparisons > 0 else 0.0
                ),
                "oracle_gain_per_extra_million_comparisons": (
                    oracle_gain * 1_000_000 / extra_comparisons
                    if extra_comparisons > 0 else 0.0
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    final_summary: list[dict], baseline_summary: list[dict], output_dir: Path
) -> None:
    test = [row for row in final_summary if row["partition"] == "test"]
    baseline = [row for row in baseline_summary if row["partition"] == "test"]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    colors = {"semantic": "#1f77b4", "native_rank": "#d62728"}
    for row_index, dataset in enumerate(("hotpotqa", "qasper")):
        for geometry in GEOMETRIES:
            for role, linestyle in (("preservation", ":"), ("exploration", "-")):
                values = sorted(
                    [
                        row for row in test
                        if row["dataset"] == dataset
                        and row["transition_geometry"] == geometry
                        and row["policy_role"] == role
                    ],
                    key=lambda row: row["fraction"],
                )
                for column, metric in enumerate(("chain_complete", "oracle_recall")):
                    axes[row_index, column].plot(
                        [100 * row["fraction"] for row in values],
                        [row[metric] for row in values],
                        marker="o", linestyle=linestyle, color=colors[geometry],
                        label=f"{role} {geometry}",
                    )
        one = sorted(
            [row for row in baseline if row["dataset"] == dataset and row["method"] == "one_shot_parent"],
            key=lambda row: row["fraction"],
        )
        for column, metric in enumerate(("chain_complete", "oracle_recall")):
            axes[row_index, column].plot(
                [100 * row["fraction"] for row in one],
                [row[metric] for row in one],
                marker="s", linestyle="--", color="#222222", label="one-shot",
            )
            axes[row_index, column].set_ylim(0, 1.02)
            axes[row_index, column].grid(alpha=0.25)
        axes[row_index, 0].set_ylabel(f"{dataset}\nrate")
    axes[0, 0].set_title("Evidence-group chain completion")
    axes[0, 1].set_title("Exact Paper-2 oracle recall")
    for axis in axes[-1]:
        axis.set_xlabel("Final parent/KV budget (%)")
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"adaptive_quality.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    primary = [
        row for row in test
        if row["fraction"] == 0.20 and row["policy_role"] == "exploration"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    labels = [f"{row['dataset']}\n{row['transition_geometry']}" for row in primary]
    axes[0].bar(labels, [row["locked_root_count"] for row in primary], label="locked root")
    axes[0].bar(
        labels,
        [row["propagation_budget"] for row in primary],
        bottom=[row["locked_root_count"] for row in primary],
        label="available propagation",
    )
    axes[0].set_title("Adaptive budget allocation")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [row["propagation_activation"] for row in primary], color="#2ca02c")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Propagation activation rate")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"adaptive_regulation.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_file, weights_only=False, map_location="cpu")
    traces = build_frozen_traces(
        features, args.seeds, args.fractions, args.projection_dir, device,
        args.candidate_fraction,
    )
    agreements = {
        (feature["example_id"], fraction): _agreements(
            feature,
            traces,
            args.seeds,
            max(1, math.ceil(len(feature["parent_spans"]) * fraction)),
        )
        for feature in features
        for fraction in args.fractions
    }

    root_rows = []
    fixed_k4 = TransitionPolicyConfig("fixed", fixed_k=4)
    for policy_name, policy in _root_policy_candidates():
        for geometry in GEOMETRIES:
            for feature in features:
                for seed in args.seeds:
                    for fraction in args.fractions:
                        root_rows.append(
                            evaluate_policy(
                                feature, seed, fraction,
                                traces[(feature["example_id"], seed)],
                                root_policy_name=policy_name, root_policy=policy,
                                transition_policy_name="fixed_k4",
                                transition_policy=fixed_k4, geometry=geometry,
                                agreement=agreements[(feature["example_id"], fraction)],
                                stage="root_selection",
                            )
                        )
    preservation_roots, preservation_selection_audit = _select_per_geometry(
        root_rows, "lock_policy"
    )
    exploratory_roots, exploratory_selection_audit = _select_per_geometry(
        root_rows, "lock_policy", minimum_propagation=0.25
    )
    root_selection_audit = []
    for row in preservation_selection_audit:
        root_selection_audit.append({**row, "selection_role": "preservation"})
    for row in exploratory_selection_audit:
        root_selection_audit.append({**row, "selection_role": "exploration"})
    root_configs = dict(_root_policy_candidates())

    transition_rows = []
    for geometry in GEOMETRIES:
        root_name = exploratory_roots[geometry]
        root_policy = root_configs[root_name]
        for transition_name, transition_policy in _transition_policy_candidates():
            for feature in features:
                for seed in args.seeds:
                    for fraction in args.fractions:
                        transition_rows.append(
                            evaluate_policy(
                                feature, seed, fraction,
                                traces[(feature["example_id"], seed)],
                                root_policy_name=root_name, root_policy=root_policy,
                                transition_policy_name=transition_name,
                                transition_policy=transition_policy, geometry=geometry,
                                agreement=agreements[(feature["example_id"], fraction)],
                                stage="transition_selection",
                            )
                        )
    selected_transitions, transition_selection_audit = _select_per_geometry(
        transition_rows, "transition_policy"
    )
    transition_configs = dict(_transition_policy_candidates())

    exploratory_final = [
        row
        for row in transition_rows
        if row["lock_policy"] == exploratory_roots[row["transition_geometry"]]
        and row["transition_policy"] == selected_transitions[row["transition_geometry"]]
    ]
    for row in exploratory_final:
        row["policy_role"] = "exploration"
    preservation_final = [
        row
        for row in root_rows
        if row["lock_policy"] == preservation_roots[row["transition_geometry"]]
    ]
    for row in preservation_final:
        row["policy_role"] = "preservation"
    final_rows = preservation_final + exploratory_final
    final_summary = _aggregate(
        final_rows,
        (
            "partition", "dataset", "fraction", "policy_role",
            "transition_geometry", "lock_policy", "transition_policy",
        ),
    )
    baseline_rows = _baseline_rows(args.prior_convergence_file)
    baseline_summary = _baseline_summary(baseline_rows)
    heldout_effects = _heldout_effects(final_summary, baseline_summary)

    frozen = [
        (
            preservation_roots[geometry], root_configs[preservation_roots[geometry]],
            "fixed_k4", fixed_k4, geometry,
        )
        for geometry in GEOMETRIES
    ] + [
        (
            exploratory_roots[geometry], root_configs[exploratory_roots[geometry]],
            selected_transitions[geometry], transition_configs[selected_transitions[geometry]],
            geometry,
        )
        for geometry in GEOMETRIES
    ]
    synthetic = _synthetic_controls(frozen)

    test_primary = [
        row for row in final_summary
        if row["partition"] == "test" and row["fraction"] == 0.20
    ]
    baseline_primary = [
        row for row in baseline_summary
        if row["partition"] == "test" and row["fraction"] == 0.20
        and row["method"] == "one_shot_parent"
    ]
    one = {row["dataset"]: row for row in baseline_primary}
    adoption = any(
        row["dataset"] == "hotpotqa"
        and row["policy_role"] == "exploration"
        and row["chain_complete"] > one["hotpotqa"]["chain_complete"]
        for row in test_primary
    ) and all(
        row["oracle_recall"] >= one["qasper"]["oracle_recall"] - 0.02
        for row in test_primary
        if row["dataset"] == "qasper" and row["policy_role"] == "exploration"
    )
    hotpot_test = [
        row for row in final_rows
        if row["partition"] == "test"
        and row["dataset"] == "hotpotqa"
        and row["policy_role"] == "exploration"
    ]
    absent_rate = statistics.fmean(
        1.0 - row["first_oracle_group_present"] for row in hotpot_test
    )
    not_locked_rate = statistics.fmean(
        row["first_oracle_group_present"] - row["first_oracle_group_locked"]
        for row in hotpot_test
    )
    recommendation = (
        "A_adopt_monotonic_adaptive_competition"
        if adoption
        else "B_refine_root_discovery_query_decomposition"
        if absent_rate >= not_locked_rate
        else "C_refine_path_scoring"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary = _aggregate(
        root_rows,
        ("partition", "dataset", "fraction", "transition_geometry", "lock_policy", "transition_policy"),
    )
    transition_summary = _aggregate(
        transition_rows,
        ("partition", "dataset", "fraction", "transition_geometry", "lock_policy", "transition_policy"),
    )
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "diagnostic_only": True,
        "production_default_changed": False,
        "training_performed": False,
        "feature_source": str(args.feature_file.relative_to(ROOT)).replace("\\", "/"),
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "candidate_pool_fraction": args.candidate_fraction,
        "native_competition_score": "per_source_raw_pre_rope_qk_rank",
        "root_policy_selection": root_selection_audit,
        "selected_preservation_root_policies": preservation_roots,
        "selected_exploratory_root_policies": exploratory_roots,
        "transition_policy_selection": transition_selection_audit,
        "selected_transition_policies": selected_transitions,
        "heldout_summary": [row for row in final_summary if row["partition"] == "test"],
        "baseline_heldout_summary": [row for row in baseline_summary if row["partition"] == "test"],
        "heldout_effects_vs_one_shot": heldout_effects,
        "synthetic_controls": synthetic,
        "hotpot_failure_decomposition": {
            "first_root_absent_rate": absent_rate,
            "first_root_present_but_not_locked_rate": not_locked_rate,
        },
        "recommendation": recommendation,
        "sdk_change_in_this_iteration": False,
    }
    (args.output_dir / "adaptive_competition_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "root_policy_rows.csv", root_rows)
    _write_csv(args.output_dir / "root_policy_summary.csv", root_summary)
    _write_csv(args.output_dir / "root_policy_selection.csv", root_selection_audit)
    _write_csv(args.output_dir / "transition_policy_rows.csv", transition_rows)
    _write_csv(args.output_dir / "transition_policy_summary.csv", transition_summary)
    _write_csv(args.output_dir / "transition_policy_selection.csv", transition_selection_audit)
    _write_csv(args.output_dir / "heldout_policy_rows.csv", [row for row in final_rows if row["partition"] == "test"])
    _write_csv(args.output_dir / "heldout_policy_summary.csv", [row for row in final_summary if row["partition"] == "test"])
    _write_csv(args.output_dir / "baseline_heldout_summary.csv", [row for row in baseline_summary if row["partition"] == "test"])
    _write_csv(args.output_dir / "heldout_effects_vs_one_shot.csv", heldout_effects)
    _write_csv(args.output_dir / "synthetic_controls.csv", synthetic)
    _plot(final_summary, baseline_summary, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--candidate-fraction", type=float, default=0.20)
    parser.add_argument(
        "--feature-file", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--projection-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--prior-convergence-file", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/oracle_competition_diagnostics/convergence_rows.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition",
    )
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    args.fractions = tuple(map(float, args.fractions.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"recommendation": result["recommendation"]}, indent=2))
