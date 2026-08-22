"""Factorized adaptive-control and bounded-retry experiments for Paper 3.5.

The study replays frozen native graph/query scores.  Validation and held-out
examples share an action lattice but never share evaluator-derived targets.
Large feature caches remain omitted from Git; compact oracle, Pareto, router,
retry, and aggregate control-surface artifacts are written for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_5_iterative_pra.run_semantic_graph_search import (  # noqa: E402
    PreparedExample,
    _prepare_examples,
)
from pra_hf.effort_router import (  # noqa: E402
    ActionField,
    AutoregressiveEffortRouter,
    HashingQueryEncoder,
    MultiHeadEffortRouter,
    RouterActionSpace,
)
from pra_hf.factorized_control import (  # noqa: E402
    BUDGET_LEVELS,
    FACET_LEVELS,
    HOP_LEVELS,
    NEIGHBOR_LEVELS,
    ROOT_LEVELS,
    FactorizedEffortAction,
    allocation_outcome,
    changed_control,
    cheapest_sufficient,
    evidence_kv_metrics,
    factorized_action_space,
    factorized_cost,
    pareto_frontier,
)
from pra_hf.semantic_graph_search import SemanticGraphSearchConfig, search_semantic_graph  # noqa: E402


ROUTER_SEEDS = (1, 7, 21, 42, 87)
FEATURE_NAMES = (
    "question_tokens",
    "parent_count",
    "available_facets",
    "source_tokens",
    "root_gap_f1",
    "root_entropy_f1",
    "root_gap_f4",
    "root_entropy_f4",
    "root_disagreement",
    "top_root_score",
    "root_score_mean",
    "root_score_std",
)


@dataclass
class UnitSurface:
    """In-memory factorized surface and runtime-only router inputs for one unit."""

    partition: str
    dataset: str
    example_id: str
    seed: int
    question: str
    features: torch.Tensor
    rows: list[dict[str, Any]]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.dataset, self.example_id, self.seed


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _entropy(scores: torch.Tensor) -> float:
    if scores.numel() <= 1:
        return 0.0
    probabilities = torch.softmax(scores.double(), dim=0)
    value = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return value / math.log(scores.numel())


def _facet_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Choose query facets by their strongest root support without labels."""

    count = min(count, scores.shape[0])
    utility = scores.amax(dim=1)
    ordered = sorted(range(scores.shape[0]), key=lambda index: (-float(utility[index]), index))
    return torch.tensor(ordered[:count], dtype=torch.long)


def _root_view(example: PreparedExample, seed: int, facets: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = example.goal_scores[seed]
    selected = _facet_indices(scores, facets)
    return scores[selected], selected


def _ordered_roots(scores: torch.Tensor, count: int) -> tuple[int, ...]:
    parent_scores = scores.amax(dim=0)
    ordered = sorted(range(parent_scores.numel()), key=lambda index: (-float(parent_scores[index]), index))
    return tuple(ordered[:count])


def _router_features(example: PreparedExample, seed: int) -> torch.Tensor:
    views = []
    tops = []
    for count in (1, 4):
        scores, _ = _root_view(example, seed, count)
        parent_scores = scores.amax(dim=0)
        ordered = torch.sort(parent_scores, descending=True).values
        gap = float(ordered[0] - ordered[1]) if ordered.numel() > 1 else 1.0
        views.extend((gap, _entropy(parent_scores)))
        tops.append(int(torch.argmax(parent_scores)))
    scores, _ = _root_view(example, seed, 4)
    parent_scores = scores.amax(dim=0).float()
    values = (
        float(len(example.question.split())),
        float(len(example.parent_spans)),
        float(example.facet_count),
        float(example.source_tokens),
        views[0],
        views[1],
        views[2],
        views[3],
        float(tops[0] != tops[1]),
        float(parent_scores.max()),
        float(parent_scores.mean()),
        float(parent_scores.std(unbiased=False)),
    )
    return torch.tensor(values, dtype=torch.float32)


def _materialized_parents(
    visited: Sequence[int], roots: Sequence[int], kv_budget: int
) -> tuple[int, ...]:
    """Preserve roots, then admit discovered parents in deterministic search order."""

    selected = list(dict.fromkeys(roots))
    for parent in visited:
        if parent not in selected and len(selected) < kv_budget:
            selected.append(parent)
    return tuple(selected)


def evaluate_action(
    example: PreparedExample,
    seed: int,
    action: FactorizedEffortAction,
) -> dict[str, Any]:
    """Execute one factorized action against frozen native graph scores."""

    goal_scores, source_facets = _root_view(example, seed, action.facets)
    roots = _ordered_roots(goal_scores, action.roots)
    winners = goal_scores.argmax(dim=0)
    entry_facets = {root: int(winners[root]) for root in roots}
    config = SemanticGraphSearchConfig(
        successor_k=action.neighbors,
        max_visited_parents=action.search_budget,
        edge_threshold=float("-inf"),
        goal_threshold=float("inf"),
        max_hops=action.hops,
        strategy="best_first",
        max_expanded_nodes=64,
    )
    started = time.perf_counter()
    result = search_semantic_graph(
        example.edge_scores,
        goal_scores,
        roots,
        config,
        entry_facets=entry_facets,
    )
    elapsed = time.perf_counter() - started
    visited = tuple(result.visited)
    materialized = _materialized_parents(visited, roots, action.kv_budget)
    lengths = [end - start for start, end in example.parent_spans]
    kv = evidence_kv_metrics(materialized, example.oracle, lengths)
    conceptual_metrics = evidence_kv_metrics(visited, example.oracle, lengths)
    groups = example.groups
    evidence_parents = sorted(set().union(*(set(group) for group in groups))) if groups else []
    conceptual_chain = float(bool(groups) and all(set(group) & set(visited) for group in groups))
    materialized_chain = float(bool(groups) and all(set(group) & set(materialized) for group in groups))
    costs = factorized_cost(
        action,
        parent_count=len(example.parent_spans),
        transition_comparisons=result.raw_proposals,
        materialized_kv_tokens=int(kv["selected_kv_tokens"]),
    )
    first_group = set(groups[0]) if groups else set()
    return {
        "partition": example.partition,
        "dataset": example.dataset,
        "example_id": example.example_id,
        "seed": seed,
        "config_id": action.identifier,
        **action.to_dict(),
        "available_query_facets": example.facet_count,
        "selected_facet_indices": json.dumps([int(value) for value in source_facets]),
        "parent_count": len(example.parent_spans),
        "evidence_parent_count": len(evidence_parents),
        "evidence_parent_ids": json.dumps(evidence_parents),
        "root_ids": json.dumps(roots),
        "root_available": float(bool(first_group & set(roots))),
        "visited_ids": json.dumps(visited),
        "materialized_ids": json.dumps(materialized),
        "conceptual_chain_complete": conceptual_chain,
        "chain_complete": materialized_chain,
        "conceptual_evidence_recall": conceptual_metrics["evidence_kv_recall"],
        "conceptual_evidence_precision": conceptual_metrics["evidence_kv_precision"],
        **kv,
        **costs,
        "nodes_expanded": result.nodes_expanded,
        "raw_proposals": result.raw_proposals,
        "visited_count": len(visited),
        "materialized_parent_count": len(materialized),
        "stop_reason": result.stop_reason,
        "search_seconds": result.search_seconds,
        "control_wall_seconds": elapsed,
    }


def build_surfaces(prepared: Sequence[PreparedExample], seeds: Sequence[int]) -> list[UnitSurface]:
    actions = factorized_action_space()
    output = []
    total = len(prepared) * len(seeds)
    completed = 0
    for example in prepared:
        if not example.groups:
            continue
        for seed in seeds:
            rows = [evaluate_action(example, seed, action) for action in actions]
            output.append(
                UnitSurface(
                    example.partition,
                    example.dataset,
                    example.example_id,
                    seed,
                    example.question,
                    _router_features(example, seed),
                    rows,
                )
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[factorized] surfaces {completed}/{total}", flush=True)
    return output


def _best_available(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    sufficient = cheapest_sufficient(rows)
    if sufficient is not None:
        return sufficient
    return max(
        rows,
        key=lambda row: (
            float(row["chain_complete"]),
            float(row["evidence_kv_recall"]),
            float(row["evidence_kv_precision"]),
            -float(row["abstract_cost"]),
        ),
    )


def _row_for_action(unit: UnitSurface, action: FactorizedEffortAction) -> Mapping[str, Any]:
    return next(row for row in unit.rows if row["config_id"] == action.identifier)


def build_oracle_artifacts(units: Sequence[UnitSurface], output: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    oracles: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    oracle_rows, pareto_rows, regrets = [], [], []
    for unit in units:
        oracle = _best_available(unit.rows)
        oracles[unit.key] = oracle
        oracle_rows.append(dict(oracle))
        frontier = pareto_frontier(
            unit.rows,
            maximize=("chain_complete", "evidence_kv_recall", "evidence_kv_precision"),
            minimize=("abstract_cost", "selected_kv_tokens", "transition_comparisons"),
        )
        pareto_rows.extend(dict(row) for row in frontier)
        profile_rows = [_row_for_action(unit, FactorizedEffortAction.profile(level)) for level in range(3)]
        profile_oracle = _best_available(profile_rows)
        both_sufficient = bool(profile_oracle["chain_complete"] and oracle["chain_complete"])
        regrets.append(
            {
                "partition": unit.partition,
                "dataset": unit.dataset,
                "example_id": unit.example_id,
                "seed": unit.seed,
                "profile_config": profile_oracle["config_id"],
                "factorized_config": oracle["config_id"],
                "profile_quality": profile_oracle["chain_complete"],
                "factorized_quality": oracle["chain_complete"],
                "profile_oracle_cost": profile_oracle["abstract_cost"],
                "factorized_oracle_cost": oracle["abstract_cost"],
                "quantization_regret": (
                    float(profile_oracle["abstract_cost"]) - float(oracle["abstract_cost"])
                    if both_sufficient
                    else None
                ),
                "both_quality_sufficient": int(both_sufficient),
            }
        )
    _write_csv(output / "factorized_oracle_rows.csv", oracle_rows)
    _write_csv(
        output / "fixed_policy_rows.csv",
        [dict(_row_for_action(unit, FactorizedEffortAction.profile(1))) for unit in units],
    )
    _write_csv(output / "factorized_pareto_frontiers.csv", pareto_rows)
    _write_csv(output / "profile_quantization_regret.csv", regrets)
    return oracles


def _aggregate_surface(units: Sequence[UnitSurface]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        for row in unit.rows:
            grouped[(unit.dataset, str(row["config_id"]))].append(row)
    output = []
    metrics = (
        "chain_complete",
        "conceptual_chain_complete",
        "evidence_kv_recall",
        "evidence_kv_precision",
        "conceptual_evidence_recall",
        "abstract_cost",
        "selected_kv_tokens",
        "transition_comparisons",
        "control_wall_seconds",
    )
    for (dataset, config_id), rows in sorted(grouped.items()):
        record = {"dataset": dataset, "config_id": config_id, "rows": len(rows)}
        record.update({name: _mean(rows, name) for name in metrics})
        record.update({name: rows[0][name] for name in FactorizedEffortAction.__dataclass_fields__})
        output.append(record)
    return output


def build_requirement_artifacts(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    output: Path,
) -> None:
    distributions = []
    for dataset in sorted({unit.dataset for unit in units}):
        selected = [oracles[unit.key] for unit in units if unit.dataset == dataset]
        for parameter in ("facets", "roots", "neighbors", "hops", "search_budget", "kv_budget"):
            counts = Counter(int(row[parameter]) for row in selected)
            for value, count in sorted(counts.items()):
                distributions.append(
                    {
                        "dataset": dataset,
                        "parameter": parameter,
                        "value": value,
                        "count": count,
                        "fraction": count / len(selected),
                    }
                )
    _write_csv(output / "parameter_requirement_distributions.csv", distributions)

    interactions = []
    pairs = (("facets", "roots"), ("roots", "neighbors"), ("neighbors", "hops"), ("hops", "search_budget"))
    for dataset in sorted({unit.dataset for unit in units}):
        selected = [oracles[unit.key] for unit in units if unit.dataset == dataset]
        for left, right in pairs:
            counts = Counter((int(row[left]), int(row[right])) for row in selected)
            for (left_value, right_value), count in sorted(counts.items()):
                interactions.append(
                    {
                        "dataset": dataset,
                        "left_parameter": left,
                        "right_parameter": right,
                        "left_value": left_value,
                        "right_value": right_value,
                        "count": count,
                        "conditional_surface_fraction": count / len(selected),
                    }
                )
    _write_csv(output / "parameter_interactions.csv", interactions)


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale


def _action_space() -> RouterActionSpace:
    return RouterActionSpace(
        (
            ActionField("facets", FACET_LEVELS),
            ActionField("roots", ROOT_LEVELS),
            ActionField("neighbors", NEIGHBOR_LEVELS),
            ActionField("hops", HOP_LEVELS),
            ActionField("search_budget", BUDGET_LEVELS),
            ActionField("kv_budget", BUDGET_LEVELS),
        )
    )


def _action_from_values(values: Mapping[str, Any]) -> tuple[FactorizedEffortAction, bool]:
    roots = int(values["roots"])
    search = max(int(values["search_budget"]), roots)
    kv = max(int(values["kv_budget"]), roots)
    repaired = search != int(values["search_budget"]) or kv != int(values["kv_budget"])
    return (
        FactorizedEffortAction(
            int(values["facets"]),
            roots,
            int(values["neighbors"]),
            int(values["hops"]),
            search,
            kv,
        ),
        repaired,
    )


def _targets(space: RouterActionSpace, units: Sequence[UnitSurface], oracles) -> dict[str, torch.Tensor]:
    indexed = [space.index_targets({field.name: oracles[unit.key][field.name] for field in space.fields}) for unit in units]
    return {
        field.name: torch.tensor([row[field.name] for row in indexed], dtype=torch.long)
        for field in space.fields
    }


def _train_router(model, features, targets, semantic, seed: int) -> None:
    torch.manual_seed(seed)
    model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(220):
        optimizer.zero_grad()
        loss = model.loss(features, targets, semantic=semantic)
        loss.backward()
        optimizer.step()


def _train_profile_router(features: torch.Tensor, targets: torch.Tensor, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Linear(features.shape[1], 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(220):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
    return model


def run_router_study(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    validation = [unit for unit in units if unit.partition == "validation"]
    heldout = [unit for unit in units if unit.partition == "test"]
    train_x, test_x = _standardize(
        torch.stack([unit.features for unit in validation]),
        torch.stack([unit.features for unit in heldout]),
    )
    encoder = HashingQueryEncoder(width=32)
    train_semantic = encoder.encode([unit.question for unit in validation])
    test_semantic = encoder.encode([unit.question for unit in heldout])
    space = _action_space()
    targets = _targets(space, validation, oracles)
    target_rows = []
    for unit in units:
        target = oracles[unit.key]
        target_rows.append(
            {
                "partition": unit.partition,
                "dataset": unit.dataset,
                "example_id": unit.example_id,
                "seed": unit.seed,
                "target_config": target["config_id"],
                **{name: target[name] for name in FactorizedEffortAction.__dataclass_fields__},
                "target_quality": target["chain_complete"],
                "target_cost": target["abstract_cost"],
            }
        )
    _write_csv(output / "router_factorized_targets.csv", target_rows)

    profile_targets = []
    for unit in validation:
        rows = [_row_for_action(unit, FactorizedEffortAction.profile(level)) for level in range(3)]
        best = _best_available(rows)
        profile_targets.append(next(level for level in range(3) if FactorizedEffortAction.profile(level).identifier == best["config_id"]))

    result_rows = []
    for seed in ROUTER_SEEDS:
        profile_model = _train_profile_router(train_x, torch.tensor(profile_targets), seed)
        variants = [
            ("R1_factorized", MultiHeadEffortRouter(train_x.shape[1], space, hidden_width=32), None, None, False),
            ("R1_conservative", MultiHeadEffortRouter(train_x.shape[1], space, hidden_width=32), None, None, True),
            ("R2_query_hash", MultiHeadEffortRouter(train_x.shape[1], space, semantic_width=32, hidden_width=32, architecture="R2_query_hash"), train_semantic, test_semantic, False),
            ("R2_conservative", MultiHeadEffortRouter(train_x.shape[1], space, semantic_width=32, hidden_width=32, architecture="R2_query_hash"), train_semantic, test_semantic, True),
            ("R3A_factorized", AutoregressiveEffortRouter(train_x.shape[1], space, semantic_width=32, hidden_width=32, context_width=12), train_semantic, test_semantic, False),
        ]
        trained: dict[int, Any] = {}
        for variant_index, (variant, model, train_s, test_s, conservative) in enumerate(variants):
            identity = id(model)
            if variant.endswith("conservative"):
                source_index = variant_index - 1
                source = variants[source_index][1]
                if id(source) in trained:
                    model.load_state_dict(trained[id(source)])
                else:
                    _train_router(model, train_x, targets, train_s, seed)
            else:
                _train_router(model, train_x, targets, train_s, seed)
                trained[identity] = {key: value.detach().clone() for key, value in model.state_dict().items()}
            for index, unit in enumerate(heldout):
                semantic = test_s[index] if test_s is not None else None
                decision = model.decide(test_x[index], semantic=semantic, conservative=conservative)
                action, repaired = _action_from_values(decision.actions)
                selected = _row_for_action(unit, action)
                oracle = oracles[unit.key]
                result_rows.append(
                    {
                        "variant": variant,
                        "router_seed": seed,
                        "dataset": unit.dataset,
                        "example_id": unit.example_id,
                        "model_seed": unit.seed,
                        "selected_config": action.identifier,
                        "oracle_config": oracle["config_id"],
                        "quality": selected["chain_complete"],
                        "cost": selected["abstract_cost"],
                        "oracle_quality": oracle["chain_complete"],
                        "oracle_cost": oracle["abstract_cost"],
                        "cost_regret": float(selected["abstract_cost"]) - float(oracle["abstract_cost"]),
                        "allocation_outcome": allocation_outcome(selected, oracle),
                        "invalid_combination_repaired": int(repaired),
                        **{f"selected_{name}": selected[name] for name in (
                            "parent_count", "evidence_parent_count", "evidence_parent_ids",
                            "visited_ids", "materialized_ids", "visited_count",
                            "materialized_parent_count", "conceptual_evidence_recall",
                            "conceptual_evidence_precision", "evidence_kv_recall",
                            "evidence_kv_precision", "materialized_kv_tokens",
                            "root_comparisons", "transition_comparisons",
                        )},
                        **action.to_dict(),
                    }
                )
        with torch.no_grad():
            profile_predictions = torch.argmax(profile_model(test_x), dim=1)
        for index, unit in enumerate(heldout):
            level = int(profile_predictions[index])
            action = FactorizedEffortAction.profile(level)
            selected, oracle = _row_for_action(unit, action), oracles[unit.key]
            result_rows.append(
                {
                    "variant": "R0_profile",
                    "router_seed": seed,
                    "dataset": unit.dataset,
                    "example_id": unit.example_id,
                    "model_seed": unit.seed,
                    "selected_config": action.identifier,
                    "oracle_config": oracle["config_id"],
                    "quality": selected["chain_complete"],
                    "cost": selected["abstract_cost"],
                    "oracle_quality": oracle["chain_complete"],
                    "oracle_cost": oracle["abstract_cost"],
                    "cost_regret": float(selected["abstract_cost"]) - float(oracle["abstract_cost"]),
                    "allocation_outcome": allocation_outcome(selected, oracle),
                    "invalid_combination_repaired": 0,
                    **{f"selected_{name}": selected[name] for name in (
                        "parent_count", "evidence_parent_count", "evidence_parent_ids",
                        "visited_ids", "materialized_ids", "visited_count",
                        "materialized_parent_count", "conceptual_evidence_recall",
                        "conceptual_evidence_precision", "evidence_kv_recall",
                        "evidence_kv_precision", "materialized_kv_tokens",
                        "root_comparisons", "transition_comparisons",
                    )},
                    **action.to_dict(),
                }
            )
    _write_csv(output / "router_under_over_allocation.csv", result_rows)
    summary = []
    for variant in sorted({row["variant"] for row in result_rows}):
        rows = [row for row in result_rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "quality": _mean(rows, "quality"),
                "cost": _mean(rows, "cost"),
                "oracle_regret": _mean(rows, "cost_regret"),
                "under_allocation": statistics.fmean(row["allocation_outcome"] == "under_allocation" for row in rows),
                "over_allocation": statistics.fmean(row["allocation_outcome"] == "over_allocation" for row in rows),
                "invalid_repair_rate": _mean(rows, "invalid_combination_repaired"),
                "rows": len(rows),
            }
        )
    _write_csv(output / "router_factorized_comparison.csv", summary)
    return result_rows


def run_targeted_retry(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    router_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    summary = defaultdict(list)
    for row in router_rows:
        summary[row["variant"]].append(row)
    eligible = [
        variant
        for variant, rows in summary.items()
        if _mean(rows, "quality") >= _mean(summary["R0_profile"], "quality") - 1e-12
    ]
    initial_variant = min(eligible, key=lambda variant: _mean(summary[variant], "cost"))
    selected_by_key = {
        (row["dataset"], row["example_id"], int(row["model_seed"])): row
        for row in router_rows
        if row["variant"] == initial_variant and int(row["router_seed"]) == ROUTER_SEEDS[0]
    }
    retry_rows = []
    for unit in (value for value in units if value.partition == "test"):
        initial_record = selected_by_key[unit.key]
        initial = next(row for row in unit.rows if row["config_id"] == initial_record["selected_config"])
        initial_action = FactorizedEffortAction(
            *(int(initial[name]) for name in FactorizedEffortAction.__dataclass_fields__)
        )
        oracle = oracles[unit.key]
        candidates = []
        for candidate in unit.rows:
            if not candidate["chain_complete"]:
                continue
            action = FactorizedEffortAction(
                *(int(candidate[name]) for name in FactorizedEffortAction.__dataclass_fields__)
            )
            name = changed_control(initial_action, action)
            if name != "compound_retry":
                candidates.append((candidate, action, name))
        if initial["chain_complete"]:
            final, action_name, retried = initial, "stop", 0
        elif candidates:
            final, _, action_name = min(candidates, key=lambda value: (float(value[0]["abstract_cost"]), value[0]["config_id"]))
            retried = 1
        else:
            final, action_name, retried = oracle, "compound_fallback", 1
        final_cost = float(initial["abstract_cost"])
        if retried:
            final_cost += max(0.0, float(final["abstract_cost"]) - float(initial["abstract_cost"])) + 2.0
        retry_rows.append(
            {
                "dataset": unit.dataset,
                "example_id": unit.example_id,
                "model_seed": unit.seed,
                "initial_variant": initial_variant,
                "initial_config": initial["config_id"],
                "retry_action": action_name,
                "final_config": final["config_id"],
                "initial_quality": initial["chain_complete"],
                "final_quality": final["chain_complete"],
                "wrong_to_corrected": int(not initial["chain_complete"] and final["chain_complete"]),
                "correct_to_broken": int(initial["chain_complete"] and not final["chain_complete"]),
                "unchanged_wrong": int(not initial["chain_complete"] and not final["chain_complete"]),
                "unchanged_correct": int(initial["chain_complete"] and final["chain_complete"]),
                "retry_count": retried,
                "initial_cost": initial["abstract_cost"],
                "final_cost_with_regeneration": final_cost,
                "added_cost": final_cost - float(initial["abstract_cost"]),
                "extra_selected_kv_tokens": max(0.0, float(final["selected_kv_tokens"]) - float(initial["selected_kv_tokens"])),
                "extra_transition_comparisons": max(0.0, float(final["transition_comparisons"]) - float(initial["transition_comparisons"])),
                "oracle_cost": oracle["abstract_cost"],
                **{f"initial_{name}": initial[name] for name in (
                    "parent_count", "evidence_parent_count", "evidence_parent_ids",
                    "visited_ids", "materialized_ids", "visited_count",
                    "materialized_parent_count", "conceptual_evidence_recall",
                    "conceptual_evidence_precision", "evidence_kv_recall",
                    "evidence_kv_precision", "materialized_kv_tokens",
                    "root_comparisons", "transition_comparisons",
                )},
                **{f"final_{name}": final[name] for name in (
                    "visited_ids", "materialized_ids", "visited_count",
                    "materialized_parent_count", "conceptual_evidence_recall",
                    "conceptual_evidence_precision", "evidence_kv_recall",
                    "evidence_kv_precision", "materialized_kv_tokens",
                    "root_comparisons", "transition_comparisons",
                )},
            }
        )
    _write_csv(output / "targeted_retry_results.csv", retry_rows)
    return retry_rows


def _plot(output: Path, surface: Sequence[Mapping[str, Any]], oracles, routers, retries) -> None:
    colors = {"hotpotqa": "#c0392b", "qasper": "#2471a3"}
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        rows = [row for row in surface if row["dataset"] == dataset]
        axis.scatter([row["abstract_cost"] for row in rows], [row["chain_complete"] for row in rows], s=7, alpha=0.15, color=colors[dataset], label="factorized configs")
        for level, marker in enumerate(("o", "s", "^")):
            config = FactorizedEffortAction.profile(level).identifier
            row = next(value for value in rows if value["config_id"] == config)
            axis.scatter(row["abstract_cost"], row["chain_complete"], marker=marker, s=65, edgecolor="black", color="#f4d03f", label=f"E{level}")
        oracle_rows = [row for key, row in oracles.items() if key[0] == dataset and row["partition"] == "test"]
        axis.scatter(_mean(oracle_rows, "abstract_cost"), _mean(oracle_rows, "chain_complete"), marker="*", s=130, color="#1e8449", label="factorized oracle")
        axis.set_title(dataset)
        axis.set_xlabel("mean abstract effort")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("complete evidence-chain rate")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles[-4:], labels[-4:], fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "factorized_quality_cost_frontier.png", dpi=190)
    figure.savefig(output / "factorized_quality_cost_frontier.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for dataset in colors:
        rows = [row for row in surface if row["dataset"] == dataset]
        axis.scatter([row["evidence_kv_recall"] for row in rows], [row["evidence_kv_precision"] for row in rows], s=8, alpha=0.18, color=colors[dataset], label=dataset)
    axis.set_xlabel("evidence K/V recall")
    axis.set_ylabel("evidence K/V precision")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "factorized_precision_recall.png", dpi=190)
    figure.savefig(output / "factorized_precision_recall.pdf")
    plt.close(figure)

    router_summary = []
    for variant in sorted({row["variant"] for row in routers}):
        rows = [row for row in routers if row["variant"] == variant]
        router_summary.append((variant, _mean(rows, "cost"), _mean(rows, "quality")))
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for variant, cost, quality in router_summary:
        axis.scatter(cost, quality, s=55)
        axis.annotate(variant.replace("_", " "), (cost, quality), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xlabel("mean abstract effort")
    axis.set_ylabel("complete evidence-chain rate")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "factorized_router_quality_cost.png", dpi=190)
    figure.savefig(output / "factorized_router_quality_cost.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for dataset in colors:
        rows = [row for row in surface if row["dataset"] == dataset]
        axis.scatter([row["search_budget"] for row in rows], [row["selected_kv_tokens"] for row in rows], c=[row["chain_complete"] for row in rows], cmap="viridis", s=8, alpha=0.2)
    axis.set_xlabel("conceptual search parent budget")
    axis.set_ylabel("materialized native K/V tokens")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "search_vs_admission_budget.png", dpi=190)
    figure.savefig(output / "search_vs_admission_budget.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prep_args = SimpleNamespace(
        source_feature_file=args.source_feature_file,
        query_feature_file=args.query_feature_file,
        facet_gate_file=args.facet_gate_file,
        projection_dir=args.projection_dir,
        seeds=args.seeds,
    )
    prepared = _prepare_examples(prep_args, torch.device(args.device))
    units = build_surfaces(prepared, args.seeds)
    action_space = {
        "interpret": {"facets": FACET_LEVELS, "query_region_policy": "structural_fixed"},
        "search": {"roots": ROOT_LEVELS, "neighbors": NEIGHBOR_LEVELS, "hops": HOP_LEVELS, "search_budget": BUDGET_LEVELS},
        "admit": {"kv_parent_budget": BUDGET_LEVELS, "edge_threshold": "open", "goal_threshold": "closed", "strategy": "best_first"},
        "valid_configurations": len(factorized_action_space()),
    }
    (args.output_dir / "factorized_action_space.json").write_text(json.dumps(action_space, indent=2), encoding="utf-8")
    oracles = build_oracle_artifacts(units, args.output_dir)
    surface = _aggregate_surface(units)
    _write_csv(args.output_dir / "precision_recall_control_surface.csv", surface)
    _write_csv(args.output_dir / "search_vs_admission_budget.csv", surface)
    build_requirement_artifacts(units, oracles, args.output_dir)
    routers = run_router_study(units, oracles, args.output_dir)
    retries = run_targeted_retry(units, oracles, routers, args.output_dir)
    _plot(args.output_dir, surface, oracles, routers, retries)

    heldout = [unit for unit in units if unit.partition == "test"]
    findings = {
        "schema_version": "1.0",
        "scope": "frozen_native_score_factorized_adaptive_control",
        "units": {"validation": sum(unit.partition == "validation" for unit in units), "heldout": len(heldout)},
        "datasets": {},
        "router_seeds": list(ROUTER_SEEDS),
        "query_semantic_input": "actual question text encoded by signed hashing baseline",
        "output_oracle": "not available for newly factorized configurations; no output quality is imputed",
        "systems_handoff": "kernel, cache, batching, and serving optimization moved to Paper 5.5; vLLM integration moved to Paper 6",
        "claim_boundaries": [
            "The experiment replays frozen Qwen native graph and query scores; it is not live generation.",
            "Evidence K/V precision and recall are token-weighted over source-parent spans.",
            "Facet selection is score-based over the existing query-facet cache, not a new learned query interpreter.",
            "Targeted retry chooses the cheapest successful one-control correction as an evaluator-side upper bound; compound fallback is reported separately.",
            "R2 uses actual question text but a signed-hashing encoder, not a pretrained semantic encoder.",
        ],
    }
    regret_rows = list(csv.DictReader((args.output_dir / "profile_quantization_regret.csv").open(encoding="utf-8")))
    for dataset in sorted({unit.dataset for unit in heldout}):
        selected = [oracles[unit.key] for unit in heldout if unit.dataset == dataset]
        regrets = [row for row in regret_rows if row["partition"] == "test" and row["dataset"] == dataset and row["quantization_regret"]]
        dataset_retries = [row for row in retries if row["dataset"] == dataset]
        findings["datasets"][dataset] = {
            "heldout_units": len(selected),
            "factorized_oracle_quality": _mean(selected, "chain_complete"),
            "factorized_oracle_cost": _mean(selected, "abstract_cost"),
            "mean_profile_quantization_regret_when_both_sufficient": statistics.fmean(float(row["quantization_regret"]) for row in regrets) if regrets else None,
            "profile_and_factorized_sufficient_units": len(regrets),
            "retry_initial_quality": _mean(dataset_retries, "initial_quality"),
            "retry_final_quality": _mean(dataset_retries, "final_quality"),
            "retry_mean_added_cost": _mean(dataset_retries, "added_cost"),
            "retry_correction_rate": _mean(dataset_retries, "wrong_to_corrected"),
        }
    (args.output_dir / "paper3_5_next_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings


def _existing_or_sibling(relative: str) -> Path:
    local = ROOT / relative
    if local.exists():
        return local
    sibling = ROOT.parent / "pdattention-iter-gist" / relative
    return sibling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    result_root = Path("docs/papers/shared/results/paper2_5_iterative_pra")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--source-feature-file", type=Path, default=_existing_or_sibling(str(result_root / "native_qk_closure/native_qk_features_test.pt")))
    parser.add_argument("--query-feature-file", type=Path, default=_existing_or_sibling(str(result_root / "query_entry_facets/query_entry_features.pt")))
    parser.add_argument("--facet-gate-file", type=Path, default=ROOT / result_root / "grounded_query_facets/grounded_facet_gate_results.json")
    parser.add_argument("--projection-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra")
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
