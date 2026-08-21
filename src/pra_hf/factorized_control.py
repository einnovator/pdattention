"""Factorized effort actions and evaluation helpers for adaptive PRA.

The profile controller chooses one coherent effort level.  Factorized control
instead exposes interpretation, search, and admission decisions separately.
This module contains only the deterministic action/accounting layer; experiment
code supplies model scores and evaluator-only evidence labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


FACET_LEVELS = (1, 2, 4)
ROOT_LEVELS = (1, 2, 4)
NEIGHBOR_LEVELS = (2, 4, 8)
HOP_LEVELS = (0, 1, 2, 3)
BUDGET_LEVELS = (2, 4, 8)


@dataclass(frozen=True)
class FactorizedEffortAction:
    """One independently configurable PRA interpretation/search/admission plan.

    ``facets`` controls query interpretation. ``roots``, ``neighbors``,
    ``hops``, and ``search_budget`` control conceptual graph exploration.
    ``kv_budget`` is a physical parent-count ceiling applied after search.
    Search and K/V admission are separate so broad discovery need not imply a
    broad attention working set.
    """

    facets: int
    roots: int
    neighbors: int
    hops: int
    search_budget: int
    kv_budget: int

    def __post_init__(self) -> None:
        supported = (
            ("facets", self.facets, FACET_LEVELS),
            ("roots", self.roots, ROOT_LEVELS),
            ("neighbors", self.neighbors, NEIGHBOR_LEVELS),
            ("hops", self.hops, HOP_LEVELS),
            ("search_budget", self.search_budget, BUDGET_LEVELS),
            ("kv_budget", self.kv_budget, BUDGET_LEVELS),
        )
        for name, value, levels in supported:
            if value not in levels:
                raise ValueError(f"Unsupported {name}={value}; expected one of {levels}.")
        if self.search_budget < self.roots:
            raise ValueError("search_budget must retain every selected root.")
        if self.kv_budget < self.roots:
            raise ValueError("kv_budget must admit every selected root.")

    @property
    def identifier(self) -> str:
        """Return a stable identifier suitable for CSV joins and traces."""

        return (
            f"F{self.facets}_R{self.roots}_K{self.neighbors}_H{self.hops}_"
            f"Bs{self.search_budget}_Bkv{self.kv_budget}"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def profile(cls, level: int) -> "FactorizedEffortAction":
        """Map E0/E1/E2 to their diagonal point in the factorized lattice."""

        if level not in (0, 1, 2):
            raise ValueError("Profile level must be 0, 1, or 2.")
        return cls(
            facets=FACET_LEVELS[level],
            roots=ROOT_LEVELS[level],
            neighbors=NEIGHBOR_LEVELS[level],
            hops=(0, 1, 3)[level],
            search_budget=BUDGET_LEVELS[level],
            kv_budget=BUDGET_LEVELS[level],
        )


def factorized_action_space() -> tuple[FactorizedEffortAction, ...]:
    """Enumerate valid actions without silently repairing impossible budgets."""

    actions = []
    for facets in FACET_LEVELS:
        for roots in ROOT_LEVELS:
            for neighbors in NEIGHBOR_LEVELS:
                for hops in HOP_LEVELS:
                    for search_budget in BUDGET_LEVELS:
                        for kv_budget in BUDGET_LEVELS:
                            if search_budget >= roots and kv_budget >= roots:
                                actions.append(
                                    FactorizedEffortAction(
                                        facets,
                                        roots,
                                        neighbors,
                                        hops,
                                        search_budget,
                                        kv_budget,
                                    )
                                )
    return tuple(actions)


def factorized_cost(
    action: FactorizedEffortAction,
    *,
    parent_count: int,
    transition_comparisons: int,
    materialized_kv_tokens: int,
) -> dict[str, float]:
    """Return exact component accounting and a normalized abstract total.

    Root interpretation scores ``facets * parent_count`` pairs.  Transition
    comparisons are measured by the search execution.  The abstract total
    normalizes root comparisons per source parent and K/V tokens in 64-token
    blocks, retaining the earlier Paper-3.5 cost convention while exposing all
    raw components alongside it.
    """

    if parent_count <= 0 or transition_comparisons < 0 or materialized_kv_tokens < 0:
        raise ValueError("Cost components must be non-negative and parent_count positive.")
    root_comparisons = action.facets * parent_count
    conceptual = action.search_budget
    kv_blocks = materialized_kv_tokens / 64.0
    abstract = root_comparisons / parent_count + transition_comparisons + conceptual + kv_blocks
    return {
        "root_comparisons": float(root_comparisons),
        "transition_comparisons": float(transition_comparisons),
        "conceptual_parent_budget": float(conceptual),
        "materialized_kv_tokens": float(materialized_kv_tokens),
        "abstract_cost": float(abstract),
    }


def evidence_kv_metrics(
    selected: Iterable[int],
    required: Iterable[int],
    parent_token_lengths: Sequence[int],
) -> dict[str, float]:
    """Compute token-weighted evidence precision and recall for admitted K/V."""

    selected_set, required_set = set(selected), set(required)
    valid = set(range(len(parent_token_lengths)))
    if not selected_set <= valid or not required_set <= valid:
        raise ValueError("Evidence metrics received an out-of-range parent index.")
    selected_tokens = sum(parent_token_lengths[index] for index in selected_set)
    required_tokens = sum(parent_token_lengths[index] for index in required_set)
    evidence_tokens = sum(parent_token_lengths[index] for index in selected_set & required_set)
    return {
        "evidence_kv_precision": evidence_tokens / selected_tokens if selected_tokens else 0.0,
        "evidence_kv_recall": evidence_tokens / required_tokens if required_tokens else 1.0,
        "selected_kv_tokens": float(selected_tokens),
        "selected_evidence_kv_tokens": float(evidence_tokens),
        "required_evidence_kv_tokens": float(required_tokens),
    }


def dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> bool:
    """Return whether ``left`` weakly dominates and strictly improves ``right``."""

    weak = all(float(left[name]) >= float(right[name]) for name in maximize) and all(
        float(left[name]) <= float(right[name]) for name in minimize
    )
    strict = any(float(left[name]) > float(right[name]) for name in maximize) or any(
        float(left[name]) < float(right[name]) for name in minimize
    )
    return weak and strict


def pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Return nondominated rows while preserving deterministic input order."""

    return [
        row
        for index, row in enumerate(rows)
        if not any(
            other_index != index
            and dominates(other, row, maximize=maximize, minimize=minimize)
            for other_index, other in enumerate(rows)
        )
    ]


def cheapest_sufficient(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_field: str = "chain_complete",
    quality_target: float = 1.0,
    cost_field: str = "abstract_cost",
) -> Mapping[str, Any] | None:
    """Select the minimum-cost quality-sufficient row with stable tie breaks."""

    sufficient = [row for row in rows if float(row[quality_field]) >= quality_target]
    if not sufficient:
        return None
    return min(
        sufficient,
        key=lambda row: (
            float(row[cost_field]),
            -float(row.get("evidence_kv_precision", 0.0)),
            float(row.get("selected_kv_tokens", 0.0)),
            str(row.get("config_id", "")),
        ),
    )


def allocation_outcome(
    selected: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    quality_field: str = "chain_complete",
    cost_field: str = "abstract_cost",
) -> str:
    """Classify a controller decision by downstream sufficiency and oracle cost."""

    selected_quality = float(selected[quality_field])
    oracle_quality = float(oracle[quality_field])
    if selected_quality < oracle_quality:
        return "under_allocation"
    if float(selected[cost_field]) > float(oracle[cost_field]) + 1e-12:
        return "over_allocation"
    return "matched"


def changed_control(before: FactorizedEffortAction, after: FactorizedEffortAction) -> str:
    """Name a single targeted retry transition or mark a compound transition."""

    names = {
        "facets": "widen_facets",
        "roots": "increase_R",
        "neighbors": "increase_K",
        "hops": "increase_H",
        "search_budget": "increase_search_budget",
        "kv_budget": "change_KV_admission",
    }
    changed = [name for name in names if getattr(before, name) != getattr(after, name)]
    return names[changed[0]] if len(changed) == 1 else "compound_retry"
