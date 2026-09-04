"""Receipt-preserving materialization policies for Paper 3.2 RAG evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MaterializationPlan:
    """A score-ordered prefix of already selected resources.

    The input selection is immutable.  This plan only decides which complete
    selected resources enter physical context, so token accounting and lost
    evidence remain auditable.
    """

    fraction: float
    selected_indices: tuple[int, ...]
    requested_token_budget: int
    materialized_tokens: int
    full_selected_tokens: int

    @property
    def materialized_fraction(self) -> float:
        return self.materialized_tokens / max(self.full_selected_tokens, 1)


@dataclass(frozen=True)
class TokenMaterializationPlan:
    """An exact token budget distributed across immutable selected resources.

    ``token_counts`` is aligned with the original selection order. A zero entry
    omits that resource; a smaller positive entry keeps its leading token span.
    ``priority`` records the order in which the budget was assigned so oracle
    and wrong-memory controls remain auditable without changing presentation
    order.
    """

    fraction: float
    token_counts: tuple[int, ...]
    priority: tuple[int, ...]
    requested_token_budget: int
    materialized_tokens: int
    full_selected_tokens: int

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(index for index, count in enumerate(self.token_counts) if count)

    @property
    def materialized_fraction(self) -> float:
        return self.materialized_tokens / max(self.full_selected_tokens, 1)


def score_prefix_plan(token_counts: Sequence[int], fraction: float) -> MaterializationPlan:
    """Keep the largest score-ranked whole-resource prefix fitting a fraction."""

    if not token_counts or any(value <= 0 for value in token_counts):
        raise ValueError("materialization requires positive selected-resource sizes")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("materialization fraction must be in (0, 1]")
    total = sum(token_counts)
    budget = max(1, round(total * fraction))
    selected: list[int] = []
    used = 0
    for index, count in enumerate(token_counts):
        if used + count <= budget or not selected:
            selected.append(index)
            used += count
        else:
            break
    return MaterializationPlan(fraction, tuple(selected), budget, used, total)


def exact_token_plan(
    token_counts: Sequence[int],
    fraction: float,
    *,
    priority: Sequence[int] | None = None,
) -> TokenMaterializationPlan:
    """Fill an exact fractional budget, truncating only the final resource.

    Resources receive tokens in ``priority`` order while the returned counts
    stay aligned with the frozen selection. This separates materialization from
    serialization and makes matched-budget causal controls possible.
    """

    if not token_counts or any(value <= 0 for value in token_counts):
        raise ValueError("materialization requires positive selected-resource sizes")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("materialization fraction must be in (0, 1]")
    order = tuple(range(len(token_counts))) if priority is None else tuple(priority)
    if sorted(order) != list(range(len(token_counts))):
        raise ValueError("materialization priority must be a permutation of resources")

    total = sum(token_counts)
    budget = max(1, round(total * fraction))
    allocated = [0] * len(token_counts)
    remaining = budget
    for index in order:
        if remaining <= 0:
            break
        allocated[index] = min(token_counts[index], remaining)
        remaining -= allocated[index]
    used = sum(allocated)
    return TokenMaterializationPlan(
        fraction=fraction,
        token_counts=tuple(allocated),
        priority=order,
        requested_token_budget=budget,
        materialized_tokens=used,
        full_selected_tokens=total,
    )


def evidence_oracle_plan(
    token_counts: Sequence[int],
    gold_indices: Sequence[int],
    fraction: float,
) -> TokenMaterializationPlan:
    """Spend the physical budget on known supporting resources first.

    This is an analysis-only upper bound: gold identities are unavailable at
    serving time. Remaining budget follows the original score-ranked order.
    """

    gold = tuple(dict.fromkeys(gold_indices))
    if any(index < 0 or index >= len(token_counts) for index in gold):
        raise ValueError("gold resource index is outside the selected resources")
    priority = gold + tuple(index for index in range(len(token_counts)) if index not in gold)
    return exact_token_plan(token_counts, fraction, priority=priority)


def wrong_memory_plan(
    token_counts: Sequence[int],
    gold_indices: Sequence[int],
    fraction: float,
) -> TokenMaterializationPlan:
    """Allocate the same budget to non-supporting resources first.

    The condition is a causal control for gains caused merely by adding the
    same number of native K/V entries. If all selected resources are gold, the
    deterministic reversed order still supplies a distinct matched-budget
    control while the receipt makes that limitation visible.
    """

    gold = set(gold_indices)
    if any(index < 0 or index >= len(token_counts) for index in gold):
        raise ValueError("gold resource index is outside the selected resources")
    non_gold = tuple(index for index in range(len(token_counts)) if index not in gold)
    priority = non_gold + tuple(index for index in reversed(range(len(token_counts))) if index in gold)
    return exact_token_plan(token_counts, fraction, priority=priority)
