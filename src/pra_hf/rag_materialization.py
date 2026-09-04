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
