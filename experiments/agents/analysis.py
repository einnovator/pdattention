"""Paired coding-agent summaries with conservative small-cohort statistics."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from .schema import CodingAgentRun


def summarize(runs: Iterable[CodingAgentRun]) -> dict[str, Any]:
    groups: dict[str, list[CodingAgentRun]] = defaultdict(list)
    for run in runs:
        key = f"{run.identity.pra_mode.value}:{run.identity.pra_profile.value}"
        groups[key].append(run)
    return {condition: _condition(rows) for condition, rows in sorted(groups.items())}


def paired_comparison(
    baseline: Sequence[CodingAgentRun], treatment: Sequence[CodingAgentRun], *, seed: int = 1729,
) -> dict[str, Any]:
    base = {_pair_key(row): row for row in baseline}
    other = {_pair_key(row): row for row in treatment}
    keys = sorted(base.keys() & other.keys())
    if not keys:
        raise ValueError("paired comparison has no shared task/repeat identities")
    pairs = [(base[key], other[key]) for key in keys]
    wins = sum((not a.outcome.success) and b.outcome.success for a, b in pairs)
    losses = sum(a.outcome.success and (not b.outcome.success) for a, b in pairs)
    ties = len(pairs) - wins - losses
    input_ratios = [
        b.tokens.input_tokens / a.tokens.input_tokens
        for a, b in pairs if a.tokens.input_tokens > 0 and b.tokens.input_tokens > 0
    ]
    wall_ratios = [
        b.timings.task_wall_ms / a.timings.task_wall_ms
        for a, b in pairs if a.timings.task_wall_ms > 0 and b.timings.task_wall_ms > 0
    ]
    return {
        "pairs": len(pairs), "wins": wins, "losses": losses, "ties": ties,
        "mcnemar_exact_p": _exact_binomial_two_sided(wins, losses),
        "input_token_geomean_ratio": _geomean(input_ratios),
        "input_token_ratio_ci95": _bootstrap_geomean(input_ratios, seed=seed),
        "wall_time_geomean_ratio": _geomean(wall_ratios),
        "wall_time_ratio_ci95": _bootstrap_geomean(wall_ratios, seed=seed + 1),
    }


def _condition(rows: Sequence[CodingAgentRun]) -> dict[str, Any]:
    successes = sum(row.outcome.success for row in rows)
    successful = [row for row in rows if row.outcome.success]
    return {
        "runs": len(rows),
        "successes": successes,
        "task_success_rate": successes / len(rows),
        "input_tokens": sum(row.tokens.input_tokens for row in rows),
        "input_tokens_per_success": _per_success(rows, successful, lambda row: row.tokens.input_tokens),
        "wall_ms_per_success": _per_success(rows, successful, lambda row: row.timings.task_wall_ms),
        "cost_per_success": _per_success(rows, successful, lambda row: float(row.cost.total or 0)),
        "median_wall_ms": statistics.median(row.timings.task_wall_ms for row in rows),
    }


def _per_success(
    rows: Sequence[CodingAgentRun], successful: Sequence[CodingAgentRun], metric: Callable[[CodingAgentRun], float],
) -> float | None:
    if not successful:
        return None
    return sum(metric(row) for row in rows) / len(successful)


def _pair_key(run: CodingAgentRun) -> tuple[str, str, int, str, str]:
    identity = run.identity
    return identity.benchmark, identity.task_id, identity.repeat, identity.agent, identity.model


def _geomean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return math.exp(statistics.mean(math.log(value) for value in values))


def _bootstrap_geomean(values: Sequence[float], *, seed: int, samples: int = 5000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        _geomean([rng.choice(values) for _ in values]) or 0 for _ in range(samples)
    )
    return [estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]]


def _exact_binomial_two_sided(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(0, min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)
