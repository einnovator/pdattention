"""Paired coding-agent summaries with conservative small-cohort statistics."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from pra_hf.canonical_evidence import (
    CanonicalEvidenceRecord,
    ConditionEvidence,
    EvidenceCondition,
    EvidenceKey,
    EvidenceProvenance,
    MeasurementState,
    MetricObservation,
    STANDARD_METRICS,
)

from .schema import CodingAgentRun, PRAMode


AGENT_EVIDENCE_METRICS = {
    name: STANDARD_METRICS[name]
    for name in (
        "official_task_success",
        "verifier_checks_passed",
        "input_tokens",
        "task_wall_ms",
        "cost_per_successful_task",
    )
}


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


def baseline_promotion_gate(
    baseline: Sequence[CodingAgentRun], *, minimum_success_rate: float = 0.30,
    maximum_success_rate: float = 0.80, minimum_runs: int = 3,
) -> dict[str, Any]:
    """Decide whether a no-PRA cohort can enter a three-condition PRA sweep.

    Floor cohorts cannot reveal a PRA regression and ceiling cohorts provide too
    little headroom.  The decision is serialized with the evidence so expensive
    treatment runs cannot silently bypass this policy.
    """

    if not baseline:
        raise ValueError("agent admission requires at least one no-PRA run")
    if any(row.identity.pra_mode != PRAMode.NONE for row in baseline):
        raise ValueError("agent admission must be computed from no-PRA runs only")
    successes = sum(row.outcome.success for row in baseline)
    rate = successes / len(baseline)
    if successes == 0:
        status, reason = "BLOCKED", "No-PRA official success is zero; PRA efficacy comparisons are floor-confounded."
    elif len(baseline) < minimum_runs:
        status, reason = "BLOCKED", f"Only {len(baseline)} no-PRA runs; at least {minimum_runs} are required."
    elif rate < minimum_success_rate:
        status, reason = "BLOCKED", f"No-PRA success {rate:.1%} is below the {minimum_success_rate:.1%} promotion floor."
    elif rate > maximum_success_rate:
        status, reason = "BLOCKED", f"No-PRA success {rate:.1%} exceeds the {maximum_success_rate:.1%} ceiling."
    else:
        status, reason = "ELIGIBLE", "No-PRA success is inside the preregistered comparison band."
    return {
        "status": status,
        "eligible": status == "ELIGIBLE",
        "runs": len(baseline),
        "successes": successes,
        "official_success_rate": rate,
        "target_range": [minimum_success_rate, maximum_success_rate],
        "reason": reason,
    }


def canonical_agent_evidence(
    baseline: Sequence[CodingAgentRun],
    *,
    pra_no_adaptor: Sequence[CodingAgentRun] = (),
    pra_adaptor_bundle: Sequence[CodingAgentRun] = (),
    profile: str = "balanced",
    mode: str = "agent-gateway",
    bundle_id: str | None = None,
    bundle_revision: str | None = None,
    date: str,
    commit: str | None = None,
) -> CanonicalEvidenceRecord:
    """Normalize one agent cohort without turning absent treatment data into zero."""

    gate = baseline_promotion_gate(baseline)
    first = baseline[0]
    identity = first.identity
    comparable_fields = ("benchmark", "agent", "model", "engine", "engine_version")
    for row in [*baseline, *pra_no_adaptor, *pra_adaptor_bundle]:
        for field in comparable_fields:
            if getattr(row.identity, field) != getattr(identity, field):
                raise ValueError(f"agent evidence identity mismatch: {field}")

    blocked_note = gate["reason"] if not gate["eligible"] else None
    conditions = {
        EvidenceCondition.NO_PRA: ConditionEvidence(metrics=_agent_metrics(baseline)),
        EvidenceCondition.PRA_NO_ADAPTOR: ConditionEvidence(
            metrics=_agent_metrics(pra_no_adaptor) if pra_no_adaptor else _missing_agent_metrics(blocked_note)
        ),
        EvidenceCondition.PRA_ADAPTOR_BUNDLE: ConditionEvidence(
            metrics=_agent_metrics(pra_adaptor_bundle) if pra_adaptor_bundle else _missing_agent_metrics(blocked_note),
            bundle_id=bundle_id,
            bundle_revision=bundle_revision,
        ),
    }
    hardware = ", ".join(f"{key}={value}" for key, value in sorted(identity.hardware.items())) or identity.host
    task_ids = sorted({row.identity.task_id for row in baseline})
    run_ids = tuple(row.identity.run_id for row in [*baseline, *pra_no_adaptor, *pra_adaptor_bundle])
    return CanonicalEvidenceRecord(
        key=EvidenceKey(
            task=f"{identity.benchmark}:{','.join(task_ids)}",
            hardware=hardware,
            engine=identity.engine,
            engine_version=identity.engine_version or "NOT_MEASURED",
            model_id=identity.model,
            model_revision=identity.model_revision or "NOT_MEASURED",
            mode=mode,
            profile=profile,
        ),
        metric_definitions=AGENT_EVIDENCE_METRICS,
        conditions=conditions,
        provenance=EvidenceProvenance(
            cohort=f"{identity.agent} official agent benchmark; admission={gate['status']}",
            run_ids=run_ids,
            commit=commit,
            date=date,
        ),
        evidence_tier="CONTROLLED" if gate["eligible"] else "BLOCKED",
    )


def _agent_metrics(rows: Sequence[CodingAgentRun]) -> dict[str, MetricObservation]:
    successes = [row for row in rows if row.outcome.success]
    passed = [row.outcome.tests_passed for row in rows if row.outcome.tests_passed is not None]
    return {
        "official_task_success": MetricObservation.measured(sum(row.outcome.success for row in rows) / len(rows)),
        "verifier_checks_passed": MetricObservation.measured(float(sum(passed))),
        "input_tokens": MetricObservation.measured(float(sum(row.tokens.input_tokens for row in rows))),
        "task_wall_ms": MetricObservation.measured(float(statistics.median(row.timings.task_wall_ms for row in rows))),
        "cost_per_successful_task": (
            MetricObservation.measured(sum(float(row.cost.total or 0) for row in rows) / len(successes))
            if successes else MetricObservation.missing(
                MeasurementState.NOT_APPLICABLE, "No successful task provides a denominator."
            )
        ),
    }


def _missing_agent_metrics(blocked_note: str | None) -> dict[str, MetricObservation]:
    state = MeasurementState.BLOCKED if blocked_note else MeasurementState.NOT_MEASURED
    return {
        name: MetricObservation.missing(state, blocked_note)
        for name in AGENT_EVIDENCE_METRICS
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
