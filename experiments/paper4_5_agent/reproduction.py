"""Compatibility review for locally graded results against published baselines."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field

from .schema import PublishedBaseline, ReproductionStatus, StrictModel


class OfficialResult(StrictModel):
    """Small normalized receipt produced after an official benchmark grader runs."""

    official_grader: bool
    score: float = Field(ge=0, le=1)
    resolved: int = Field(ge=0)
    total: int = Field(ge=1)
    task_ids: tuple[str, ...] = ()
    configuration_differences: tuple[str, ...] = ()
    grader_artifact: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "OfficialResult":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ReproductionReview(StrictModel):
    status: ReproductionStatus
    compatible: bool
    published_score: float
    observed_score: float | None = None
    score_delta: float | None = None
    observed_interval_95: tuple[float, float] | None = None
    reasons: tuple[str, ...]


def review_result(
    baseline: PublishedBaseline,
    result: OfficialResult,
    *,
    absolute_tolerance: float,
    require_exact_cohort: bool,
) -> ReproductionReview:
    """Admit only officially graded, identity-compatible baseline results."""

    reasons: list[str] = []
    if not result.official_grader:
        reasons.append("The result was not produced by the benchmark's official grader.")
    if result.configuration_differences:
        reasons.extend(f"Configuration difference: {item}" for item in result.configuration_differences)
    if require_exact_cohort and result.total != baseline.published_total:
        reasons.append(
            f"Cohort size differs: observed {result.total}, published {baseline.published_total}."
        )
    if baseline.task_ids and tuple(result.task_ids) != tuple(baseline.task_ids):
        reasons.append("Frozen task IDs or their order differ from the published cohort.")

    interval = _wilson_interval(result.resolved, result.total)
    score_delta = result.score - baseline.published_score
    score_compatible = (
        abs(score_delta) <= absolute_tolerance
        or interval[0] <= baseline.published_score <= interval[1]
    )
    if not score_compatible:
        reasons.append(
            f"Observed score {result.score:.3f} is not compatible with published "
            f"{baseline.published_score:.3f} at tolerance {absolute_tolerance:.3f}."
        )

    identity_compatible = not reasons
    if identity_compatible and score_compatible:
        status = ReproductionStatus.BASELINE_REPRODUCED
    elif result.official_grader:
        status = ReproductionStatus.BASELINE_ATTEMPTED
    else:
        status = ReproductionStatus.BASELINE_FAILED
    return ReproductionReview(
        status=status,
        compatible=status == ReproductionStatus.BASELINE_REPRODUCED,
        published_score=baseline.published_score,
        observed_score=result.score,
        score_delta=score_delta,
        observed_interval_95=interval,
        reasons=tuple(reasons) or ("Official result matches the pinned baseline contract.",),
    )


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
