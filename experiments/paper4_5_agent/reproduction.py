"""Compatibility review for locally graded results against published baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from pydantic import Field, model_validator

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
    execution_identity: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def counts_and_score_are_consistent(self) -> "OfficialResult":
        if self.resolved > self.total:
            raise ValueError("resolved count cannot exceed total")
        expected_score = self.resolved / self.total
        if not math.isclose(self.score, expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("score must equal resolved / total")
        if self.task_ids:
            if len(self.task_ids) != self.total:
                raise ValueError("task_ids length must equal total")
            if len(set(self.task_ids)) != len(self.task_ids):
                raise ValueError("task_ids must be unique")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "OfficialResult":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ReproductionReview(StrictModel):
    status: ReproductionStatus
    compatible: bool
    published_score: float | None
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
    if baseline.task_ids_sha256:
        if result.execution_identity is None:
            reasons.append("Fixed-cohort result is missing its structured execution identity.")
        else:
            expected_identity = {
                "cohort_sha256": baseline.task_ids_sha256,
                "benchmark_revision": baseline.benchmark_revision,
                "harness": baseline.harness,
                "harness_version": baseline.harness_version,
                "model": baseline.model,
                "engine": baseline.engine,
                "engine_version": baseline.engine_version,
                "dtype": baseline.dtype,
                "quantization": baseline.quantization,
                "kv_cache_dtype": baseline.kv_cache_dtype,
                "scaffold": baseline.scaffold,
                "context_limit": baseline.context_limit,
                "max_steps": baseline.max_steps,
                "temperature": baseline.temperature,
                "function_calling": baseline.function_calling,
                "prefix_caching": baseline.prefix_caching,
                "grading": baseline.grading,
            }
            for key, expected in expected_identity.items():
                observed = result.execution_identity.get(key)
                if observed != expected:
                    reasons.append(
                        f"Execution identity differs for {key}: observed {observed!r}, "
                        f"expected {expected!r}."
                    )

    interval = _wilson_interval(result.resolved, result.total)
    if baseline.admission_kind == "local_calibration":
        score_delta = None
        score_compatible = (
            baseline.minimum_admission_score
            <= result.score
            <= baseline.maximum_admission_score
        )
        if not score_compatible:
            reasons.append(
                f"Observed score {result.score:.3f} is outside the local admission band "
                f"[{baseline.minimum_admission_score:.3f}, "
                f"{baseline.maximum_admission_score:.3f}]."
            )
    else:
        assert baseline.published_score is not None
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
        reasons=tuple(reasons) or (
            "Official result satisfies the pinned local admission contract."
            if baseline.admission_kind == "local_calibration"
            else "Official result matches the pinned baseline contract.",
        ),
    )


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
