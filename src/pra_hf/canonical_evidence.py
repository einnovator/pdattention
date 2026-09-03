"""Canonical three-condition evidence shared by PRA product surfaces.

The schema keeps measurements separate from rendering.  A metric is always
identified by its direction, unit, and aggregation, while absent measurements
carry an explicit state instead of a numeric placeholder.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceCondition(str, Enum):
    NO_PRA = "no_pra"
    PRA_NO_ADAPTOR = "pra_no_adaptor"
    PRA_ADAPTOR_BUNDLE = "pra_adaptor_bundle"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


class MetricGroup(str, Enum):
    QUALITY = "quality"
    CONTEXT = "context"
    SERVING = "serving"
    RESOURCES = "resources"
    COST = "cost"
    ROUTING = "routing"


class MeasurementState(str, Enum):
    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"
    NEEDS_RUN = "NEEDS_RUN"
    CALIBRATION_PENDING = "CALIBRATION_PENDING"
    NO_QUALIFIED_ADAPTER = "NO_QUALIFIED_ADAPTER"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED = "BLOCKED"


class AuditState(str, Enum):
    AVAILABLE_EXISTING = "AVAILABLE_EXISTING"
    PARTIAL = "PARTIAL"
    NEEDS_RUN = "NEEDS_RUN"
    CALIBRATION_PENDING = "CALIBRATION_PENDING"
    NO_QUALIFIED_ADAPTER = "NO_QUALIFIED_ADAPTER"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED = "BLOCKED"


class MetricDefinition(StrictModel):
    """Meaning and display contract for one scalar metric."""

    name: str
    group: MetricGroup
    unit: str
    direction: MetricDirection
    aggregation: str | None = None
    description: str

    @model_validator(mode="after")
    def require_unambiguous_serving_units(self) -> "MetricDefinition":
        lowered = self.name.lower()
        if "ttft" in lowered and self.aggregation not in {"p50", "p95", "p99", "mean"}:
            raise ValueError("TTFT metrics require an explicit p50, p95, p99, or mean aggregation")
        if "tokens_per_second" in lowered and self.unit not in {"output_token/s", "total_token/s"}:
            raise ValueError("tokens/s must say whether output or total tokens are counted")
        if "requests_per_second" in lowered and self.unit != "request/s":
            raise ValueError("requests/s metrics must use request/s")
        return self


class MetricObservation(StrictModel):
    """One metric value or an explicit reason that no value exists."""

    value: float | None = None
    state: MeasurementState = MeasurementState.MEASURED
    note: str | None = None

    @model_validator(mode="after")
    def value_matches_state(self) -> "MetricObservation":
        if self.state == MeasurementState.MEASURED and self.value is None:
            raise ValueError("MEASURED observations require a value")
        if self.state != MeasurementState.MEASURED and self.value is not None:
            raise ValueError("missing observations must not contain numeric values")
        return self

    @classmethod
    def measured(cls, value: float) -> "MetricObservation":
        return cls(value=value)

    @classmethod
    def missing(cls, state: MeasurementState, note: str | None = None) -> "MetricObservation":
        if state == MeasurementState.MEASURED:
            raise ValueError("Use measured() for numeric observations")
        return cls(state=state, note=note)


class ConditionEvidence(StrictModel):
    """Measurements for one canonical execution condition."""

    metrics: dict[str, MetricObservation] = Field(default_factory=dict)
    bundle_id: str | None = None
    bundle_revision: str | None = None

    @field_validator("metrics", mode="before")
    @classmethod
    def normalize_metric_values(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            str(name): ({"value": observation} if isinstance(observation, (int, float)) else observation)
            for name, observation in value.items()
        }


class EvidenceKey(StrictModel):
    """Identity dimensions that must match before conditions are compared."""

    task: str
    hardware: str
    engine: str
    engine_version: str
    model_id: str
    model_revision: str
    model_fingerprint: str | None = None
    mode: str
    profile: str


class EvidenceProvenance(StrictModel):
    cohort: str
    seeds: tuple[int, ...] = ()
    concurrency: int | None = Field(default=None, ge=1)
    prompt_length_distribution: Mapping[str, float] = Field(default_factory=dict)
    output_length_distribution: Mapping[str, float] = Field(default_factory=dict)
    run_ids: tuple[str, ...] = ()
    commit: str | None = None
    date: str
    artifacts: tuple[str, ...] = ()


class MetricDelta(StrictModel):
    metric: str
    condition: EvidenceCondition
    baseline: float | None
    candidate: float | None
    delta: float | None
    percent_delta: float | None
    state: MeasurementState


class CanonicalEvidenceRecord(StrictModel):
    """A matched No-PRA / PRA / PRA-bundle comparison for one exact key."""

    schema_version: int = 1
    key: EvidenceKey
    metric_definitions: dict[str, MetricDefinition]
    conditions: dict[EvidenceCondition, ConditionEvidence]
    provenance: EvidenceProvenance
    evidence_tier: str

    @model_validator(mode="after")
    def validate_condition_and_metric_contract(self) -> "CanonicalEvidenceRecord":
        required = set(EvidenceCondition)
        if set(self.conditions) != required:
            missing = sorted(value.value for value in required - set(self.conditions))
            extra = sorted(str(value) for value in set(self.conditions) - required)
            raise ValueError(f"canonical evidence requires exactly three conditions; missing={missing}, extra={extra}")
        bundle = self.conditions[EvidenceCondition.PRA_ADAPTOR_BUNDLE]
        if bool(bundle.bundle_id) != bool(bundle.bundle_revision):
            raise ValueError("bundle ID and immutable revision must be supplied together")
        for condition, evidence in self.conditions.items():
            if condition != EvidenceCondition.PRA_ADAPTOR_BUNDLE and (
                evidence.bundle_id or evidence.bundle_revision
            ):
                raise ValueError("bundle identity belongs only to PRA Adaptor Bundle")
            unknown = set(evidence.metrics) - set(self.metric_definitions)
            if unknown:
                raise ValueError(f"condition {condition.value} uses undefined metrics: {sorted(unknown)}")
        return self

    def delta(self, metric: str, condition: EvidenceCondition) -> MetricDelta:
        if condition == EvidenceCondition.NO_PRA:
            raise ValueError("No PRA is the baseline, not a delta condition")
        baseline = self.conditions[EvidenceCondition.NO_PRA].metrics.get(metric)
        candidate = self.conditions[condition].metrics.get(metric)
        if baseline is None or candidate is None:
            return MetricDelta(
                metric=metric, condition=condition, baseline=None, candidate=None,
                delta=None, percent_delta=None, state=MeasurementState.NOT_MEASURED,
            )
        if baseline.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, condition=condition, baseline=None, candidate=candidate.value,
                delta=None, percent_delta=None, state=baseline.state,
            )
        if candidate.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, condition=condition, baseline=baseline.value, candidate=None,
                delta=None, percent_delta=None, state=candidate.state,
            )
        assert baseline.value is not None and candidate.value is not None
        delta = candidate.value - baseline.value
        percent = None if baseline.value == 0 else 100.0 * delta / baseline.value
        return MetricDelta(
            metric=metric, condition=condition, baseline=baseline.value,
            candidate=candidate.value, delta=delta, percent_delta=percent,
            state=MeasurementState.MEASURED,
        )

    def incremental_adaptor_delta(self, metric: str) -> MetricDelta:
        baseline = self.conditions[EvidenceCondition.PRA_NO_ADAPTOR].metrics.get(metric)
        candidate = self.conditions[EvidenceCondition.PRA_ADAPTOR_BUNDLE].metrics.get(metric)
        if baseline is None or candidate is None:
            return MetricDelta(
                metric=metric, condition=EvidenceCondition.PRA_ADAPTOR_BUNDLE,
                baseline=None, candidate=None, delta=None, percent_delta=None,
                state=MeasurementState.NOT_MEASURED,
            )
        if baseline.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, condition=EvidenceCondition.PRA_ADAPTOR_BUNDLE,
                baseline=None, candidate=candidate.value, delta=None, percent_delta=None,
                state=baseline.state,
            )
        if candidate.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, condition=EvidenceCondition.PRA_ADAPTOR_BUNDLE,
                baseline=baseline.value, candidate=None, delta=None, percent_delta=None,
                state=candidate.state,
            )
        assert baseline.value is not None and candidate.value is not None
        delta = candidate.value - baseline.value
        percent = None if baseline.value == 0 else 100.0 * delta / baseline.value
        return MetricDelta(
            metric=metric, condition=EvidenceCondition.PRA_ADAPTOR_BUNDLE,
            baseline=baseline.value, candidate=candidate.value, delta=delta,
            percent_delta=percent, state=MeasurementState.MEASURED,
        )

    def serialize_for_control_plane(self) -> dict[str, Any]:
        """Return the same normalized document consumed by CLI and web views."""

        payload = self.model_dump(mode="json")
        payload["deltas"] = {
            condition.value: {
                metric: self.delta(metric, condition).model_dump(mode="json")
                for metric in self.metric_definitions
            }
            for condition in (
                EvidenceCondition.PRA_NO_ADAPTOR,
                EvidenceCondition.PRA_ADAPTOR_BUNDLE,
            )
        }
        payload["incremental_adaptor_deltas"] = {
            metric: self.incremental_adaptor_delta(metric).model_dump(mode="json")
            for metric in self.metric_definitions
        }
        return payload


def render_markdown_table(
    record: CanonicalEvidenceRecord,
    group: MetricGroup | None = None,
    *,
    compact_missing: bool = False,
) -> str:
    """Render grouped canonical values, optionally collapsing an unrun adaptor arm."""

    metric_names = [
        name
        for name, definition in record.metric_definitions.items()
        if group is None or definition.group == group
    ]
    adaptor_measured = any(
        record.conditions[EvidenceCondition.PRA_ADAPTOR_BUNDLE]
        .metrics.get(name, MetricObservation.missing(MeasurementState.NEEDS_RUN))
        .state
        == MeasurementState.MEASURED
        for name in metric_names
    )
    include_adaptor = not compact_missing or adaptor_measured
    if include_adaptor:
        lines = [
            "| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    else:
        lines = [
            "| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    for name in metric_names:
        definition = record.metric_definitions[name]
        no_pra = record.conditions[EvidenceCondition.NO_PRA].metrics.get(name)
        no_adapter = record.conditions[EvidenceCondition.PRA_NO_ADAPTOR].metrics.get(name)
        bundle = record.conditions[EvidenceCondition.PRA_ADAPTOR_BUNDLE].metrics.get(name)
        delta_no_adapter = record.delta(name, EvidenceCondition.PRA_NO_ADAPTOR)
        delta_bundle = record.delta(name, EvidenceCondition.PRA_ADAPTOR_BUNDLE)
        cells = [
            _metric_label(name), definition.unit, definition.direction.value,
            _format_observation(no_pra), _format_observation(no_adapter),
        ]
        if include_adaptor:
            cells.append(_format_observation(bundle))
        cells.append(_format_delta(delta_no_adapter))
        if include_adaptor:
            cells.append(_format_delta(delta_bundle))
        lines.append(
            "| " + " | ".join(cells) + " |"
        )
    if compact_missing and not include_adaptor:
        lines += [
            "",
            "PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.",
        ]
    return "\n".join(lines) + "\n"


def render_latex_table(record: CanonicalEvidenceRecord, group: MetricGroup | None = None) -> str:
    """Render a compact LaTeX table with canonical grouped headers."""

    rows = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Metric & Unit & Direction & No PRA & PRA no adaptor & PRA bundle & $\Delta$ no adaptor & $\Delta$ bundle \\",
        r"\midrule",
    ]
    for name, definition in record.metric_definitions.items():
        if group is not None and definition.group != group:
            continue
        values = [
            _format_observation(record.conditions[condition].metrics.get(name))
            for condition in EvidenceCondition
        ]
        deltas = [
            _format_delta(record.delta(name, condition))
            for condition in (EvidenceCondition.PRA_NO_ADAPTOR, EvidenceCondition.PRA_ADAPTOR_BUNDLE)
        ]
        cells = [_metric_label(name).replace("_", r"\_"), definition.unit, definition.direction.value.replace("_", r"\_"), *values, *deltas]
        rows.append(" & ".join(cells) + r" \\")
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(rows) + "\n"


def _format_observation(value: MetricObservation | None) -> str:
    if value is None:
        return MeasurementState.NOT_MEASURED.value
    if value.state != MeasurementState.MEASURED:
        return value.state.value
    assert value.value is not None
    return f"{value.value:.6g}"


def _format_delta(value: MetricDelta) -> str:
    if value.state != MeasurementState.MEASURED or value.delta is None:
        return value.state.value
    percent = "" if value.percent_delta is None else f" ({value.percent_delta:+.2f}%)"
    return f"{value.delta:+.6g}{percent}"


def _metric_label(name: str) -> str:
    """Return a compact public label without obscuring aggregation semantics."""

    initialisms = {"ttft": "TTFT", "itl": "ITL", "kv": "K/V", "f1": "F1"}
    parts = name.split("_")
    unit_suffix = " (ms)" if parts[-1:] == ["ms"] else ""
    if unit_suffix:
        parts.pop()
    words = [
        initialisms.get(part, part if part.startswith("p") and part[1:].isdigit() else part.title())
        for part in parts
    ]
    return " ".join(words) + unit_suffix


STANDARD_METRICS: dict[str, MetricDefinition] = {
    row.name: row
    for row in (
        MetricDefinition(name="official_task_success", group=MetricGroup.QUALITY, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of tasks passing the official verifier."),
        MetricDefinition(name="token_f1", group=MetricGroup.QUALITY, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Token-overlap F1 against the reference answer."),
        MetricDefinition(name="exact_match", group=MetricGroup.QUALITY, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of answers exactly matching the reference."),
        MetricDefinition(name="gold_answer_log_probability", group=MetricGroup.QUALITY, unit="log_probability", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Mean log-probability assigned to the gold answer."),
        MetricDefinition(name="verifier_checks_passed", group=MetricGroup.QUALITY, unit="count", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="sum", description="Official verifier checks passed."),
        MetricDefinition(name="input_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.LOWER_IS_BETTER, aggregation="sum", description="Cumulative model input tokens."),
        MetricDefinition(name="visible_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Tokens visible in the sequential prompt."),
        MetricDefinition(name="selected_native_kv_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.NEUTRAL, aggregation="mean", description="Native K/V tokens selected outside the sequential prompt."),
        MetricDefinition(name="ttft_p50_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p50", description="Median time to first generated token."),
        MetricDefinition(name="ttft_p95_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p95", description="95th-percentile time to first generated token."),
        MetricDefinition(name="ttft_p99_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p99", description="99th-percentile time to first generated token."),
        MetricDefinition(name="itl_p50_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p50", description="Median inter-token latency after the first generated token."),
        MetricDefinition(name="itl_p95_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p95", description="95th-percentile inter-token latency after the first generated token."),
        MetricDefinition(name="itl_p99_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="p99", description="99th-percentile inter-token latency after the first generated token."),
        MetricDefinition(name="output_tokens_per_second", group=MetricGroup.SERVING, unit="output_token/s", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Mean decode-only output-token rate, excluding time to first token."),
        MetricDefinition(name="requests_per_second", group=MetricGroup.SERVING, unit="request/s", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Successful and failed completed requests per second."),
        MetricDefinition(name="queue_time_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean engine or gateway queue time per benchmark task."),
        MetricDefinition(name="inference_time_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean cumulative model-inference time per benchmark task."),
        MetricDefinition(name="task_wall_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="median", description="End-to-end task wall time."),
        MetricDefinition(name="completion_latency_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean end-to-end completion latency."),
        MetricDefinition(name="peak_accelerator_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak accelerator-resident memory."),
        MetricDefinition(name="peak_memory_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak memory reported by the engine-specific allocator."),
        MetricDefinition(name="active_detail_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Native detail bytes active for the request."),
        MetricDefinition(name="retained_detail_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Native detail bytes retained for reuse."),
        MetricDefinition(name="cost_per_successful_task", group=MetricGroup.COST, unit="USD/task", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Total cost divided by successful tasks."),
        MetricDefinition(name="evidence_recall", group=MetricGroup.ROUTING, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of annotated evidence recovered at the stated budget."),
    )
}
