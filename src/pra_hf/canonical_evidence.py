"""Canonical staged PRA evidence shared by product surfaces.

Execution depth and bundle use are independent dimensions.  In particular,
Selected Context is PRA: it must never be serialized as the ordinary No-PRA
baseline.  A metric is identified by its direction, unit, aggregation, source
condition, and target condition; absent measurements retain an explicit state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .precision import PRECISION_FAMILIES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceCondition(str, Enum):
    NO_PRA = "NO_PRA"
    PRA_SELECTED_CONTEXT_NO_ADAPTOR = "PRA_SELECTED_CONTEXT_NO_ADAPTOR"
    PRA_NATIVE_MEMORY_NO_ADAPTOR = "PRA_NATIVE_MEMORY_NO_ADAPTOR"
    PRA_NATIVE_SERVING_NO_ADAPTOR = "PRA_NATIVE_SERVING_NO_ADAPTOR"
    PRA_SELECTED_CONTEXT_BUNDLE = "PRA_SELECTED_CONTEXT_BUNDLE"
    PRA_NATIVE_MEMORY_BUNDLE = "PRA_NATIVE_MEMORY_BUNDLE"
    PRA_NATIVE_SERVING_BUNDLE = "PRA_NATIVE_SERVING_BUNDLE"


CONDITION_LABELS: dict[EvidenceCondition, str] = {
    EvidenceCondition.NO_PRA: "No PRA",
    EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR: "Selected Context",
    EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR: "Native Memory",
    EvidenceCondition.PRA_NATIVE_SERVING_NO_ADAPTOR: "Native Serving",
    EvidenceCondition.PRA_SELECTED_CONTEXT_BUNDLE: "Selected Context + Bundle",
    EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE: "Native Memory + Bundle",
    EvidenceCondition.PRA_NATIVE_SERVING_BUNDLE: "Native Serving + Bundle",
}

CONDITION_ORDER = tuple(EvidenceCondition)
BUNDLE_CONDITIONS = frozenset(
    {
        EvidenceCondition.PRA_SELECTED_CONTEXT_BUNDLE,
        EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE,
        EvidenceCondition.PRA_NATIVE_SERVING_BUNDLE,
    }
)
PRA_ONLY_METRICS = frozenset(
    {
        "selected_native_kv_tokens",
        "active_detail_bytes",
        "retained_detail_bytes",
        "pra_cache_hit_rate",
    }
)

DELTA_PAIRS: dict[str, tuple[EvidenceCondition, EvidenceCondition]] = {
    "delta_sc_vs_no_pra": (
        EvidenceCondition.NO_PRA,
        EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
    ),
    "delta_nm_vs_no_pra": (
        EvidenceCondition.NO_PRA,
        EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
    ),
    "delta_ns_vs_no_pra": (
        EvidenceCondition.NO_PRA,
        EvidenceCondition.PRA_NATIVE_SERVING_NO_ADAPTOR,
    ),
    "delta_nm_vs_sc": (
        EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
    ),
    "delta_ns_vs_nm": (
        EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        EvidenceCondition.PRA_NATIVE_SERVING_NO_ADAPTOR,
    ),
    "delta_sc_bundle_vs_sc": (
        EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        EvidenceCondition.PRA_SELECTED_CONTEXT_BUNDLE,
    ),
    "delta_nm_bundle_vs_nm": (
        EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE,
    ),
    "delta_ns_bundle_vs_ns": (
        EvidenceCondition.PRA_NATIVE_SERVING_NO_ADAPTOR,
        EvidenceCondition.PRA_NATIVE_SERVING_BUNDLE,
    ),
}


def condition_for_mode(mode: str, *, bundle: bool = False) -> EvidenceCondition:
    """Resolve one public execution mode to its canonical evidence condition."""

    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "e0": "selected_context",
        "selected": "selected_context",
        "agent_gateway": "selected_context",
        "e2": "native_memory",
        "native": "native_memory",
        "e3": "native_serving",
    }
    normalized = aliases.get(normalized, normalized)
    lookup = {
        ("selected_context", False): EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        ("native_memory", False): EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
        ("native_serving", False): EvidenceCondition.PRA_NATIVE_SERVING_NO_ADAPTOR,
        ("selected_context", True): EvidenceCondition.PRA_SELECTED_CONTEXT_BUNDLE,
        ("native_memory", True): EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE,
        ("native_serving", True): EvidenceCondition.PRA_NATIVE_SERVING_BUNDLE,
    }
    try:
        return lookup[(normalized, bundle)]
    except KeyError as exc:
        raise ValueError(f"unknown PRA evidence mode: {mode!r}") from exc


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


class LegacyConditionClass(str, Enum):
    TRUE_NO_PRA = "TRUE_NO_PRA"
    PRA_SELECTED_CONTEXT = "PRA_SELECTED_CONTEXT"
    PRA_NATIVE_MEMORY = "PRA_NATIVE_MEMORY"
    PRA_NATIVE_SERVING = "PRA_NATIVE_SERVING"
    AMBIGUOUS = "AMBIGUOUS"


def classify_legacy_condition(
    label: str,
    *,
    provenance: str = "",
) -> LegacyConditionClass:
    """Classify a legacy label without treating ``baseline`` as No PRA."""

    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"e0", "selected", "selected_context", "e0_selected_text"}:
        return LegacyConditionClass.PRA_SELECTED_CONTEXT
    if normalized in {"e2", "native", "native_memory", "e2_native_kv"}:
        return LegacyConditionClass.PRA_NATIVE_MEMORY
    if normalized in {"e3", "native_serving"}:
        return LegacyConditionClass.PRA_NATIVE_SERVING
    if normalized == "no_pra":
        source = provenance.lower()
        if any(term in source for term in ("ordinary inference", "standard rag", "pra disabled")):
            return LegacyConditionClass.TRUE_NO_PRA
    return LegacyConditionClass.AMBIGUOUS


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
    precision_family: str = "UNSPECIFIED"
    precision_encoding: str = "UNSPECIFIED"
    mode: str
    profile: str

    @field_validator("precision_family")
    @classmethod
    def normalize_precision_family(cls, value: str) -> str:
        family = value.strip().upper()
        if family not in PRECISION_FAMILIES:
            raise ValueError(f"unknown precision family: {value!r}")
        return family

    @model_validator(mode="after")
    def require_paired_precision_identity(self) -> "EvidenceKey":
        if bool(self.precision_family == "UNSPECIFIED") != bool(
            self.precision_encoding == "UNSPECIFIED"
        ):
            raise ValueError("precision family and encoding must be supplied together")
        if not self.precision_encoding.strip():
            raise ValueError("precision encoding cannot be empty")
        return self


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
    tokenizer_revision: str | None = None
    config_hash: str | None = None
    quantization_config_hash: str | None = None
    conversion_revision: str | None = None
    conversion_tool: str | None = None
    quantization_recipe: str | None = None
    artifact_checksum: str | None = None
    feature_extraction_precision: str | None = None
    adaptor_parameter_precision: str | None = None


class MetricDelta(StrictModel):
    metric: str
    name: str | None = None
    source_condition: EvidenceCondition
    target_condition: EvidenceCondition
    baseline: float | None
    candidate: float | None
    delta: float | None
    percent_delta: float | None
    state: MeasurementState


class CanonicalEvidenceRecord(StrictModel):
    """A matched staged comparison for one exact evidence identity."""

    schema_version: int = 3
    key: EvidenceKey
    metric_definitions: dict[str, MetricDefinition]
    conditions: dict[EvidenceCondition, ConditionEvidence]
    provenance: EvidenceProvenance
    evidence_tier: str

    @model_validator(mode="before")
    @classmethod
    def migrate_schema_two_conditions(cls, value: Any) -> Any:
        """Relabel the known E0/E2 schema-2 layout without inventing No PRA.

        The old serializer called the E0 arm ``no_pra`` and the E2 arm
        ``pra_no_adaptor``.  We can repair that layout only when the record's
        mode identifies a Native Memory comparison.  Other legacy baselines
        are rejected as ambiguous and must be audited from provenance.
        """

        if not isinstance(value, Mapping) or int(value.get("schema_version", 2)) >= 3:
            return value
        raw_conditions = value.get("conditions")
        if not isinstance(raw_conditions, Mapping):
            return value
        normalized = {
            key.value if isinstance(key, EvidenceCondition) else str(key): item
            for key, item in raw_conditions.items()
        }
        legacy = {"no_pra", "pra_no_adaptor", "pra_adaptor_bundle"}
        if not legacy.intersection(normalized):
            return value
        mode = str(dict(value.get("key", {})).get("mode", "")).lower().replace("_", "-")
        if "native-memory" not in mode:
            raise ValueError(
                "AMBIGUOUS_LEGACY_CONDITION: schema-2 baseline cannot be mapped "
                "to NO_PRA without provenance; run the evidence-condition audit"
            )
        mapping = {
            "no_pra": EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value,
            "pra_no_adaptor": EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value,
            "pra_adaptor_bundle": EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE.value,
        }
        migrated = dict(value)
        migrated["schema_version"] = 3 if int(value.get("schema_version", 2)) >= 2 else 1
        migrated["conditions"] = {
            mapping.get(name, name): item for name, item in normalized.items()
        }
        return migrated

    @model_validator(mode="after")
    def validate_condition_and_metric_contract(self) -> "CanonicalEvidenceRecord":
        if self.schema_version >= 2 and self.key.precision_family == "UNSPECIFIED":
            raise ValueError("canonical evidence schema 2 requires explicit precision identity")
        if not self.conditions:
            raise ValueError("canonical evidence requires at least one explicit condition")
        for condition, evidence in self.conditions.items():
            is_bundle = condition in BUNDLE_CONDITIONS
            if bool(evidence.bundle_id) != bool(evidence.bundle_revision):
                raise ValueError("bundle ID and immutable revision must be supplied together")
            if not is_bundle and (
                evidence.bundle_id or evidence.bundle_revision
            ):
                raise ValueError("bundle identity belongs only to a bundle condition")
            if is_bundle and any(
                observation.state == MeasurementState.MEASURED
                for observation in evidence.metrics.values()
            ) and not evidence.bundle_id:
                raise ValueError("measured bundle evidence requires an exact bundle ID and revision")
            unknown = set(evidence.metrics) - set(self.metric_definitions)
            if unknown:
                raise ValueError(f"condition {condition.value} uses undefined metrics: {sorted(unknown)}")
        no_pra = self.conditions.get(EvidenceCondition.NO_PRA)
        if no_pra is not None:
            invalid = sorted(PRA_ONLY_METRICS.intersection(no_pra.metrics))
            if invalid:
                raise ValueError(f"NO_PRA cannot contain PRA-only metrics: {invalid}")
        return self

    def compare(
        self,
        metric: str,
        source: EvidenceCondition,
        target: EvidenceCondition,
        *,
        name: str | None = None,
    ) -> MetricDelta:
        """Compute target minus source for an explicit condition pair."""

        baseline = self.conditions.get(source, ConditionEvidence()).metrics.get(metric)
        candidate = self.conditions.get(target, ConditionEvidence()).metrics.get(metric)
        if baseline is None or candidate is None:
            return MetricDelta(
                metric=metric, name=name, source_condition=source, target_condition=target,
                baseline=None, candidate=None,
                delta=None, percent_delta=None, state=MeasurementState.NOT_MEASURED,
            )
        if baseline.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, name=name, source_condition=source, target_condition=target,
                baseline=None, candidate=candidate.value,
                delta=None, percent_delta=None, state=baseline.state,
            )
        if candidate.state != MeasurementState.MEASURED:
            return MetricDelta(
                metric=metric, name=name, source_condition=source, target_condition=target,
                baseline=baseline.value, candidate=None,
                delta=None, percent_delta=None, state=candidate.state,
            )
        assert baseline.value is not None and candidate.value is not None
        delta = candidate.value - baseline.value
        percent = None if baseline.value == 0 else 100.0 * delta / baseline.value
        return MetricDelta(
            metric=metric, name=name, source_condition=source, target_condition=target,
            baseline=baseline.value,
            candidate=candidate.value, delta=delta, percent_delta=percent,
            state=MeasurementState.MEASURED,
        )

    def named_delta(self, metric: str, name: str) -> MetricDelta:
        """Compute one canonical attribution delta by stable public name."""

        try:
            source, target = DELTA_PAIRS[name]
        except KeyError as exc:
            raise ValueError(f"unknown canonical delta: {name}") from exc
        return self.compare(metric, source, target, name=name)

    def delta(
        self,
        metric: str,
        condition: EvidenceCondition,
        source_condition: EvidenceCondition | None = None,
    ) -> MetricDelta:
        """Compatibility wrapper requiring an unambiguous target condition."""

        if source_condition is None:
            pairs = [pair for pair in DELTA_PAIRS.items() if pair[1][1] == condition]
            if len(pairs) != 1:
                raise ValueError("delta source is ambiguous; use compare() or named_delta()")
            name, (source_condition, _) = pairs[0]
        else:
            name = next(
                (key for key, pair in DELTA_PAIRS.items() if pair == (source_condition, condition)),
                None,
            )
        return self.compare(metric, source_condition, condition, name=name)

    def incremental_adaptor_delta(
        self, metric: str, mode: str = "native_memory"
    ) -> MetricDelta:
        names = {
            "selected_context": "delta_sc_bundle_vs_sc",
            "native_memory": "delta_nm_bundle_vs_nm",
            "native_serving": "delta_ns_bundle_vs_ns",
        }
        return self.named_delta(metric, names[mode.lower().replace(" ", "_").replace("-", "_")])

    def serialize_for_control_plane(self) -> dict[str, Any]:
        """Return the same normalized document consumed by CLI and web views."""

        payload = self.model_dump(mode="json")
        payload["deltas"] = {
            name: {
                metric: self.named_delta(metric, name).model_dump(mode="json")
                for metric in self.metric_definitions
            }
            for name, pair in DELTA_PAIRS.items()
            if pair[0] in self.conditions and pair[1] in self.conditions
        }
        return payload


def render_markdown_table(
    record: CanonicalEvidenceRecord,
    group: MetricGroup | None = None,
    *,
    compact_missing: bool = False,
) -> str:
    """Render only conditions and pairwise deltas present in this record."""

    metric_names = [
        name
        for name, definition in record.metric_definitions.items()
        if group is None or definition.group == group
    ]
    conditions = [condition for condition in CONDITION_ORDER if condition in record.conditions]
    if compact_missing:
        conditions = [
            condition for condition in conditions
            if any(
                record.conditions[condition].metrics.get(name) is not None
                for name in metric_names
            )
        ]
    deltas = [
        (name, pair) for name, pair in DELTA_PAIRS.items()
        if pair[0] in conditions and pair[1] in conditions
    ]
    headers = ["Metric", "Unit", "Direction"] + [
        CONDITION_LABELS[condition] for condition in conditions
    ] + [_delta_label(name) for name, _ in deltas]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index < 3 else "---:" for index in range(len(headers))) + " |",
    ]
    for name in metric_names:
        definition = record.metric_definitions[name]
        cells = [
            _metric_label(name), definition.unit, definition.direction.value,
            *[
                _format_observation(record.conditions[condition].metrics.get(name))
                for condition in conditions
            ],
            *[_format_delta(record.named_delta(name, delta_name)) for delta_name, _ in deltas],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_latex_table(record: CanonicalEvidenceRecord, group: MetricGroup | None = None) -> str:
    """Render a compact condition-aware LaTeX table."""

    conditions = [condition for condition in CONDITION_ORDER if condition in record.conditions]
    deltas = [
        (name, pair) for name, pair in DELTA_PAIRS.items()
        if pair[0] in conditions and pair[1] in conditions
    ]
    column_count = 3 + len(conditions) + len(deltas)
    rows = [
        rf"\begin{{tabular}}{{lll{'r' * (column_count - 3)}}}",
        r"\toprule",
        " & ".join(
            ["Metric", "Unit", "Direction"]
            + [CONDITION_LABELS[condition] for condition in conditions]
            + [_delta_label(name) for name, _ in deltas]
        ) + r" \\",
        r"\midrule",
    ]
    for name, definition in record.metric_definitions.items():
        if group is not None and definition.group != group:
            continue
        values = [
            _format_observation(record.conditions[condition].metrics.get(name))
            for condition in conditions
        ]
        delta_values = [
            _format_delta(record.named_delta(name, delta_name))
            for delta_name, _ in deltas
        ]
        cells = [_metric_label(name).replace("_", r"\_"), definition.unit, definition.direction.value.replace("_", r"\_"), *values, *delta_values]
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


def _delta_label(name: str) -> str:
    labels = {
        "delta_sc_vs_no_pra": "Delta SC vs No PRA",
        "delta_nm_vs_no_pra": "Delta NM vs No PRA",
        "delta_ns_vs_no_pra": "Delta NS vs No PRA",
        "delta_nm_vs_sc": "Delta NM vs SC",
        "delta_ns_vs_nm": "Delta NS vs NM",
        "delta_sc_bundle_vs_sc": "Delta Bundle vs SC",
        "delta_nm_bundle_vs_nm": "Delta Bundle vs NM",
        "delta_ns_bundle_vs_ns": "Delta Bundle vs NS",
    }
    return labels[name]


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
        MetricDefinition(name="logical_candidate_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.NEUTRAL, aggregation="mean", description="Tokens in the logical candidate set before selection."),
        MetricDefinition(name="visible_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Tokens visible in the sequential prompt."),
        MetricDefinition(name="selected_native_kv_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.NEUTRAL, aggregation="mean", description="Native K/V tokens selected outside the sequential prompt."),
        MetricDefinition(name="selected_full_ratio", group=MetricGroup.CONTEXT, unit="fraction", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Selected physical tokens divided by logical candidate tokens."),
        MetricDefinition(name="newly_materialized_tokens", group=MetricGroup.CONTEXT, unit="token", direction=MetricDirection.LOWER_IS_BETTER, aggregation="sum", description="Selected tokens materialized for the first time."),
        MetricDefinition(name="materialization_avoidance", group=MetricGroup.CONTEXT, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of otherwise repeated materialization avoided."),
        MetricDefinition(name="visible_reuse", group=MetricGroup.CONTEXT, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of visible context reused by the ordinary engine cache."),
        MetricDefinition(name="native_reuse", group=MetricGroup.CONTEXT, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of selected native K/V reused without re-encoding."),
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
        MetricDefinition(name="prefill_time_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean model prefill time per request."),
        MetricDefinition(name="decode_time_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean decode time after prefill."),
        MetricDefinition(name="load_time_mean_ms", group=MetricGroup.SERVING, unit="ms", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Mean model artifact load time."),
        MetricDefinition(name="peak_accelerator_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak accelerator-resident memory."),
        MetricDefinition(name="peak_memory_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak memory reported by the engine-specific allocator."),
        MetricDefinition(name="peak_unified_memory_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak Apple unified-memory residency."),
        MetricDefinition(name="peak_host_memory_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak host-memory residency."),
        MetricDefinition(name="kv_memory_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="K/V memory attributable to the measured request or retained resource."),
        MetricDefinition(name="temporary_allocation_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Peak temporary allocation outside persistent model and K/V state."),
        MetricDefinition(name="transfer_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="sum", description="Bytes transferred across the measured device or storage boundary."),
        MetricDefinition(name="storage_reloads", group=MetricGroup.RESOURCES, unit="count", direction=MetricDirection.LOWER_IS_BETTER, aggregation="sum", description="Storage-to-active residency reloads."),
        MetricDefinition(name="model_artifact_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="max", description="Size of the exact model artifact used by the run."),
        MetricDefinition(name="max_successful_context_tokens", group=MetricGroup.RESOURCES, unit="token", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="max", description="Largest context that passed the deterministic memory gate."),
        MetricDefinition(name="active_detail_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Native detail bytes active for the request."),
        MetricDefinition(name="retained_detail_bytes", group=MetricGroup.RESOURCES, unit="byte", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Native detail bytes retained for reuse."),
        MetricDefinition(name="cost_per_successful_task", group=MetricGroup.COST, unit="USD/task", direction=MetricDirection.LOWER_IS_BETTER, aggregation="mean", description="Total cost divided by successful tasks."),
        MetricDefinition(name="evidence_recall", group=MetricGroup.ROUTING, unit="fraction", direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean", description="Fraction of annotated evidence recovered at the stated budget."),
    )
}
