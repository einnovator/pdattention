"""Canonical evidence contract and renderer tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pra_hf.canonical_evidence import (
    CanonicalEvidenceRecord,
    ConditionEvidence,
    EvidenceCondition,
    EvidenceKey,
    EvidenceProvenance,
    MeasurementState,
    MetricDefinition,
    MetricDirection,
    MetricGroup,
    MetricObservation,
    STANDARD_METRICS,
    render_latex_table,
    render_markdown_table,
)


def _record() -> CanonicalEvidenceRecord:
    metrics = {
        name: STANDARD_METRICS[name]
        for name in ("official_task_success", "ttft_p95_ms", "output_tokens_per_second")
    }
    return CanonicalEvidenceRecord(
        key=EvidenceKey(
            task="qasper", hardware="m5-48gb", engine="mlx", engine_version="0.29",
            model_id="Qwen/Qwen3-32B", model_revision="abc", mode="native-memory",
            precision_family="INT4", precision_encoding="MLX-4bit", profile="balanced",
        ),
        metric_definitions=metrics,
        conditions={
            EvidenceCondition.NO_PRA: ConditionEvidence(metrics={
                "official_task_success": 0.5,
                "ttft_p95_ms": 1000,
                "output_tokens_per_second": 20,
            }),
            EvidenceCondition.PRA_NO_ADAPTOR: ConditionEvidence(metrics={
                "official_task_success": 0.6,
                "ttft_p95_ms": 800,
                "output_tokens_per_second": 22,
            }),
            EvidenceCondition.PRA_ADAPTOR_BUNDLE: ConditionEvidence(
                bundle_id="EInnovator/pra-qwen3-32b",
                bundle_revision="def",
                metrics={
                    "official_task_success": 0.7,
                    "ttft_p95_ms": 750,
                    "output_tokens_per_second": 21,
                },
            ),
        },
        provenance=EvidenceProvenance(cohort="paired natural QA", date="2026-09-03"),
        evidence_tier="ENGINE_QUALIFIED",
    )


def test_three_condition_deltas_preserve_mathematical_sign() -> None:
    record = _record()
    latency = record.delta("ttft_p95_ms", EvidenceCondition.PRA_ADAPTOR_BUNDLE)
    quality = record.delta("official_task_success", EvidenceCondition.PRA_NO_ADAPTOR)
    incremental = record.incremental_adaptor_delta("official_task_success")
    assert latency.delta == -250
    assert latency.percent_delta == -25
    assert quality.delta == pytest.approx(0.1)
    assert incremental.delta == pytest.approx(0.1)


def test_missing_state_is_not_encoded_as_zero() -> None:
    record = _record()
    value = record.model_copy(update={
        "conditions": {
            **record.conditions,
            EvidenceCondition.PRA_NO_ADAPTOR: ConditionEvidence(metrics={
                "official_task_success": MetricObservation.missing(MeasurementState.BLOCKED, "baseline floor")
            }),
        }
    })
    delta = value.delta("official_task_success", EvidenceCondition.PRA_NO_ADAPTOR)
    assert delta.delta is None
    assert delta.state == MeasurementState.BLOCKED


def test_renderers_use_canonical_condition_grammar() -> None:
    markdown = render_markdown_table(_record(), MetricGroup.SERVING)
    latex = render_latex_table(_record(), MetricGroup.QUALITY)
    assert "No PRA | PRA - No Adaptor | PRA - Adaptor Bundle" in markdown
    assert "-250 (-25.00%)" in markdown
    assert "PRA no adaptor & PRA bundle" in latex
    assert "Official Task Success" in latex


def test_compact_card_table_collapses_uniformly_unrun_adaptor_metrics() -> None:
    record = _record()
    missing = {
        name: MetricObservation.missing(
            MeasurementState.NEEDS_RUN, "Exact learned-adaptor arm was not run."
        )
        for name in record.metric_definitions
    }
    record = record.model_copy(update={
        "conditions": {
            **record.conditions,
            EvidenceCondition.PRA_ADAPTOR_BUNDLE: ConditionEvidence(metrics=missing),
        }
    })

    markdown = render_markdown_table(
        record, MetricGroup.SERVING, compact_missing=True
    )

    assert "| No PRA | PRA - No Adaptor | Delta No Adaptor |" in markdown
    assert "Delta Bundle" not in markdown
    assert markdown.count("NEEDS_RUN") == 1


def test_control_plane_serialization_contains_computed_deltas() -> None:
    payload = _record().serialize_for_control_plane()
    assert payload["conditions"]["no_pra"]["metrics"]["ttft_p95_ms"]["value"] == 1000
    assert payload["deltas"]["pra_adaptor_bundle"]["ttft_p95_ms"]["delta"] == -250
    assert payload["incremental_adaptor_deltas"]["official_task_success"]["delta"] == pytest.approx(0.1)


def test_metric_definitions_reject_ambiguous_ttft_and_token_rate() -> None:
    with pytest.raises(ValidationError, match="TTFT metrics require"):
        MetricDefinition(
            name="ttft_ms", group=MetricGroup.SERVING, unit="ms",
            direction=MetricDirection.LOWER_IS_BETTER, description="Ambiguous TTFT.",
        )
    with pytest.raises(ValidationError, match="output or total"):
        MetricDefinition(
            name="tokens_per_second", group=MetricGroup.SERVING, unit="token/s",
            direction=MetricDirection.HIGHER_IS_BETTER, aggregation="mean",
            description="Ambiguous token rate.",
        )


def test_record_requires_exactly_three_conditions() -> None:
    payload = _record().model_dump()
    del payload["conditions"][EvidenceCondition.PRA_NO_ADAPTOR]
    with pytest.raises(ValidationError, match="exactly three conditions"):
        CanonicalEvidenceRecord.model_validate(payload)


def test_schema_two_requires_exact_precision_but_legacy_schema_remains_readable() -> None:
    payload = _record().model_dump(mode="json")
    payload["key"].update(
        precision_family="UNSPECIFIED", precision_encoding="UNSPECIFIED"
    )
    with pytest.raises(ValidationError, match="explicit precision"):
        CanonicalEvidenceRecord.model_validate(payload)

    payload["schema_version"] = 1
    legacy = CanonicalEvidenceRecord.model_validate(payload)
    assert legacy.key.precision_family == "UNSPECIFIED"
