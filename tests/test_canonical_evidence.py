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
from experiments.paper4_5_runtime.audit_evidence_conditions import audit_document


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
                "ttft_p95_ms": 1200,
                "output_tokens_per_second": 18,
            }),
            EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR: ConditionEvidence(metrics={
                "official_task_success": 0.6,
                "ttft_p95_ms": 1000,
                "output_tokens_per_second": 20,
            }),
            EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR: ConditionEvidence(metrics={
                "official_task_success": 0.6,
                "ttft_p95_ms": 800,
                "output_tokens_per_second": 22,
            }),
            EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE: ConditionEvidence(
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


def test_staged_deltas_preserve_mathematical_sign_and_attribution() -> None:
    record = _record()
    latency = record.named_delta("ttft_p95_ms", "delta_nm_vs_sc")
    quality = record.named_delta("official_task_success", "delta_sc_vs_no_pra")
    incremental = record.incremental_adaptor_delta("official_task_success")
    assert latency.delta == -200
    assert latency.percent_delta == -20
    assert latency.source_condition == EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR
    assert latency.target_condition == EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR
    assert quality.delta == pytest.approx(0.1)
    assert incremental.delta == pytest.approx(0.1)


def test_missing_state_is_not_encoded_as_zero() -> None:
    record = _record()
    value = record.model_copy(update={
        "conditions": {
            **record.conditions,
            EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR: ConditionEvidence(metrics={
                "official_task_success": MetricObservation.missing(MeasurementState.BLOCKED, "baseline floor")
            }),
        }
    })
    delta = value.named_delta("official_task_success", "delta_sc_vs_no_pra")
    assert delta.delta is None
    assert delta.state == MeasurementState.BLOCKED


def test_renderers_use_canonical_condition_grammar() -> None:
    markdown = render_markdown_table(_record(), MetricGroup.SERVING)
    latex = render_latex_table(_record(), MetricGroup.QUALITY)
    assert "No PRA | Selected Context | Native Memory | Native Memory + Bundle" in markdown
    assert "-200 (-20.00%)" in markdown
    assert "Selected Context & Native Memory & Native Memory + Bundle" in latex
    assert "Official Task Success" in latex


def test_card_table_does_not_create_unsupported_bundle_cells() -> None:
    record = _record()
    record = record.model_copy(update={
        "conditions": {
            condition: evidence
            for condition, evidence in record.conditions.items()
            if condition != EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE
        }
    })

    markdown = render_markdown_table(
        record, MetricGroup.SERVING, compact_missing=True
    )

    assert "Native Memory + Bundle" not in markdown
    assert "Delta Bundle vs NM" not in markdown
    assert "Delta NM vs SC" in markdown


def test_control_plane_serialization_contains_computed_deltas() -> None:
    payload = _record().serialize_for_control_plane()
    assert payload["conditions"]["NO_PRA"]["metrics"]["ttft_p95_ms"]["value"] == 1200
    assert payload["deltas"]["delta_nm_vs_sc"]["ttft_p95_ms"]["delta"] == -200
    assert payload["deltas"]["delta_nm_bundle_vs_nm"]["official_task_success"]["delta"] == pytest.approx(0.1)


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


def test_record_accepts_only_applicable_conditions_but_not_an_empty_set() -> None:
    payload = _record().model_dump()
    payload["conditions"] = {}
    with pytest.raises(ValidationError, match="at least one explicit condition"):
        CanonicalEvidenceRecord.model_validate(payload)


def test_schema_three_requires_exact_precision_but_legacy_schema_remains_readable() -> None:
    payload = _record().model_dump(mode="json")
    payload["key"].update(
        precision_family="UNSPECIFIED", precision_encoding="UNSPECIFIED"
    )
    with pytest.raises(ValidationError, match="explicit precision"):
        CanonicalEvidenceRecord.model_validate(payload)

    payload["schema_version"] = 1
    legacy = CanonicalEvidenceRecord.model_validate(payload)
    assert legacy.key.precision_family == "UNSPECIFIED"


def test_no_pra_rejects_pra_only_metrics() -> None:
    payload = _record().model_dump()
    payload["metric_definitions"]["selected_native_kv_tokens"] = STANDARD_METRICS[
        "selected_native_kv_tokens"
    ]
    payload["conditions"][EvidenceCondition.NO_PRA]["metrics"][
        "selected_native_kv_tokens"
    ] = MetricObservation.measured(0)
    with pytest.raises(ValidationError, match="NO_PRA cannot contain PRA-only"):
        CanonicalEvidenceRecord.model_validate(payload)


def test_legacy_e0_e2_record_maps_to_selected_and_native_not_no_pra() -> None:
    payload = _record().model_dump(mode="json")
    payload["schema_version"] = 2
    payload["conditions"] = {
        "no_pra": payload["conditions"]["NO_PRA"],
        "pra_no_adaptor": payload["conditions"]["PRA_NATIVE_MEMORY_NO_ADAPTOR"],
        "pra_adaptor_bundle": payload["conditions"]["PRA_NATIVE_MEMORY_BUNDLE"],
    }
    migrated = CanonicalEvidenceRecord.model_validate(payload)
    assert EvidenceCondition.NO_PRA not in migrated.conditions
    assert EvidenceCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR in migrated.conditions
    assert EvidenceCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR in migrated.conditions


def test_legacy_baseline_without_native_mode_is_rejected_as_ambiguous() -> None:
    payload = _record().model_dump(mode="json")
    payload["schema_version"] = 2
    payload["key"]["mode"] = "selected-context"
    payload["conditions"] = {"no_pra": payload["conditions"]["NO_PRA"]}
    with pytest.raises(ValidationError, match="AMBIGUOUS_LEGACY_CONDITION"):
        CanonicalEvidenceRecord.model_validate(payload)


def test_measured_bundle_requires_exact_bundle_identity() -> None:
    payload = _record().model_dump()
    payload["conditions"][EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE]["bundle_id"] = None
    payload["conditions"][EvidenceCondition.PRA_NATIVE_MEMORY_BUNDLE]["bundle_revision"] = None
    with pytest.raises(ValidationError, match="measured bundle evidence requires"):
        CanonicalEvidenceRecord.model_validate(payload)


def test_audit_rejects_mislabeled_e0_e2_and_anonymous_delta() -> None:
    findings = audit_document({
        "schema_version": 2,
        "key": {"mode": "native-memory"},
        "conditions": {
            "no_pra": {"metrics": {"visible_tokens": 396}},
            "pra_no_adaptor": {"metrics": {"visible_tokens": 33.5}},
        },
        "quality_delta": 0.0,
    })
    codes = {finding.code for finding in findings}
    assert "LEGACY_E0_E2_ATTRIBUTION" in codes
    assert "ANONYMOUS_DELTA" in codes


def test_audit_accepts_explicit_selected_to_native_conditions() -> None:
    findings = audit_document({
        "conditions": {
            "PRA_SELECTED_CONTEXT_NO_ADAPTOR": {"metrics": {"visible_tokens": 396}},
            "PRA_NATIVE_MEMORY_NO_ADAPTOR": {"metrics": {"visible_tokens": 33.5}},
        },
        "delta_nm_vs_sc": {"visible_tokens": -362.5},
    })
    assert not [finding for finding in findings if finding.severity == "ERROR"]
