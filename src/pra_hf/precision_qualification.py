"""Automated, identity-safe precision qualification artifact generation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .canonical_evidence import (
    CanonicalEvidenceRecord,
    ConditionEvidence,
    EvidenceCondition,
    EvidenceKey,
    EvidenceProvenance,
    MeasurementState,
    MetricObservation,
    STANDARD_METRICS,
    render_latex_table,
    render_markdown_table,
)
from .onboarding import ModelInspector
from .precision import (
    MemoryGateObservation,
    MemoryGateStatus,
    PrecisionDescriptor,
    classify_memory_gate,
)


QUALIFICATION_METRICS = (
    "token_f1",
    "exact_match",
    "evidence_recall",
    "logical_candidate_tokens",
    "visible_tokens",
    "selected_native_kv_tokens",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "output_tokens_per_second",
    "requests_per_second",
    "completion_latency_mean_ms",
    "prefill_time_mean_ms",
    "decode_time_mean_ms",
    "peak_accelerator_bytes",
    "peak_unified_memory_bytes",
    "peak_host_memory_bytes",
    "kv_memory_bytes",
    "temporary_allocation_bytes",
    "transfer_bytes",
    "storage_reloads",
    "model_artifact_bytes",
    "load_time_mean_ms",
    "max_successful_context_tokens",
    "cost_per_successful_task",
)


@dataclass(frozen=True)
class PrecisionQualificationRequest:
    model_id: str
    revision: str | None
    tokenizer_revision: str | None
    engine: str
    engine_version: str
    dataset: str
    profile: str
    mode: str
    precision: PrecisionDescriptor
    feature_extraction_precision: str | None = None
    adaptor_parameter_precision: str | None = None
    config_hash: str | None = None
    quantization_config_hash: str | None = None
    conversion_revision: str | None = None
    conversion_tool: str | None = None
    quantization_recipe: str | None = None
    artifact_checksum: str | None = None
    bundle_id: str | None = None
    bundle_revision: str | None = None
    evidence_tier: str = "NOT_MEASURED"
    date: str = "NOT_MEASURED"
    commit: str | None = None

    def __post_init__(self) -> None:
        if bool(self.bundle_id) != bool(self.bundle_revision):
            raise ValueError("bundle ID and immutable revision must be supplied together")
        if not self.precision.is_explicit:
            raise ValueError("precision qualification requires an explicit precision family")


class PrecisionQualificationService:
    """Resolve identity, validate evidence, and emit all publication fragments."""

    def __init__(self, inspector: ModelInspector | None = None) -> None:
        self.inspector = inspector or ModelInspector()

    def qualify(
        self,
        request: PrecisionQualificationRequest,
        *,
        output: str | Path,
        evidence: str | Path | None = None,
        memory_gate: str | Path | None = None,
    ) -> dict[str, Any]:
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        inspected = self.inspector.inspect(request.model_id, revision=request.revision)
        revision = request.revision or str(inspected["model"]["revision"])
        if revision in {"", "main", "master", "latest", "unresolved"}:
            raise ValueError("precision qualification requires an immutable model revision")
        config_hash = request.config_hash or hashlib.sha256(
            json.dumps(inspected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        observations = self._load_memory_gate(memory_gate)
        memory_status = (
            classify_memory_gate(observations).value
            if observations
            else MeasurementState.NOT_MEASURED.value
        )
        record = (
            self._load_evidence(evidence, request, revision)
            if evidence
            else self._empty_record(request, revision, config_hash)
        )
        workflow = self._workflow(record, observations, memory_status)
        result = {
            "schema_version": 1,
            "identity": {
                "model_id": request.model_id,
                "model_revision": revision,
                "tokenizer_revision": request.tokenizer_revision or revision,
                "config_hash": config_hash,
                "precision": request.precision.to_dict(),
                "engine": request.engine,
                "engine_version": request.engine_version,
                "dataset": request.dataset,
                "profile": request.profile,
                "mode": request.mode,
            },
            "memory_gate": {
                "status": memory_status,
                "observations": [row.to_dict() for row in observations],
            },
            "workflow": workflow,
            "canonical_evidence": record.serialize_for_control_plane(),
        }
        self._write_outputs(target, result, record)
        return result

    @staticmethod
    def _load_memory_gate(path: str | Path | None) -> tuple[MemoryGateObservation, ...]:
        if path is None:
            return ()
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = document.get("observations", document.get("memory_gate", document))
        if isinstance(rows, Mapping):
            rows = rows.get("observations", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("memory-gate artifact requires an observations list")
        return tuple(MemoryGateObservation(**dict(row)) for row in rows)

    @staticmethod
    def _load_evidence(
        path: str | Path,
        request: PrecisionQualificationRequest,
        revision: str,
    ) -> CanonicalEvidenceRecord:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = document.get("canonical_evidence", document)
        if isinstance(raw, Mapping) and "schema_version" not in raw and "key" not in raw:
            raise ValueError("evidence artifact does not contain canonical_evidence")
        fields = CanonicalEvidenceRecord.model_fields
        record = CanonicalEvidenceRecord.model_validate(
            {name: raw[name] for name in fields if name in raw}
        )
        expected = {
            "model_id": request.model_id,
            "model_revision": revision,
            "engine": request.engine,
            "task": request.dataset,
            "profile": request.profile,
            "mode": request.mode,
            "precision_family": request.precision.precision_family,
            "precision_encoding": request.precision.precision_encoding,
        }
        actual = {name: getattr(record.key, name) for name in expected}
        mismatches = [
            f"{name}: expected {value!r}, got {actual[name]!r}"
            for name, value in expected.items()
            if actual[name] != value
        ]
        if mismatches:
            raise ValueError("precision evidence identity mismatch: " + "; ".join(mismatches))
        return record

    @staticmethod
    def _empty_record(
        request: PrecisionQualificationRequest,
        revision: str,
        config_hash: str,
    ) -> CanonicalEvidenceRecord:
        missing = {
            name: MetricObservation.missing(
                MeasurementState.NEEDS_RUN,
                "The exact precision-qualified condition has not been run.",
            )
            for name in QUALIFICATION_METRICS
        }
        adaptor_state = (
            MeasurementState.NEEDS_RUN
            if request.bundle_id
            else MeasurementState.NO_QUALIFIED_ADAPTER
        )
        adaptor = {
            name: MetricObservation.missing(
                adaptor_state,
                "No immutable precision-qualified adaptor bundle was supplied."
                if not request.bundle_id
                else "The immutable bundle condition still needs a matched run.",
            )
            for name in QUALIFICATION_METRICS
        }
        return CanonicalEvidenceRecord(
            schema_version=2,
            key=EvidenceKey(
                task=request.dataset,
                hardware="NOT_MEASURED",
                engine=request.engine,
                engine_version=request.engine_version,
                model_id=request.model_id,
                model_revision=revision,
                precision_family=request.precision.precision_family,
                precision_encoding=request.precision.precision_encoding,
                mode=request.mode,
                profile=request.profile,
            ),
            metric_definitions={name: STANDARD_METRICS[name] for name in QUALIFICATION_METRICS},
            conditions={
                EvidenceCondition.NO_PRA: ConditionEvidence(metrics=missing),
                EvidenceCondition.PRA_NO_ADAPTOR: ConditionEvidence(metrics=missing),
                EvidenceCondition.PRA_ADAPTOR_BUNDLE: ConditionEvidence(
                    metrics=adaptor,
                    bundle_id=request.bundle_id,
                    bundle_revision=request.bundle_revision,
                ),
            },
            provenance=EvidenceProvenance(
                cohort="precision qualification plan; measurements pending",
                commit=request.commit,
                date=request.date,
                tokenizer_revision=request.tokenizer_revision or revision,
                config_hash=config_hash,
                quantization_config_hash=request.quantization_config_hash,
                conversion_revision=request.conversion_revision,
                conversion_tool=request.conversion_tool,
                quantization_recipe=request.quantization_recipe,
                artifact_checksum=request.artifact_checksum,
                feature_extraction_precision=(
                    request.feature_extraction_precision
                    or request.precision.feature_extraction_precision
                ),
                adaptor_parameter_precision=(
                    request.adaptor_parameter_precision
                    or request.precision.adaptor_parameter_precision
                ),
            ),
            evidence_tier=request.evidence_tier,
        )

    @staticmethod
    def _workflow(
        record: CanonicalEvidenceRecord,
        observations: Sequence[MemoryGateObservation],
        memory_status: str,
    ) -> list[dict[str, str]]:
        measured = any(
            observation.state == MeasurementState.MEASURED
            for evidence in record.conditions.values()
            for observation in evidence.metrics.values()
        )
        gate_passed = memory_status == MemoryGateStatus.QUALIFIABLE.value
        return [
            {"stage": "resolve_model", "status": "PASS"},
            {
                "stage": "memory_gate",
                "status": memory_status if observations else "NOT_MEASURED",
            },
            {"stage": "extract_features", "status": "RECORDED" if measured else "NEEDS_RUN"},
            {"stage": "train_or_reuse_adaptor", "status": "RECORDED" if measured else "NEEDS_RUN"},
            {
                "stage": "canonical_three_condition_run",
                "status": "RECORDED" if measured and gate_passed else "NEEDS_RUN",
            },
            {"stage": "emit_publication_artifacts", "status": "PASS"},
        ]

    @staticmethod
    def _write_outputs(
        target: Path,
        result: Mapping[str, Any],
        record: CanonicalEvidenceRecord,
    ) -> None:
        (target / "evidence.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (target / "manifest.yaml").write_text(
            yaml.safe_dump(dict(result["identity"]), sort_keys=False), encoding="utf-8"
        )
        (target / "card_fragment.md").write_text(
            PrecisionQualificationService._card_fragment(record), encoding="utf-8"
        )
        (target / "table.tex").write_text(render_latex_table(record), encoding="utf-8")
        with (target / "precision_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "task", "hardware", "engine", "model", "precision_family",
                    "precision_encoding", "mode", "profile", "condition", "metric",
                    "value", "state",
                )
            )
            for condition, evidence in record.conditions.items():
                for metric, observation in evidence.metrics.items():
                    writer.writerow(
                        (
                            record.key.task,
                            record.key.hardware,
                            record.key.engine,
                            record.key.model_id,
                            record.key.precision_family,
                            record.key.precision_encoding,
                            record.key.mode,
                            record.key.profile,
                            condition.value,
                            metric,
                            observation.value,
                            observation.state.value,
                        )
                    )

    @staticmethod
    def _card_fragment(record: CanonicalEvidenceRecord) -> str:
        lines = [
            "## Precision qualification",
            "",
            "Qualification is scoped to the exact model revision, encoding, engine, mode, and profile.",
            "",
            "| Task | HW/Engine | Precision | Mode | Profile | Evidence |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {record.key.task} | {record.key.hardware} / {record.key.engine} "
            f"| {record.key.precision_family} / {record.key.precision_encoding} "
            f"| {record.key.mode} | {record.key.profile} | {record.evidence_tier} |",
            "",
            render_markdown_table(record).rstrip(),
            "",
        ]
        return "\n".join(lines)
