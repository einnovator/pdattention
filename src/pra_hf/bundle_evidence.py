"""Identity-safe qualification evidence for public PRA runtime bundles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .canonical_evidence import (
    CanonicalEvidenceRecord,
    ConditionEvidence,
    EvidenceCondition,
    EvidenceKey,
    EvidenceProvenance,
    MeasurementState,
    MetricObservation,
    STANDARD_METRICS,
)


EVIDENCE_TIERS = frozenset(
    {
        "PRODUCTION_QUALIFIED",
        "ENGINE_QUALIFIED",
        "CONTROLLED",
        "RESEARCH",
        "SMOKE",
        "NOT_MEASURED",
        "NOT_APPLICABLE",
        "BLOCKED",
    }
)
METRIC_CLASSES = frozenset(
    {"END_TASK", "SEMANTIC_EQUIVALENCE", "ROUTING_DIAGNOSTIC", "SERVING_ECONOMICS"}
)
PROFILE_STATUSES = frozenset({"QUALIFIED", "CALIBRATION_PENDING", "RESEARCH", "BLOCKED"})


class EvidenceValidationError(ValueError):
    """Raised when evidence cannot be attributed to an exact bundle identity."""


@dataclass(frozen=True)
class EvidenceIdentity:
    """Fields that must match before an artifact can qualify a bundle."""

    model_id: str
    model_revision: str
    quantization: str
    engine: str
    engine_version: str
    profile: str
    execution_mode: str


def file_sha256(path: str | Path) -> str:
    """Return a stable checksum for evidence provenance."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quantization(value: object) -> str:
    if isinstance(value, Mapping):
        bits = value.get("bits")
        return f"{bits}bit" if bits is not None else str(value.get("name", "")).lower()
    return str(value or "").replace("-", "").lower()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    ttft = values("ttft_ms")
    completion = values("completion_latency_ms")
    visible = values("visible_prompt_tokens")
    return {
        "quality_metric": "token_f1",
        "quality": mean(values("token_f1")),
        "exact_match": mean(values("exact_match")),
        "gold_answer_logprob": mean(values("gold_answer_logprob")),
        "visible_tokens": mean(visible),
        "selected_native_kv_tokens": mean(values("selected_native_kv_tokens")),
        "active_detail_bytes": mean(values("active_detail_bytes")),
        "retained_detail_bytes": mean(values("retained_detail_bytes")),
        "ttft_ms": {"p50": median(ttft), "p95": _percentile(ttft, 0.95), "p99": _percentile(ttft, 0.99)},
        "completion_latency_ms": {
            "mean": mean(completion),
            "p50": median(completion),
            "p95": _percentile(completion, 0.95),
            "p99": _percentile(completion, 0.99),
        },
        "peak_memory_bytes": max(values("peak_unified_memory_bytes")),
    }


def _pct_delta(pra: float | None, baseline: float | None) -> float | None:
    if pra is None or baseline in {None, 0}:
        return None
    return 100.0 * (pra - baseline) / baseline


def import_mlx_paired_evidence(
    artifact: str | Path,
    identity: EvidenceIdentity,
    *,
    hardware: str | None = None,
    evidence_date: str = "2026-09-01",
    artifact_reference: str | None = None,
) -> list[dict[str, Any]]:
    """Import selector-frozen E0/E2 rows from one exact MLX profile artifact.

    The importer deliberately accepts only the all-consumer concatenated native
    condition. Segmented and reduced-layer variants remain research diagnostics.
    """

    path = Path(artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    actual = {
        "model_id": payload.get("model_id"),
        "model_revision": payload.get("model_revision"),
        "quantization": _quantization(next(iter(payload.get("rows", [{}])), {}).get("quantization")),
        "engine": "mlx-lm",
        "engine_version": str(runtime.get("mlx_lm", "")),
    }
    expected = {
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "quantization": _quantization(identity.quantization),
        "engine": identity.engine,
        "engine_version": identity.engine_version,
    }
    mismatches = [f"{key}: expected {expected[key]!r}, got {actual[key]!r}" for key in expected if actual[key] != expected[key]]
    if mismatches:
        raise EvidenceValidationError("Evidence identity mismatch: " + "; ".join(mismatches))
    if identity.profile.lower() != "balanced" or identity.execution_mode != "Native Memory":
        raise EvidenceValidationError("MLX profile evidence qualifies only BALANCED Native Memory.")

    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    datasets = sorted({str(row.get("dataset")) for row in rows})
    imported: list[dict[str, Any]] = []
    for dataset in [*datasets, "combined"]:
        cohort = rows if dataset == "combined" else [row for row in rows if row.get("dataset") == dataset]
        baseline_rows = [row for row in cohort if row.get("condition") == "E0_SELECTED"]
        pra_rows = [row for row in cohort if row.get("condition") == "E2_CONCAT_ALL"]
        baseline_keys = {(row.get("dataset"), row.get("seed"), row.get("example_id")) for row in baseline_rows}
        pra_keys = {(row.get("dataset"), row.get("seed"), row.get("example_id")) for row in pra_rows}
        if not baseline_rows or baseline_keys != pra_keys:
            raise EvidenceValidationError(f"E0/E2 paired cohort mismatch for {dataset}.")
        baseline = _summary(baseline_rows)
        pra = _summary(pra_rows)
        exact_pairs = sum(
            1 for row in pra_rows if float(row.get("sequence_agreement_vs_e0", 0.0)) == 1.0
        )
        imported.append(
            {
                "metric_class": "END_TASK",
                "model_id": identity.model_id,
                "model_revision": identity.model_revision,
                "quantization": identity.quantization,
                "engine": identity.engine,
                "engine_version": identity.engine_version,
                "profile": identity.profile,
                "execution_mode": identity.execution_mode,
                "dataset": dataset,
                "sample_count": len(baseline_rows),
                "seed_count": len({row.get("seed") for row in baseline_rows}),
                "baseline": baseline,
                "pra": pra,
                "deltas": {
                    "quality": pra["quality"] - baseline["quality"],
                    "visible_tokens_pct": _pct_delta(pra["visible_tokens"], baseline["visible_tokens"]),
                    "ttft_pct": _pct_delta(pra["ttft_ms"]["p50"], baseline["ttft_ms"]["p50"]),
                    "completion_latency_pct": _pct_delta(
                        pra["completion_latency_ms"]["mean"], baseline["completion_latency_ms"]["mean"]
                    ),
                },
                "semantic_equivalence": {
                    "exact_output_pairs": exact_pairs,
                    "paired_examples": len(pra_rows),
                },
                "evidence_tier": "ENGINE_QUALIFIED",
                "recommendation": "RECOMMENDED",
                "hardware": hardware or f"{runtime.get('hardware_model', 'Apple Silicon')}, 48 GB",
                "cohort": "selector-frozen natural QA",
                "date": evidence_date,
                "pra_commit": runtime.get("git_commit"),
                "artifact": artifact_reference or path.name,
                "artifact_sha256": file_sha256(path),
            }
        )
    return imported


def canonicalize_paired_transport_evidence(row: Mapping[str, Any]) -> CanonicalEvidenceRecord:
    """Map a legacy selector-frozen E0/E2 row without inventing a bundle run.

    The original engine path is No PRA and the native path is PRA without a
    learned adaptor.  Because those runs predate immutable bundle resolution,
    the bundle condition remains explicitly unmeasured.
    """

    baseline = row.get("baseline")
    pra = row.get("pra")
    if not isinstance(baseline, Mapping) or not isinstance(pra, Mapping):
        raise EvidenceValidationError("paired transport evidence requires baseline and pra mappings")

    def observations(source: Mapping[str, Any]) -> dict[str, MetricObservation]:
        ttft = source.get("ttft_ms", {})
        completion = source.get("completion_latency_ms", {})
        values = {
            "token_f1": source.get("quality"),
            "exact_match": source.get("exact_match"),
            "gold_answer_log_probability": source.get("gold_answer_logprob"),
            "visible_tokens": source.get("visible_tokens"),
            "selected_native_kv_tokens": source.get("selected_native_kv_tokens"),
            "active_detail_bytes": source.get("active_detail_bytes"),
            "retained_detail_bytes": source.get("retained_detail_bytes"),
            "ttft_p50_ms": ttft.get("p50") if isinstance(ttft, Mapping) else None,
            "ttft_p95_ms": ttft.get("p95") if isinstance(ttft, Mapping) else None,
            "ttft_p99_ms": ttft.get("p99") if isinstance(ttft, Mapping) else None,
            "completion_latency_mean_ms": completion.get("mean") if isinstance(completion, Mapping) else None,
            "peak_memory_bytes": source.get("peak_memory_bytes"),
        }
        return {
            name: MetricObservation.measured(float(value))
            for name, value in values.items()
            if value is not None
        }

    baseline_observations = observations(baseline)
    pra_observations = observations(pra)
    metric_names = tuple(dict.fromkeys((*baseline_observations, *pra_observations)))
    missing_bundle = {
        name: MetricObservation.missing(
            MeasurementState.NOT_MEASURED,
            "The paired engine run did not resolve and record the immutable Runtime Bundle.",
        )
        for name in metric_names
    }
    return CanonicalEvidenceRecord(
        key=EvidenceKey(
            task=str(row.get("dataset", "NOT_MEASURED")),
            hardware=str(row.get("hardware", "NOT_MEASURED")),
            engine=str(row.get("engine", "NOT_MEASURED")),
            engine_version=str(row.get("engine_version", "NOT_MEASURED")),
            model_id=str(row.get("model_id", "NOT_MEASURED")),
            model_revision=str(row.get("model_revision", "NOT_MEASURED")),
            mode=str(row.get("execution_mode", "native-memory")).lower().replace(" ", "-"),
            profile=str(row.get("profile", "balanced")).lower(),
        ),
        metric_definitions={name: STANDARD_METRICS[name] for name in metric_names},
        conditions={
            EvidenceCondition.NO_PRA: ConditionEvidence(metrics=baseline_observations),
            EvidenceCondition.PRA_NO_ADAPTOR: ConditionEvidence(metrics=pra_observations),
            EvidenceCondition.PRA_ADAPTOR_BUNDLE: ConditionEvidence(metrics=missing_bundle),
        },
        provenance=EvidenceProvenance(
            cohort=f"{row.get('cohort', 'selector-frozen paired transport')}; seed_count={row.get('seed_count', 'NOT_MEASURED')}",
            run_ids=(),
            commit=row.get("pra_commit"),
            date=str(row.get("date", "NOT_MEASURED")),
            artifacts=tuple(str(value) for value in (row.get("artifact"),) if value),
        ),
        evidence_tier=str(row.get("evidence_tier", "CONTROLLED")),
    )


def import_product_matrix_evidence(
    artifact: str | Path, identity: EvidenceIdentity
) -> list[dict[str, Any]]:
    """Import exact-identity rows from the Paper 4.5 product matrix.

    This path intentionally does not relax missing engine versions or transfer
    family-level rows. Callers may use the returned controlled measurements as
    supporting evidence; only paired rows should be promoted to headlines.
    """

    path = Path(artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    model_rows = [row for row in all_rows if row.get("model_id") == identity.model_id]
    if model_rows and not any(row.get("model_revision") == identity.model_revision for row in model_rows):
        raise EvidenceValidationError("Evidence identity mismatch: model_revision")
    matched = []
    for row in model_rows:
        if row.get("model_revision") != identity.model_revision:
            continue
        if _quantization(row.get("quantization")) != _quantization(identity.quantization):
            continue
        if str(row.get("engine")) != identity.engine:
            continue
        if str(row.get("engine_version") or "") != identity.engine_version:
            continue
        if str(row.get("profile", "")).lower() != identity.profile.lower():
            continue
        public_mode = {"E0": "Selected Context", "E2": "Native Memory", "E3": "Native Serving"}.get(
            str(row.get("integration_level")), str(row.get("integration_level"))
        )
        if public_mode != identity.execution_mode:
            continue
        matched.append(
            {
                "metric_class": "END_TASK" if any(row.get(key) is not None for key in ("f1", "em", "task_success")) else "SERVING_ECONOMICS",
                "model_id": identity.model_id,
                "model_revision": identity.model_revision,
                "quantization": identity.quantization,
                "engine": identity.engine,
                "engine_version": identity.engine_version,
                "profile": identity.profile,
                "execution_mode": identity.execution_mode,
                "dataset": row.get("dataset"),
                "sample_count": row.get("sample_count"),
                "metrics": {
                    key: row.get(key) for key in (
                        "f1", "em", "task_success", "quality_score", "quality_reference",
                        "visible_tokens", "active_kv_tokens", "active_kv_bytes", "ttft_ms",
                        "itl_ms", "completion_ms", "requests_per_second", "peak_memory_bytes",
                    ) if row.get(key) is not None
                },
                "evidence_tier": row.get("evidence_tier", "NOT_MEASURED"),
                "hardware": row.get("hardware"),
                "artifact": path.name,
                "artifact_sha256": file_sha256(path),
            }
        )
    return matched


def validate_selector_manifest(artifact: str | Path) -> dict[str, Any]:
    """Validate the shared selector-frozen qualification contract."""

    path = Path(artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    selections = payload.get("selections", [])
    if not selections or any(not row.get("digest") for row in selections):
        raise EvidenceValidationError("Qualification manifest requires frozen selection digests.")
    if "once" not in str(payload.get("selector_contract", "")).lower():
        raise EvidenceValidationError("Qualification manifest does not freeze selector output once.")
    return payload


def validate_bundle_evidence(bundle: Any) -> None:
    """Apply strict release gates to bundles using evidence contract v1."""

    qualification = bundle.qualification
    if not isinstance(qualification, Mapping) or qualification.get("contract_version") != 1:
        return
    errors: list[str] = []
    tier = str(qualification.get("status", "NOT_MEASURED"))
    if tier not in EVIDENCE_TIERS:
        errors.append(f"unknown evidence tier {tier!r}")
    recommended = []
    for name, raw in bundle.profiles.items():
        profile = raw if isinstance(raw, Mapping) else {}
        status = str(profile.get("status", ""))
        if status not in PROFILE_STATUSES:
            errors.append(f"profile {name!r} has unknown status {status!r}")
        if profile.get("recommended") is True:
            recommended.append(str(name))
            if status != "QUALIFIED":
                errors.append(f"recommended profile {name!r} is not QUALIFIED")
    if len(recommended) != 1:
        errors.append("exactly one recommended profile is required")

    headlines = qualification.get("headline", [])
    if not isinstance(headlines, Sequence):
        errors.append("headline evidence must be a sequence")
    for row in headlines if isinstance(headlines, Sequence) else []:
        if not isinstance(row, Mapping):
            errors.append("headline evidence rows must be mappings")
            continue
        if row.get("metric_class") == "ROUTING_DIAGNOSTIC":
            errors.append("routing diagnostics cannot be headline evidence")
        if not isinstance(row.get("baseline"), Mapping) or not isinstance(row.get("pra"), Mapping):
            errors.append("headline evidence requires paired baseline and PRA measurements")
        for key, expected in (
            ("model_id", bundle.base_model.get("id")),
            ("model_revision", bundle.base_model.get("revision")),
        ):
            if row.get(key) != expected:
                errors.append(f"headline {key} disagrees with bundle")
        if _quantization(row.get("quantization")) != _quantization(bundle.base_model.get("quantization")):
            errors.append("headline quantization disagrees with bundle")
        if row.get("profile") not in bundle.profiles:
            errors.append(f"headline profile {row.get('profile')!r} is absent from bundle")
        if row.get("evidence_tier") not in EVIDENCE_TIERS:
            errors.append("headline evidence tier is invalid")
    canonical = qualification.get("canonical_evidence", [])
    canonical_rows = (
        canonical
        if isinstance(canonical, Sequence) and not isinstance(canonical, (str, bytes, Mapping))
        else [canonical]
        if canonical
        else []
    )
    for raw in canonical_rows:
        if not isinstance(raw, Mapping):
            errors.append("canonical evidence rows must be mappings")
            continue
        try:
            fields = CanonicalEvidenceRecord.model_fields
            record = CanonicalEvidenceRecord.model_validate(
                {name: raw[name] for name in fields if name in raw}
            )
        except ValueError as error:
            errors.append(f"invalid canonical evidence: {error}")
            continue
        if record.key.model_id != bundle.base_model.get("id"):
            errors.append("canonical evidence model_id disagrees with bundle")
        if record.key.model_revision != bundle.base_model.get("revision"):
            errors.append("canonical evidence model_revision disagrees with bundle")
    if errors:
        raise EvidenceValidationError("; ".join(errors))
