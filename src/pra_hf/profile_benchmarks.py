"""Versioned product-profile benchmark registry and inspection helpers.

Semantic profile evidence is keyed by model revision and workload. Physical
runtime measurements add engine, hardware, and precision without redefining
the semantic profile. Missing physical metrics remain ``None`` in JSON and are
rendered as ``NOT_MEASURED`` at presentation boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class ProductProfile(str, Enum):
    """Stable user-facing semantic objectives."""

    REFERENCE_CORRECTNESS = "REFERENCE_CORRECTNESS"
    QUALITY_MAX_CANDIDATE = "QUALITY_MAX_CANDIDATE"
    QUALITY_MAX = "QUALITY_MAX"
    BALANCED = "BALANCED"
    ECONOMY = "ECONOMY"


class EvidenceTier(str, Enum):
    """Strength of evidence supporting one registry row."""

    SMOKE = "SMOKE"
    CONTROLLED = "CONTROLLED"
    BENCHMARK = "BENCHMARK"
    SERVING = "SERVING"


class MeasurementStatus(str, Enum):
    """Explicit completeness or capability status for one measurement."""

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    NOT_MEASURED = "NOT_MEASURED"
    UNSUPPORTED = "UNSUPPORTED"
    PARTIAL_TOPOLOGY = "PARTIAL_TOPOLOGY"
    HARDWARE_GATED = "HARDWARE_GATED"
    CALIBRATION_PENDING = "CALIBRATION_PENDING"


RUNTIME_METRIC_FIELDS = (
    "ttft_ms",
    "inter_token_ms",
    "tokens_per_second",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput",
    "peak_hbm_bytes",
    "peak_ram_bytes",
    "h2d_bytes",
    "h2d_ms",
    "cache_hit_rate",
    "index_build_time_ms",
    "prefetch_overlap",
    "concurrent_sessions",
    "prefix_tokens_reusable",
    "prefix_reuse_fraction",
    "prefix_invalidations",
    "message_bytes_sent",
    "resource_bytes_sent",
    "session_delta_bytes",
    "engine_prefix_cache_hit",
    "engine_session_reuse",
)

REQUIRED_BENCHMARK_FIELDS = (
    "model_family",
    "model_id",
    "model_revision",
    "parameter_count",
    "num_layers",
    "workload",
    "dataset",
    "split",
    "profile",
    "profile_registry_version",
    "quality_metric",
    "quality_absolute",
    "quality_reference",
    "quality_retention",
    "quality_delta",
    "visible_initial_tokens",
    "visible_recovered_tokens",
    "materialized_tokens",
    "active_kv_tokens",
    "active_kv_bytes",
    "active_kv_saving",
    "detail_kv_bytes",
    "detail_kv_saving",
    "address_index_bytes",
    "backing_bytes",
    "compression_policy",
    "search_policy",
    "materialization_profile",
    "address_layers",
    "detail_kv_layers",
    "routing_layers",
    "consumer_layers",
    "eligible_consumer_layers",
    "consumer_layer_fraction",
    "engine",
    "engine_version",
    "hardware",
    "dtype",
    *RUNTIME_METRIC_FIELDS,
    "sample_count",
    "seed_count",
    "evidence_tier",
    "profile_status",
    "measurement_status",
    "runtime_measurement_status",
    "recommended_use",
    "artifact_path",
    "commit",
    "timestamp",
    "notes",
)


def normalized_quality(value: float, reference: float) -> tuple[float, float]:
    """Return retention and absolute delta for a positive quality metric."""

    if reference <= 0:
        raise ValueError("quality_reference must be positive for normalization.")
    return value / reference, value - reference


def normalized_saving(value: float, reference: float) -> float:
    """Return fractional savings relative to a positive reference cost."""

    if reference <= 0:
        raise ValueError("reference cost must be positive for normalization.")
    return 1.0 - value / reference


@dataclass(frozen=True)
class ProfileResolution:
    """A resolved profile row and the provenance used by the runtime."""

    row: Mapping[str, Any]
    profile_requested: str
    profile_resolved: str
    profile_source: str
    registry_version: str

    def trace(self) -> dict[str, str]:
        return {
            "profile_requested": self.profile_requested,
            "profile_resolved": self.profile_resolved,
            "profile_source": self.profile_source,
            "registry_version": self.registry_version,
        }


class ProfileBenchmarkRegistry:
    """Validated append-only model/profile/engine benchmark registry."""

    def __init__(self, payload: Mapping[str, Any], *, source: str = "memory") -> None:
        self.payload = dict(payload)
        self.source = source
        self.schema_version = str(self.payload.get("schema_version", ""))
        self.registry_version = str(self.payload.get("registry_version", ""))
        self.rows = tuple(dict(row) for row in self.payload.get("benchmarks", ()))
        self._validate()

    @classmethod
    def from_path(cls, path: str | Path) -> "ProfileBenchmarkRegistry":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), source=str(path))

    @classmethod
    def default(cls) -> "ProfileBenchmarkRegistry":
        path = Path(__file__).with_name("model_profiles") / "pra_profile_benchmarks.json"
        return cls.from_path(path)

    def _validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported benchmark schema: {self.schema_version!r}.")
        if not self.registry_version:
            raise ValueError("registry_version is required.")
        identities: set[tuple[Any, ...]] = set()
        for index, row in enumerate(self.rows):
            missing = [field for field in REQUIRED_BENCHMARK_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Benchmark row {index} is missing fields: {missing}")
            ProductProfile(str(row["profile"]))
            EvidenceTier(str(row["evidence_tier"]))
            MeasurementStatus(str(row["profile_status"]))
            MeasurementStatus(str(row["measurement_status"]))
            MeasurementStatus(str(row["runtime_measurement_status"]))
            identity = (
                row["model_id"], row["model_revision"], row["workload"],
                row["profile"], row["engine"], row["engine_version"],
                row["hardware"], row["dtype"],
            )
            if identity in identities:
                raise ValueError(f"Duplicate benchmark realization: {identity}")
            identities.add(identity)
            self._validate_derived(row, index)

    @staticmethod
    def _validate_derived(row: Mapping[str, Any], index: int) -> None:
        value = row["quality_absolute"]
        reference = row["quality_reference"]
        if value is not None and reference is not None:
            retention, delta = normalized_quality(float(value), float(reference))
            if abs(retention - float(row["quality_retention"])) > 1e-9:
                raise ValueError(f"Benchmark row {index} has stale quality_retention.")
            if abs(delta - float(row["quality_delta"])) > 1e-9:
                raise ValueError(f"Benchmark row {index} has stale quality_delta.")
        for value_field, reference_field, saving_field in (
            ("active_kv_tokens", "active_kv_reference_tokens", "active_kv_saving"),
            ("detail_kv_bytes", "detail_kv_reference_bytes", "detail_kv_saving"),
        ):
            value = row.get(value_field)
            reference = row.get(reference_field)
            saving = row.get(saving_field)
            if value is not None and reference is not None:
                expected = normalized_saving(float(value), float(reference))
                if saving is None or abs(expected - float(saving)) > 1e-6:
                    raise ValueError(f"Benchmark row {index} has stale {saving_field}.")

    def find(
        self,
        model_id: str,
        *,
        workload: str | None = None,
        profile: ProductProfile | str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return all matching physical realizations without collapsing engines."""

        wanted_profile = None if profile is None else _product_profile(profile).value
        return tuple(
            row for row in self.rows
            if (row["model_id"] == model_id or row["model_family"] == model_id)
            and (workload is None or row["workload"] == workload)
            and (wanted_profile is None or row["profile"] == wanted_profile)
        )

    def resolve(
        self,
        model_id: str,
        *,
        workload: str | None,
        profile: ProductProfile | str,
    ) -> ProfileResolution:
        """Resolve one semantic profile, preferring an exact model ID."""

        requested = _product_profile(profile).value
        rows = self.find(model_id, workload=workload, profile=requested)
        if not rows:
            raise KeyError(
                f"No profile evidence for model={model_id!r}, workload={workload!r}, "
                f"profile={requested!r}."
            )
        exact = [row for row in rows if row["model_id"] == model_id]
        row = (exact or list(rows))[0]
        return ProfileResolution(
            row=row,
            profile_requested=requested,
            profile_resolved=str(row["profile"]),
            profile_source="model_workload_registry",
            registry_version=self.registry_version,
        )

    def inspect(
        self,
        model_id: str,
        *,
        workload: str | None = None,
    ) -> dict[str, Any]:
        """Return user-facing quality, economy, policy, and provenance rows."""

        rows = self.find(model_id, workload=workload)
        statuses = {str(row["measurement_status"]) for row in rows}
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "source": self.source,
            "model": model_id,
            "workload": workload,
            "measurement_status": (
                next(iter(statuses))
                if len(statuses) == 1
                else (
                    None if rows else MeasurementStatus.CALIBRATION_PENDING.value
                )
            ),
            "profiles": [self._inspection_row(row) for row in rows],
        }

    @staticmethod
    def _inspection_row(row: Mapping[str, Any]) -> dict[str, Any]:
        runtime = {
            field: row[field] if row[field] is not None else MeasurementStatus.NOT_MEASURED.value
            for field in RUNTIME_METRIC_FIELDS
        }
        return {
            "profile": row["profile"],
            "quality": {
                "metric": row["quality_metric"],
                "absolute": row["quality_absolute"],
                "retention": row["quality_retention"],
                "delta": row["quality_delta"],
            },
            "economy": {
                "visible_initial_tokens": _display(row["visible_initial_tokens"]),
                "visible_recovered_tokens": _display(row["visible_recovered_tokens"]),
                "materialized_tokens": row["materialized_tokens"],
                "active_kv_tokens": row["active_kv_tokens"],
                "active_kv_saving": row["active_kv_saving"],
                "detail_kv_bytes": row["detail_kv_bytes"],
                "detail_kv_saving": row["detail_kv_saving"],
            },
            "policies": {
                "compression": row["compression_policy"],
                "search": row["search_policy"],
                "materialization": row["materialization_profile"],
                "address_layers": row["address_layers"],
                "detail_kv_layers": row["detail_kv_layers"],
                "routing_layers": row["routing_layers"],
                "consumer_layers": row["consumer_layers"],
            },
            "evidence_tier": row["evidence_tier"],
            "profile_status": row["profile_status"],
            "measurement_status": row["measurement_status"],
            "runtime_measurement_status": row["runtime_measurement_status"],
            "recommended_use": row["recommended_use"],
            "artifact_path": row["artifact_path"],
            "runtime": runtime,
        }


def profile_objective(profile: ProductProfile | str) -> str:
    """Translate a product profile name to the layer-registry objective."""

    resolved = _product_profile(profile)
    if resolved == ProductProfile.QUALITY_MAX_CANDIDATE:
        return "quality_max"
    return resolved.value.casefold()


def explicit_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply explicit request values after modular profile defaults."""

    result = dict(base)
    result.update({key: value for key, value in overrides.items() if value is not None})
    return result


def _product_profile(profile: ProductProfile | str) -> ProductProfile:
    return profile if isinstance(profile, ProductProfile) else ProductProfile(str(profile).upper())


def _display(value: Any) -> Any:
    return MeasurementStatus.NOT_MEASURED.value if value is None else value
