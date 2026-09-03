"""Versioned product evidence shared by PRA runtimes and engine papers.

Numeric values are intentionally separate from measurement status and
provenance. A missing metric is represented by ``None`` plus an explicit
``NOT_MEASURED`` (or ``NOT_APPLICABLE``) status, never by a synthetic zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence, get_type_hints

from .precision import infer_precision


PRODUCT_MATRIX_SCHEMA_VERSION = "2.1"
PRODUCT_MATRIX_STATUSES = {
    "MEASURED",
    "CONTROLLED",
    "MODEL_BACKED",
    "NATURAL_WORKLOAD",
    "CANDIDATE",
    "CALIBRATION_PENDING",
    "RESEARCH_ONLY",
    "BLOCKED",
    "NOT_MEASURED",
    "NOT_APPLICABLE",
}
INTEGRATION_LEVELS = {"E0", "E1", "E2", "E3"}


def optional_number(value: object) -> float | None:
    """Normalize a metric to a finite float or explicit unknown ``None``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ProductMatrixRow:
    """One model, engine, profile, hardware, workload, and representation row."""

    row_id: str
    model_family: str
    model_id: str
    model_revision: str | None
    model_size: int | None
    model_variant: str | None
    engine: str
    engine_version: str | None
    hardware: str
    profile: str
    profile_status: str
    workload: str
    dataset: str
    quality_metric: str

    integration_level: str = "E0"
    representation: str = "E0_SELECTED"
    selector_digest: str | None = None
    quantization: str | None = None
    precision_family: str = "UNSPECIFIED"
    precision_encoding: str = "UNSPECIFIED"
    serving_precision: str | None = None
    feature_extraction_precision: str | None = None
    adaptor_parameter_precision: str | None = None
    cpu: str | None = None
    accelerator: str | None = None
    vram_bytes: float | None = None
    ram_bytes: float | None = None
    storage: str | None = None
    os: str | None = None
    driver_version: str | None = None
    runtime_version: str | None = None

    quality_score: float | None = None
    quality_reference: float | None = None
    quality_delta: float | None = None
    task_success: float | None = None
    em: float | None = None
    f1: float | None = None
    exact_pair_parity: float | None = None
    gold_answer_log_probability: float | None = None
    evidence_recall: float | None = None

    source_tokens: float | None = None
    full_visible_tokens: float | None = None
    visible_tokens: float | None = None
    visible_token_reduction: float | None = None
    context_amplification: float | None = None
    active_kv_tokens: float | None = None
    active_kv_bytes: float | None = None
    reference_kv_tokens: float | None = None
    active_kv_reduction: float | None = None
    consumer_layers: tuple[int, ...] = ()

    model_memory_bytes: float | None = None
    local_kv_bytes: float | None = None
    pra_hot_kv_bytes: float | None = None
    temporary_attention_bytes: float | None = None
    engine_workspace_bytes: float | None = None
    hot_bytes: float | None = None
    warm_bytes: float | None = None
    cold_bytes: float | None = None
    persistence_mode: str | None = None
    peak_memory_bytes: float | None = None
    peak_device_memory_bytes: float | None = None
    peak_host_memory_bytes: float | None = None
    model_artifact_bytes: float | None = None
    load_time_ms: float | None = None
    max_successful_context_tokens: float | None = None

    ttft_ms: float | None = None
    itl_ms: float | None = None
    completion_ms: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    completion_p50_ms: float | None = None
    completion_p95_ms: float | None = None
    completion_p99_ms: float | None = None
    queue_delay_ms: float | None = None
    requests_per_second: float | None = None
    output_tokens_per_second: float | None = None
    successful_requests_per_second: float | None = None
    successful_tasks_per_accelerator_hour: float | None = None

    transfer_bytes: float | None = None
    transfer_h2d_bytes: float | None = None
    transfer_h2d_ms: float | None = None
    transfer_d2h_bytes: float | None = None
    transfer_d2h_ms: float | None = None
    network_bytes: float | None = None
    network_ms: float | None = None
    disk_bytes: float | None = None
    disk_ms: float | None = None

    prefix_cache_hit_rate: float | None = None
    pra_cache_hit_rate: float | None = None
    warm_hit_rate: float | None = None
    evictions: float | None = None
    reloads: float | None = None
    reload_amplification: float | None = None
    preemptions: float | None = None
    block_occupancy: float | None = None
    batch_occupancy: float | None = None

    queries_per_resource: float | None = None
    shared_resource_count: float | None = None
    physical_copies: float | None = None
    logical_references: float | None = None
    shared_bytes_saved: float | None = None
    duplicate_kv_bytes_avoided: float | None = None
    re_prefill_avoided: float | None = None
    promotion_avoided: float | None = None

    routing_method: str | None = None
    routing_recall: float | None = None
    accelerator_cost_per_hour: float | None = None
    hourly_cost_source: str | None = None
    cost_per_successful_task: float | None = None
    sample_count: int = 0
    seed_count: int = 0
    evidence_tier: str = "UNKNOWN"
    evidence_provenance: str = ""
    experiment_status: str = "NOT_MEASURED"
    verified_invariants: tuple[str, ...] = ()
    metric_statuses: Mapping[str, str] | None = None
    metric_provenance: Mapping[str, str] | None = None
    notes: str = ""

    @classmethod
    def metric_fields(cls) -> tuple[str, ...]:
        """Return fields that require value/status/provenance separation."""

        hints = get_type_hints(cls)
        return tuple(name for name, hint in hints.items() if hint == float | None)

    def __post_init__(self) -> None:
        required = {
            "row_id": self.row_id,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "engine": self.engine,
            "hardware": self.hardware,
            "profile": self.profile,
            "workload": self.workload,
            "dataset": self.dataset,
            "quality_metric": self.quality_metric,
            "evidence_tier": self.evidence_tier,
            "evidence_provenance": self.evidence_provenance,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Product-matrix row is missing: {', '.join(missing)}")
        if self.profile_status not in PRODUCT_MATRIX_STATUSES:
            raise ValueError(f"Unknown profile status: {self.profile_status}")
        if self.experiment_status not in PRODUCT_MATRIX_STATUSES:
            raise ValueError(f"Unknown experiment status: {self.experiment_status}")
        if self.integration_level not in INTEGRATION_LEVELS:
            raise ValueError(f"Unknown integration level: {self.integration_level}")
        if self.sample_count < 0 or self.seed_count < 0:
            raise ValueError("Product-matrix sample and seed counts cannot be negative.")

        object.__setattr__(self, "consumer_layers", tuple(self.consumer_layers))
        object.__setattr__(self, "verified_invariants", tuple(self.verified_invariants))
        precision = infer_precision(
            self.quantization,
            engine=self.engine,
            precision_family=(
                None if self.precision_family == "UNSPECIFIED" else self.precision_family
            ),
            precision_encoding=(
                None if self.precision_encoding == "UNSPECIFIED" else self.precision_encoding
            ),
            feature_extraction_precision=self.feature_extraction_precision,
            adaptor_parameter_precision=self.adaptor_parameter_precision,
        )
        object.__setattr__(self, "precision_family", precision.precision_family)
        object.__setattr__(self, "precision_encoding", precision.precision_encoding)
        object.__setattr__(self, "serving_precision", self.serving_precision or precision.serving_precision)
        self._derive_metrics()

        metric_names = self.metric_fields()
        for name in metric_names:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Product-matrix metric {name} must be finite or null.")
        if self.requests_per_second is not None and (
            self.task_success is None and self.quality_score is None
        ):
            raise ValueError("Throughput rows must include task_success or quality_score.")

        supplied_statuses = dict(self.metric_statuses or {})
        supplied_provenance = dict(self.metric_provenance or {})
        unknown_status = sorted(set(supplied_statuses) - set(metric_names))
        unknown_provenance = sorted(set(supplied_provenance) - set(metric_names))
        if unknown_status or unknown_provenance:
            names = ", ".join([*unknown_status, *unknown_provenance])
            raise ValueError(f"Unknown product-matrix metric annotations: {names}")
        statuses: dict[str, str] = {}
        provenance: dict[str, str] = {}
        for name in metric_names:
            value = getattr(self, name)
            status = supplied_statuses.get(
                name,
                self.experiment_status if value is not None else "NOT_MEASURED",
            )
            if status not in PRODUCT_MATRIX_STATUSES:
                raise ValueError(f"Unknown metric status for {name}: {status}")
            if value is None and status in {
                "MEASURED", "CONTROLLED", "MODEL_BACKED", "NATURAL_WORKLOAD"
            }:
                raise ValueError(f"Metric {name} is {status} but has no value.")
            statuses[name] = status
            provenance[name] = supplied_provenance.get(
                name, self.evidence_provenance if value is not None else ""
            )
        object.__setattr__(self, "metric_statuses", statuses)
        object.__setattr__(self, "metric_provenance", provenance)

    def _derive_metrics(self) -> None:
        """Populate algebraic metrics only when every input is available."""

        if (
            self.visible_token_reduction is None
            and self.full_visible_tokens is not None
            and self.visible_tokens is not None
            and self.full_visible_tokens > 0
        ):
            object.__setattr__(
                self,
                "visible_token_reduction",
                1.0 - self.visible_tokens / self.full_visible_tokens,
            )
        if (
            self.context_amplification is None
            and self.full_visible_tokens is not None
            and self.visible_tokens is not None
            and self.visible_tokens > 0
        ):
            object.__setattr__(
                self,
                "context_amplification",
                self.full_visible_tokens / self.visible_tokens,
            )
        if (
            self.active_kv_reduction is None
            and self.reference_kv_tokens is not None
            and self.active_kv_tokens is not None
            and self.reference_kv_tokens > 0
        ):
            object.__setattr__(
                self,
                "active_kv_reduction",
                1.0 - self.active_kv_tokens / self.reference_kv_tokens,
            )
        if (
            self.successful_requests_per_second is None
            and self.requests_per_second is not None
            and self.task_success is not None
        ):
            object.__setattr__(
                self,
                "successful_requests_per_second",
                self.requests_per_second * self.task_success,
            )
        if self.successful_requests_per_second is not None:
            object.__setattr__(
                self,
                "successful_tasks_per_accelerator_hour",
                3600.0 * self.successful_requests_per_second,
            )
        if (
            self.cost_per_successful_task is None
            and self.accelerator_cost_per_hour is not None
            and self.successful_requests_per_second is not None
            and self.successful_requests_per_second > 0
        ):
            object.__setattr__(
                self,
                "cost_per_successful_task",
                self.accelerator_cost_per_hour
                / (3600.0 * self.successful_requests_per_second),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductMatrixRow":
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise ValueError(f"Unknown product-matrix fields: {', '.join(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductMatrix:
    """Versioned collection consumed by papers, CLI inspection, and releases."""

    registry_version: str
    rows: tuple[ProductMatrixRow, ...]
    schema_version: str = PRODUCT_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCT_MATRIX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported product-matrix schema: {self.schema_version}")
        if not self.registry_version:
            raise ValueError("Product-matrix registry_version is required.")
        object.__setattr__(self, "rows", tuple(self.rows))
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("Product-matrix row IDs must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "rows": [row.to_dict() for row in self.rows],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductMatrix":
        unknown = sorted(set(value) - {"schema_version", "registry_version", "rows"})
        if unknown:
            raise ValueError(f"Unknown product-matrix document fields: {', '.join(unknown)}")
        rows: Sequence[Mapping[str, Any]] = value.get("rows", ())
        schema_version = str(value.get("schema_version", ""))
        if schema_version in {"1.0", "2.0"}:
            schema_version = PRODUCT_MATRIX_SCHEMA_VERSION
        return cls(
            schema_version=schema_version,
            registry_version=str(value.get("registry_version", "")),
            rows=tuple(ProductMatrixRow.from_dict(row) for row in rows),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ProductMatrix":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
