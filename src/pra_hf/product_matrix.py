"""Stable product-matrix schema shared by PRA runtimes and engine papers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


PRODUCT_MATRIX_SCHEMA_VERSION = "1.0"
PRODUCT_MATRIX_STATUSES = {
    "MEASURED",
    "CALIBRATION_PENDING",
    "RESEARCH_ONLY",
    "BLOCKED",
    "NOT_MEASURED",
}


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
    """One measured model, engine, profile, hardware, and workload condition."""

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
    quality_score: float | None = None
    quality_reference: float | None = None
    quality_delta: float | None = None
    source_tokens: float | None = None
    visible_tokens: float | None = None
    visible_token_reduction: float | None = None
    active_kv_tokens: float | None = None
    reference_kv_tokens: float | None = None
    active_kv_reduction: float | None = None
    hot_bytes: float | None = None
    warm_bytes: float | None = None
    cold_bytes: float | None = None
    persistence_mode: str | None = None
    ttft_ms: float | None = None
    itl_ms: float | None = None
    completion_ms: float | None = None
    requests_per_second: float | None = None
    peak_memory_bytes: float | None = None
    transfer_bytes: float | None = None
    routing_method: str | None = None
    routing_recall: float | None = None
    sample_count: int = 0
    seed_count: int = 0
    evidence_tier: str = "UNKNOWN"
    evidence_provenance: str = ""
    experiment_status: str = "NOT_MEASURED"
    notes: str = ""

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
        if self.sample_count < 0 or self.seed_count < 0:
            raise ValueError("Product-matrix sample and seed counts cannot be negative.")
        numeric_fields = (
            "quality_score", "quality_reference", "quality_delta", "source_tokens",
            "visible_tokens", "visible_token_reduction", "active_kv_tokens",
            "reference_kv_tokens", "active_kv_reduction", "hot_bytes", "warm_bytes",
            "cold_bytes", "ttft_ms", "itl_ms", "completion_ms",
            "requests_per_second", "peak_memory_bytes", "transfer_bytes", "routing_recall",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Product-matrix metric {name} must be finite or null.")

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
        return cls(
            schema_version=str(value.get("schema_version", "")),
            registry_version=str(value.get("registry_version", "")),
            rows=tuple(ProductMatrixRow.from_dict(row) for row in rows),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ProductMatrix":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
