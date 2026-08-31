"""Evidence gates and selector-frozen manifests for PRA engine qualification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .product_matrix import ProductMatrixRow


class IntegrationLevel(str, Enum):
    """Progressively deeper PRA integration with an inference engine."""

    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"


class Representation(str, Enum):
    """Execution conditions in the common qualification benchmark."""

    FULL = "FULL"
    E0_SELECTED = "E0_SELECTED"
    E2_HOT = "E2_HOT"
    E2_WARM = "E2_WARM"
    E3_PREFETCH = "E3_PREFETCH"
    E3_REMOTE_WARM = "E3_REMOTE_WARM"
    E3_COLD = "E3_COLD"
    SOURCE = "SOURCE"


@dataclass(frozen=True)
class FrozenSelection:
    """One selector result reused verbatim by E0, E2, and E3 execution."""

    example_id: str
    query_sha256: str
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    selected_intervals: tuple[tuple[str, int, int], ...]

    @classmethod
    def create(
        cls,
        *,
        example_id: str,
        query: str,
        candidate_ids: Sequence[str],
        selected_ids: Sequence[str],
        selected_intervals: Sequence[tuple[str, int, int]],
    ) -> "FrozenSelection":
        return cls(
            example_id=example_id,
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            candidate_ids=tuple(candidate_ids),
            selected_ids=tuple(selected_ids),
            selected_intervals=tuple(tuple(value) for value in selected_intervals),
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QualificationManifest:
    """Restartable cross-engine benchmark contract with frozen selections."""

    manifest_id: str
    context_sizes: tuple[str, ...] = ("small", "medium", "large")
    resource_reuse: tuple[str, ...] = ("shared", "independent")
    concurrency: tuple[int, ...] = (1, 4, 8, 16)
    representations: tuple[str, ...] = (
        Representation.FULL.value,
        Representation.E0_SELECTED.value,
        Representation.E2_HOT.value,
        Representation.E2_WARM.value,
    )
    selections: tuple[FrozenSelection, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.manifest_id:
            raise ValueError("Qualification manifest ID is required.")
        allowed = {item.value for item in Representation}
        unknown = sorted(set(self.representations) - allowed)
        if unknown:
            raise ValueError(f"Unknown qualification representations: {', '.join(unknown)}")
        example_ids = [selection.example_id for selection in self.selections]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Frozen selection example IDs must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "dimensions": {
                "context_sizes": list(self.context_sizes),
                "resource_reuse": list(self.resource_reuse),
                "concurrency": list(self.concurrency),
                "representations": list(self.representations),
            },
            "selector_contract": (
                "Compute candidate IDs and selected intervals once; reuse each "
                "selection digest across every representation."
            ),
            "selections": [
                {**asdict(selection), "digest": selection.digest}
                for selection in self.selections
            ],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


E0_REQUIRED_INVARIANTS = frozenset({"selector_frozen", "pressure_curve"})
E1_REQUIRED_INVARIANTS = frozenset(
    {"logical_resource_identity", "versioning_authorization", "fallback_semantics"}
)
E2_REQUIRED_INVARIANTS = frozenset(
    {
        "geometry_matched",
        "position_mask_topology",
        "single_normalization",
        "request_cleanup",
        "cross_request_isolation",
        "no_duplicate_materialization",
    }
)
E3_REQUIRED_INVARIANTS = frozenset(
    {"scheduler_owned_lifecycle", "promotion_eviction_reload", "concurrent_batching"}
)


def qualification_gaps(
    row: ProductMatrixRow,
    target: IntegrationLevel | str,
) -> tuple[str, ...]:
    """Return missing evidence without silently promoting a runtime."""

    level = IntegrationLevel(target)
    gaps: list[str] = []
    quality_available = row.task_success is not None or row.quality_score is not None
    if not quality_available:
        gaps.append("quality")
    for name in ("visible_tokens", "ttft_p50_ms", "requests_per_second"):
        if getattr(row, name) is None:
            gaps.append(name)
    required = set(E0_REQUIRED_INVARIANTS)
    if level in {IntegrationLevel.E1, IntegrationLevel.E2, IntegrationLevel.E3}:
        required.update(E1_REQUIRED_INVARIANTS)
    if level in {IntegrationLevel.E2, IntegrationLevel.E3}:
        required.update(E2_REQUIRED_INVARIANTS)
        for name in ("active_kv_tokens", "active_kv_bytes", "consumer_layers"):
            if not getattr(row, name):
                gaps.append(name)
        if row.exact_pair_parity is None and row.task_success is None:
            gaps.append("native_parity_or_task_success")
        if not row.selector_digest:
            gaps.append("selector_digest")
    if level is IntegrationLevel.E3:
        required.update(E3_REQUIRED_INVARIANTS)
        for name in (
            "peak_device_memory_bytes",
            "ttft_p95_ms",
            "completion_p95_ms",
            "batch_occupancy",
        ):
            if getattr(row, name) is None:
                gaps.append(name)
    gaps.extend(
        f"invariant:{name}"
        for name in sorted(required - set(row.verified_invariants))
    )
    return tuple(dict.fromkeys(gaps))


def claimed_level_is_supported(row: ProductMatrixRow) -> bool:
    """Return whether the row closes every gate for its claimed level."""

    return not qualification_gaps(row, row.integration_level)


def assert_selector_frozen(rows: Sequence[ProductMatrixRow]) -> None:
    """Reject matched E0/E2 rows that were produced by different selections."""

    digests = {row.selector_digest for row in rows}
    if None in digests or len(digests) != 1:
        raise ValueError("Matched representation rows require one non-null selector digest.")


def status_summary(row: ProductMatrixRow) -> Mapping[str, object]:
    """Compact evidence summary consumed by generated paper tables."""

    gaps = qualification_gaps(row, row.integration_level)
    return {
        "row_id": row.row_id,
        "engine": row.engine,
        "claimed_level": row.integration_level,
        "supported": not gaps,
        "gaps": list(gaps),
    }
