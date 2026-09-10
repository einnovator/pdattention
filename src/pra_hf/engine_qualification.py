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


class SelectedKVSource(str, Enum):
    """Where selected K/V came from before the current request consumed it."""

    TEXT_REMATERIALIZED = "text_rematerialized"
    DETACHED_RESOURCE_ENCODING = "detached_resource_encoding"
    LIVE_PREFIX_CAPTURE = "live_prefix_capture"


@dataclass(frozen=True)
class SelectedKVRange:
    """Stable logical record mapped to an interval in one live prefix."""

    record_id: str
    parent_record_id: str
    causal_group_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.record_id or not self.parent_record_id or not self.causal_group_id:
            raise ValueError("Selected K/V ranges require stable record and causal IDs.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Selected K/V ranges must be non-empty half-open intervals.")


@dataclass(frozen=True)
class LivePrefixKVObservation:
    """Machine-checkable provenance for one agent-history K/V attachment.

    Token counts alone cannot distinguish true live-cache reuse from selected
    text that an engine silently encoded again. A qualifying observation names
    the source prefix, lists selected record intervals, and reports re-encoding
    and physical-copy work separately.
    """

    source: SelectedKVSource | str
    source_prefix_id: str
    source_token_count: int
    ranges: tuple[SelectedKVRange, ...]
    selected_kv_tokens: int
    reused_kv_tokens: int
    selected_text_reencoded_tokens: int
    physical_kv_copy_tokens: int = 0
    source_positions_preserved: bool = False
    single_attention_normalization: bool = False
    exact_live_prefix_continuation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", SelectedKVSource(self.source))
        object.__setattr__(self, "ranges", tuple(self.ranges))
        for name in (
            "source_token_count",
            "selected_kv_tokens",
            "reused_kv_tokens",
            "selected_text_reencoded_tokens",
            "physical_kv_copy_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")

    @property
    def interval_tokens(self) -> int:
        return sum(span.end - span.start for span in self.ranges)

    @property
    def full_retention(self) -> bool:
        if self.source_token_count == 0:
            return False
        merged = sorted((span.start, span.end) for span in self.ranges)
        cursor = 0
        for start, end in merged:
            if start != cursor:
                return False
            cursor = end
        return cursor == self.source_token_count

    def qualification_gaps(
        self, *, require_full_retention: bool = False
    ) -> tuple[str, ...]:
        gaps: list[str] = []
        if self.source is not SelectedKVSource.LIVE_PREFIX_CAPTURE:
            gaps.append(f"source:{self.source.value}")
        if not self.source_prefix_id:
            gaps.append("source_prefix_id")
        if not self.ranges:
            gaps.append("selected_ranges")
        ordered = sorted(self.ranges, key=lambda span: (span.start, span.end))
        cursor = -1
        for span in ordered:
            if span.end > self.source_token_count:
                gaps.append(f"range_out_of_bounds:{span.record_id}")
            if span.start < cursor:
                gaps.append(f"overlapping_range:{span.record_id}")
            cursor = max(cursor, span.end)
        if self.selected_kv_tokens != self.interval_tokens:
            gaps.append("selected_token_count")
        if self.reused_kv_tokens != self.selected_kv_tokens:
            gaps.append("reused_token_count")
        if self.selected_text_reencoded_tokens:
            gaps.append(f"selected_text_reencoded:{self.selected_text_reencoded_tokens}")
        if not self.source_positions_preserved:
            gaps.append("source_positions_preserved")
        if not self.single_attention_normalization:
            gaps.append("single_attention_normalization")
        if require_full_retention:
            if not self.full_retention:
                gaps.append("full_retention_coverage")
            if not self.exact_live_prefix_continuation:
                gaps.append("full_retention_prefix_equivalence")
        return tuple(dict.fromkeys(gaps))


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
STATEFUL_AGENT_REQUIRED_INVARIANTS = frozenset(
    {
        "generated_history_replay_exact",
        "live_state_continuation_exact",
        "boundary_token_preserved",
        "multi_turn_cleanup",
        "live_prefix_kv_subset",
        "zero_selected_text_reencoding",
        "source_positions_preserved",
        "multiple_record_attach",
        "full_retention_prefix_equivalence",
    }
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


def stateful_agent_qualification_gaps(
    *,
    exact_pairs: int,
    total_pairs: int,
    verified_invariants: Sequence[str],
) -> tuple[str, ...]:
    """Gate stateful agent reuse beyond ordinary static E2 parity.

    Dense-vs-native parity on an immutable document does not establish that a
    generated multi-turn trajectory can be transferred safely.  Stateful
    agents additionally require exact live continuation, preservation of the
    sampled-but-not-yet-evaluated boundary token, and lifecycle cleanup.
    """

    gaps = [
        f"invariant:{name}"
        for name in sorted(
            STATEFUL_AGENT_REQUIRED_INVARIANTS - set(verified_invariants)
        )
    ]
    if total_pairs <= 0:
        gaps.append("sequential_state_pairs")
    elif exact_pairs != total_pairs:
        gaps.append(f"sequential_state_exact:{exact_pairs}/{total_pairs}")
    return tuple(gaps)


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
