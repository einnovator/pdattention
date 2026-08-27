"""Lazy model-side encoding and activation for typed capability records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from .context_records import ContextRecord, RecordType, RecordViewName, serialize_record


class CapabilityEncodingState(str, Enum):
    """Highest encoding/activation state reached by one backing record."""

    UNENCODED = "unencoded"
    SELECTION_ENCODED = "selection_encoded"
    FULL_ENCODED = "full_encoded"
    ACTIVE_SELECTION = "active_selection"
    ACTIVE_FULL = "active_full"


@dataclass(frozen=True)
class CapabilityEncodingPolicy:
    """Control when model-visible selection and full views are encoded."""

    lazy_selection: bool = True
    lazy_full: bool = True
    cache_encoded_views: bool = True
    initial_view: RecordViewName | str = RecordViewName.SELECTION
    selected_view: RecordViewName | str = RecordViewName.FULL

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_view", RecordViewName(self.initial_view))
        object.__setattr__(self, "selected_view", RecordViewName(self.selected_view))
        if self.selected_view != RecordViewName.FULL:
            raise ValueError("Selected capability records must activate their full view.")


@dataclass(frozen=True)
class ToolRecordPolicy(CapabilityEncodingPolicy):
    """Lazy-by-default policy for complete tool schemas."""


@dataclass(frozen=True)
class SkillRecordPolicy(CapabilityEncodingPolicy):
    """Lazy-by-default policy for complete skill instructions."""


@dataclass(frozen=True)
class EncodedCapabilityView:
    """Cached model encoding metadata for one immutable named record view."""

    record_id: str
    view: RecordViewName
    source_fingerprint: str
    token_count: int
    encoded_bytes: int
    encode_seconds: float


@dataclass(frozen=True)
class CapabilityActivation:
    """One exact-identity local activation with cold/warm cost accounting."""

    record_id: str
    record_type: RecordType
    view: RecordViewName
    state: CapabilityEncodingState
    cache_hit: bool
    token_count: int
    encoded_bytes: int
    active_bytes: int
    encode_seconds: float
    activation_seconds: float
    semantic_rediscovery_calls: int = 0


@dataclass(frozen=True)
class CapabilityPaletteActivation:
    """Bounded Phase-A palette admitted under count and token budgets."""

    requested_record_ids: tuple[str, ...]
    admitted_record_ids: tuple[str, ...]
    dropped_record_ids: tuple[str, ...]
    selection_tokens: int
    active_bytes: int
    cache_hits: int
    cold_encodes: int


@dataclass
class _RuntimeEntry:
    record: ContextRecord
    state: CapabilityEncodingState = CapabilityEncodingState.UNENCODED
    encoded: dict[RecordViewName, EncodedCapabilityView] = field(default_factory=dict)
    active_view: RecordViewName | None = None


def _default_token_counter(text: str) -> int:
    return len(text.split())


class LazyCapabilityRuntime:
    """Own backing records, cached encodings, and exact local view transitions.

    Discovery supplies record identities once. ``activate_selected`` never
    reruns lexical, embedding, union, or semantic ranking; it resolves the
    already-visible identity against this runtime's immutable backing registry.
    """

    def __init__(
        self,
        records: Sequence[ContextRecord] = (),
        *,
        policy: CapabilityEncodingPolicy | None = None,
        token_counter: Callable[[str], int] | None = None,
        encoder: Callable[[str], object] | None = None,
        native_kv_bytes_per_token: int = 0,
    ) -> None:
        self.policy = policy or CapabilityEncodingPolicy()
        self.token_counter = token_counter or _default_token_counter
        self.encoder = encoder
        self.native_kv_bytes_per_token = native_kv_bytes_per_token
        self._entries: dict[str, _RuntimeEntry] = {}
        self.register(records)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def register(self, records: Sequence[ContextRecord]) -> None:
        """Register backing records and eagerly encode only explicit overrides."""

        for record in records:
            if record.record_type not in {RecordType.TOOL_RECORD, RecordType.SKILL_RECORD}:
                raise ValueError("Lazy capability runtime accepts only tool and skill records.")
            previous = self._entries.get(record.record_id)
            if previous and previous.record.source_fingerprint != record.source_fingerprint:
                self._entries[record.record_id] = _RuntimeEntry(record)
            elif previous:
                continue
            else:
                self._entries[record.record_id] = _RuntimeEntry(record)
            if not self.policy.lazy_selection:
                self._ensure_encoded(record.record_id, RecordViewName.SELECTION)
            if not self.policy.lazy_full or self.policy.initial_view == RecordViewName.FULL:
                self._ensure_encoded(record.record_id, RecordViewName.FULL)

    def state(self, record_id: str) -> CapabilityEncodingState:
        return self._entry(record_id).state

    def cached_views(self, record_id: str) -> frozenset[RecordViewName]:
        return frozenset(self._entry(record_id).encoded)

    def active_view(self, record_id: str) -> RecordViewName | None:
        return self._entry(record_id).active_view

    def _entry(self, record_id: str) -> _RuntimeEntry:
        try:
            return self._entries[record_id]
        except KeyError as error:
            raise ValueError(f"Unknown capability record: {record_id}") from error

    def _encoded_size(self, value: object, tokens: int) -> int:
        if self.native_kv_bytes_per_token:
            return tokens * self.native_kv_bytes_per_token
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value)
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel() * value.element_size())
        return tokens * 4

    def _ensure_encoded(
        self, record_id: str, view: RecordViewName
    ) -> tuple[EncodedCapabilityView, bool]:
        entry = self._entry(record_id)
        cached = entry.encoded.get(view)
        if cached is not None:
            return cached, True
        text = serialize_record(entry.record, view=view)
        started = time.perf_counter()
        encoded_value = self.encoder(text) if self.encoder is not None else text.encode("utf-8")
        elapsed = time.perf_counter() - started
        tokens = self.token_counter(text)
        encoded = EncodedCapabilityView(
            record_id=record_id,
            view=view,
            source_fingerprint=entry.record.source_fingerprint,
            token_count=tokens,
            encoded_bytes=self._encoded_size(encoded_value, tokens),
            encode_seconds=elapsed,
        )
        if self.policy.cache_encoded_views:
            entry.encoded[view] = encoded
        if view == RecordViewName.FULL:
            entry.state = CapabilityEncodingState.FULL_ENCODED
        elif entry.state == CapabilityEncodingState.UNENCODED:
            entry.state = CapabilityEncodingState.SELECTION_ENCODED
        return encoded, False

    def activate_selection_palette(
        self,
        record_ids: Sequence[str],
        *,
        max_candidates: int | None = None,
        selection_view_token_budget: int | None = None,
    ) -> CapabilityPaletteActivation:
        """Encode and activate a compact palette without touching full views."""

        requested = tuple(dict.fromkeys(record_ids))
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        if selection_view_token_budget is not None and selection_view_token_budget <= 0:
            raise ValueError("selection_view_token_budget must be positive when provided.")
        admitted: list[str] = []
        total_tokens = cache_hits = cold_encodes = 0
        for record_id in requested:
            if max_candidates is not None and len(admitted) >= max_candidates:
                break
            entry = self._entry(record_id)
            text = serialize_record(entry.record, view=RecordViewName.SELECTION)
            tokens = self.token_counter(text)
            if selection_view_token_budget is not None and total_tokens + tokens > selection_view_token_budget:
                continue
            encoded, cache_hit = self._ensure_encoded(record_id, RecordViewName.SELECTION)
            entry.active_view = RecordViewName.SELECTION
            entry.state = CapabilityEncodingState.ACTIVE_SELECTION
            admitted.append(record_id)
            total_tokens += encoded.token_count
            cache_hits += int(cache_hit)
            cold_encodes += int(not cache_hit)
        admitted_set = set(admitted)
        return CapabilityPaletteActivation(
            requested_record_ids=requested,
            admitted_record_ids=tuple(admitted),
            dropped_record_ids=tuple(record_id for record_id in requested if record_id not in admitted_set),
            selection_tokens=total_tokens,
            active_bytes=sum(
                self._entries[record_id].encoded[RecordViewName.SELECTION].encoded_bytes
                for record_id in admitted
                if RecordViewName.SELECTION in self._entries[record_id].encoded
            ),
            cache_hits=cache_hits,
            cold_encodes=cold_encodes,
        )

    def activate_selected(self, record_id: str) -> CapabilityActivation:
        """Activate the exact full view selected from the current Phase-A palette."""

        entry = self._entry(record_id)
        if entry.active_view != RecordViewName.SELECTION:
            raise ValueError("Selected capability must be active in the Phase-A palette.")
        started = time.perf_counter()
        encoded, cache_hit = self._ensure_encoded(record_id, RecordViewName.FULL)
        activation_seconds = time.perf_counter() - started
        entry.active_view = RecordViewName.FULL
        entry.state = CapabilityEncodingState.ACTIVE_FULL
        return CapabilityActivation(
            record_id=record_id,
            record_type=entry.record.record_type,
            view=RecordViewName.FULL,
            state=entry.state,
            cache_hit=cache_hit,
            token_count=encoded.token_count,
            encoded_bytes=encoded.encoded_bytes,
            active_bytes=encoded.encoded_bytes,
            encode_seconds=0.0 if cache_hit else encoded.encode_seconds,
            activation_seconds=activation_seconds,
        )

    def deactivate(self) -> None:
        """Remove active views while retaining policy-allowed encoded caches."""

        for entry in self._entries.values():
            entry.active_view = None
            if RecordViewName.FULL in entry.encoded:
                entry.state = CapabilityEncodingState.FULL_ENCODED
            elif RecordViewName.SELECTION in entry.encoded:
                entry.state = CapabilityEncodingState.SELECTION_ENCODED
            else:
                entry.state = CapabilityEncodingState.UNENCODED

    def accounting(self) -> Mapping[str, int]:
        """Return distinct resident-cache and currently active byte totals."""

        resident = sum(
            view.encoded_bytes for entry in self._entries.values() for view in entry.encoded.values()
        )
        active = sum(
            entry.encoded[entry.active_view].encoded_bytes
            for entry in self._entries.values()
            if entry.active_view is not None and entry.active_view in entry.encoded
        )
        return {
            "records": len(self._entries),
            "encoded_views": sum(len(entry.encoded) for entry in self._entries.values()),
            "resident_encoded_bytes": resident,
            "active_encoded_bytes": active,
        }
