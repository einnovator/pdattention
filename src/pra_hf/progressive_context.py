"""Progressive model-controlled disclosure over typed PRA records.

The backing runtime owns exact bytes, authorization, cursors, and transport.
This module gives every visible record view a stable PRA document identity and
executes a bounded context-control decision without asking the model to invent
payloads or rediscover a record whose identity is already known.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from .adaptive_context_runtime import (
    AdaptiveContextRuntime,
    CursorAction,
    CursorOperation,
    MaterializationEvent,
)
from .context_records import RecordType, RecordViewName
from .context_store import RecordAccessDenied
from .large_record_index import (
    IndexComponentState,
    IndexLifecycleState,
    LargeRecordChannel,
    LargeRecordIndex,
    LargeRecordSearchPolicy,
    LargeRecordSearchResult,
    RecordIndexLifecycle,
)

if TYPE_CHECKING:
    from .model import PRAForCausalLM, ReferenceHandle, RoutingResult
    from .typed_context import AdaptiveContextRecord


_TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]+")


class ContextAction(str, Enum):
    """Bounded decisions available to a model controlling visible context."""

    CONTINUE = "CONTINUE"
    MATERIALIZE_FULL = "MATERIALIZE_FULL"
    MATERIALIZE_MORE = "MATERIALIZE_MORE"
    SEARCH_RECORD = "SEARCH_RECORD"
    CURSOR_NEXT = "CURSOR_NEXT"
    CURSOR_QUERY = "CURSOR_QUERY"
    CALL_TOOL = "CALL_TOOL"


class PRAViewKind(str, Enum):
    """Views of one backing identity that can re-enter PRA independently."""

    SUMMARY = "summary"
    FULL = "full"
    DETAIL = "detail"
    SEARCH_RESULT = "search"
    CURSOR_PAGE = "cursor"


class NativeIndexState(str, Enum):
    """Auditable lifecycle state of a full-backing native PRA index."""

    NOT_REQUESTED = "NOT_REQUESTED"
    BUILT = "BUILT"
    SKIPPED_SIZE_LIMIT = "SKIPPED_SIZE_LIMIT"
    DEFERRED = "DEFERRED"


class ContextTransport(str, Enum):
    """Interface used to execute one logical context-control decision."""

    NATIVE = "native"
    TOOL = "tool"
    MIXED = "mixed"


class ControllerDescriptionLevel(str, Enum):
    """Validation-controlled verbosity of the fixed PRA operation block."""

    D0_MINIMAL = "D0"
    D1_CONCISE = "D1"
    D2_RICH = "D2"


class ControllerProtocol(str, Enum):
    """Flat or factorized insufficiency/operation decision protocol."""

    FLAT = "flat"
    HIERARCHICAL = "hierarchical"


_PRA_OPERATION_DESCRIPTIONS = {
    ContextAction.CONTINUE: (
        "Current visible context is sufficient.",
        "Continue only when the requested evidence is already visible; do not use it merely because a record summary exists.",
        "Example: the compact row already contains the requested answer code.",
    ),
    ContextAction.MATERIALIZE_FULL: (
        "Load the complete known backing record.",
        "Use when the record is bounded and the whole result is needed or no narrower interface suffices; avoid it for identifiable subsets.",
        "Example: load a small stored API response whose omitted fields are all relevant.",
    ),
    ContextAction.MATERIALIZE_MORE: (
        "Expose a selected field, row, line, chunk, or neighborhood.",
        "Prefer this over full materialization when a declared typed selector identifies the needed subset.",
        "Example: expose rows 40 through 48 from a known database result.",
    ),
    ContextAction.SEARCH_RECORD: (
        "Search inside one existing large backing record.",
        "Use when evidence may occur somewhere in a known searchable result; this does not discover a different record or call a new tool.",
        "Example: search one retained log for the affected account identifier.",
    ),
    ContextAction.CURSOR_NEXT: (
        "Advance an authorized cursor to its next bounded page.",
        "Use for sequential continuation when the current page is insufficient; avoid it when a targeted cursor query is available.",
        "Example: inspect the next page after no matching row appears in the current page.",
    ),
    ContextAction.CURSOR_QUERY: (
        "Search, filter, aggregate, or otherwise refine an authorized cursor.",
        "Use for a targeted operation over a cursor-backed result rather than loading the complete collection.",
        "Example: search the cursor for rows matching a known lookup key.",
    ),
    ContextAction.CALL_TOOL: (
        "Call an ordinary application tool for information absent from retained backing state.",
        "Use only when full, partial, search, and cursor operations cannot recover the required information from the current record.",
        "Example: query an external service for a status not present in the stored response.",
    ),
}


def controller_description_block(
    level: ControllerDescriptionLevel | str,
) -> str:
    """Render the fixed, always-visible PRA context-operation descriptions."""

    level = ControllerDescriptionLevel(level)
    line_count = {
        ControllerDescriptionLevel.D0_MINIMAL: 1,
        ControllerDescriptionLevel.D1_CONCISE: 2,
        ControllerDescriptionLevel.D2_RICH: 3,
    }[level]
    return "\n".join(
        f"{action.value}: {' '.join(parts[:line_count])}"
        for action, parts in _PRA_OPERATION_DESCRIPTIONS.items()
    )


@dataclass(frozen=True)
class ControllerConfig:
    """Frozen controller choice shared by MODEL_ONLY and PRA_ADAPTIVE."""

    model: str
    description_level: ControllerDescriptionLevel | str
    protocol: ControllerProtocol | str = ControllerProtocol.FLAT
    thinking: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "description_level", ControllerDescriptionLevel(self.description_level)
        )
        object.__setattr__(self, "protocol", ControllerProtocol(self.protocol))

    @property
    def description_block(self) -> str:
        return controller_description_block(self.description_level)

    @property
    def fingerprint(self) -> str:
        payload = {
            "model": self.model,
            "description_level": self.description_level.value,
            "protocol": self.protocol.value,
            "thinking": self.thinking,
            "description_block": self.description_block,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ContextDecision:
    """Validated structured model output for one progressive-context step."""

    action: ContextAction | str
    record_id: str | None = None
    selector: Mapping[str, object] | None = None
    query: str | None = None
    cursor_id: str | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        action = ContextAction(self.action)
        object.__setattr__(self, "action", action)
        if self.selector is not None:
            object.__setattr__(self, "selector", dict(self.selector))
        if action in {
            ContextAction.MATERIALIZE_FULL,
            ContextAction.MATERIALIZE_MORE,
            ContextAction.SEARCH_RECORD,
        } and not self.record_id:
            raise ValueError(f"{action.value} requires record_id.")
        if action == ContextAction.MATERIALIZE_MORE and not self.selector:
            raise ValueError("MATERIALIZE_MORE requires a bounded typed selector.")
        if action == ContextAction.SEARCH_RECORD and not self.query:
            raise ValueError("SEARCH_RECORD requires a query.")
        if action in {ContextAction.CURSOR_NEXT, ContextAction.CURSOR_QUERY} and not self.cursor_id:
            raise ValueError(f"{action.value} requires cursor_id.")
        if action == ContextAction.CURSOR_QUERY and not self.selector:
            raise ValueError("CURSOR_QUERY requires a structured cursor selector.")
        if action == ContextAction.CALL_TOOL and not self.tool_name:
            raise ValueError("CALL_TOOL requires tool_name.")
        if action == ContextAction.CONTINUE and any(
            value is not None
            for value in (self.record_id, self.selector, self.query, self.cursor_id, self.tool_name)
        ):
            raise ValueError("CONTINUE cannot carry retrieval arguments.")


def parse_context_decision(
    value: Mapping[str, object],
    *,
    allowed_record_ids: Sequence[str] = (),
    allowed_cursor_ids: Sequence[str] = (),
    allowed_tools: Sequence[str] = (),
) -> ContextDecision:
    """Parse untrusted model JSON and reject identities outside the active scope."""

    decision = ContextDecision(
        action=str(value.get("context_action", value.get("action", ""))).upper(),
        record_id=str(value["record_id"]) if value.get("record_id") is not None else None,
        selector=value.get("selector") if isinstance(value.get("selector"), Mapping) else None,
        query=str(value["query"]) if value.get("query") is not None else None,
        cursor_id=str(value["cursor_id"]) if value.get("cursor_id") is not None else None,
        tool_name=str(value["tool_name"]) if value.get("tool_name") is not None else None,
    )
    if decision.record_id and decision.record_id not in set(allowed_record_ids):
        raise ValueError(f"Unknown or unauthorized record_id: {decision.record_id}")
    if decision.cursor_id and decision.cursor_id not in set(allowed_cursor_ids):
        raise ValueError(f"Unknown or unauthorized cursor_id: {decision.cursor_id}")
    if decision.tool_name and decision.tool_name not in set(allowed_tools):
        raise ValueError(f"Unknown or unauthorized tool_name: {decision.tool_name}")
    return decision


@dataclass(frozen=True)
class RecordCapabilities:
    """Prompt-visible operations supported by one typed backing record."""

    full_available: bool = True
    full_bounded: bool = True
    searchable: bool = False
    partial_selectors: tuple[str, ...] = ()
    cursor_available: bool = False
    cursor_id: str | None = None
    has_more: bool = False
    allowed_cursor_operations: tuple[CursorOperation | str, ...] = ()
    partial_context: bool = False
    native_index_state: NativeIndexState | str = NativeIndexState.NOT_REQUESTED
    native_index_skipped_reason: str | None = None
    lazy_native_available: bool = False

    def __post_init__(self) -> None:
        operations = tuple(CursorOperation(value) for value in self.allowed_cursor_operations)
        object.__setattr__(self, "allowed_cursor_operations", operations)
        object.__setattr__(self, "partial_selectors", tuple(self.partial_selectors))
        object.__setattr__(self, "native_index_state", NativeIndexState(self.native_index_state))
        if self.cursor_available and not self.cursor_id:
            raise ValueError("cursor_available requires cursor_id.")

    def prompt_descriptor(self) -> dict[str, object]:
        """Return capabilities without exposing backing content."""

        return {
            "full_backing_available": self.full_available,
            "full_backing_bounded": self.full_bounded,
            "search_available": self.searchable,
            "partial_selectors": list(self.partial_selectors),
            "cursor_available": self.cursor_available,
            "cursor_id": self.cursor_id,
            "has_more": self.has_more,
            "allowed_cursor_operations": [value.value for value in self.allowed_cursor_operations],
            "partial_context": self.partial_context,
            "more_data_available": self.has_more,
            "full_materialization_available": self.full_available,
            "native_index_state": self.native_index_state.value,
            "native_index_skipped_reason": self.native_index_skipped_reason,
            "lazy_native_encoding_available": self.lazy_native_available,
        }


@dataclass(frozen=True)
class ImplicitPRAChunk:
    """One addressable chunk of a typed view before model-specific gist encoding."""

    chunk_id: str
    document_uri: str
    record_id: str
    ordinal: int
    text: str
    token_start: int
    token_end: int
    lexical_gist: tuple[str, ...]


@dataclass(frozen=True)
class ImplicitPRADocument:
    """A PRA-addressable view that preserves its native backing identity."""

    uri: str
    record_id: str
    record_type: RecordType
    view: PRAViewKind
    payload: object
    text: str
    chunks: tuple[ImplicitPRAChunk, ...]
    address_terms: tuple[str, ...]
    provenance: Mapping[str, object]
    scope_fingerprint: str
    parent_uri: str | None = None
    pra_reference_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class PRASelection:
    """Bounded automatic selection over active implicit PRA documents."""

    query: str
    chunks: tuple[ImplicitPRAChunk, ...]
    scores: tuple[float, ...]
    compared_chunks: int

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(chunk.record_id for chunk in self.chunks))


@dataclass(frozen=True)
class NativePRASelection:
    """Production PRA result over retrieval-only full-backing references."""

    query: str
    routing: RoutingResult
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class NativeIndexAudit:
    """Lifecycle and cost record for one full-backing native index request."""

    record_id: str
    native_index_state: NativeIndexState
    native_index_requested: bool
    native_index_built: bool
    native_index_skipped_reason: str | None
    native_index_tokens: int | None
    native_index_bytes: int
    native_index_latency_ms: float
    cheap_index_modes_built: tuple[str, ...]
    lazy_native_regions_encoded: int = 0
    lazy_native_tokens: int = 0

    def prompt_descriptor(self) -> dict[str, object]:
        """Return bounded controller metadata without exposing backing content."""

        return {
            "native_index_state": self.native_index_state.value,
            "native_index_built": self.native_index_built,
            "native_index_skipped_reason": self.native_index_skipped_reason,
            "search_available": bool(self.cheap_index_modes_built),
            "lazy_native_regions_encoded": self.lazy_native_regions_encoded,
        }


@dataclass(frozen=True)
class LazyNativeRegion:
    """One authorized selected region encoded through the native PRA path."""

    record_id: str
    selector: Mapping[str, object]
    reference_uri: str
    native_tokens: int
    latency_ms: float
    payload_bytes: int


class NativeIndexSizeLimitExceeded(RuntimeError):
    """Raised when callers request PRA_NATIVE after a size-gated skip."""

    def __init__(self, audit: NativeIndexAudit) -> None:
        super().__init__(
            f"Full-body native index for {audit.record_id!r} was skipped: "
            f"{audit.native_index_skipped_reason}. Use cheap search/cursor or "
            "lazy selected-region native encoding instead."
        )
        self.audit = audit


def _terms(value: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token.casefold() for token in _TOKEN.findall(str(value))))


def _payload_text(payload: object) -> str:
    return payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, ensure_ascii=True, default=str
    )


def _view_uri(record_id: str, view: PRAViewKind, sequence: int = 0) -> str:
    suffix = view.value if sequence == 0 else f"{view.value}-{sequence}"
    return f"{record_id}/views/{suffix}"


class PRARecordRegistry:
    """Stable record/view mapping with optional registration in a real PRA model.

    The lexical scorer is a transparent CPU fallback used by tests and mechanism
    benchmarks. ``bind_model`` and ``register_document`` call
    :meth:`PRAForCausalLM.add_reference`, which runs the production tokenizer,
    chunker, hidden-state encoder, gist construction, and native-K/V cache path.
    """

    def __init__(
        self,
        runtime: AdaptiveContextRuntime,
        *,
        chunk_tokens: int = 64,
        pra_model: PRAForCausalLM | None = None,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive.")
        self.runtime = runtime
        self.chunk_tokens = chunk_tokens
        self.pra_model = pra_model
        self.documents: dict[str, ImplicitPRADocument] = {}
        self.views_by_record: dict[str, list[str]] = {}
        self.capabilities: dict[str, RecordCapabilities] = {}
        self.reference_handles: dict[str, ReferenceHandle] = {}
        self.backing_reference_handles: dict[str, ReferenceHandle] = {}
        self.backing_index_metrics: dict[str, dict[str, float | int]] = {}
        self.native_index_audits: dict[str, NativeIndexAudit] = {}
        self.lazy_reference_handles: dict[str, ReferenceHandle] = {}
        self.large_record_indexes: dict[str, LargeRecordIndex] = {}

    def _ensure_large_record_index(self, record_id: str) -> LargeRecordIndex:
        """Build cheap indexes independently of full-body native admission."""

        existing = self.large_record_indexes.get(record_id)
        if existing is not None:
            return existing
        payload = self.runtime.store.get(record_id, scope=self.runtime.scope)
        index = LargeRecordIndex(payload)
        self.large_record_indexes[record_id] = index
        self.runtime._audit(
            "large_record_indexes_built",
            record_id,
            channels=[channel.value for channel in index.component_states],
            index_bytes=dict(index.index_bytes),
            indexed_units=len(index.units),
        )
        return index

    def search_large_record(
        self,
        record_id: str,
        query: str,
        *,
        policy: LargeRecordSearchPolicy | str = LargeRecordSearchPolicy.AUTO,
        channels: Sequence[LargeRecordChannel | str] | None = None,
        top_k: int = 4,
        candidate_limit: int = 64,
        native_ranking: Mapping[str, int] | None = None,
    ) -> LargeRecordSearchResult:
        """Return exact backing selectors using available independent indexes."""

        if record_id not in self.runtime.records:
            raise KeyError(record_id)
        result = self._ensure_large_record_index(record_id).search(
            query,
            policy=policy,
            channels=channels,
            top_k=top_k,
            candidate_limit=candidate_limit,
            native_ranking=native_ranking,
        )
        self.runtime._audit(
            "large_record_search",
            record_id,
            requested_policy=result.trace.requested_policy.value,
            resolved_channels=[channel.value for channel in result.trace.resolved_channels],
            fusion=result.trace.fusion,
            candidates_scored=dict(result.trace.candidates_scored),
            query_latency_ms=result.trace.query_latency_ms,
        )
        return result

    def large_record_lifecycle(self, record_id: str) -> RecordIndexLifecycle:
        """Report prompt, cheap-index, native-Q/K, and detail-K/V state separately."""

        index = self._ensure_large_record_index(record_id)
        record = self.runtime.records[record_id]
        audit = self.native_index_audits.get(record_id)
        if audit is None:
            native_state = IndexLifecycleState.NOT_BUILT
        else:
            native_state = {
                NativeIndexState.BUILT: IndexLifecycleState.BUILT,
                NativeIndexState.SKIPPED_SIZE_LIMIT: IndexLifecycleState.SKIPPED_SIZE_LIMIT,
                NativeIndexState.DEFERRED: IndexLifecycleState.DEFERRED,
                NativeIndexState.NOT_REQUESTED: IndexLifecycleState.NOT_BUILT,
            }[audit.native_index_state]
        native = IndexComponentState(
            native_state,
            audit.native_index_latency_ms if audit else 0.0,
            audit.native_index_bytes if audit and audit.native_index_built else 0,
            audit.native_index_tokens or 0 if audit else 0,
        )
        detail_state = (
            IndexLifecycleState.BUILT
            if audit and audit.native_index_built
            else IndexLifecycleState.LAZY
        )
        detail_bytes = int(self.backing_index_metrics.get(record_id, {}).get("resident_detail_kv_bytes", 0))
        summary = IndexComponentState(
            IndexLifecycleState.BUILT,
            index_bytes=record.compact_bytes,
            units=record.retained_units,
        )
        return RecordIndexLifecycle(
            index.component_states[LargeRecordChannel.TYPED],
            index.component_states[LargeRecordChannel.BM25],
            index.component_states[LargeRecordChannel.EMBEDDING],
            summary,
            native,
            IndexComponentState(detail_state, index_bytes=detail_bytes),
        )

    def bind_model(self, model: PRAForCausalLM) -> None:
        """Register existing views through the production PRA memory path."""

        self.pra_model = model
        for document in tuple(self.documents.values()):
            self._register_model_reference(document)

    def _chunks(self, record_id: str, uri: str, text: str) -> tuple[ImplicitPRAChunk, ...]:
        tokens = _TOKEN.findall(text)
        chunks = []
        for ordinal, start in enumerate(range(0, max(len(tokens), 1), self.chunk_tokens)):
            values = tokens[start : start + self.chunk_tokens]
            chunk_text = " ".join(values)
            gist = tuple(dict.fromkeys(token.casefold() for token in values))[:12]
            chunks.append(ImplicitPRAChunk(
                chunk_id=f"{uri}#chunk={ordinal}",
                document_uri=uri,
                record_id=record_id,
                ordinal=ordinal,
                text=chunk_text,
                token_start=start,
                token_end=start + len(values),
                lexical_gist=gist,
            ))
        return tuple(chunks)

    def _register_model_reference(self, document: ImplicitPRADocument) -> str:
        if self.pra_model is None:
            return ""
        previous = self.reference_handles.get(document.uri)
        if previous is not None:
            self.pra_model.remove_reference(previous)
        handle = self.pra_model.add_reference(document.uri, text=document.text)
        self.reference_handles[document.uri] = handle
        return handle.id

    def register_document(
        self,
        record_id: str,
        *,
        view: PRAViewKind | str,
        payload: object,
        parent_uri: str | None = None,
        sequence: int = 0,
    ) -> ImplicitPRADocument:
        """Create one typed PRA view and preserve scope/provenance identity."""

        if record_id not in self.runtime.records:
            raise KeyError(record_id)
        record = self.runtime.records[record_id]
        view = PRAViewKind(view)
        uri = _view_uri(record_id, view, sequence)
        text = _payload_text(payload)
        address_terms = tuple(dict.fromkeys(
            term
            for value in record.address_views().values()
            for term in _terms(value)
        ))
        document = ImplicitPRADocument(
            uri=uri,
            record_id=record_id,
            record_type=record.record_type,
            view=view,
            payload=payload,
            text=text,
            chunks=self._chunks(record_id, uri, text),
            address_terms=address_terms,
            provenance=record.backing.provenance,
            scope_fingerprint=record.backing.scope.fingerprint,
            parent_uri=parent_uri,
        )
        reference_id = self._register_model_reference(document)
        if reference_id:
            document = ImplicitPRADocument(**{
                **document.__dict__, "pra_reference_id": reference_id
            })
        self.documents[uri] = document
        record_views = self.views_by_record.setdefault(record_id, [])
        if uri not in record_views:
            record_views.append(uri)
        return document

    def register_compact_record(
        self,
        record_id: str,
        *,
        capabilities: RecordCapabilities | None = None,
    ) -> ImplicitPRADocument:
        """Register ``record/.../summary`` as the first implicit PRA document."""

        record = self.runtime.records[record_id]
        self._ensure_large_record_index(record_id)
        capabilities = capabilities or RecordCapabilities()
        compact_is_partial = bool(record.metadata.get("lossy", True))
        capabilities = replace(
            capabilities,
            partial_context=capabilities.partial_context or compact_is_partial,
            has_more=capabilities.has_more or compact_is_partial,
            searchable=True,
        )
        audit = self.native_index_audits.get(record_id)
        if audit is not None:
            capabilities = replace(
                capabilities,
                partial_context=capabilities.partial_context or compact_is_partial,
                searchable=capabilities.searchable or bool(audit.cheap_index_modes_built),
                has_more=capabilities.has_more or compact_is_partial,
                native_index_state=audit.native_index_state,
                native_index_skipped_reason=audit.native_index_skipped_reason,
                lazy_native_available=(
                    capabilities.lazy_native_available
                    or audit.native_index_state in {
                        NativeIndexState.SKIPPED_SIZE_LIMIT,
                        NativeIndexState.DEFERRED,
                    }
                ),
            )
        self.capabilities[record_id] = capabilities
        payload = {
            "record_id": record_id,
            "record_type": record.record_type.value,
            "view": "summary",
            "compact": record.compact_view(),
            "backing_size_bytes": record.backing.size_bytes,
            "capabilities": capabilities.prompt_descriptor(),
            "native_index": (
                audit.prompt_descriptor()
                if audit is not None
                else NativeIndexAudit(
                    record_id,
                    NativeIndexState.NOT_REQUESTED,
                    False,
                    False,
                    None,
                    0,
                    record.backing.size_bytes,
                    0.0,
                    tuple(channel.value for channel in self._ensure_large_record_index(record_id).component_states),
                ).prompt_descriptor()
            ),
            "provenance": dict(record.backing.provenance),
        }
        return self.register_document(record_id, view=PRAViewKind.SUMMARY, payload=payload)

    def _native_token_count(self, text: str) -> int:
        if self.pra_model is None:
            raise RuntimeError("A PRA model must be bound before native token accounting.")
        return len(self.pra_model.tokenizer(text, add_special_tokens=False).input_ids)

    def _refresh_compact_metadata(self, record_id: str) -> None:
        capabilities = self.capabilities.get(record_id, RecordCapabilities())
        self.register_compact_record(record_id, capabilities=capabilities)

    def prepare_backing_index(
        self, record_id: str, *, force: bool = False
    ) -> NativeIndexAudit:
        """Apply the configured size gate and build a full native index if allowed.

        The backing reference is retrieval-only: it is not inserted into
        ``documents`` and therefore does not become prompt-visible context.
        """

        if self.pra_model is None:
            raise RuntimeError("A PRA model must be bound before indexing backing state.")
        if record_id not in self.runtime.records:
            raise KeyError(record_id)
        record = self.runtime.records[record_id]
        byte_count = record.backing.size_bytes
        token_limit, byte_limit, deferred = self.runtime.policy.native_index_policy(
            record.record_type
        )
        cheap_index = self._ensure_large_record_index(record_id)
        cheap_modes = tuple(channel.value for channel in cheap_index.component_states)
        reason = None
        state = NativeIndexState.BUILT
        token_count: int | None = None
        if deferred and not force:
            state = NativeIndexState.DEFERRED
            reason = "native index deferred by policy"
        elif byte_limit is not None and byte_count > byte_limit:
            state = NativeIndexState.SKIPPED_SIZE_LIMIT
            reason = f"{byte_count} bytes exceed configured limit {byte_limit}"
        else:
            payload = self.runtime.store.get(record_id, scope=self.runtime.scope)
            text = _payload_text(payload)
            token_count = self._native_token_count(text)
            if token_limit is not None and token_count > token_limit:
                state = NativeIndexState.SKIPPED_SIZE_LIMIT
                reason = f"{token_count} tokens exceed configured limit {token_limit}"
        if state is not NativeIndexState.BUILT:
            audit = NativeIndexAudit(
                record_id,
                state,
                True,
                False,
                reason,
                token_count,
                byte_count,
                0.0,
                cheap_modes,
            )
            self.native_index_audits[record_id] = audit
            self.runtime._audit(
                "native_index_skipped",
                record_id,
                state=state.value,
                reason=reason,
                native_index_tokens=token_count,
                native_index_bytes=byte_count,
                cheap_index_modes_built=list(cheap_modes),
            )
            self._refresh_compact_metadata(record_id)
            return audit
        payload = self.runtime.store.get(record_id, scope=self.runtime.scope)
        text = _payload_text(payload)
        if token_count is None:
            token_count = self._native_token_count(text)
        previous = self.backing_reference_handles.get(record_id)
        if previous is not None:
            self.pra_model.remove_reference(previous)
        uri = f"{record_id}/views/backing"
        started = time.perf_counter()
        handle = self.pra_model.add_reference(uri, text=text)
        elapsed = time.perf_counter() - started
        stats = self.pra_model.stats()
        self.backing_reference_handles[record_id] = handle
        self.backing_index_metrics[record_id] = {
            "backing_tokens": handle.tokens,
            "backing_chunks": handle.chunks,
            "ingestion_seconds": elapsed,
            "routing_index_bytes": int(stats["routing_index_bytes"]),
            "resident_detail_kv_bytes": int(stats["resident_detail_kv_bytes"]),
            "full_backing_bytes": byte_count,
        }
        audit = NativeIndexAudit(
            record_id,
            NativeIndexState.BUILT,
            True,
            True,
            None,
            int(handle.tokens),
            byte_count,
            elapsed * 1000.0,
            cheap_modes,
        )
        self.native_index_audits[record_id] = audit
        self.runtime._audit(
            "native_index_built",
            record_id,
            native_index_tokens=int(handle.tokens),
            native_index_bytes=byte_count,
            native_index_latency_ms=elapsed * 1000.0,
            cheap_index_modes_built=list(cheap_modes),
        )
        self._refresh_compact_metadata(record_id)
        return audit

    def register_backing_record(self, record_id: str) -> ReferenceHandle:
        """Build PRA_NATIVE or fail explicitly when the configured gate skips it."""

        audit = self.prepare_backing_index(record_id)
        if not audit.native_index_built:
            raise NativeIndexSizeLimitExceeded(audit)
        return self.backing_reference_handles[record_id]

    def encode_selected_region_native(
        self,
        record_id: str,
        selector: Mapping[str, object],
    ) -> LazyNativeRegion:
        """Authorize, select, and natively encode only one bounded backing region."""

        if self.pra_model is None:
            raise RuntimeError("A PRA model must be bound before lazy native encoding.")
        materialized = self.runtime.materialize(MaterializationEvent(
            record_id,
            level=RecordViewName.SELECTED,
            selector=selector,
        ))
        text = _payload_text(materialized.payload)
        sequence = sum(
            uri.startswith(f"{record_id}/views/lazy-native-")
            for uri in self.lazy_reference_handles
        ) + 1
        uri = f"{record_id}/views/lazy-native-{sequence}"
        started = time.perf_counter()
        handle = self.pra_model.add_reference(uri, text=text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.lazy_reference_handles[uri] = handle
        previous = self.native_index_audits.get(record_id)
        if previous is None:
            record = self.runtime.records[record_id]
            previous = NativeIndexAudit(
                record_id,
                NativeIndexState.NOT_REQUESTED,
                False,
                False,
                None,
                0,
                record.backing.size_bytes,
                0.0,
                tuple(channel.value for channel in self._ensure_large_record_index(record_id).component_states),
            )
        audit = replace(
            previous,
            lazy_native_regions_encoded=previous.lazy_native_regions_encoded + 1,
            lazy_native_tokens=previous.lazy_native_tokens + int(handle.tokens),
        )
        self.native_index_audits[record_id] = audit
        self.runtime._audit(
            "lazy_native_region_encoded",
            record_id,
            selector=dict(selector),
            native_tokens=int(handle.tokens),
            latency_ms=latency_ms,
            source_fingerprint=self.runtime.records[record_id].backing.content_hash,
        )
        self._refresh_compact_metadata(record_id)
        return LazyNativeRegion(
            record_id,
            dict(selector),
            uri,
            int(handle.tokens),
            latency_ms,
            materialized.payload_bytes,
        )

    def route_backing(self, query: str) -> NativePRASelection:
        """Run production PRA routing over registered full-backing references."""

        if self.pra_model is None:
            raise RuntimeError("A PRA model must be bound before native routing.")
        if not self.backing_reference_handles:
            raise RuntimeError("No full-backing PRA references are registered.")
        routing = self.pra_model.route(query)
        by_uri = {
            handle.uri: record_id
            for record_id, handle in self.backing_reference_handles.items()
        }
        record_ids = tuple(dict.fromkeys(
            by_uri[str(row["reference_uri"])]
            for row in routing.selected
            if str(row["reference_uri"]) in by_uri
        ))
        return NativePRASelection(query, routing, record_ids)

    def route(
        self,
        query: str,
        *,
        top_k: int = 4,
        views: Sequence[PRAViewKind | str] | None = None,
    ) -> PRASelection:
        """Select active chunks by compact gist/address overlap under a hard K."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        allowed = {PRAViewKind(value) for value in views} if views else None
        query_terms = set(_terms(query))
        ranked: list[tuple[float, str, ImplicitPRAChunk]] = []
        compared = 0
        for document in self.documents.values():
            if allowed is not None and document.view not in allowed:
                continue
            address_overlap = len(query_terms & set(document.address_terms))
            for chunk in document.chunks:
                compared += 1
                chunk_overlap = len(query_terms & set(chunk.lexical_gist))
                score = float(chunk_overlap + 0.25 * address_overlap)
                if score > 0:
                    ranked.append((score, chunk.chunk_id, chunk))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        chosen = ranked[:top_k]
        return PRASelection(
            query=query,
            chunks=tuple(row[2] for row in chosen),
            scores=tuple(row[0] for row in chosen),
            compared_chunks=compared,
        )


@dataclass(frozen=True)
class ContextExecutionResult:
    """One context-control transition and the typed view it produced."""

    decision: ContextDecision
    success: bool
    produced_document: ImplicitPRADocument | None
    payload: object | None
    payload_bytes: int
    network_bytes: int
    round_trips: int
    model_passes: int
    latency_seconds: float
    transport: ContextTransport
    error: str | None = None


class ProgressiveContextRuntime:
    """Recursive PRA selection plus bounded model-controlled escalation."""

    def __init__(
        self,
        runtime: AdaptiveContextRuntime,
        *,
        registry: PRARecordRegistry | None = None,
        pra_model: PRAForCausalLM | None = None,
        chunk_tokens: int = 64,
    ) -> None:
        self.runtime = runtime
        self.registry = registry or PRARecordRegistry(
            runtime, chunk_tokens=chunk_tokens, pra_model=pra_model
        )
        self._view_sequences: dict[tuple[str, PRAViewKind], int] = {}

    def ingest(
        self,
        payload: object,
        *,
        record_type: RecordType | str,
        capabilities: RecordCapabilities | None = None,
        provenance: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> AdaptiveContextRecord:
        """Persist a typed result and immediately register its compact PRA view."""

        record = self.runtime.ingest(
            payload, record_type=record_type, provenance=provenance, **kwargs
        )
        self.registry.register_compact_record(record.record_id, capabilities=capabilities)
        return record

    def register_compact_record(
        self, record_id: str, *, capabilities: RecordCapabilities | None = None
    ) -> ImplicitPRADocument:
        return self.registry.register_compact_record(record_id, capabilities=capabilities)

    def _register_result(
        self,
        record_id: str,
        view: PRAViewKind,
        payload: object,
    ) -> ImplicitPRADocument:
        key = (record_id, view)
        sequence = self._view_sequences.get(key, 0) + 1
        self._view_sequences[key] = sequence
        parent = self.registry.views_by_record[record_id][0]
        return self.registry.register_document(
            record_id, view=view, payload=payload, parent_uri=parent, sequence=sequence
        )

    def materialize_full(self, record_id: str) -> ContextExecutionResult:
        """Activate an exact known backing record without semantic rediscovery."""

        return self.execute(ContextDecision(ContextAction.MATERIALIZE_FULL, record_id=record_id))

    def materialize_more(
        self, record_id: str, selector: Mapping[str, object]
    ) -> ContextExecutionResult:
        return self.execute(ContextDecision(
            ContextAction.MATERIALIZE_MORE, record_id=record_id, selector=selector
        ))

    def search_record(self, record_id: str, query: str) -> ContextExecutionResult:
        return self.execute(ContextDecision(
            ContextAction.SEARCH_RECORD, record_id=record_id, query=query
        ))

    def cursor_next(self, cursor_id: str) -> ContextExecutionResult:
        return self.execute(ContextDecision(ContextAction.CURSOR_NEXT, cursor_id=cursor_id))

    def cursor_query(
        self, cursor_id: str, selector: Mapping[str, object]
    ) -> ContextExecutionResult:
        return self.execute(ContextDecision(
            ContextAction.CURSOR_QUERY, cursor_id=cursor_id, selector=selector
        ))

    def automatic_select(self, query: str, *, top_k: int = 4) -> PRASelection:
        """Run PRA selection over currently active typed views before escalation."""

        return self.registry.route(query, top_k=top_k)

    def register_backing_record(self, record_id: str) -> ReferenceHandle:
        """Create retrieval-only production PRA addresses for exact backing state."""

        return self.registry.register_backing_record(record_id)

    def prepare_native_index(
        self, record_id: str, *, force: bool = False
    ) -> NativeIndexAudit:
        """Return an auditable built, gated, or deferred native-index result."""

        return self.registry.prepare_backing_index(record_id, force=force)

    def encode_selected_region_native(
        self, record_id: str, selector: Mapping[str, object]
    ) -> LazyNativeRegion:
        """Encode one authorized bounded region after cheap selection."""

        return self.registry.encode_selected_region_native(record_id, selector)

    def search_large_record(
        self,
        record_id: str,
        query: str,
        *,
        policy: LargeRecordSearchPolicy | str = LargeRecordSearchPolicy.AUTO,
        channels: Sequence[LargeRecordChannel | str] | None = None,
        top_k: int = 4,
        candidate_limit: int = 64,
        native_ranking: Mapping[str, int] | None = None,
    ) -> LargeRecordSearchResult:
        """Discover bounded selectors without requiring a full native index."""

        return self.registry.search_large_record(
            record_id,
            query,
            policy=policy,
            channels=channels,
            top_k=top_k,
            candidate_limit=candidate_limit,
            native_ranking=native_ranking,
        )

    def recover_large_record_native(
        self,
        record_id: str,
        query: str,
        *,
        policy: LargeRecordSearchPolicy | str = LargeRecordSearchPolicy.AUTO,
        top_k: int = 1,
    ) -> tuple[LargeRecordSearchResult, tuple[LazyNativeRegion, ...]]:
        """Search cheaply, then lazily encode only the selected exact regions."""

        result = self.search_large_record(record_id, query, policy=policy, top_k=top_k)
        regions = tuple(
            self.encode_selected_region_native(record_id, hit.selector)
            for hit in result.hits
        )
        return result, regions

    def native_select(self, query: str) -> NativePRASelection:
        """Run production PRA routing over exact backing state."""

        return self.registry.route_backing(query)

    def materialize_native_selection(
        self,
        selection: NativePRASelection,
        *,
        top_k: int | None = None,
    ) -> ContextExecutionResult:
        """Expose selected original spans after production native-K/V selection."""

        if self.registry.pra_model is None:
            raise RuntimeError("A PRA model must be bound before native materialization.")
        selected = selection.routing.selected[:top_k]
        if not selected:
            decision = ContextDecision(
                ContextAction.SEARCH_RECORD,
                record_id=next(iter(self.runtime.records)),
                query=selection.query,
            )
            return ContextExecutionResult(
                decision, False, None, None, 0, 0, 0, 1, 0.0,
                ContextTransport.NATIVE, "PRA selected no backing chunk.",
            )
        by_uri = {
            handle.uri: record_id
            for record_id, handle in self.registry.backing_reference_handles.items()
        }
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in selected:
            uri = str(row["reference_uri"])
            if uri not in by_uri:
                continue
            grouped.setdefault(by_uri[uri], []).append(dict(row))
        if not grouped:
            raise ValueError("Native selection did not resolve to an authorized backing record.")
        started = time.perf_counter()
        record_id = next(iter(grouped))
        payload = self.runtime.store.get(record_id, scope=self.runtime.scope)
        text = _payload_text(payload)
        tokenizer = self.registry.pra_model.tokenizer
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        chunks = []
        for row in grouped[record_id]:
            start = int(row["logical_start"])
            stop = int(row["logical_end"])
            chunks.append({
                "chunk_id": row["chunk_id"],
                "logical_start": start,
                "logical_end": stop,
                "text": tokenizer.decode(token_ids[start:stop], skip_special_tokens=True),
            })
        detail = {
            "record_id": record_id,
            "source_view": "backing",
            "selection_policy": selection.routing.stats["selection_policy"],
            "selected_chunks": chunks,
        }
        payload_bytes, network_bytes, round_trips = self.runtime.account_selected_payload(
            detail, action="native_pra_materialize"
        )
        document = self._register_result(record_id, PRAViewKind.DETAIL, detail)
        decision = ContextDecision(
            ContextAction.SEARCH_RECORD, record_id=record_id, query=selection.query
        )
        return ContextExecutionResult(
            decision, True, document, detail, payload_bytes, network_bytes,
            round_trips, 1, time.perf_counter() - started, ContextTransport.NATIVE,
        )

    def materialize_backing_chunks(
        self,
        record_id: str,
        query: str,
        chunks: Sequence[Mapping[str, object]],
        *,
        selection_policy: str,
    ) -> ContextExecutionResult:
        """Replay a frozen production routing result through typed materialization.

        Experiments may route once and evaluate several controller seeds. This
        method avoids recomputing the frozen Qwen index while preserving the
        exact record identity, selected original spans, transport accounting,
        and recursive PRA document registration used by live native selection.
        """

        if record_id not in self.runtime.records:
            raise KeyError(record_id)
        started = time.perf_counter()
        selected = tuple(dict(chunk) for chunk in chunks)
        detail = {
            "record_id": record_id,
            "source_view": "backing",
            "selection_policy": selection_policy,
            "selected_chunks": selected,
        }
        payload_bytes, network_bytes, round_trips = self.runtime.account_selected_payload(
            detail, action="native_pra_materialize"
        )
        document = self._register_result(record_id, PRAViewKind.DETAIL, detail)
        decision = ContextDecision(
            ContextAction.SEARCH_RECORD, record_id=record_id, query=query
        )
        return ContextExecutionResult(
            decision, bool(selected), document, detail, payload_bytes, network_bytes,
            round_trips, 1, time.perf_counter() - started, ContextTransport.NATIVE,
            None if selected else "PRA selected no backing chunk.",
        )

    def execute(
        self,
        decision: ContextDecision,
        *,
        transport: ContextTransport | str = ContextTransport.NATIVE,
    ) -> ContextExecutionResult:
        """Execute one decision and recursively register any returned typed view."""

        transport = ContextTransport(transport)
        started = time.perf_counter()
        if decision.action in {ContextAction.CONTINUE, ContextAction.CALL_TOOL}:
            return ContextExecutionResult(
                decision, True, None, None, 0, 0, 0, 1,
                time.perf_counter() - started, transport,
            )
        try:
            if decision.action == ContextAction.MATERIALIZE_FULL:
                result = self.runtime.materialize(MaterializationEvent(
                    decision.record_id or "", level=RecordViewName.FULL
                ))
                document = self._register_result(
                    result.record_id, PRAViewKind.FULL, result.payload
                )
                return ContextExecutionResult(
                    decision, True, document, result.payload, result.payload_bytes,
                    result.network_bytes, result.round_trips, 2,
                    time.perf_counter() - started, transport,
                )
            if decision.action == ContextAction.MATERIALIZE_MORE:
                result = self.runtime.materialize(MaterializationEvent(
                    decision.record_id or "",
                    level=RecordViewName.SELECTED,
                    selector=decision.selector,
                ))
                document = self._register_result(
                    result.record_id, PRAViewKind.DETAIL, result.payload
                )
                return ContextExecutionResult(
                    decision, True, document, result.payload, result.payload_bytes,
                    result.network_bytes, result.round_trips, 2,
                    time.perf_counter() - started, transport,
                )
            if decision.action == ContextAction.SEARCH_RECORD:
                record_id = decision.record_id or ""
                result = self.runtime.search_record(
                    record_id, decision.query or "", limit=4
                )
                selected = result.payload
                remote = self.runtime.policy.topology.value != "same_process"
                document = self._register_result(
                    record_id, PRAViewKind.SEARCH_RESULT, selected
                )
                return ContextExecutionResult(
                    decision, True, document, selected, result.payload_bytes,
                    result.payload_bytes if remote else 0, int(remote), 2,
                    time.perf_counter() - started, transport,
                )
            cursor = self.runtime.cursors.describe(
                decision.cursor_id or "", scope=self.runtime.scope
            )
            if decision.action == ContextAction.CURSOR_NEXT:
                cursor_action = CursorAction(cursor.cursor_id, CursorOperation.NEXT)
            else:
                selector = dict(decision.selector or {})
                operation = CursorOperation(str(selector.pop("operation", "search")))
                cursor_action = CursorAction(cursor.cursor_id, operation, selector)
            result = self.runtime.execute_cursor_action(cursor_action)
            if not result.success:
                raise ValueError(result.error or "cursor action failed")
            document = self._register_result(
                cursor.record_id, PRAViewKind.CURSOR_PAGE, result.payload
            )
            return ContextExecutionResult(
                decision, True, document, result.payload, result.payload_bytes,
                result.payload_bytes if self.runtime.policy.topology.value != "same_process" else 0,
                int(self.runtime.policy.topology.value != "same_process"), 2,
                time.perf_counter() - started, transport,
            )
        except (KeyError, TypeError, ValueError, RecordAccessDenied) as exc:
            return ContextExecutionResult(
                decision, False, None, None, 0, 0, 0, 1,
                time.perf_counter() - started, transport,
                f"{type(exc).__name__}: {exc}",
            )

    def run_loop(
        self,
        query: str,
        decide: Callable[[tuple[ImplicitPRADocument, ...], PRASelection], ContextDecision],
        *,
        max_steps: int = 3,
        top_k: int = 4,
    ) -> tuple[ContextExecutionResult, ...]:
        """Re-enter PRA after each expansion until CONTINUE/CALL_TOOL or the bound."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        transitions = []
        for _ in range(max_steps):
            selection = self.automatic_select(query, top_k=top_k)
            active = tuple(self.registry.documents.values())
            decision = decide(active, selection)
            result = self.execute(decision)
            transitions.append(result)
            if not result.success or decision.action in {
                ContextAction.CONTINUE, ContextAction.CALL_TOOL
            }:
                break
        return tuple(transitions)
