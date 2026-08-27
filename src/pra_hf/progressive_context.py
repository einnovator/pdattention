"""Progressive model-controlled disclosure over typed PRA records.

The backing runtime owns exact bytes, authorization, cursors, and transport.
This module gives every visible record view a stable PRA document identity and
executes a bounded context-control decision without asking the model to invent
payloads or rediscover a record whose identity is already known.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
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

if TYPE_CHECKING:
    from .model import PRAForCausalLM, ReferenceHandle
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


class ContextTransport(str, Enum):
    """Interface used to execute one logical context-control decision."""

    NATIVE = "native"
    TOOL = "tool"
    MIXED = "mixed"


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

    def __post_init__(self) -> None:
        operations = tuple(CursorOperation(value) for value in self.allowed_cursor_operations)
        object.__setattr__(self, "allowed_cursor_operations", operations)
        object.__setattr__(self, "partial_selectors", tuple(self.partial_selectors))
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
        self.views_by_record.setdefault(record_id, []).append(uri)
        return document

    def register_compact_record(
        self,
        record_id: str,
        *,
        capabilities: RecordCapabilities | None = None,
    ) -> ImplicitPRADocument:
        """Register ``record/.../summary`` as the first implicit PRA document."""

        record = self.runtime.records[record_id]
        capabilities = capabilities or RecordCapabilities()
        self.capabilities[record_id] = capabilities
        payload = {
            "record_id": record_id,
            "record_type": record.record_type.value,
            "view": "summary",
            "compact": record.compact_view(),
            "backing_size_bytes": record.backing.size_bytes,
            "capabilities": capabilities.prompt_descriptor(),
            "provenance": dict(record.backing.provenance),
        }
        return self.register_document(record_id, view=PRAViewKind.SUMMARY, payload=payload)

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
                payload = self.runtime.store.get(record_id, scope=self.runtime.scope)
                selected = _search_known_payload(payload, decision.query or "", limit=4)
                payload_bytes, network_bytes, round_trips = (
                    self.runtime.account_selected_payload(
                        selected, action="search_record"
                    )
                )
                document = self._register_result(
                    record_id, PRAViewKind.SEARCH_RESULT, selected
                )
                return ContextExecutionResult(
                    decision, True, document, selected, payload_bytes,
                    network_bytes, round_trips, 2,
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


def _search_known_payload(payload: object, query: str, *, limit: int) -> object:
    """Return bounded exact units from one already identified backing object."""

    if limit <= 0:
        raise ValueError("search limit must be positive.")
    collection = "items"
    values: list[object]
    if isinstance(payload, Mapping):
        for name in ("rows", "nodes", "edges", "events", "results", "chunks", "items"):
            candidate = payload.get(name)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                collection, values = name, list(candidate)
                break
        else:
            collection, values = "fields", [
                {"field": key, "value": value} for key, value in payload.items()
            ]
    elif isinstance(payload, str):
        collection, values = "lines", payload.splitlines()
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
    else:
        values = [payload]
    terms = set(_terms(query))
    ranked = []
    for index, value in enumerate(values):
        overlap = len(terms & set(_terms(_payload_text(value))))
        if overlap:
            ranked.append((overlap, index, value))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected = [row[2] for row in ranked[:limit]]
    return {"collection": collection, "query": query, "matches": selected}
