"""Typed context records and record-aware PRA admission/materialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Mapping, Sequence

from pra_hf.agent_execution import resource_tool_schema
from pra_hf.agent_resources import AgentResource
from pra_hf.union_discovery import CandidateSet


class RecordType(str, Enum):
    TOOL_RECORD = "tool_record"
    TOOL_DEFINITION = "tool_record"
    TOOL_CATALOG_SLICE = "tool_catalog_slice"
    SKILL_CATALOG_SLICE = "skill_catalog_slice"
    CAPABILITY_SLICE = "capability_slice"
    TOOL_RESPONSE = "tool_response"
    LOG_BLOCK = "log_block"
    TERMINAL_OUTPUT = "terminal_output"
    DB_RESULT = "db_result"
    GRAPH_RESULT = "graph_result"
    RAG_RESULT = "rag_result"
    RAG_CHUNK_SET = "rag_chunk_set"
    RAG_CHUNK = "rag_chunk"
    FILE_READ = "file_read"
    API_RESULT = "api_result"
    GENERIC_TEXT = "generic_text"
    CURSOR = "cursor"
    SYSTEM_INSTRUCTION = "system_instruction"
    SKILL_RECORD = "skill_record"
    SKILL = "skill_record"
    SESSION_RECORD = "session_record"
    GENERIC_DOCUMENT = "generic_document"

    @classmethod
    def _missing_(cls, value: object):
        # Read pre-lazy-record manifests without emitting legacy type names.
        legacy = {"tool_definition": cls.TOOL_RECORD, "skill": cls.SKILL_RECORD}
        return legacy.get(value)


class RecordSelectionPolicy(str, Enum):
    EXPLICIT = "explicit"
    AUTHORITATIVE_PARENT = "authoritative_parent"
    LEXICAL = "lexical"
    INDEXED = "indexed"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    UNION = "union"
    ADAPTIVE = "adaptive"
    GRAPH = "graph"
    ALL_CHILDREN = "all_children"
    TOP_K_CHILDREN = "top_k_children"
    NONE = "none"


class MaterializationPolicy(str, Enum):
    FULL = "full"
    CHUNKED = "chunked"
    SELECTED_CHUNKS = "selected_chunks"
    FIELDS = "fields"
    HEAD_ONLY = "head_only"
    BOUNDED = "bounded"
    NONE = "none"


class SelectionAuthority(str, Enum):
    AUTHORITATIVE = "authoritative"
    PREFERRED = "preferred"
    ADVISORY = "advisory"


class RecordAtomicity(str, Enum):
    RECORD = "record"
    CHUNK = "chunk"
    FIELD = "field"


class OverflowBehavior(str, Enum):
    REQUEST_NARROW = "request_narrow"
    ERROR = "error"
    EXPAND_TEMPORARILY = "expand_temporarily"
    DROP_WHOLE_RECORDS = "drop_whole_records"


class RecordViewName(str, Enum):
    """Predefined typed representations of one semantic record."""

    METADATA = "metadata"
    COMPACT = "compact"
    SELECTED = "selected"
    SELECTION = "selection"
    FULL = "full"


@dataclass(frozen=True)
class RecordView:
    """One deterministic, type-aware representation of a record."""

    name: RecordViewName | str
    payload: Mapping[str, object] | str
    fields: tuple[str, ...]
    token_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", RecordViewName(self.name))
        object.__setattr__(self, "fields", tuple(dict.fromkeys(self.fields)))
        if self.token_count < 0:
            raise ValueError("Record-view token_count cannot be negative.")


@dataclass(frozen=True)
class RecordPolicy:
    """Orthogonal selection, authority, atomicity, and materialization policy."""

    record_type: RecordType | str
    selection: RecordSelectionPolicy | str
    authority: SelectionAuthority | str
    atomicity: RecordAtomicity | str
    materialization: MaterializationPolicy | str
    allow_partial_tools: bool = False
    initial_view: RecordViewName | str = RecordViewName.FULL
    selected_view: RecordViewName | str = RecordViewName.FULL

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_type", RecordType(self.record_type))
        object.__setattr__(self, "selection", RecordSelectionPolicy(self.selection))
        object.__setattr__(self, "authority", SelectionAuthority(self.authority))
        object.__setattr__(self, "atomicity", RecordAtomicity(self.atomicity))
        object.__setattr__(self, "materialization", MaterializationPolicy(self.materialization))
        object.__setattr__(self, "initial_view", RecordViewName(self.initial_view))
        object.__setattr__(self, "selected_view", RecordViewName(self.selected_view))
        if self.record_type in {
            RecordType.TOOL_CATALOG_SLICE,
            RecordType.SKILL_CATALOG_SLICE,
            RecordType.CAPABILITY_SLICE,
        }:
            if self.selection != RecordSelectionPolicy.ALL_CHILDREN:
                raise ValueError("Capability slices must select all authoritative children.")
            if self.materialization != MaterializationPolicy.FULL:
                raise ValueError("Capability slices materialize complete named child views.")
        if self.record_type in {RecordType.TOOL_DEFINITION, RecordType.SKILL}:
            partial = self.atomicity != RecordAtomicity.RECORD or self.materialization != MaterializationPolicy.FULL
            if partial and not self.allow_partial_tools:
                raise ValueError("Partial capability records require an explicit experimental override.")
            if not self.allow_partial_tools and self.selected_view != RecordViewName.FULL:
                raise ValueError("Selected capabilities must transition to the full view.")


def default_record_policy(record_type: RecordType | str) -> RecordPolicy:
    """Return safe defaults while leaving future Paper 7 reductions unimplemented."""

    record_type = RecordType(record_type)
    if record_type in {
        RecordType.TOOL_CATALOG_SLICE,
        RecordType.SKILL_CATALOG_SLICE,
        RecordType.CAPABILITY_SLICE,
    }:
        return RecordPolicy(
            record_type, RecordSelectionPolicy.ALL_CHILDREN, SelectionAuthority.AUTHORITATIVE,
            RecordAtomicity.RECORD, MaterializationPolicy.FULL,
        )
    if record_type in {RecordType.TOOL_DEFINITION, RecordType.SKILL}:
        return RecordPolicy(
            record_type, RecordSelectionPolicy.AUTHORITATIVE_PARENT, SelectionAuthority.AUTHORITATIVE,
            RecordAtomicity.RECORD, MaterializationPolicy.FULL,
            initial_view=RecordViewName.SELECTION,
            selected_view=RecordViewName.FULL,
        )
    if record_type == RecordType.SYSTEM_INSTRUCTION:
        return RecordPolicy(
            record_type, RecordSelectionPolicy.AUTHORITATIVE_PARENT, SelectionAuthority.AUTHORITATIVE,
            RecordAtomicity.RECORD, MaterializationPolicy.FULL,
        )
    if record_type == RecordType.GENERIC_DOCUMENT:
        return RecordPolicy(
            record_type, RecordSelectionPolicy.ADAPTIVE, SelectionAuthority.ADVISORY,
            RecordAtomicity.CHUNK, MaterializationPolicy.CHUNKED,
        )
    return RecordPolicy(
        record_type, RecordSelectionPolicy.EXPLICIT, SelectionAuthority.PREFERRED,
        RecordAtomicity.RECORD, MaterializationPolicy.FULL,
    )


@dataclass(frozen=True)
class RecordBoundary:
    """Character boundary for one serialized record, child, or field."""

    name: str
    start: int
    end: int
    boundary_type: str = "field"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Record boundaries must be ordered nonnegative offsets.")


@dataclass(frozen=True)
class ContextRecord:
    """Provider-independent context object whose boundaries survive admission."""

    record_id: str
    record_type: RecordType | str
    payload: Mapping[str, object] | str
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    boundaries: tuple[RecordBoundary, ...] = ()
    selection_provenance: Mapping[str, object] = field(default_factory=dict)
    policy: RecordPolicy | None = None
    version: str = "v1"
    source_fingerprint: str = ""
    views: Mapping[RecordViewName | str, RecordView] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_type = RecordType(self.record_type)
        object.__setattr__(self, "record_type", record_type)
        object.__setattr__(self, "child_ids", tuple(dict.fromkeys(self.child_ids)))
        object.__setattr__(self, "selection_provenance", dict(self.selection_provenance))
        if self.policy is None:
            object.__setattr__(self, "policy", default_record_policy(record_type))
        elif self.policy.record_type != record_type:
            raise ValueError("Record policy type does not match ContextRecord type.")
        if not self.record_id:
            raise ValueError("record_id is required.")
        normalized_views = {
            RecordViewName(name): view if isinstance(view, RecordView) else RecordView(name, view, ())
            for name, view in self.views.items()
        }
        if RecordViewName.FULL not in normalized_views:
            fields = tuple(self.payload) if isinstance(self.payload, Mapping) else ("body",)
            normalized_views[RecordViewName.FULL] = RecordView(
                RecordViewName.FULL, self.payload, fields
            )
        if normalized_views[RecordViewName.FULL].payload != self.payload:
            raise ValueError("ContextRecord payload must equal its full view payload.")
        for required_view in (self.policy.initial_view, self.policy.selected_view):
            if required_view not in normalized_views:
                raise ValueError(f"Record policy requires missing {required_view.value!r} view.")
        object.__setattr__(self, "views", normalized_views)
        if not self.source_fingerprint:
            source = json.dumps(self.payload, sort_keys=True, default=str) if not isinstance(self.payload, str) else self.payload
            object.__setattr__(self, "source_fingerprint", hashlib.sha256(source.encode("utf-8")).hexdigest())

    @property
    def size_bytes(self) -> int:
        return len(serialize_record(self, view=RecordViewName.FULL).encode("utf-8"))

    def materialize(
        self,
        view: RecordViewName | str = RecordViewName.FULL,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> RecordView:
        """Return one complete named view with optional serialized token cost."""

        view_name = RecordViewName(view)
        if view_name not in self.views:
            raise ValueError(f"Record {self.record_id!r} has no {view_name.value!r} view.")
        value = self.views[view_name]
        tokens = token_counter(serialize_record(self, view=view_name)) if token_counter else value.token_count
        return replace(value, token_count=tokens)


def _payload_text(payload: Mapping[str, object] | str) -> str:
    return payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=True)


def serialize_record(
    record: ContextRecord,
    view: RecordViewName | str = RecordViewName.FULL,
) -> str:
    """Serialize one record with explicit identity/type/version delimiters."""

    view_name = RecordViewName(view)
    if view_name not in record.views:
        raise ValueError(f"Record {record.record_id!r} has no {view_name.value!r} view.")
    header_fields = {
        "id": record.record_id,
        "type": record.record_type.value,
        "version": record.version,
        "view": view_name.value,
    }
    if view_name == RecordViewName.FULL:
        header_fields["fingerprint"] = record.source_fingerprint
    header = json.dumps(header_fields, sort_keys=True)
    payload = record.views[view_name].payload
    return f"<<<PRA_RECORD {header}>>>\n{_payload_text(payload)}\n<<<END_PRA_RECORD {record.record_id}>>>"


def _tool_selection_signature(resource: AgentResource, schema: Mapping[str, object]) -> str:
    """Build a compact provider-neutral signature from typed parameters."""

    declared = resource.metadata.get("signature")
    if declared:
        return f"{resource.name}{declared}" if str(declared).startswith("(") else str(declared)
    function = schema.get("function", {})
    parameters = function.get("parameters", {}) if isinstance(function, Mapping) else {}
    properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    required = set(parameters.get("required", ())) if isinstance(parameters, Mapping) else set()
    rendered = []
    for name, value in properties.items() if isinstance(properties, Mapping) else ():
        type_name = value.get("type", "any") if isinstance(value, Mapping) else "any"
        rendered.append(f"{name}: {type_name}{'' if name in required else ' = optional'}")
    return f"{resource.name}({', '.join(rendered)})"


def tool_definition_record(
    resource: AgentResource,
    *,
    parent_id: str | None = None,
    selection_provenance: Mapping[str, object] | None = None,
) -> ContextRecord:
    """Preserve one complete provider-neutral tool schema as an atomic record."""

    schema = resource_tool_schema(resource)
    payload = {
        "uri": resource.uri,
        "version": resource.version,
        "schema": schema,
        "side_effect": resource.side_effect_class.value,
    }
    selection_payload = "\n".join((
        _tool_selection_signature(resource, schema),
        resource.description,
        f"Effect: {resource.side_effect_class.value}",
    ))
    text = json.dumps(payload, sort_keys=True)
    boundaries = []
    for name in ("uri", "version", "schema", "side_effect"):
        marker = json.dumps(name) + ":"
        start = text.find(marker)
        if start >= 0:
            next_starts = [text.find(json.dumps(other) + ":", start + len(marker)) for other in ("uri", "version", "schema", "side_effect")]
            next_starts = [value for value in next_starts if value >= 0]
            boundaries.append(RecordBoundary(name, start, min(next_starts) if next_starts else len(text)))
    return ContextRecord(
        record_id=resource.uri,
        record_type=RecordType.TOOL_DEFINITION,
        payload=payload,
        parent_id=parent_id,
        boundaries=tuple(boundaries),
        selection_provenance=dict(selection_provenance or {}),
        version=resource.version,
        source_fingerprint=hashlib.sha256(
            json.dumps(resource.fingerprint_payload(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        views={
            RecordViewName.SELECTION: RecordView(
                RecordViewName.SELECTION,
                selection_payload,
                ("name", "signature", "description", "side_effect"),
            ),
            RecordViewName.FULL: RecordView(
                RecordViewName.FULL,
                payload,
                ("uri", "version", "schema", "side_effect"),
            ),
        },
    )


def tool_catalog_slice_records(
    candidates: CandidateSet,
    resources: Sequence[AgentResource],
    *,
    slice_id: str,
    child_view: RecordViewName | str = RecordViewName.SELECTION,
) -> tuple[ContextRecord, tuple[ContextRecord, ...]]:
    """Convert one external candidate decision into an authoritative record tree."""

    child_view = RecordViewName(child_view)
    by_uri = {resource.uri: resource for resource in resources}
    children = []
    for uri in candidates.candidate_uris:
        provenance = candidates.provenance_for(uri)
        children.append(tool_definition_record(
            by_uri[uri],
            parent_id=slice_id,
            selection_provenance={
                "admission_source": provenance.admission_source,
                "admission_rank": provenance.admission_rank,
                "channel_hits": [asdict(row) for row in provenance.sources],
            },
        ))
    parent = ContextRecord(
        record_id=slice_id,
        record_type=RecordType.TOOL_CATALOG_SLICE,
        payload={
            "candidate_uris": list(candidates.candidate_uris),
            "mode": candidates.mode.value,
            "strategy": candidates.strategy.value,
            "max_candidates": candidates.max_candidates,
            "child_view": child_view.value,
        },
        child_ids=tuple(row.record_id for row in children),
        selection_provenance={"owner": "agent_resolver", "authoritative": True},
    )
    return parent, tuple(children)


def capability_slice_records(
    records: Sequence[ContextRecord],
    *,
    slice_id: str,
    selected_record_ids: Sequence[str] | None = None,
    child_view: RecordViewName | str = RecordViewName.SELECTION,
) -> tuple[ContextRecord, tuple[ContextRecord, ...]]:
    """Create an authoritative tool, skill, or mixed capability slice.

    The slice contains identities and view policy only. Full payloads remain in
    the backing records and are activated by exact identity after model choice.
    """

    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Capability record IDs must be unique.")
    selected_ids = tuple(selected_record_ids or by_id)
    if any(record_id not in by_id for record_id in selected_ids):
        raise ValueError("Every selected capability ID must resolve to a backing record.")
    selected = tuple(replace(by_id[record_id], parent_id=slice_id) for record_id in selected_ids)
    record_types = {record.record_type for record in selected}
    if not record_types <= {RecordType.TOOL_RECORD, RecordType.SKILL_RECORD}:
        raise ValueError("Capability slices accept only TOOL_RECORD and SKILL_RECORD children.")
    if record_types == {RecordType.TOOL_RECORD}:
        slice_type = RecordType.TOOL_CATALOG_SLICE
    elif record_types == {RecordType.SKILL_RECORD}:
        slice_type = RecordType.SKILL_CATALOG_SLICE
    else:
        slice_type = RecordType.CAPABILITY_SLICE
    child_view = RecordViewName(child_view)
    parent = ContextRecord(
        record_id=slice_id,
        record_type=slice_type,
        payload={
            "candidate_uris": list(selected_ids),
            "child_view": child_view.value,
            "capability_types": sorted(record_type.value for record_type in record_types),
        },
        child_ids=selected_ids,
        selection_provenance={"owner": "capability_runtime", "authoritative": True},
    )
    return parent, selected


class RecordBudgetExceeded(ValueError):
    """Raised when authoritative whole records cannot fit the declared budget."""


@dataclass(frozen=True)
class RecordMaterializationResult:
    """Auditable output of preserving or explicitly rejecting selected records."""

    status: str
    serialized_payload: str
    selected_record_ids: tuple[str, ...]
    materialized_record_ids: tuple[str, ...]
    records_selected: int
    records_materialized: int
    child_records_selected: int
    child_records_materialized: int
    record_coverage: float
    serialized_bytes: int
    serialized_tokens: int
    native_kv_bytes: int
    partial_record_count: int
    atomicity_violations: int
    upstream_selection_preserved: bool
    overflow_behavior: OverflowBehavior
    materialized_view: RecordViewName = RecordViewName.FULL


def materialize_authoritative_slice(
    parent: ContextRecord,
    children: Sequence[ContextRecord],
    *,
    max_bytes: int,
    overflow: OverflowBehavior | str = OverflowBehavior.REQUEST_NARROW,
    token_counter: Callable[[str], int] | None = None,
    native_kv_bytes_per_token: int = 0,
    temporary_max_bytes: int | None = None,
    view: RecordViewName | str | None = None,
) -> RecordMaterializationResult:
    """Materialize selected tools as whole records or return an explicit overflow."""

    overflow = OverflowBehavior(overflow)
    if parent.record_type not in {
        RecordType.TOOL_CATALOG_SLICE,
        RecordType.SKILL_CATALOG_SLICE,
        RecordType.CAPABILITY_SLICE,
    }:
        raise ValueError("Authoritative materialization requires a typed capability slice.")
    by_id = {row.record_id: row for row in children}
    selected = [by_id[record_id] for record_id in parent.child_ids if record_id in by_id]
    if len(selected) != len(parent.child_ids):
        raise ValueError("Every selected child ID must resolve to one ContextRecord.")
    for row in selected:
        if row.parent_id != parent.record_id:
            raise ValueError("Capability child parent_id does not match the catalog slice.")
        if row.policy.authority != SelectionAuthority.AUTHORITATIVE:
            raise ValueError("Capability children in an authoritative slice must remain authoritative.")
        if row.policy.atomicity != RecordAtomicity.RECORD or row.policy.materialization != MaterializationPolicy.FULL:
            raise ValueError("Authoritative capability children must use full record atomicity.")
    selected_view = RecordViewName(
        view or parent.payload.get("child_view", parent.policy.initial_view.value)
    )
    serialized = [serialize_record(row, view=selected_view) for row in selected]
    total = sum(len(value.encode("utf-8")) for value in serialized)
    effective_limit = max_bytes
    if total > max_bytes and overflow == OverflowBehavior.EXPAND_TEMPORARILY:
        if temporary_max_bytes is None or total > temporary_max_bytes:
            raise RecordBudgetExceeded("Temporary materialization budget cannot fit authoritative records.")
        effective_limit = temporary_max_bytes
    if total > effective_limit and overflow == OverflowBehavior.ERROR:
        raise RecordBudgetExceeded("Authoritative tool slice exceeds the materialization budget.")
    admitted = list(selected)
    status = "materialized"
    if total > effective_limit and overflow == OverflowBehavior.REQUEST_NARROW:
        admitted = []
        status = "narrow_required"
    elif total > effective_limit and overflow == OverflowBehavior.DROP_WHOLE_RECORDS:
        admitted = []
        used = 0
        for row, text in zip(selected, serialized):
            size = len(text.encode("utf-8"))
            if used + size <= effective_limit:
                admitted.append(row)
                used += size
        status = "whole_records_dropped"
    admitted_text = [serialize_record(row, view=selected_view) for row in admitted]
    payload = "\n".join(admitted_text)
    byte_count = len(payload.encode("utf-8"))
    token_count = token_counter(payload) if token_counter is not None and payload else 0
    return RecordMaterializationResult(
        status=status,
        serialized_payload=payload,
        selected_record_ids=tuple(row.record_id for row in selected),
        materialized_record_ids=tuple(row.record_id for row in admitted),
        records_selected=1 + len(selected),
        records_materialized=(1 if admitted else 0) + len(admitted),
        child_records_selected=len(selected),
        child_records_materialized=len(admitted),
        record_coverage=len(admitted) / max(len(selected), 1),
        serialized_bytes=byte_count,
        serialized_tokens=token_count,
        native_kv_bytes=token_count * native_kv_bytes_per_token,
        partial_record_count=0,
        atomicity_violations=0,
        upstream_selection_preserved=len(admitted) == len(selected),
        overflow_behavior=overflow,
        materialized_view=selected_view,
    )


def _future_record(record_id: str, record_type: RecordType, payload: Mapping[str, object], *, parent_id: str | None = None) -> ContextRecord:
    return ContextRecord(record_id, record_type, dict(payload), parent_id=parent_id)


def tool_response_record(record_id: str, *, producer_tool_uri: str, call_id: str, result_schema: Mapping[str, object], timestamp: str, payload: object) -> ContextRecord:
    return _future_record(record_id, RecordType.TOOL_RESPONSE, {"producer_tool_uri": producer_tool_uri, "call_id": call_id, "result_schema": dict(result_schema), "timestamp": timestamp, "payload": payload})


def log_block_record(record_id: str, *, source: str, severity: str, time_range: tuple[str, str], events: Sequence[str]) -> ContextRecord:
    return _future_record(record_id, RecordType.LOG_BLOCK, {"source": source, "severity": severity, "time_range": list(time_range), "events": list(events)})


def terminal_output_record(record_id: str, *, command: str, exit_status: int, stdout: str, stderr: str, working_directory: str) -> ContextRecord:
    return _future_record(record_id, RecordType.TERMINAL_OUTPUT, {"command": command, "exit_status": exit_status, "stdout": stdout, "stderr": stderr, "working_directory": working_directory})


def db_result_record(record_id: str, *, query_id: str, columns: Sequence[str], rows: Sequence[Sequence[object]], source: str) -> ContextRecord:
    return _future_record(record_id, RecordType.DB_RESULT, {"query_id": query_id, "columns": list(columns), "rows": [list(row) for row in rows], "row_count": len(rows), "source": source})


def rag_chunk_record(record_id: str, *, document_uri: str, chunk_id: str, source_offsets: tuple[int, int], retrieval_score: float, text: str, metadata: Mapping[str, object] | None = None) -> ContextRecord:
    return _future_record(record_id, RecordType.RAG_CHUNK, {"document_uri": document_uri, "chunk_id": chunk_id, "source_offsets": list(source_offsets), "retrieval_score": retrieval_score, "text": text, "metadata": dict(metadata or {})})


def unsafe_partial_tool_control(record: ContextRecord, *, keep_fields: Sequence[str]) -> ContextRecord:
    """Construct an explicitly marked partial-tool control for E5/E6 only."""

    if record.record_type != RecordType.TOOL_DEFINITION or not isinstance(record.payload, Mapping):
        raise ValueError("Partial-tool controls require a tool definition record.")
    payload = {name: value for name, value in record.payload.items() if name in set(keep_fields)}
    policy = RecordPolicy(
        RecordType.TOOL_DEFINITION,
        RecordSelectionPolicy.AUTHORITATIVE_PARENT,
        SelectionAuthority.ADVISORY,
        RecordAtomicity.FIELD,
        MaterializationPolicy.FIELDS,
        allow_partial_tools=True,
    )
    return ContextRecord(
        record_id=record.record_id + "#partial-control",
        record_type=record.record_type,
        payload=payload,
        parent_id=record.parent_id,
        selection_provenance={**record.selection_provenance, "experimental_partial_control": True},
        policy=policy,
        version=record.version,
    )
