"""Typed compact views and exact selective materialization for agent results."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from .context_records import RecordType, RecordViewName
from .context_store import BackingRecord, LocalBackingStore, RecordScope


_TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]+")
_ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*|[A-Z]{2,})\b")
_ERROR = re.compile(r"\b(error|exception|failed|failure|fatal|panic|denied|timeout)\b", re.I)


class AddressViewKind(str, Enum):
    """Independent indices used to rediscover compacted backing state."""

    LEXICAL = "lexical"
    ENTITY = "entity"
    RARE_TERM = "rare_term"
    SCHEMA = "schema"
    SUMMARY = "summary"
    DENSE = "dense"
    NATIVE_QK = "native_qk"


@dataclass(frozen=True)
class AddressView:
    """One non-prompt retrieval representation for a backing record."""

    kind: AddressViewKind | str
    value: object
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AddressViewKind(self.kind))


@dataclass(frozen=True)
class CompressionResult:
    """Compact prompt view plus orthogonal retrieval addresses and accounting."""

    compact_payload: object
    address_views: tuple[AddressView, ...]
    strategy: str
    original_bytes: int
    compact_bytes: int
    lossy: bool
    retained_units: int
    total_units: int

    @property
    def byte_savings_fraction(self) -> float:
        return 1.0 - self.compact_bytes / max(self.original_bytes, 1)


@dataclass(frozen=True)
class AdaptiveContextRecord:
    """Prompt-safe descriptor joining compact and address views to exact state."""

    backing: BackingRecord
    compact_payload: object
    address_index: tuple[AddressView, ...]
    compression_strategy: str
    compact_bytes: int
    retained_units: int
    total_units: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "address_index", tuple(self.address_index))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def record_id(self) -> str:
        return self.backing.record_id

    @property
    def record_type(self) -> RecordType:
        return self.backing.record_type

    def compact_view(self) -> object:
        """Return the bounded payload intended for initial model visibility."""

        return self.compact_payload

    def address_views(self) -> Mapping[str, object]:
        """Return retrieval-only views keyed by index kind."""

        return {view.kind.value: view.value for view in self.address_index}

    def materialize(
        self,
        store: LocalBackingStore,
        *,
        scope: RecordScope,
        level: RecordViewName | str = RecordViewName.FULL,
        selector: Mapping[str, object] | None = None,
    ) -> object:
        """Resolve metadata, compact, selected, or exact full state.

        Selection is deterministic and is applied only after the store has
        authorized the record scope and hash-verified its full payload.
        """

        view = RecordViewName(level)
        if view == RecordViewName.METADATA:
            return {
                "record_id": self.record_id,
                "record_type": self.record_type.value,
                "size_bytes": self.backing.size_bytes,
                "content_hash": self.backing.content_hash,
                "provenance": dict(self.backing.provenance),
                **dict(self.metadata),
            }
        if view in {RecordViewName.COMPACT, RecordViewName.SELECTION}:
            return self.compact_payload
        payload = store.get(self.record_id, scope=scope)
        if view == RecordViewName.FULL:
            if selector:
                raise ValueError("selector is valid only for selected materialization.")
            return payload
        if view == RecordViewName.SELECTED:
            if not selector:
                raise ValueError("selected materialization requires a selector.")
            return select_payload(payload, selector)
        raise ValueError(f"Unsupported materialization level: {view.value}")


Compressor = Callable[[object, int, Callable[[str], str] | None], CompressionResult]


def _encoded_bytes(value: object) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _address_views(value: object, summary: str | None = None) -> tuple[AddressView, ...]:
    text = _text(value)
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    counts = Counter(tokens)
    lexical = tuple(dict.fromkeys(tokens))
    entities = tuple(dict.fromkeys(_ENTITY.findall(text)))
    rare = tuple(token for token in lexical if counts[token] == 1 and len(token) >= 5)
    schema: object = ()
    if isinstance(value, Mapping):
        schema = tuple(sorted(str(key) for key in value))
    views = [
        AddressView(AddressViewKind.LEXICAL, lexical, "exact-token-index"),
        AddressView(AddressViewKind.ENTITY, entities, "deterministic-entity-pattern"),
        AddressView(AddressViewKind.RARE_TERM, rare, "within-record-frequency"),
    ]
    if schema:
        views.append(AddressView(AddressViewKind.SCHEMA, schema, "structural-keys"))
    if summary:
        views.append(AddressView(AddressViewKind.SUMMARY, summary, "optional-summary"))
    return tuple(views)


def _finish(
    original: object,
    compact: object,
    *,
    strategy: str,
    retained: int,
    total: int,
    summary_fn: Callable[[str], str] | None,
) -> CompressionResult:
    summary = summary_fn(_text(original)) if summary_fn is not None else None
    return CompressionResult(
        compact_payload=compact,
        address_views=_address_views(original, summary),
        strategy=strategy,
        original_bytes=_encoded_bytes(original),
        compact_bytes=_encoded_bytes(compact),
        lossy=compact != original,
        retained_units=retained,
        total_units=total,
    )


def _representative_indices(length: int, limit: int) -> tuple[int, ...]:
    if length <= limit:
        return tuple(range(length))
    if limit <= 1:
        return (0,)
    return tuple(sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)}))


def _compress_db(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), Sequence):
        return _compress_structured(payload, limit, summary_fn)
    rows = list(payload["rows"])
    columns = list(payload.get("columns", ()))
    indices = _representative_indices(len(rows), limit)
    selected = [rows[index] for index in indices]
    numeric: dict[str, dict[str, float]] = {}
    for column_index, name in enumerate(columns):
        values = []
        for row in rows:
            try:
                value = row.get(name) if isinstance(row, Mapping) else row[column_index]
            except (IndexError, KeyError, TypeError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            numeric[str(name)] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    compact = {
        "columns": columns,
        "row_count": len(rows),
        "representative_row_indices": list(indices),
        "representative_rows": selected,
        "numeric_statistics": numeric,
        "has_more": len(selected) < len(rows),
    }
    return _finish(
        payload, compact, strategy="schema_representative_rows_stats",
        retained=len(selected), total=len(rows), summary_fn=summary_fn,
    )


def _log_lines(payload: object) -> tuple[list[str], dict[str, object]]:
    metadata: dict[str, object] = {}
    if isinstance(payload, Mapping):
        metadata = {str(key): value for key, value in payload.items() if key not in {"events", "lines", "text"}}
        source = payload.get("events", payload.get("lines", payload.get("text", "")))
    else:
        source = payload
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return [str(line) for line in source], metadata
    return _text(source).splitlines(), metadata


def _compress_log(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    lines, metadata = _log_lines(payload)
    important = [index for index, line in enumerate(lines) if _ERROR.search(line)]
    chosen = important[:limit]
    if len(chosen) < limit:
        for index in _representative_indices(len(lines), limit):
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= limit:
                break
    chosen.sort()
    severity = Counter(
        match.group(1).casefold()
        for line in lines
        for match in [_ERROR.search(line)]
        if match is not None
    )
    compact = {
        **metadata,
        "line_count": len(lines),
        "selected_line_indices": chosen,
        "selected_lines": [lines[index] for index in chosen],
        "error_counts": dict(severity),
        "has_more": len(chosen) < len(lines),
    }
    return _finish(
        payload, compact, strategy="error_preserving_log",
        retained=len(chosen), total=len(lines), summary_fn=summary_fn,
    )


def _compress_terminal(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    if not isinstance(payload, Mapping):
        return _compress_log(payload, limit, summary_fn)
    stdout = _text(payload.get("stdout", "")).splitlines()
    stderr = _text(payload.get("stderr", "")).splitlines()
    keep = max(1, limit // 2)
    compact = {
        "command": payload.get("command"),
        "exit_status": payload.get("exit_status"),
        "working_directory": payload.get("working_directory"),
        "stdout_line_count": len(stdout),
        "stdout_head": stdout[:keep],
        "stdout_tail": stdout[-keep:] if len(stdout) > keep else [],
        "stderr": stderr[:limit],
        "has_more": len(stdout) > 2 * keep or len(stderr) > limit,
    }
    retained = min(len(stdout), 2 * keep) + min(len(stderr), limit)
    return _finish(
        payload, compact, strategy="terminal_status_head_tail_errors",
        retained=retained, total=len(stdout) + len(stderr), summary_fn=summary_fn,
    )


def _compress_graph(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    if not isinstance(payload, Mapping):
        return _compress_structured(payload, limit, summary_fn)
    nodes = list(payload.get("nodes", ()))
    edges = list(payload.get("edges", ()))
    node_indices = _representative_indices(len(nodes), max(1, limit // 2))
    edge_indices = _representative_indices(len(edges), max(1, limit // 2))
    compact = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "representative_nodes": [nodes[index] for index in node_indices],
        "representative_edges": [edges[index] for index in edge_indices],
        "has_more": len(node_indices) < len(nodes) or len(edge_indices) < len(edges),
    }
    return _finish(
        payload, compact, strategy="graph_schema_representatives",
        retained=len(node_indices) + len(edge_indices), total=len(nodes) + len(edges),
        summary_fn=summary_fn,
    )


def _compress_text(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    lines = _text(payload).splitlines()
    indices = _representative_indices(len(lines), limit)
    compact = {
        "line_count": len(lines),
        "selected_line_indices": list(indices),
        "selected_lines": [lines[index] for index in indices],
        "has_more": len(indices) < len(lines),
    }
    return _finish(
        payload, compact, strategy="deterministic_text_sampling",
        retained=len(indices), total=len(lines), summary_fn=summary_fn,
    )


def _compress_structured(
    payload: object, limit: int, summary_fn: Callable[[str], str] | None
) -> CompressionResult:
    if isinstance(payload, Mapping):
        items = list(payload.items())
        retained = items[:limit]
        compact = {
            "fields": {str(key): value for key, value in retained},
            "field_count": len(items),
            "has_more": len(retained) < len(items),
        }
        return _finish(
            payload, compact, strategy="bounded_structural_projection",
            retained=len(retained), total=len(items), summary_fn=summary_fn,
        )
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
        indices = _representative_indices(len(values), limit)
        compact = {
            "item_count": len(values),
            "representative_indices": list(indices),
            "representative_items": [values[index] for index in indices],
            "has_more": len(indices) < len(values),
        }
        return _finish(
            payload, compact, strategy="deterministic_item_sampling",
            retained=len(indices), total=len(values), summary_fn=summary_fn,
        )
    return _compress_text(payload, limit, summary_fn)


class CompressorRegistry:
    """Record-type dispatch for deterministic, auditable compact views."""

    def __init__(self) -> None:
        self._compressors: dict[RecordType, Compressor] = {}
        self._default: Compressor = _compress_structured
        for record_type in (RecordType.DB_RESULT,):
            self.register(record_type, _compress_db)
        for record_type in (RecordType.LOG_BLOCK,):
            self.register(record_type, _compress_log)
        self.register(RecordType.TERMINAL_OUTPUT, _compress_terminal)
        self.register(RecordType.GRAPH_RESULT, _compress_graph)
        for record_type in (
            RecordType.RAG_RESULT,
            RecordType.RAG_CHUNK_SET,
            RecordType.RAG_CHUNK,
            RecordType.FILE_READ,
            RecordType.GENERIC_TEXT,
            RecordType.GENERIC_DOCUMENT,
        ):
            self.register(record_type, _compress_text)

    def register(self, record_type: RecordType | str, compressor: Compressor) -> None:
        self._compressors[RecordType(record_type)] = compressor

    def compress(
        self,
        record_type: RecordType | str,
        payload: object,
        *,
        unit_limit: int = 8,
        summary_fn: Callable[[str], str] | None = None,
    ) -> CompressionResult:
        if unit_limit <= 0:
            raise ValueError("unit_limit must be positive.")
        return self._compressors.get(RecordType(record_type), self._default)(
            payload, unit_limit, summary_fn
        )


def create_adaptive_record(
    payload: object,
    *,
    record_type: RecordType | str,
    store: LocalBackingStore,
    scope: RecordScope,
    registry: CompressorRegistry | None = None,
    unit_limit: int = 8,
    provenance: Mapping[str, object] | None = None,
    ttl_seconds: float | None = None,
    summary_fn: Callable[[str], str] | None = None,
) -> AdaptiveContextRecord:
    """Persist exact state, then build bounded visible and retrieval views."""

    descriptor = store.put(
        payload,
        record_type=record_type,
        scope=scope,
        provenance=provenance,
        ttl_seconds=ttl_seconds,
    )
    compressed = (registry or CompressorRegistry()).compress(
        record_type, payload, unit_limit=unit_limit, summary_fn=summary_fn
    )
    return AdaptiveContextRecord(
        backing=descriptor,
        compact_payload=compressed.compact_payload,
        address_index=compressed.address_views,
        compression_strategy=compressed.strategy,
        compact_bytes=compressed.compact_bytes,
        retained_units=compressed.retained_units,
        total_units=compressed.total_units,
        metadata={
            "lossy": compressed.lossy,
            "byte_savings_fraction": compressed.byte_savings_fraction,
        },
    )


def select_payload(payload: object, selector: Mapping[str, object]) -> object:
    """Apply a bounded deterministic field/range selector to exact state."""

    if "fields" in selector:
        if not isinstance(payload, Mapping):
            raise TypeError("fields selector requires a mapping payload.")
        fields = tuple(str(value) for value in selector["fields"])
        return {name: payload[name] for name in fields if name in payload}
    if "rows" in selector:
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("rows selector requires a sequence or mapping with rows.")
        start, stop = _range(selector["rows"], len(rows))
        selected = list(rows[start:stop])
        if isinstance(payload, Mapping):
            return {"columns": payload.get("columns", ()), "rows": selected, "range": [start, stop]}
        return selected
    if "lines" in selector:
        lines = _text(payload).splitlines()
        start, stop = _range(selector["lines"], len(lines))
        return "\n".join(lines[start:stop])
    if "items" in selector:
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("items selector requires a sequence.")
        start, stop = _range(selector["items"], len(payload))
        return list(payload[start:stop])
    raise ValueError("selector must define fields, rows, lines, or items.")


def _range(value: object, length: int) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("range selector must contain [start, stop].")
    start, stop = int(value[0]), int(value[1])
    if start < 0 or stop < start or stop > length:
        raise ValueError(f"range [{start}, {stop}] is outside payload length {length}.")
    return start, stop
