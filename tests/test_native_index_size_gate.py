from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordAccessDenied, RecordScope
from pra_hf.progressive_context import (
    NativeIndexSizeLimitExceeded,
    NativeIndexState,
    ProgressiveContextRuntime,
)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


class _FakePRA:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.references = {}

    def add_reference(self, reference, *, text):
        tokens = len(text.split())
        self.references[reference] = text
        return SimpleNamespace(
            id=reference,
            uri=reference,
            tokens=tokens,
            chunks=max(1, (tokens + 7) // 8),
        )

    def remove_reference(self, handle):
        self.references.pop(handle.uri, None)

    def stats(self):
        tokens = sum(len(value.split()) for value in self.references.values())
        return {
            "routing_index_bytes": tokens * 8,
            "resident_detail_kv_bytes": tokens * 32,
        }


class _BatchedTokenizer(_Tokenizer):
    def __call__(self, text, add_special_tokens=False):
        encoded = super().__call__(text, add_special_tokens=add_special_tokens)
        return SimpleNamespace(input_ids=[encoded.input_ids])


def _progressive(tmp_path, policy, *, scope=None):
    runtime = AdaptiveContextRuntime(
        scope or RecordScope("tenant", "session"),
        policy,
    )
    return ProgressiveContextRuntime(runtime, pra_model=_FakePRA(), chunk_tokens=8)


def test_token_gate_skips_without_truncating_or_claiming_pra_native(tmp_path):
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=3),
    )
    record = progressive.ingest(
        "one two three four hidden-evidence",
        record_type=RecordType.TOOL_RESPONSE,
        provenance={"tool": "lookup", "tenant": "tenant"},
    )

    audit = progressive.prepare_native_index(record.record_id)

    assert audit.native_index_state == NativeIndexState.SKIPPED_SIZE_LIMIT
    assert audit.native_index_requested and not audit.native_index_built
    assert audit.native_index_tokens == 5
    assert record.record_id not in progressive.registry.backing_reference_handles
    with pytest.raises(NativeIndexSizeLimitExceeded):
        progressive.register_backing_record(record.record_id)
    summary = progressive.registry.documents[f"{record.record_id}/views/summary"]
    assert summary.payload["capabilities"]["partial_context"] is True
    assert summary.payload["capabilities"]["more_data_available"] is True
    assert summary.payload["capabilities"]["search_available"] is True
    assert summary.payload["native_index"]["native_index_state"] == "SKIPPED_SIZE_LIMIT"


def test_exact_token_boundary_builds_full_native_index(tmp_path):
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=4),
    )
    record = progressive.ingest("one two three four", record_type=RecordType.LOG_BLOCK)

    audit = progressive.prepare_native_index(record.record_id)

    assert audit.native_index_state == NativeIndexState.BUILT
    assert audit.native_index_built
    assert audit.native_index_tokens == 4
    assert progressive.registry.backing_reference_handles[record.record_id].tokens == 4


def test_token_gate_counts_inner_huggingface_batch_dimension(tmp_path):
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=3),
    )
    progressive.registry.pra_model.tokenizer = _BatchedTokenizer()
    record = progressive.ingest(
        "one two three four", record_type=RecordType.GENERIC_TEXT
    )

    audit = progressive.prepare_native_index(record.record_id)

    assert audit.native_index_tokens == 4
    assert audit.native_index_state == NativeIndexState.SKIPPED_SIZE_LIMIT


def test_byte_gate_and_type_override_are_resolved_independently(tmp_path):
    policy = ContextPolicy(
        local_store=tmp_path,
        max_native_index_bytes=10_000,
        record_policies={
            RecordType.DB_RESULT: TypeContextPolicy(max_native_index_bytes=32),
            RecordType.LOG_BLOCK: TypeContextPolicy(
                override_native_index_limits=True,
                max_native_index_tokens=None,
                max_native_index_bytes=None,
            ),
        },
    )
    progressive = _progressive(tmp_path, policy)
    db = progressive.ingest(
        {"rows": [{"value": "x" * 64}]}, record_type=RecordType.DB_RESULT
    )
    log = progressive.ingest("x" * 128, record_type=RecordType.LOG_BLOCK)

    assert progressive.prepare_native_index(db.record_id).native_index_state == (
        NativeIndexState.SKIPPED_SIZE_LIMIT
    )
    assert progressive.prepare_native_index(log.record_id).native_index_built


def test_cheap_search_and_cursor_remain_available_after_gate(tmp_path):
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=1),
    )
    record = progressive.ingest(
        {
            "columns": ["account", "status"],
            "rows": [
                {"account": "A", "status": "normal"},
                {"account": "B", "status": "hidden-anomaly"},
            ],
        },
        record_type=RecordType.DB_RESULT,
    )
    progressive.prepare_native_index(record.record_id)

    assert progressive.runtime.search_records("hidden-anomaly") == (record,)
    result = progressive.search_record(record.record_id, "hidden-anomaly")
    cursor = progressive.runtime.open_cursor(record.record_id, collection="rows")

    assert result.success and result.payload["matches"][0]["account"] == "B"
    assert progressive.runtime.fetch_cursor(cursor.cursor_id).items


def test_lazy_region_encoding_preserves_scope_identity_and_audit(tmp_path):
    scope = RecordScope("tenant", "authorized-session")
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, max_native_index_tokens=1),
        scope=scope,
    )
    record = progressive.ingest(
        {
            "rows": [
                {"id": 1, "value": "ordinary"},
                {"id": 2, "value": "ANSWER_CODE=ZX-7"},
                {"id": 3, "value": "ordinary"},
            ]
        },
        record_type=RecordType.DB_RESULT,
        provenance={"source": "authorized-tool"},
    )
    progressive.prepare_native_index(record.record_id)

    region = progressive.encode_selected_region_native(
        record.record_id, {"rows": [1, 2]}
    )
    audit = progressive.registry.native_index_audits[record.record_id]

    assert region.record_id == record.record_id
    assert region.native_tokens > 0
    assert region.reference_uri.endswith("/views/lazy-native-1")
    assert audit.native_index_state == NativeIndexState.SKIPPED_SIZE_LIMIT
    assert audit.lazy_native_regions_encoded == 1
    assert audit.lazy_native_tokens == region.native_tokens
    assert progressive.runtime.audit_events[-1]["source_fingerprint"] == (
        record.backing.content_hash
    )
    with pytest.raises(RecordAccessDenied):
        record.materialize(
            progressive.runtime.store,
            scope=RecordScope("tenant", "other-session"),
            level="selected",
            selector={"rows": [1, 2]},
        )
    with pytest.raises(RecordAccessDenied):
        progressive.runtime.search_record(
            record.record_id,
            "ZX-7",
            scope=RecordScope("tenant", "other-session"),
        )


def test_deferred_index_registers_compact_view_and_can_be_forced(tmp_path):
    progressive = _progressive(
        tmp_path,
        ContextPolicy(local_store=tmp_path, defer_native_index=True),
    )
    record = progressive.ingest("alpha beta gamma", record_type=RecordType.RAG_RESULT)

    deferred = progressive.prepare_native_index(record.record_id)
    built = progressive.prepare_native_index(record.record_id, force=True)

    assert deferred.native_index_state == NativeIndexState.DEFERRED
    assert built.native_index_state == NativeIndexState.BUILT
    assert f"{record.record_id}/views/summary" in progressive.registry.documents
