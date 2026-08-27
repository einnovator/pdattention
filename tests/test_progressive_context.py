from __future__ import annotations

import pytest

from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorPolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordScope
from pra_hf.progressive_context import (
    ControllerConfig,
    ControllerDescriptionLevel,
    ControllerProtocol,
    ContextAction,
    ContextDecision,
    PRAViewKind,
    ProgressiveContextRuntime,
    RecordCapabilities,
    parse_context_decision,
)


def _progressive(tmp_path):
    runtime = AdaptiveContextRuntime(
        RecordScope("paper7", "progressive-test"),
        ContextPolicy(
            local_store=tmp_path,
            cursor_policy=CursorPolicy(page_size=2, max_page_size=8),
            persistent_store=True,
        ),
    )
    return ProgressiveContextRuntime(runtime, chunk_tokens=12)


def _db_payload():
    return {
        "columns": ["account", "status", "amount"],
        "rows": [
            {"account": "A", "status": "settled", "amount": 10},
            {"account": "B", "status": "failed", "amount": 90},
            {"account": "C", "status": "settled", "amount": 30},
            {"account": "D", "status": "pending", "amount": 50},
        ],
    }


def test_compact_record_is_stable_implicit_pra_document(tmp_path):
    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        _db_payload(),
        record_type=RecordType.DB_RESULT,
        provenance={"tool": "ledger.query", "security_scope": "billing-read"},
        capabilities=RecordCapabilities(
            searchable=True,
            partial_selectors=("rows", "fields"),
        ),
    )

    uri = progressive.registry.views_by_record[record.record_id][0]
    document = progressive.registry.documents[uri]

    assert uri == f"{record.record_id}/views/summary"
    assert document.view == PRAViewKind.SUMMARY
    assert document.record_id == record.record_id
    assert document.scope_fingerprint == progressive.runtime.scope.fingerprint
    assert document.provenance["tool"] == "ledger.query"
    assert document.chunks
    assert document.chunks[0].document_uri == uri
    assert document.chunks[0].record_id == record.record_id
    assert document.payload["capabilities"]["search_available"] is True


def test_bounded_decision_parser_rejects_unregistered_identities():
    decision = parse_context_decision(
        {"context_action": "MATERIALIZE_FULL", "record_id": "R17"},
        allowed_record_ids=("R17",),
    )
    assert decision.action == ContextAction.MATERIALIZE_FULL

    with pytest.raises(ValueError, match="unauthorized"):
        parse_context_decision(
            {"context_action": "MATERIALIZE_FULL", "record_id": "R18"},
            allowed_record_ids=("R17",),
        )
    tool = parse_context_decision(
        {"context_action": "CALL_TOOL", "tool_name": "billing.lookup"},
        allowed_tools=("billing.lookup",),
    )
    assert tool.tool_name == "billing.lookup"
    with pytest.raises(ValueError, match="unauthorized"):
        parse_context_decision(
            {"context_action": "CALL_TOOL", "tool_name": "billing.delete"}
        )
    with pytest.raises(ValueError, match="requires a bounded"):
        ContextDecision(ContextAction.MATERIALIZE_MORE, record_id="R17")
    with pytest.raises(ValueError, match="cannot carry"):
        ContextDecision(ContextAction.CONTINUE, record_id="R17")


def test_full_partial_and_search_views_reenter_pra(tmp_path):
    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        _db_payload(),
        record_type=RecordType.DB_RESULT,
        capabilities=RecordCapabilities(
            searchable=True,
            partial_selectors=("rows",),
        ),
    )

    more = progressive.materialize_more(record.record_id, {"rows": [1, 2]})
    search = progressive.search_record(record.record_id, "failed account")
    full = progressive.materialize_full(record.record_id)

    assert more.success and more.produced_document.view == PRAViewKind.DETAIL
    assert more.payload["rows"][0]["account"] == "B"
    assert search.success and search.produced_document.view == PRAViewKind.SEARCH_RESULT
    assert search.payload["matches"][0]["status"] == "failed"
    assert full.success and full.produced_document.view == PRAViewKind.FULL
    assert all(
        document.record_id == record.record_id
        for document in progressive.registry.documents.values()
    )
    assert len(progressive.registry.views_by_record[record.record_id]) == 4


def test_cursor_result_reenters_pra_and_preserves_record_identity(tmp_path):
    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        _db_payload(), record_type=RecordType.DB_RESULT
    )
    cursor = progressive.runtime.open_cursor(record.record_id, collection="rows")
    assert progressive.runtime.cursors.cursor_ids == (cursor.cursor_id,)
    assert progressive.runtime.cursors.describe(
        cursor.cursor_id, scope=progressive.runtime.scope
    ) == cursor

    page = progressive.cursor_next(cursor.cursor_id)
    query = progressive.cursor_query(
        cursor.cursor_id,
        {"operation": "search", "query": "pending", "limit": 2},
    )

    assert page.success and len(page.payload.items) == 2
    assert page.produced_document.record_id == record.record_id
    assert page.produced_document.view == PRAViewKind.CURSOR_PAGE
    assert query.success and query.payload[0]["status"] == "pending"
    assert query.produced_document.parent_uri.endswith("/views/summary")


def test_automatic_selection_precedes_bounded_recursive_escalation(tmp_path):
    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        _db_payload(),
        record_type=RecordType.DB_RESULT,
        capabilities=RecordCapabilities(searchable=True),
    )
    decisions = iter((
        ContextDecision(
            ContextAction.SEARCH_RECORD,
            record_id=record.record_id,
            query="failed account",
        ),
        ContextDecision(ContextAction.CONTINUE),
    ))
    observed_view_counts = []

    def decide(documents, selection):
        observed_view_counts.append(len(documents))
        assert selection.compared_chunks >= 1
        return next(decisions)

    transitions = progressive.run_loop("failed account", decide, max_steps=3)

    assert [row.decision.action for row in transitions] == [
        ContextAction.SEARCH_RECORD,
        ContextAction.CONTINUE,
    ]
    assert observed_view_counts == [1, 2]


def test_real_pra_binding_uses_add_reference_for_every_view(tmp_path):
    class FakePRA:
        def __init__(self):
            self.added = []
            self.removed = []

        def add_reference(self, reference, *, text):
            self.added.append((reference, text))
            return type("Handle", (), {"id": reference, "uri": reference})()

        def remove_reference(self, handle):
            self.removed.append(handle.uri)

    progressive = _progressive(tmp_path)
    record = progressive.ingest(_db_payload(), record_type=RecordType.DB_RESULT)
    model = FakePRA()

    progressive.registry.bind_model(model)
    progressive.materialize_more(record.record_id, {"rows": [0, 1]})

    assert [uri for uri, _ in model.added] == [
        f"{record.record_id}/views/summary",
        f"{record.record_id}/views/detail-1",
    ]


def test_controller_descriptions_are_fixed_visible_and_fingerprinted():
    minimal = ControllerConfig("qwen3:0.6b", ControllerDescriptionLevel.D0_MINIMAL)
    rich = ControllerConfig(
        "gemma3:4b", ControllerDescriptionLevel.D2_RICH,
        protocol=ControllerProtocol.HIERARCHICAL,
    )

    for action in ContextAction:
        assert action.value in minimal.description_block
        assert action.value in rich.description_block
    assert len(rich.description_block) > len(minimal.description_block)
    assert rich.fingerprint == ControllerConfig(
        "gemma3:4b", "D2", protocol="hierarchical"
    ).fingerprint


def test_backing_reference_is_retrieval_only_and_preserves_exact_identity(tmp_path):
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Encoded", (), {"input_ids": list(range(len(text.split())))})()

        def decode(self, values, skip_special_tokens=True):
            return " ".join(f"token-{value}" for value in values)

    class FakePRA:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.references = {}

        def add_reference(self, reference, *, text):
            self.references[reference] = text
            return type("Handle", (), {
                "id": reference, "uri": reference,
                "tokens": len(text.split()), "chunks": 2,
            })()

        def remove_reference(self, handle):
            self.references.pop(handle.uri)

        def stats(self):
            return {"routing_index_bytes": 64, "resident_detail_kv_bytes": 512}

        def route(self, query):
            uri = next(iter(self.references))
            selected = ({
                "reference_uri": uri, "chunk_id": f"{uri}#0",
                "logical_start": 0, "logical_end": 4,
            },)
            return type("Routing", (), {
                "selected": selected,
                "stats": {"selection_policy": "top_k"},
            })()

    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        "alpha beta exact evidence omega", record_type=RecordType.GENERIC_TEXT
    )
    visible_uris = tuple(progressive.registry.documents)
    model = FakePRA()
    progressive.registry.pra_model = model

    handle = progressive.register_backing_record(record.record_id)
    selection = progressive.native_select("exact evidence")
    result = progressive.materialize_native_selection(selection)

    assert handle.uri.endswith("/views/backing")
    assert tuple(progressive.registry.documents)[:len(visible_uris)] == visible_uris
    assert handle.uri not in progressive.registry.documents
    assert selection.record_ids == (record.record_id,)
    assert result.success
    assert result.produced_document.record_id == record.record_id
    assert result.payload["selected_chunks"][0]["logical_start"] == 0


def test_frozen_native_chunks_replay_through_typed_materialization(tmp_path):
    progressive = _progressive(tmp_path)
    record = progressive.ingest(
        "alpha beta exact evidence omega", record_type=RecordType.GENERIC_TEXT
    )
    chunks = ({
        "chunk_id": "native-7",
        "logical_start": 7,
        "logical_end": 11,
        "text": "exact evidence ANSWER_CODE=ZX-7",
    },)

    result = progressive.materialize_backing_chunks(
        record.record_id,
        "find ZX-7",
        chunks,
        selection_policy="PRA_NATIVE_HYBRID",
    )

    assert result.success
    assert result.produced_document is not None
    assert result.produced_document.record_id == record.record_id
    assert result.payload["selected_chunks"][0]["text"].endswith("ZX-7")
    assert result.payload_bytes > 0
