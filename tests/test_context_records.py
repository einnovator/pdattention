"""Tests for record-aware authoritative tool materialization."""

import pytest

from data.agent_workflows import realistic_tool_catalog
from pra_hf.context_records import (
    ContextRecord,
    MaterializationPolicy,
    OverflowBehavior,
    RecordAtomicity,
    RecordBudgetExceeded,
    RecordPolicy,
    RecordSelectionPolicy,
    RecordType,
    RecordViewName,
    SelectionAuthority,
    db_result_record,
    default_record_policy,
    log_block_record,
    materialize_authoritative_slice,
    rag_chunk_record,
    serialize_record,
    terminal_output_record,
    tool_catalog_slice_records,
    tool_response_record,
    unsafe_partial_tool_control,
)
from pra_hf.union_discovery import CandidateProvenance, CandidateSet, ChannelHit, ToolDiscoveryMode, UnionStrategy


def _slice(count=2):
    resources = realistic_tool_catalog()
    uris = tuple(row.uri for row in resources[:count])
    candidates = CandidateSet(
        ToolDiscoveryMode.UNION,
        UnionStrategy.DIVERSITY_UNION,
        uris,
        tuple(CandidateProvenance(uri, (ChannelHit("tags", rank, 1.0 / rank),), "tags", rank) for rank, uri in enumerate(uris, 1)),
        count,
    )
    parent, children = tool_catalog_slice_records(candidates, resources, slice_id="slice:test")
    return parent, children


def test_tool_slice_and_definition_defaults_are_authoritative_and_atomic() -> None:
    parent, children = _slice()

    assert parent.policy.selection == RecordSelectionPolicy.ALL_CHILDREN
    assert parent.policy.authority == SelectionAuthority.AUTHORITATIVE
    assert all(row.policy.atomicity == RecordAtomicity.RECORD for row in children)
    assert all(row.policy.materialization == MaterializationPolicy.FULL for row in children)


def test_selection_view_materialization_preserves_boundaries_and_upstream_selection() -> None:
    parent, children = _slice()
    result = materialize_authoritative_slice(
        parent,
        children,
        max_bytes=100_000,
        token_counter=lambda text: len(text.split()),
        native_kv_bytes_per_token=64,
    )

    assert result.status == "materialized"
    assert result.materialized_record_ids == parent.child_ids
    assert result.record_coverage == 1.0
    assert result.partial_record_count == result.atomicity_violations == 0
    assert result.upstream_selection_preserved
    assert result.materialized_view == RecordViewName.SELECTION
    assert result.native_kv_bytes == result.serialized_tokens * 64
    assert all("<<<PRA_RECORD" in serialize_record(row) for row in children)
    assert all(row.boundaries for row in children)


def test_budget_overflow_requests_narrowing_instead_of_partial_tools() -> None:
    parent, children = _slice()
    result = materialize_authoritative_slice(parent, children, max_bytes=10)

    assert result.status == "narrow_required"
    assert result.materialized_record_ids == ()
    assert result.partial_record_count == 0
    assert not result.upstream_selection_preserved


def test_budget_error_and_explicit_whole_record_drop() -> None:
    parent, children = _slice()
    with pytest.raises(RecordBudgetExceeded):
        materialize_authoritative_slice(parent, children, max_bytes=10, overflow=OverflowBehavior.ERROR)

    one_size = len(serialize_record(children[0], view="selection").encode("utf-8")) + 1
    dropped = materialize_authoritative_slice(
        parent, children, max_bytes=one_size, overflow=OverflowBehavior.DROP_WHOLE_RECORDS
    )
    assert dropped.child_records_materialized == 1
    assert dropped.partial_record_count == 0


def test_invalid_partial_tool_policy_requires_explicit_control_override() -> None:
    with pytest.raises(ValueError):
        RecordPolicy(
            RecordType.TOOL_DEFINITION,
            RecordSelectionPolicy.AUTHORITATIVE_PARENT,
            SelectionAuthority.AUTHORITATIVE,
            RecordAtomicity.FIELD,
            MaterializationPolicy.FIELDS,
        )
    _, children = _slice(1)
    partial = unsafe_partial_tool_control(children[0], keep_fields=("uri",))
    assert partial.policy.allow_partial_tools
    assert partial.selection_provenance["experimental_partial_control"]


def test_record_type_policy_mismatch_and_slice_policy_are_rejected() -> None:
    with pytest.raises(ValueError):
        ContextRecord(
            "bad",
            RecordType.LOG_BLOCK,
            {},
            policy=default_record_policy(RecordType.TOOL_DEFINITION),
        )
    with pytest.raises(ValueError):
        RecordPolicy(
            RecordType.TOOL_CATALOG_SLICE,
            RecordSelectionPolicy.TOP_K_CHILDREN,
            SelectionAuthority.AUTHORITATIVE,
            RecordAtomicity.RECORD,
            MaterializationPolicy.FULL,
        )


def test_future_record_adapters_are_constructible_and_serializable() -> None:
    records = (
        tool_response_record("response:1", producer_tool_uri="tool:1", call_id="call:1", result_schema={}, timestamp="2026-01-01T00:00:00Z", payload={"ok": True}),
        log_block_record("log:1", source="worker", severity="info", time_range=("a", "b"), events=("ready",)),
        terminal_output_record("terminal:1", command="echo ok", exit_status=0, stdout="ok", stderr="", working_directory="/tmp"),
        db_result_record("db:1", query_id="q1", columns=("id",), rows=((1,),), source="memory"),
        rag_chunk_record("rag:1", document_uri="doc:1", chunk_id="c1", source_offsets=(0, 10), retrieval_score=0.9, text="evidence"),
    )

    assert {row.record_type for row in records} == {
        RecordType.TOOL_RESPONSE,
        RecordType.LOG_BLOCK,
        RecordType.TERMINAL_OUTPUT,
        RecordType.DB_RESULT,
        RecordType.RAG_CHUNK,
    }
    assert all("<<<END_PRA_RECORD" in serialize_record(row) for row in records)
