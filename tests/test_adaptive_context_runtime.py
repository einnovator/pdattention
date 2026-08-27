import pytest

from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorPolicy,
    DeploymentTopology,
    MaterializationEvent,
    StoragePolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import RecordType, RecordViewName
from pra_hf.context_store import LocalBackingStore, RecordAccessDenied, RecordScope


def _runtime(tmp_path):
    policy = ContextPolicy(
        storage="adaptive",
        local_store=tmp_path,
        topology=DeploymentTopology.REMOTE_MODEL,
        upfront_max_bytes=100,
        adaptive_reuse_max_bytes=500,
        native_kv_bytes_per_token=16,
        cursor_policy=CursorPolicy(page_size=2, max_page_size=4),
        record_policies={RecordType.DB_RESULT: TypeContextPolicy(unit_limit=2)},
        persistent_store=True,
    )
    return AdaptiveContextRuntime(RecordScope("tenant", "session"), policy)


def test_adaptive_remote_transport_fetches_large_payload_once(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest(
        {"text": "x" * 1000, "trigger": "deploy_canary"},
        record_type=RecordType.TOOL_RESPONSE,
    )

    assert runtime.decisions[record.record_id].storage == StoragePolicy.ON_DEMAND
    first = runtime.materialize(MaterializationEvent(record.record_id))
    second = runtime.retrieve_record(record.record_id)

    assert first.network_bytes == first.payload_bytes
    assert first.round_trips == 1
    assert not first.cache_hit
    assert second.network_bytes == 0
    assert second.cache_hit
    assert runtime.accounting().cache_hits == 1


def test_address_search_repairs_hidden_trigger_and_scope_is_enforced(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest(
        {"message": "ordinary result", "next_action": "ROTATE_CREDENTIAL_Z91"},
        record_type=RecordType.API_RESULT,
    )

    assert runtime.search_records("rotate_credential_z91") == (record,)
    with pytest.raises(RecordAccessDenied):
        runtime.retrieve_record(
            record.record_id, scope=RecordScope("tenant", "other-session")
        )
    assert runtime.audit_events[-1]["action"] == "materialize_denied"


def test_cursor_pages_filters_aggregates_and_closes(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest(
        {
            "columns": ["group", "value", "note"],
            "rows": [
                {"group": "a", "value": 1, "note": "normal"},
                {"group": "b", "value": 5, "note": "inspect"},
                {"group": "a", "value": 3, "note": "normal"},
                {"group": "b", "value": 9, "note": "inspect anomaly"},
                {"group": "b", "value": 7, "note": "inspect"},
            ],
        },
        record_type=RecordType.DB_RESULT,
    )
    cursor = runtime.open_cursor(record.record_id, filters={"group": "b"})

    first = runtime.fetch_cursor(cursor.cursor_id)
    second = runtime.fetch_cursor(cursor.cursor_id)

    assert [row["value"] for row in first.items] == [5, 9]
    assert [row["value"] for row in second.items] == [7]
    assert runtime.cursors.search(cursor.cursor_id, "anomaly", scope=runtime.scope) == (
        {"group": "b", "value": 9, "note": "inspect anomaly"},
    )
    assert runtime.cursors.aggregate(cursor.cursor_id, "value", scope=runtime.scope)["mean"] == 7
    assert len(runtime.cursors.materialize_fields(
        cursor.cursor_id, ("value",), scope=runtime.scope
    )) <= runtime.policy.cursor_policy.max_page_size
    runtime.cursors.close(cursor.cursor_id, scope=runtime.scope)
    with pytest.raises(KeyError):
        runtime.fetch_cursor(cursor.cursor_id)
