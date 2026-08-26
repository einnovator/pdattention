import pytest

from pra_hf.context_records import RecordType, RecordViewName
from pra_hf.context_store import LocalBackingStore, RecordScope
from pra_hf.typed_context import CompressorRegistry, create_adaptive_record


def test_database_compaction_keeps_schema_stats_and_exact_backing(tmp_path):
    payload = {
        "columns": ["id", "latency_ms", "status"],
        "rows": [[index, index * 2, "ok"] for index in range(20)]
        + [[999, 9001, "RARE_TRIGGER_X7"]],
        "source": "analytics",
    }
    scope = RecordScope("tenant", "db-session")
    store = LocalBackingStore(tmp_path, persistent=True)
    record = create_adaptive_record(
        payload,
        record_type=RecordType.DB_RESULT,
        store=store,
        scope=scope,
        unit_limit=4,
    )

    compact = record.compact_view()
    assert compact["row_count"] == 21
    assert compact["columns"] == payload["columns"]
    assert compact["numeric_statistics"]["latency_ms"]["max"] == 9001.0
    assert record.materialize(store, scope=scope) == payload
    assert record.materialize(
        store,
        scope=scope,
        level=RecordViewName.SELECTED,
        selector={"rows": [19, 21]},
    )["rows"] == payload["rows"][19:21]
    assert "rare_trigger_x7" in record.address_views()["lexical"]


def test_log_compactor_preserves_errors_before_representative_lines():
    payload = {"source": "worker", "events": [f"INFO item {i}" for i in range(30)]}
    payload["events"][17] = "FATAL payment queue denied RARE_ACTION_77"

    result = CompressorRegistry().compress(RecordType.LOG_BLOCK, payload, unit_limit=4)

    assert "FATAL payment queue denied RARE_ACTION_77" in result.compact_payload["selected_lines"]
    assert result.compact_payload["error_counts"] == {"fatal": 1}
    assert result.lossy


def test_selected_materialization_rejects_invalid_selectors(tmp_path):
    scope = RecordScope("tenant", "session")
    store = LocalBackingStore(tmp_path, persistent=True)
    record = create_adaptive_record(
        [1, 2, 3], record_type=RecordType.API_RESULT, store=store, scope=scope
    )

    with pytest.raises(ValueError, match="requires a selector"):
        record.materialize(store, scope=scope, level=RecordViewName.SELECTED)
    with pytest.raises(ValueError, match="outside payload length"):
        record.materialize(
            store,
            scope=scope,
            level=RecordViewName.SELECTED,
            selector={"items": [0, 9]},
        )
