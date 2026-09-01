from experiments.paper4_5_runtime.run_session_aware_selected_context import run


def test_session_aware_selected_context_avoids_only_compatible_visible_reuse() -> None:
    payload = run(seeds=(11,))
    rows = {row["condition"]: row for row in payload["aggregates"]}

    unaware = rows["selected_context_without_logical_reuse"]
    aware = rows["session_aware_selected_context"]
    native = rows["native_memory"]
    assert unaware["new_materialized_tokens"] == 28
    assert aware["new_materialized_tokens"] == 12
    assert aware["logical_reuse_tokens"] == 16
    assert aware["visible_tokens"] == 72
    assert native["native_reuse_tokens"] is None
    assert native["measurement_status"] == "NOT_MEASURED"
