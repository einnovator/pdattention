from dataclasses import replace
import json

import pytest

from data.latent_trigger_cases import latent_trigger_cases

from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorAction,
    CursorOperation,
    CursorPolicy,
    DeploymentTopology,
    MaterializationEvent,
    StoragePolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_recovery import (
    CCRStyleBaseline,
    LatentRecoveryEngine,
    TriggerCase,
    action_conditioned_probes,
    build_action_probe,
    parse_cursor_action,
    trigger_case_fingerprint,
)
from pra_hf.context_store import RecordAccessDenied, RecordScope


def _case() -> TriggerCase:
    return TriggerCase(
        case_id="backend-legacy",
        family="backend_type",
        record_type=RecordType.GENERIC_TEXT,
        payload="storage_backend=glacier-legacy requires inspect_legacy_archive",
        hidden_trigger="glacier-legacy",
        hidden_evidence="legacy archives require inspection before migration",
        explicit_query="What should I do with glacier-legacy storage?",
        latent_query="Migrate this stored object safely.",
        candidate_actions=("migrate_standard", "inspect_legacy_archive", "manual_review"),
        required_action="inspect_legacy_archive",
        misleading_action="migrate_standard",
    )


def _runtime(tmp_path, *, proactive=False, topology="same_process"):
    return AdaptiveContextRuntime(
        RecordScope("tenant", "session"),
        ContextPolicy(
            local_store=tmp_path,
            allow_proactive_expansion=proactive,
            topology=topology,
            cursor_policy=CursorPolicy(page_size=2, max_page_size=8),
        ),
    )


def test_trigger_cases_are_paired_and_fingerprint_is_deterministic():
    case = _case()
    assert case.hidden_trigger in case.explicit_query
    assert case.hidden_trigger not in case.latent_query
    assert trigger_case_fingerprint([case]) == trigger_case_fingerprint([_case()])
    with pytest.raises(ValueError, match="latent queries"):
        TriggerCase(**{**case.__dict__, "latent_query": case.explicit_query})


def test_trigger_benchmark_is_deterministic_and_compact_views_omit_triggers(tmp_path):
    cases = latent_trigger_cases()
    assert len(cases) == 13
    assert len({case.family for case in cases}) == 13
    assert trigger_case_fingerprint(cases) == trigger_case_fingerprint(latent_trigger_cases())
    runtime = AdaptiveContextRuntime(
        RecordScope("tenant", "session"),
        ContextPolicy(local_store=tmp_path, record_policies={
            record_type: {"unit_limit": 3} for record_type in RecordType
        }),
    )
    for case in cases:
        record = runtime.ingest(case.payload, record_type=case.record_type)
        compact = json.dumps(record.compact_view(), sort_keys=True, default=str)
        assert case.hidden_trigger.casefold() not in compact.casefold(), case.case_id


def test_action_conditioned_probes_are_bounded_and_name_the_action():
    probe = build_action_probe("migrate safely", "inspect_legacy_archive")
    assert "inspect legacy archive" in probe.text
    probes = action_conditioned_probes("migrate safely", _case().candidate_actions, limit=2)
    assert len(probes) == 2
    assert [probe.action for probe in probes] == ["migrate_standard", "inspect_legacy_archive"]


def test_ccr_handle_recovers_exact_backing_and_enforces_scope(tmp_path):
    runtime = _runtime(tmp_path)
    case = _case()
    record = runtime.ingest(case.payload, record_type=case.record_type)
    ccr = CCRStyleBaseline(runtime)
    handle = ccr.register(record.record_id)
    assert ccr.retrieve(handle.marker).payload == case.payload
    with pytest.raises(RecordAccessDenied):
        ccr.retrieve(handle.marker, scope=RecordScope("other", "session"))


def test_proactive_expansion_is_deny_by_default(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest(_case().payload, record_type=RecordType.GENERIC_TEXT)
    with pytest.raises(RecordAccessDenied, match="disabled"):
        runtime.proactive_materialize(MaterializationEvent(record.record_id), reason="test")
    assert runtime.audit_events[-1]["action"] == "proactive_materialize_denied"


def test_probe_false_positive_accounting(tmp_path):
    runtime = _runtime(tmp_path, proactive=True)
    case = _case()
    expected = runtime.ingest(case.payload, record_type=case.record_type)
    runtime.ingest(
        "inspect legacy archive documentation but storage_backend=standard",
        record_type=RecordType.GENERIC_TEXT,
        provenance={"distractor": True},
    )
    probes = (build_action_probe(case.latent_query, case.required_action),)
    result = LatentRecoveryEngine(runtime).execute_probes(
        probes,
        expected_record_id=expected.record_id,
        hidden_trigger=case.hidden_trigger,
        per_probe_k=2,
    )
    assert result.expected_record_retrieved
    assert result.expected_trigger_materialized
    assert result.false_positive_expansions == 1
    assert result.expansion_precision == 0.5


def test_model_cursor_actions_filter_aggregate_and_reject_bad_arguments(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest(
        {"rows": [
            {"group": "a", "value": 2},
            {"group": "b", "value": 100},
            {"group": "a", "value": 4},
        ]},
        record_type=RecordType.DB_RESULT,
    )
    cursor = runtime.open_cursor(record.record_id)
    filtered = runtime.execute_cursor_action(CursorAction(
        cursor.cursor_id, CursorOperation.FILTER, {"filters": {"group": "a"}}
    ))
    assert filtered.success
    aggregate = runtime.execute_cursor_action(CursorAction(
        cursor.cursor_id, CursorOperation.AGGREGATE, {"field": "value"}
    ))
    assert aggregate.success
    assert aggregate.payload["mean"] == 3
    invalid = runtime.execute_cursor_action(CursorAction(
        cursor.cursor_id, CursorOperation.RANGE, {"start": 0}
    ))
    assert not invalid.success
    assert "KeyError" in invalid.error


def test_cursor_action_scope_closure_and_expiry(tmp_path):
    runtime = _runtime(tmp_path)
    record = runtime.ingest({"rows": [{"x": 1}, {"x": 2}]}, record_type="db_result")
    cursor = runtime.open_cursor(record.record_id)
    denied = runtime.execute_cursor_action(
        CursorAction(cursor.cursor_id, "next"), scope=RecordScope("other", "session")
    )
    assert not denied.success
    assert runtime.execute_cursor_action(CursorAction(cursor.cursor_id, "close")).success
    assert not runtime.execute_cursor_action(CursorAction(cursor.cursor_id, "next")).success

    cursor = runtime.open_cursor(record.record_id)
    runtime.cursors._cursors[cursor.cursor_id] = replace(cursor, expires_at=0)
    assert not runtime.execute_cursor_action(CursorAction(cursor.cursor_id, "next")).success


def test_cursor_parser_pins_identity_and_allowed_operations():
    action = parse_cursor_action(
        {"cursor_id": "attacker", "operation": "search", "arguments": {"query": "x"}},
        cursor_id="authorized",
        allowed_operations=("search",),
    )
    assert action.cursor_id == "authorized"
    with pytest.raises(ValueError, match="not allowed"):
        parse_cursor_action(
            {"operation": "close"}, cursor_id="authorized", allowed_operations=("search",)
        )


def test_transport_policy_thresholds_and_full_context_accounting(tmp_path):
    runtime = AdaptiveContextRuntime(
        RecordScope("tenant", "session"),
        ContextPolicy(
            local_store=tmp_path,
            topology=DeploymentTopology.REMOTE_MODEL,
            storage=StoragePolicy.ADAPTIVE,
            upfront_max_bytes=100,
            adaptive_reuse_max_bytes=1000,
        ),
    )
    small = runtime.ingest("x" * 50, record_type="generic_text")
    medium = runtime.ingest("x" * 500, record_type="generic_text", expected_reuse=0.9)
    large = runtime.ingest("x" * 2000, record_type="generic_text", expected_reuse=0.9)
    assert runtime.decisions[small.record_id].storage == StoragePolicy.UPFRONT
    assert runtime.decisions[medium.record_id].storage == StoragePolicy.UPFRONT
    assert runtime.decisions[large.record_id].storage == StoragePolicy.ON_DEMAND

    full = runtime.materialize(MaterializationEvent(large.record_id))
    assert full.payload == "x" * 2000
    assert full.payload_bytes == large.backing.size_bytes
    assert full.network_bytes == large.backing.size_bytes
