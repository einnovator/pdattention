"""Contracts for deterministic two-phase capability disclosure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from data.large_tool_schemas import LARGE_SCHEMA_CALLABLES
from pra_hf.context_records import RecordAtomicity, RecordViewName, serialize_record, tool_definition_record
from pra_hf.progressive_disclosure import (
    CapabilityTransition,
    capability_choice_accounting,
    disclosure_cost,
    native_kv_cost,
    bounded_candidate_ids,
    minimum_candidate_budget,
    materialize_capability_views,
    transition_selected_capability,
)
from pra_hf.tool_records import tool_record_from_callable


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class RoutingRule:
    pattern: str
    priority: Priority


def configure_router(router_id: str, rule: RoutingRule, enabled: bool = True) -> dict[str, object]:
    """Configure a router with one nested typed routing rule.

    Args:
        router_id: Stable router identifier.
        rule: Nested match pattern and priority configuration.
        enabled: Whether the rule should be active immediately.

    Returns:
        Persisted router configuration.
    """

    return {"router_id": router_id, "enabled": enabled}


def _records():
    first = tool_record_from_callable(configure_router, namespace="test").to_agent_resource()

    def inspect_router(router_id: str) -> dict[str, object]:
        """Inspect one router configuration."""

        return {"router_id": router_id}

    second = tool_record_from_callable(inspect_router, namespace="test").to_agent_resource()
    return tuple(tool_definition_record(resource) for resource in (first, second))


def test_tool_selection_and_full_views_have_stable_typed_boundaries() -> None:
    record = _records()[0]
    selection = record.materialize("selection", token_counter=lambda text: len(text.split()))
    full = record.materialize("full", token_counter=lambda text: len(text.split()))

    assert selection.fields == ("name", "signature", "description", "side_effect")
    assert "parameters" not in selection.payload
    assert selection.payload.startswith("configure_router(")
    assert "schema" in full.payload
    assert full.payload["schema"]["function"]["parameters"]["properties"]["rule"]
    assert '"view": "selection"' in serialize_record(record, view="selection")
    assert '"view": "full"' in serialize_record(record, view="full")
    assert selection.token_count < full.token_count


def test_selected_full_record_is_atomic_and_non_selected_full_schema_is_absent() -> None:
    records = _records()
    phase_a = materialize_capability_views(
        records, view="selection", phase="A", token_counter=lambda text: len(text.split())
    )
    phase_b = transition_selected_capability(
        records,
        CapabilityTransition(records[0].record_id),
        token_counter=lambda text: len(text.split()),
    )

    assert phase_a.record_ids == tuple(record.record_id for record in records)
    assert phase_b.record_ids == (records[0].record_id,)
    assert records[1].record_id not in phase_b.serialized_payload
    assert records[0].policy.atomicity == RecordAtomicity.RECORD
    assert phase_a.atomicity_violations == phase_b.atomicity_violations == 0


def test_capability_transition_and_view_policy_reject_invalid_routes() -> None:
    records = _records()
    with pytest.raises(ValueError):
        CapabilityTransition(records[0].record_id, from_view="full", to_view="selection")
    with pytest.raises(ValueError):
        transition_selected_capability(
            records,
            CapabilityTransition("missing"),
            token_counter=lambda text: len(text.split()),
        )


def test_disclosure_cost_accounts_for_each_named_view() -> None:
    records = _records()
    cost = disclosure_cost(
        records,
        selected_record_id=records[0].record_id,
        token_counter=lambda text: len(text.split()),
    )

    assert cost.candidate_count == 2
    assert cost.phase_a_selection_tokens < cost.all_candidate_full_tokens
    assert cost.phase_b_selected_full_tokens < cost.all_candidate_full_tokens
    assert cost.progressive_tokens == (
        cost.phase_a_selection_tokens + cost.phase_b_selected_full_tokens
    )
    assert cost.tokens_saved == cost.all_candidate_full_tokens - cost.progressive_tokens
    assert cost.token_savings_fraction == pytest.approx(
        1 - cost.progressive_tokens / cost.all_candidate_full_tokens
    )
    assert 0 < cost.disclosure_ratio < 1
    assert cost.full_candidate_tokens_avoided > 0

    kv = native_kv_cost(cost, native_kv_bytes_per_token=256)
    assert kv.full_all_active_bytes == cost.all_candidate_full_tokens * 256
    assert kv.progressive_active_bytes == cost.progressive_tokens * 256
    assert kv.bytes_saved == kv.full_all_active_bytes - kv.progressive_active_bytes
    assert kv.savings_fraction == pytest.approx(cost.token_savings_fraction)


def test_native_kv_cost_rejects_negative_bytes_per_token() -> None:
    cost = disclosure_cost(
        _records(),
        selected_record_id=_records()[0].record_id,
        token_counter=lambda text: len(text.split()),
    )
    with pytest.raises(ValueError):
        native_kv_cost(cost, native_kv_bytes_per_token=-1)


def test_capability_choice_accounting_separates_retrieval_from_model_choice() -> None:
    value = capability_choice_accounting(((True, True), (True, False), (False, False)))

    assert value.examples == 3
    assert value.retrieval_recall == pytest.approx(2 / 3)
    assert value.conditional_choice_accuracy == pytest.approx(1 / 2)
    assert value.end_to_end_choice_accuracy == pytest.approx(1 / 3)


def test_capability_choice_accounting_rejects_impossible_correct_choice() -> None:
    with pytest.raises(ValueError):
        capability_choice_accounting(((False, True),))


def test_candidate_budget_cap_and_k_min_are_exact_and_deterministic() -> None:
    ranked = ("a", "b", "b", "target", "c")

    assert bounded_candidate_ids(ranked, 3) == ("a", "b", "target")
    assert minimum_candidate_budget("target", ranked, (1, 2, 4, 8)) == 4
    assert minimum_candidate_budget("missing", ranked, (1, 2, 4, 8)) is None
    with pytest.raises(ValueError):
        bounded_candidate_ids(ranked, 0)
    with pytest.raises(ValueError):
        minimum_candidate_budget("target", ranked, (0, 4))


def test_large_nested_callable_schemas_remain_atomic_and_executable() -> None:
    records = [
        tool_definition_record(
            tool_record_from_callable(function, namespace="stress").to_agent_resource()
        )
        for function in LARGE_SCHEMA_CALLABLES
    ]

    for record in records:
        schema = record.materialize("full").payload["schema"]["function"]["parameters"]
        assert record.policy.atomicity == RecordAtomicity.RECORD
        assert schema["properties"]
        assert any("properties" in str(value) for value in schema["properties"].values())
        assert record.materialize("selection").payload != record.materialize("full").payload
