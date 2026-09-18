from types import SimpleNamespace

from experiments.paper8_5_agent_memory import run_persistent_first_divergence_oracle as oracle
from experiments.paper8_5_agent_memory.run_persistent_first_divergence_oracle import (
    _causal_group_addback_batches,
    _completed_epoch_addback_batches,
    _full_control_config,
)
from experiments.paper8_5_agent_memory.autonomous_proxy import (
    AutonomousSelectionConfig,
)
from experiments.paper8_5_agent_memory.materialization import MaterializationMode
from experiments.paper8_5_agent_memory.multi_issue_session import BoundaryMode
from experiments.paper8_5_agent_memory.negative_receipts import NegativeRealizationMode


def test_generation_row_preserves_content_and_finish_reason(monkeypatch):
    monkeypatch.setattr(
        oracle,
        "_post",
        lambda endpoint, payload, timeout: {
            "choices": [{
                "message": {"content": "analysis without a command"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )
    transformation = SimpleNamespace(
        payload={"messages": []},
        trace={"selected_messages_sha256": "selected", "oracle_addback": None},
        plan=SimpleNamespace(selected_tokens=10),
        materialized_tokens=10,
    )

    row = oracle._generation_row(
        arm="candidate",
        repeat=1,
        transformation=transformation,
        endpoint="http://unused",
        timeout=1,
    )

    assert row["response_content"] == "analysis without a command"
    assert row["finish_reason"] == "length"
    assert row["action_valid"] is False


def test_full_control_clone_disables_candidate_only_retirement_options():
    candidate = AutonomousSelectionConfig(
        policy="persistent_instruction_epoch_retirement",
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        completed_finalization_turns=1,
        compact_completed_finalizations=True,
        retire_closed_instructions=True,
        materialization_mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
        negative_realization=NegativeRealizationMode.DROP,
        negative_fallback="none",
    )

    control = _full_control_config(candidate)

    assert control.policy == "full"
    assert control.budget_fraction == 1.0
    assert control.materialization_mode is MaterializationMode.WHOLE_RECORD
    assert control.negative_realization is NegativeRealizationMode.DROP
    assert control.negative_fallback == "none"
    assert control.compact_completed_finalizations is False
    assert control.retire_closed_instructions is False


def test_completed_epoch_batches_cover_each_exclusion_once():
    composed = {
        "episodes": [
            {"episode_index": 1, "model_visible_messages": 3},
            {"episode_index": 2, "model_visible_messages": 2},
        ]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(5)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000001", "record-000002"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000003", "record-000004"),
            excluded_tokens=13,
        ),
    ]

    assert _completed_epoch_addback_batches(
        composed=composed, history=history, exclusions=exclusions,
    ) == [
        {
            "epoch_index": 1,
            "causal_group_ids": ("turn-000000",),
            "excluded_tokens": 11,
        },
        {
            "epoch_index": 2,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        },
    ]


def test_causal_group_batches_can_be_limited_to_one_source_epoch():
    composed = {
        "episodes": [
            {"episode_index": 1, "model_visible_messages": 3},
            {"episode_index": 2, "model_visible_messages": 2},
        ]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(5)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000001", "record-000002"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000003", "record-000004"),
            excluded_tokens=13,
        ),
    ]

    assert _causal_group_addback_batches(
        composed=composed,
        history=history,
        exclusions=exclusions,
        addback_epoch=2,
    ) == [
        {
            "epoch_index": 2,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        }
    ]
