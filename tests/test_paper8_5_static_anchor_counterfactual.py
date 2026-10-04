import copy

import pytest

from experiments.paper8_5_agent_memory.autonomous_proxy import STATIC_WORKFLOW_ANCHORS
from experiments.paper8_5_agent_memory.run_static_anchor_counterfactual import (
    replace_static_anchor,
)


def request(anchor: str = "coding_bounded_evidence_v2"):
    return {
        "model": "locked-model",
        "messages": [
            {
                "role": "system",
                "content": "Use one action.\n\n" + STATIC_WORKFLOW_ANCHORS[anchor],
            },
            {"role": "user", "content": "Fix it."},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 0,
    }


def test_replace_static_anchor_changes_only_the_declared_system_substring():
    source = request()
    original = copy.deepcopy(source)
    transformed = replace_static_anchor(
        source,
        source_anchor="coding_bounded_evidence_v2",
        candidate_anchor="coding_compact_state_machine_v4",
    )

    assert source == original
    assert transformed["messages"][1:] == original["messages"][1:]
    assert STATIC_WORKFLOW_ANCHORS["coding_bounded_evidence_v2"] not in transformed[
        "messages"
    ][0]["content"]
    assert STATIC_WORKFLOW_ANCHORS["coding_compact_state_machine_v4"] in transformed[
        "messages"
    ][0]["content"]
    for key in ("model", "temperature", "top_p", "top_k", "seed"):
        assert transformed[key] == original[key]


def test_replace_static_anchor_fails_closed_on_ambiguous_or_missing_source():
    duplicated = request()
    duplicated["messages"][0]["content"] += (
        "\n" + STATIC_WORKFLOW_ANCHORS["coding_bounded_evidence_v2"]
    )
    with pytest.raises(ValueError, match="more than once"):
        replace_static_anchor(
            duplicated,
            source_anchor="coding_bounded_evidence_v2",
            candidate_anchor="coding_compact_state_machine_v4",
        )
    with pytest.raises(ValueError, match="found 0"):
        replace_static_anchor(
            request("coding_guarded_edit_v3"),
            source_anchor="coding_bounded_evidence_v2",
            candidate_anchor="coding_compact_state_machine_v4",
        )


def test_replace_static_anchor_can_move_candidate_to_declared_user_message():
    source = request()
    transformed = replace_static_anchor(
        source,
        source_anchor="coding_bounded_evidence_v2",
        candidate_anchor="coding_minimal_transaction_v7",
        candidate_message_index=1,
    )

    assert STATIC_WORKFLOW_ANCHORS["coding_bounded_evidence_v2"] not in transformed[
        "messages"
    ][0]["content"]
    assert STATIC_WORKFLOW_ANCHORS["coding_minimal_transaction_v7"] not in transformed[
        "messages"
    ][0]["content"]
    assert transformed["messages"][1]["content"].startswith("Fix it.")
    assert STATIC_WORKFLOW_ANCHORS["coding_minimal_transaction_v7"] in transformed[
        "messages"
    ][1]["content"]

    with pytest.raises(ValueError, match="user message"):
        replace_static_anchor(
            source,
            source_anchor="coding_bounded_evidence_v2",
            candidate_anchor="coding_minimal_transaction_v7",
            candidate_message_index=0,
        )
