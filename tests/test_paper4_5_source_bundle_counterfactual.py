from __future__ import annotations

from experiments.paper4_5_agent.run_source_bundle_counterfactual import (
    SOURCE_BUNDLE_MESSAGE_INDICES,
    SPARSE_MESSAGE_INDICES,
    build_fixture,
)


def _request_with_task04_wording() -> dict[str, object]:
    messages = [
        {"role": "system" if index == 0 else "assistant" if index % 2 == 0 else "user",
         "content": f"message-{index}"}
        for index in range(12)
    ]
    messages[10]["content"] = "frozen Task04 request-5 reasoning from the sparse run"
    return {"logical_payload": {"messages": messages}}


def test_counterfactual_has_three_conditions_and_only_restores_source_bundle() -> None:
    fixture = build_fixture(_request_with_task04_wording())
    conditions = {row["name"]: row for row in fixture["conditions"]}

    assert set(conditions) == {
        "recorded_sparse_subset",
        "restore_only_source_bundle_4_5",
        "matched_full_history",
    }
    assert conditions["recorded_sparse_subset"]["selected_message_indices"] == list(
        SPARSE_MESSAGE_INDICES
    )
    assert conditions["restore_only_source_bundle_4_5"][
        "selected_message_indices"
    ] == list(range(12))
    assert not conditions["recorded_sparse_subset"]["source_bundle_included"]
    assert conditions["restore_only_source_bundle_4_5"]["source_bundle_included"]
    assert set(range(12)) - set(SPARSE_MESSAGE_INDICES) == set(
        SOURCE_BUNDLE_MESSAGE_INDICES
    )


def test_task04_request5_wording_is_identical_in_restored_and_full_controls() -> None:
    fixture = build_fixture(_request_with_task04_wording())
    conditions = {row["name"]: row for row in fixture["conditions"]}
    restored = conditions["restore_only_source_bundle_4_5"]
    full = conditions["matched_full_history"]

    assert restored["prompt_sha256"] == full["prompt_sha256"]
    assert restored["messages"][10]["content"] == (
        "frozen Task04 request-5 reasoning from the sparse run"
    )
    assert restored["messages"][10] == full["messages"][10]
    assert fixture["request5_action_message_sha256"] == fixture[
        "logical_message_manifest"
    ][10]["content_sha256"]
