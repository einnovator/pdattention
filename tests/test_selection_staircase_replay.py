from experiments.paper4_5_agent.run_selection_staircase_replay import (
    ablate_physical_payload,
    forced_old_causal_bundle,
    oldest_eligible_causal_bundles,
)


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "inspect old"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "inspect middle"},
        {"role": "user", "content": "middle observation"},
        {"role": "assistant", "content": "inspect recent"},
        {"role": "user", "content": "recent observation"},
        {"role": "assistant", "content": "current action"},
        {"role": "user", "content": "current observation"},
    ]


def _request_row():
    messages = _messages()
    resources = []
    for index in range(1, 8):
        resources.append({
            "resource_id": f"m{index}-0-{messages[index]['role']}",
            "text": messages[index]["content"],
            "metadata": {
                "message_index": index,
                "segment_index": 0,
                "role": messages[index]["role"],
            },
        })
    return {
        "logical_payload": {"messages": messages},
        "physical_payload": {
            "messages": [messages[0], messages[8], messages[9]],
            "pra": {
                "resources": resources,
                "budget": {"max_resources": len(resources), "max_selected_tokens": 99},
                "metadata": {"selection_complete": True},
            },
        },
    }


def test_staircase_omits_oldest_bundle_but_never_task_or_recent_tail():
    assert oldest_eligible_causal_bundles(_messages(), 1) == [[2, 3]]
    physical, audit = ablate_physical_payload(
        _request_row(), max_omitted_bundles=1,
    )
    retained = {row["metadata"]["message_index"] for row in physical["pra"]["resources"]}
    assert retained == {1, 4, 5, 6, 7}
    assert audit["omitted_message_indices"] == [2, 3]
    assert physical["pra"]["metadata"]["selection_complete"] is False
    assert physical["pra"]["metadata"]["selection_ablation"] == (
        "oldest_complete_causal_bundles_v1"
    )


def test_zero_step_is_exact_full_selection():
    physical, audit = ablate_physical_payload(
        _request_row(), max_omitted_bundles=0,
    )
    assert len(physical["pra"]["resources"]) == 7
    assert audit["omitted_message_indices"] == []
    assert physical["pra"]["metadata"]["selection_complete"] is True


def test_forced_bundle_can_ablate_early_progress_after_it_leaves_recent_tail():
    assert forced_old_causal_bundle(_messages(), 2) == [[2, 3]]
    physical, audit = ablate_physical_payload(
        _request_row(), max_omitted_bundles=0, forced_bundle_start=2,
    )
    retained = {row["metadata"]["message_index"] for row in physical["pra"]["resources"]}
    assert retained == {1, 4, 5, 6, 7}
    assert audit["forced_bundle_start"] == 2
    assert physical["pra"]["metadata"]["selection_ablation"] == (
        "forced_old_complete_causal_bundle_v1"
    )
